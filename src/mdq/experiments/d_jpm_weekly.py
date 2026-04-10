"""Experiment D: JPM daily-bar touches of prior-week volume levels.

Pipeline:
  1. Load JPM 1-min bars -> RTH -> group into weekly profiles
  2. For each trading day, detect touches of the PRIOR week's levels
     (poc, vah, val, high, low, close, hvn_1, hvn_2, lvn_1, lvn_2)
  3. Measure forward reactions at 1, 3, 5 trading days
  4. Compute random-bar baseline for comparison
  5. Run grid search over fade vs momentum x target/stop % combinations
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth
from mdq.levels.weekly_profile import compute_all_weekly_profiles


def bars_to_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min RTH bars into daily OHLCV."""
    rth = filter_rth(bars)
    if rth.empty:
        return pd.DataFrame()
    rth = rth.copy()
    rth["session_date"] = pd.to_datetime(rth["session_date"]).dt.date

    grouped = rth.groupby("session_date").agg(
        o=("o", "first"),
        h=("h", "max"),
        l=("l", "min"),
        c=("c", "last"),
        v=("v", "sum"),
    ).reset_index()
    return grouped.sort_values("session_date").reset_index(drop=True)


def _empty_touches() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "session_date", "level_name", "level", "approach", "touch_num",
        "open", "high", "low", "close", "prev_close",
    ])


def detect_daily_touches(
    daily_bars: pd.DataFrame,
    weekly_profiles: pd.DataFrame,
    tolerance_pct: float = 0.001,  # 0.1% of price
) -> pd.DataFrame:
    """Detect daily touches of prior-week levels.

    For each trading day, uses the most recent completed week's profile (from
    strictly before that day) as the source of levels. Touches are tracked per
    (level, week) with touch_num incrementing across days of that week.
    """
    if daily_bars.empty or weekly_profiles.empty:
        return _empty_touches()

    profiles = weekly_profiles.copy()
    profiles["week_end"] = pd.to_datetime(profiles["week_end"]).dt.date
    profiles = profiles.sort_values("week_end").reset_index(drop=True)

    level_cols = ["poc", "vah", "val", "high", "low", "close",
                  "hvn_1", "hvn_2", "lvn_1", "lvn_2"]

    rows: list[dict] = []
    # Track, per (prior_week_end, level_name), touches consumed so far
    # (so touch_num increments correctly within one week of usage)
    touch_state: dict[tuple, int] = {}

    prev_close: float | None = None

    for _, bar in daily_bars.iterrows():
        sd = bar["session_date"]
        # Find most recent profile whose week_end < sd
        valid = profiles[profiles["week_end"] < sd]
        if valid.empty:
            prev_close = bar["c"]
            continue
        prof = valid.iloc[-1]
        prior_week_end = prof["week_end"]

        high = bar["h"]
        low = bar["l"]
        open_px = bar["o"]
        close_px = bar["c"]
        tol = max(0.05, open_px * tolerance_pct)  # at least $0.05

        for lname in level_cols:
            lvl = prof[lname]
            if lvl is None or pd.isna(lvl):
                continue
            # Did this daily bar touch the level?
            if low <= lvl + tol and high >= lvl - tol:
                key = (prior_week_end, lname)
                touch_num = touch_state.get(key, 0) + 1
                touch_state[key] = touch_num

                ref = prev_close if prev_close is not None else open_px
                if ref < lvl:
                    approach = "from_below"
                elif ref > lvl:
                    approach = "from_above"
                else:
                    approach = "at"

                rows.append({
                    "session_date": sd,
                    "level_name": lname,
                    "level": float(lvl),
                    "approach": approach,
                    "touch_num": touch_num,
                    "open": float(open_px),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close_px),
                    "prev_close": float(ref),
                    "prior_week_end": prior_week_end,
                })

        prev_close = close_px

    return pd.DataFrame(rows) if rows else _empty_touches()


def measure_daily_reactions(
    daily_bars: pd.DataFrame,
    touches: pd.DataFrame,
    horizons_days: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Compute forward N-day MFE/MAE in percent terms from each touch.

    Entry is NEXT day's open (after the touch day), so the reaction measurement
    starts on session_date + 1 trading day.
    """
    if touches.empty:
        return touches.copy()

    daily_bars = daily_bars.sort_values("session_date").reset_index(drop=True)
    daily_bars["session_date"] = pd.to_datetime(daily_bars["session_date"]).dt.date
    idx_map = {sd: i for i, sd in enumerate(daily_bars["session_date"])}

    out = touches.copy()
    max_h = max(horizons_days)

    for h in horizons_days:
        out[f"mfe_up_{h}d"] = np.nan   # max up-move from entry as %
        out[f"mfe_dn_{h}d"] = np.nan   # max down-move as %
        out[f"close_{h}d"] = np.nan    # close-to-entry return as %

    out["entry_price"] = np.nan

    highs = daily_bars["h"].to_numpy()
    lows = daily_bars["l"].to_numpy()
    closes = daily_bars["c"].to_numpy()
    opens = daily_bars["o"].to_numpy()

    for i, row in out.iterrows():
        td_idx = idx_map.get(row["session_date"])
        if td_idx is None:
            continue
        entry_idx = td_idx + 1
        if entry_idx >= len(daily_bars):
            continue
        entry_price = opens[entry_idx]
        out.at[i, "entry_price"] = entry_price

        for h in horizons_days:
            end = min(entry_idx + h, len(daily_bars))
            if end <= entry_idx:
                continue
            window_highs = highs[entry_idx:end]
            window_lows = lows[entry_idx:end]
            mfe_up = (window_highs.max() - entry_price) / entry_price
            mfe_dn = (entry_price - window_lows.min()) / entry_price
            close_ret = (closes[end - 1] - entry_price) / entry_price
            out.at[i, f"mfe_up_{h}d"] = mfe_up
            out.at[i, f"mfe_dn_{h}d"] = mfe_dn
            out.at[i, f"close_{h}d"] = close_ret

    return out


def first_passage_daily(
    daily_bars: pd.DataFrame,
    reactions: pd.DataFrame,
    target_pct: float,
    stop_pct: float,
    horizon_days: int,
    direction: str,  # 'fade' or 'momentum'
) -> pd.Series:
    """First-passage target/stop race on daily bars.

    direction='fade'     — resistance touch => short, support touch => long
    direction='momentum' — resistance touch => long,  support touch => short
    """
    if reactions.empty:
        return pd.Series(dtype=object)

    daily_bars = daily_bars.sort_values("session_date").reset_index(drop=True)
    daily_bars["session_date"] = pd.to_datetime(daily_bars["session_date"]).dt.date
    idx_map = {sd: i for i, sd in enumerate(daily_bars["session_date"])}

    highs = daily_bars["h"].to_numpy()
    lows = daily_bars["l"].to_numpy()
    opens = daily_bars["o"].to_numpy()

    outcomes: list[str] = []

    for _, row in reactions.iterrows():
        td_idx = idx_map.get(row["session_date"])
        if td_idx is None:
            outcomes.append("no_data")
            continue
        entry_idx = td_idx + 1
        if entry_idx >= len(daily_bars):
            outcomes.append("no_data")
            continue
        entry_price = opens[entry_idx]
        if np.isnan(entry_price):
            outcomes.append("no_data")
            continue

        is_short = (
            (direction == "fade" and row["approach"] == "from_below")
            or (direction == "momentum" and row["approach"] == "from_above")
        )
        is_long = (
            (direction == "fade" and row["approach"] == "from_above")
            or (direction == "momentum" and row["approach"] == "from_below")
        )
        if not (is_short or is_long):
            outcomes.append("skip")
            continue

        if is_short:
            target_px = entry_price * (1 - target_pct)
            stop_px = entry_price * (1 + stop_pct)
        else:
            target_px = entry_price * (1 + target_pct)
            stop_px = entry_price * (1 - stop_pct)

        end = min(entry_idx + horizon_days, len(daily_bars))
        outcome = "timeout"
        for i in range(entry_idx, end):
            h = highs[i]
            l = lows[i]
            if is_short:
                hit_stop = h >= stop_px
                hit_target = l <= target_px
            else:
                hit_stop = l <= stop_px
                hit_target = h >= target_px
            if hit_stop and hit_target:
                outcome = "stop"  # conservative: both hit same bar => stop wins
                break
            if hit_stop:
                outcome = "stop"
                break
            if hit_target:
                outcome = "target"
                break
        outcomes.append(outcome)

    return pd.Series(outcomes, index=reactions.index)


@dataclass
class ExperimentDResult:
    profiles: pd.DataFrame
    daily: pd.DataFrame
    touches: pd.DataFrame
    reactions: pd.DataFrame


def run_experiment_d(start: str, end: str) -> ExperimentDResult:
    bars = load_bars("JPM", start, end)
    profiles = compute_all_weekly_profiles(bars, bin_size=0.10)
    daily = bars_to_daily(bars)
    touches = detect_daily_touches(daily, profiles, tolerance_pct=0.001)
    reactions = measure_daily_reactions(daily, touches, horizons_days=(1, 3, 5))
    return ExperimentDResult(profiles, daily, touches, reactions)
