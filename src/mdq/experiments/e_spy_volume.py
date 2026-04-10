"""Experiment E: volume-confirmed SPY rules with train/OOS split.

Pipeline:
  1. Load SPY 1-min bars
  2. Split into train (2023-01-03 to 2025-12-31) and OOS (2026-01-01+)
  3. For each session, load prior-session volume profile, evaluate all rules
  4. Measure forward reactions with ATR-based target/stop geometry
  5. Run a grid search over target_mult x stop_mult (fractions of ATR-20)
  6. Compare vs matched random-bar baseline
  7. Report: train edge, OOS edge, n_train, n_oos, year splits

This is the most rigorous study we've done — train/test split, OOS holdout,
multi-rule family, explicit baseline for every cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.levels.volume_profile import compute_all_profiles
from mdq.levels.volume_rules import evaluate_all_rules
from mdq.stats.atr import compute_atr


TARGET_MULTS = (0.3, 0.5, 0.75, 1.0, 1.5)
STOP_MULTS   = (0.5, 0.75, 1.0, 1.5)
HORIZON_BARS = 15
MIN_ATR_BARS_PRIOR = 20  # need at least this many bars of history to compute ATR
WINDOW = ("09:30", "11:30")  # fade rules
WINDOW_MOMENTUM = ("09:30", "15:55")  # momentum rule


def _load_split(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bars_with_ts, daily_profile_cache)."""
    bars = load_bars("SPY", start, end)
    profiles = compute_all_profiles(bars, bin_size=0.05)
    return bars, profiles


def _levels_for_session(profiles: pd.DataFrame, session_date) -> list[tuple[str, float]] | None:
    prior = profiles[profiles["session_date"] < session_date]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    named = []
    for lname in ("poc", "vah", "val", "high", "low", "close"):
        v = row.get(lname)
        if v is not None and not pd.isna(v):
            named.append((f"prior_{lname}" if lname in ("high", "low", "close") else lname, float(v)))
    return named


def collect_events(
    bars: pd.DataFrame,
    profiles: pd.DataFrame,
    window: tuple[str, str],
) -> pd.DataFrame:
    """Run all 4 rules across all sessions, return combined event frame."""
    bars_sorted = bars.sort_values(["session_date", "t"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for sd, session in bars_sorted.groupby("session_date", sort=True):
        named = _levels_for_session(profiles, sd)
        if not named:
            continue
        session_win = filter_window(session, window[0], window[1]).reset_index(drop=True)
        if session_win.empty:
            continue
        events = evaluate_all_rules(session_win, named)
        if not events.empty:
            frames.append(events)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def measure_events_with_atr(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int = HORIZON_BARS,
) -> pd.DataFrame:
    """For each event, attach entry ATR (used for sizing) and forward bars."""
    if events.empty:
        return events.copy()

    bars_sorted = bars.sort_values(["session_date", "t"]).reset_index(drop=True)

    # Compute ATR per session on ALL bars (not window-filtered), so rules that
    # entered during the window still see correct ATR context
    out_events = events.copy()
    out_events["atr"] = np.nan
    out_events["event_idx_in_session"] = np.nan

    sess_cache: dict = {}
    for sd, sess in bars_sorted.groupby("session_date", sort=True):
        sess = sess.reset_index(drop=True)
        atr = compute_atr(sess, window=20)
        sess_cache[sd] = {
            "bars": sess,
            "atr": atr,
            "t_to_idx": {int(t): i for i, t in enumerate(sess["t"].to_numpy())},
        }

    for i, row in out_events.iterrows():
        cache = sess_cache.get(row["session_date"])
        if cache is None:
            continue
        idx = cache["t_to_idx"].get(int(row["t"]))
        if idx is None:
            continue
        atr_val = cache["atr"][idx] if not np.isnan(cache["atr"][idx]) else np.nan
        out_events.at[i, "atr"] = atr_val
        out_events.at[i, "event_idx_in_session"] = idx

    return out_events


def first_passage_grid(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    target_mult: float,
    stop_mult: float,
    horizon: int = HORIZON_BARS,
) -> pd.Series:
    """First-passage target/stop race. Returns outcome per event."""
    if events.empty:
        return pd.Series(dtype=object)

    bars_sorted = bars.sort_values(["session_date", "t"]).reset_index(drop=True)
    sess_cache: dict = {}
    for sd, sess in bars_sorted.groupby("session_date", sort=True):
        sess = sess.reset_index(drop=True)
        sess_cache[sd] = {
            "h": sess["h"].to_numpy(),
            "l": sess["l"].to_numpy(),
            "c": sess["c"].to_numpy(),
            "t_to_idx": {int(t): i for i, t in enumerate(sess["t"].to_numpy())},
            "n": len(sess),
        }

    outcomes: list[str] = []
    for _, row in events.iterrows():
        cache = sess_cache.get(row["session_date"])
        if cache is None:
            outcomes.append("no_data")
            continue
        idx = cache["t_to_idx"].get(int(row["t"]))
        if idx is None:
            outcomes.append("no_data")
            continue
        atr_val = row.get("atr")
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            outcomes.append("no_atr")
            continue
        entry_price = float(row["entry_price"])
        target = atr_val * target_mult
        stop = atr_val * stop_mult
        direction = row["direction"]

        start = idx + 1
        end = min(start + horizon, cache["n"])
        if start >= end:
            outcomes.append("no_forward")
            continue

        if direction == "short":
            target_px = entry_price - target
            stop_px = entry_price + stop
        else:  # long
            target_px = entry_price + target
            stop_px = entry_price - stop

        outcome = "timeout"
        for i in range(start, end):
            hit_stop = (
                cache["h"][i] >= stop_px if direction == "short"
                else cache["l"][i] <= stop_px
            )
            hit_target = (
                cache["l"][i] <= target_px if direction == "short"
                else cache["h"][i] >= target_px
            )
            if hit_stop:
                outcome = "stop"
                break
            if hit_target:
                outcome = "target"
                break
        outcomes.append(outcome)

    return pd.Series(outcomes, index=events.index)


def compute_expectancy_atr(
    outcomes: pd.Series,
    target_mult: float,
    stop_mult: float,
) -> dict:
    valid = outcomes[~outcomes.isin(["no_data", "no_atr", "no_forward"])]
    if valid.empty:
        return {"n": 0, "hit": np.nan, "stop": np.nan,
                "timeout": np.nan, "expectancy_atr": np.nan}
    n = len(valid)
    n_t = (valid == "target").sum()
    n_s = (valid == "stop").sum()
    n_to = (valid == "timeout").sum()
    exp = (n_t * target_mult - n_s * stop_mult) / n
    return {
        "n": n,
        "hit": n_t / n,
        "stop": n_s / n,
        "timeout": n_to / n,
        "expectancy_atr": exp,
    }


def make_pseudo_events(
    bars: pd.DataFrame,
    window: tuple[str, str],
    every_nth: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Random pseudo-events for baseline comparison.

    Samples every Nth bar in the window, assigns random direction, same
    entry_price = bar close.
    """
    bars_win = filter_window(bars, window[0], window[1]).reset_index(drop=True)
    picks = bars_win.iloc[::every_nth].copy().reset_index(drop=True)
    if picks.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    picks["direction"] = rng.choice(["short", "long"], size=len(picks))
    picks["level_name"] = "baseline"
    picks["level"] = picks["c"]
    picks["rule"] = "baseline"
    picks["approach"] = "at"
    picks["entry_ts"] = picks["ts_et"]
    picks["entry_price"] = picks["c"]
    picks["bar_idx"] = picks.index
    return picks[[
        "session_date", "ts_et", "t", "bar_idx", "level", "level_name",
        "approach", "rule", "direction", "entry_ts", "entry_price",
    ]]
