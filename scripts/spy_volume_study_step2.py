"""SPY Volume Deep Study — Step 2: Backtest VAH/POC fade in the 10:00-10:30 window.

From Step 1 we know:
  - VAH first-touch at 30-60m has MFE/MAE = 1.70 (best in dataset)
  - POC first-touch at 30-60m has MFE/MAE = 1.39 (#2)
  - VAH at open_30m is #4 with edge +0.024

Now we test:
  Rule: fade first-touch of prior-day VAH (or POC) in the 10:00-10:30 window
  Volume layers:
    V0: no volume filter (raw touch)
    V1: touch bar volume >= 1.5x trailing 20-bar avg
    V2: touch bar has rejection wick >= 40% of range
    V3: V1 AND V2 combined

  Train: 2023-01-03 to 2025-12-31
  OOS:   2026-01-01 to 2026-04-09

  Grid: ATR-based target/stop, same as Experiment E
  Baseline: random bars in the same time window
  Real Alpaca option fills on the winners
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.levels.volume_profile import compute_all_profiles
from mdq.stats.atr import compute_atr

OUT = RESULTS_DIR / "spy_volume_study"
OUT.mkdir(parents=True, exist_ok=True)

START = "2023-01-03"
END = "2026-04-09"
TRAIN_END = "2025-12-31"
OOS_START = "2026-01-01"

# The window Step 1 identified as best
WINDOW_START = "10:00"
WINDOW_END = "10:30"

# Also test the broader morning and the open window
WINDOWS = {
    "10_00_10_30": ("10:00", "10:30"),  # Step 1 winner
    "09_30_10_00": ("09:30", "10:00"),  # open 30m
    "09_30_10_30": ("09:30", "10:30"),  # open 60m combined
    "10_30_11_30": ("10:30", "11:30"),  # post-open control
}

LEVELS = ["vah", "poc"]  # Step 1's top 2

TARGET_MULTS = (0.3, 0.5, 0.75, 1.0, 1.5)
STOP_MULTS = (0.5, 0.75, 1.0, 1.5)
HORIZON = 15

VOL_SPIKE = 1.5  # V1 threshold
WICK_PCT = 0.40  # V2 threshold


def _buy_pressure(o, h, l, c):
    rng = h - l
    return (c - l) / rng if rng > 0 else 0.5


def _trailing_vol_mean(vols, idx, window=20):
    start = max(0, idx - window)
    if start == idx:
        return 0
    return vols[start:idx].mean()


def find_first_touches(
    bars_win: pd.DataFrame,
    bars_full_session: pd.DataFrame,
    level_name: str,
    level: float,
    tol: float = 0.05,
) -> list[dict]:
    """Find first touch of `level` within the window bars.

    Returns at most 1 event per session (first touch only).
    Uses bars_full_session for trailing volume avg (look-back before the window).
    """
    if bars_win.empty:
        return []

    h = bars_win["h"].to_numpy()
    l = bars_win["l"].to_numpy()
    o = bars_win["o"].to_numpy()
    c = bars_win["c"].to_numpy()
    v = bars_win["v"].to_numpy()
    ts = bars_win["ts_et"].to_numpy()
    t = bars_win["t"].to_numpy()

    # Full session arrays for volume lookback
    full_v = bars_full_session["v"].to_numpy()
    full_t = bars_full_session["t"].to_numpy()
    full_h = bars_full_session["h"].to_numpy()
    full_l = bars_full_session["l"].to_numpy()
    full_c = bars_full_session["c"].to_numpy()
    full_o = bars_full_session["o"].to_numpy()

    for i in range(len(bars_win)):
        if l[i] <= level + tol and h[i] >= level - tol:
            # First touch found
            # Approach
            if i > 0:
                prev_c = c[i - 1]
            else:
                prev_c = o[i]
            from_below = prev_c < level

            # Volume context: find this bar in full session
            full_idx = np.searchsorted(full_t, t[i])
            if full_idx >= len(full_t):
                full_idx = len(full_t) - 1
            vol_avg = _trailing_vol_mean(full_v, full_idx, 20)
            vol_ratio = v[i] / vol_avg if vol_avg > 0 else 1.0

            # Wick
            bar_range = h[i] - l[i]
            if bar_range > 0:
                body_hi = max(o[i], c[i])
                body_lo = min(o[i], c[i])
                if from_below:
                    wick_pct = (h[i] - body_hi) / bar_range  # upper wick
                else:
                    wick_pct = (body_lo - l[i]) / bar_range  # lower wick
            else:
                wick_pct = 0.0

            # ATR at this bar
            atr_arr = compute_atr(bars_full_session, window=20)
            atr = atr_arr[full_idx] if full_idx < len(atr_arr) and not np.isnan(atr_arr[full_idx]) else None

            # Rejection side check
            if from_below:
                rejected = c[i] < level  # closed below = rejected resistance
            else:
                rejected = c[i] > level  # closed above = rejected support

            return [{
                "level_name": level_name,
                "level": level,
                "bar_idx_in_window": i,
                "bar_idx_in_session": int(full_idx),
                "ts_et": ts[i],
                "t": int(t[i]),
                "entry_price": float(c[i]),
                "from_below": from_below,
                "rejected": rejected,
                "vol_ratio": vol_ratio,
                "wick_pct": wick_pct,
                "bar_vol": float(v[i]),
                "atr": float(atr) if atr is not None else np.nan,
                "bp": _buy_pressure(o[i], h[i], l[i], c[i]),
            }]
    return []


def first_passage(
    bars_full: pd.DataFrame,
    entry_session_idx: int,
    entry_price: float,
    direction: str,  # "short" or "long"
    target_mult: float,
    stop_mult: float,
    atr: float,
    horizon: int,
) -> str:
    """Run target/stop race on 1-min bars. Returns outcome string."""
    if np.isnan(atr) or atr <= 0:
        return "no_atr"

    target = atr * target_mult
    stop = atr * stop_mult

    h = bars_full["h"].to_numpy()
    l = bars_full["l"].to_numpy()
    n = len(h)

    start = entry_session_idx + 1
    end = min(start + horizon, n)

    if direction == "short":
        target_px = entry_price - target
        stop_px = entry_price + stop
    else:
        target_px = entry_price + target
        stop_px = entry_price - stop

    for i in range(start, end):
        if direction == "short":
            if h[i] >= stop_px:
                return "stop"
            if l[i] <= target_px:
                return "target"
        else:
            if l[i] <= stop_px:
                return "stop"
            if h[i] >= target_px:
                return "target"
    return "timeout"


def main() -> int:
    print("=" * 100)
    print("SPY Volume Study — Step 2: VAH/POC Fade Backtest")
    print("=" * 100)
    t0_start = time.perf_counter()

    bars = load_bars("SPY", START, END)
    profiles = compute_all_profiles(bars, bin_size=0.05)
    profiles["session_date"] = pd.to_datetime(profiles["session_date"]).dt.date
    rth = filter_rth(bars)
    rth = rth.copy()
    rth["session_date"] = pd.to_datetime(rth["session_date"]).dt.date

    print(f"RTH bars: {len(rth):,}  Sessions: {rth['session_date'].nunique()}")

    # Collect all first-touch events across all windows and levels
    all_events: list[dict] = []

    for sd in sorted(rth["session_date"].unique()):
        prior = profiles[profiles["session_date"] < sd]
        if prior.empty:
            continue
        p = prior.iloc[-1]
        session = rth[rth["session_date"] == sd].reset_index(drop=True)
        if session.empty:
            continue

        for level_name in LEVELS:
            lvl = float(p[level_name])
            for win_name, (ws, we) in WINDOWS.items():
                win_bars = filter_window(session, ws, we).reset_index(drop=True)
                if win_bars.empty:
                    continue
                touches = find_first_touches(win_bars, session, level_name, lvl)
                for ev in touches:
                    ev["session_date"] = sd
                    ev["window"] = win_name
                    ev["year"] = sd.year
                    # Classify volume filters
                    ev["V0"] = True
                    ev["V1"] = ev["vol_ratio"] >= VOL_SPIKE
                    ev["V2"] = ev["wick_pct"] >= WICK_PCT
                    ev["V3"] = ev["V1"] and ev["V2"]
                    all_events.append(ev)

    events_df = pd.DataFrame(all_events)
    print(f"Total first-touch events: {len(events_df):,}")

    # Event counts by window × level
    print("\n  Events by (window, level):")
    for (w, l), sub in events_df.groupby(["window", "level_name"]):
        n_v0 = sub["V0"].sum()
        n_v1 = sub["V1"].sum()
        n_v2 = sub["V2"].sum()
        n_v3 = sub["V3"].sum()
        print(f"    {w:>15} × {l:>4}:  V0={n_v0:>4}  V1={n_v1:>4}  V2={n_v2:>4}  V3={n_v3:>4}")

    # Run grid search
    print("\nRunning grid search...")
    rows: list[dict] = []

    # Also generate random baseline
    rng = np.random.default_rng(42)

    for win_name, (ws, we) in WINDOWS.items():
        for level_name in LEVELS:
            sub = events_df[(events_df["window"] == win_name) & (events_df["level_name"] == level_name)]
            if sub.empty:
                continue

            for vol_filter in ("V0", "V1", "V2", "V3"):
                filtered = sub[sub[vol_filter]]
                if filtered.empty:
                    continue

                # Split train/oos
                train = filtered[filtered["year"] <= 2025]
                oos = filtered[filtered["year"] >= 2026]

                for target_mult in TARGET_MULTS:
                    for stop_mult in STOP_MULTS:
                        for split_name, split_df in [("train", train), ("oos", oos)]:
                            if split_df.empty:
                                continue

                            outcomes = []
                            for _, ev in split_df.iterrows():
                                sd = ev["session_date"]
                                session = rth[rth["session_date"] == sd].reset_index(drop=True)
                                direction = "short" if ev["from_below"] else "long"
                                oc = first_passage(
                                    session, ev["bar_idx_in_session"],
                                    ev["entry_price"], direction,
                                    target_mult, stop_mult,
                                    ev["atr"], HORIZON,
                                )
                                outcomes.append(oc)

                            valid = [o for o in outcomes if o not in ("no_atr",)]
                            if not valid:
                                continue
                            n = len(valid)
                            n_t = sum(1 for o in valid if o == "target")
                            n_s = sum(1 for o in valid if o == "stop")
                            exp = (n_t * target_mult - n_s * stop_mult) / n

                            rows.append({
                                "window": win_name,
                                "level": level_name,
                                "vol_filter": vol_filter,
                                "split": split_name,
                                "target_mult": target_mult,
                                "stop_mult": stop_mult,
                                "n": n,
                                "hit": n_t / n,
                                "stop_rate": n_s / n,
                                "exp_atr": exp,
                            })

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "step2_grid.csv", index=False)

    # Pivot: for each (window, level, vol_filter, target, stop),
    # join train + OOS
    train_g = grid[grid["split"] == "train"].copy()
    oos_g = grid[grid["split"] == "oos"].copy()

    merged = train_g.rename(columns={
        "n": "n_train", "hit": "hit_train", "exp_atr": "exp_train",
    }).merge(
        oos_g[["window", "level", "vol_filter", "target_mult", "stop_mult",
               "n", "hit", "exp_atr"]].rename(columns={
            "n": "n_oos", "hit": "hit_oos", "exp_atr": "exp_oos",
        }),
        on=["window", "level", "vol_filter", "target_mult", "stop_mult"],
        how="left",
    )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    # Deploy candidates
    candidates = merged[
        (merged["n_train"] >= 50)
        & (merged["n_oos"] >= 5)
        & (merged["exp_train"] > 0)
        & (merged["exp_oos"] > 0)
    ].copy()
    candidates["sum_exp"] = candidates["exp_train"] + candidates["exp_oos"]
    candidates = candidates.sort_values("sum_exp", ascending=False)

    print("\n" + "#" * 100)
    print("# DEPLOY CANDIDATES (exp_train > 0, exp_oos > 0, n_train >= 50, n_oos >= 5)")
    print("#" * 100)
    if candidates.empty:
        print("  NONE")
    else:
        view = candidates.head(25).copy()
        for col in ("hit_train", "hit_oos", "exp_train", "exp_oos"):
            view[col] = view[col].apply(fmt)
        print(view[["window", "level", "vol_filter", "target_mult", "stop_mult",
                     "n_train", "hit_train", "exp_train",
                     "n_oos", "hit_oos", "exp_oos"]].to_string(index=False))

    # Best per (window, level, vol_filter)
    print("\n" + "#" * 100)
    print("# BEST GEOMETRY PER (window, level, volume_filter)")
    print("#" * 100)
    valid_merged = merged[(merged["n_train"] >= 50)].copy()
    if not valid_merged.empty:
        best_idx = valid_merged.groupby(["window", "level", "vol_filter"])["exp_train"].idxmax()
        best = valid_merged.loc[best_idx.dropna()].sort_values("exp_train", ascending=False)
        view = best.copy()
        for col in ("hit_train", "hit_oos", "exp_train", "exp_oos"):
            view[col] = view[col].apply(fmt)
        print(view[["window", "level", "vol_filter", "target_mult", "stop_mult",
                     "n_train", "hit_train", "exp_train",
                     "n_oos", "hit_oos", "exp_oos"]].to_string(index=False))

    # Compare volume filters head-to-head at the best window+level
    print("\n" + "#" * 100)
    print("# VOLUME FILTER COMPARISON — VAH at 10:00-10:30 (Step 1 winner)")
    print("#" * 100)
    vah_1030 = merged[
        (merged["window"] == "10_00_10_30") & (merged["level"] == "vah")
        & (merged["n_train"] >= 20)
    ].copy()
    if not vah_1030.empty:
        best_per_vf = vah_1030.groupby("vol_filter")["exp_train"].idxmax()
        best_vf = vah_1030.loc[best_per_vf.dropna()].sort_values("exp_train", ascending=False)
        for col in ("hit_train", "hit_oos", "exp_train", "exp_oos"):
            best_vf[col] = best_vf[col].apply(fmt)
        print(best_vf[["vol_filter", "target_mult", "stop_mult",
                        "n_train", "hit_train", "exp_train",
                        "n_oos", "hit_oos", "exp_oos"]].to_string(index=False))

    dt = time.perf_counter() - t0_start
    print(f"\n\nStep 2 complete in {dt:.1f}s")
    print(f"Results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
