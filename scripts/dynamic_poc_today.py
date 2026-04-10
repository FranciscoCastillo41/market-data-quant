"""Compute today's developing POC / rolling POC for SPY at various points in time.

Shows how the POC evolves through the session — helps us understand whether
a dynamic POC would have given the bot a better level to watch.
"""

from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd

from mdq.data.bars import download_range_async, load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.levels.volume_profile import compute_session_profile
from mdq.levels.weekly_profile import _build_volume_at_price, _value_area


def compute_poc_from_bars(bars: pd.DataFrame, bin_size: float = 0.05) -> tuple[float, float, float] | None:
    """Return (poc, vah, val) for a DataFrame of bars. None if empty."""
    if bars.empty:
        return None
    lows = bars["l"].to_numpy()
    highs = bars["h"].to_numpy()
    vols = bars["v"].to_numpy()
    bin_centers, volumes, _, _ = _build_volume_at_price(lows, highs, vols, bin_size)
    if volumes.size == 0 or volumes.sum() == 0:
        return None
    poc_idx = int(np.argmax(volumes))
    val_idx, vah_idx = _value_area(volumes, poc_idx, 0.70)
    return float(bin_centers[poc_idx]), float(bin_centers[vah_idx]), float(bin_centers[val_idx])


def main() -> int:
    today = date.today()
    print("=" * 75)
    print(f"Dynamic POC analysis for SPY — {today}")
    print("=" * 75)

    # Refresh bars
    print("\nPulling latest SPY bars from Massive...")
    asyncio.run(
        download_range_async(
            ticker="SPY",
            start=today.isoformat(),
            end=today.isoformat(),
            overwrite=True,
        )
    )
    bars = load_bars("SPY", today.isoformat(), today.isoformat())
    rth = filter_rth(bars).reset_index(drop=True)
    print(f"RTH bars so far: {len(rth)}")
    if rth.empty:
        return 1
    print(f"Range: {rth['ts_et'].min()} -> {rth['ts_et'].max()}")

    # 1. Prior-day POC (what the bot used)
    print("\n[1] Prior-day POC (what the bot is watching):")
    prior_profs = pd.read_parquet("data/results/experiment_c/profiles__SPY.parquet")
    prior_profs["session_date"] = pd.to_datetime(prior_profs["session_date"]).dt.date
    prior_row = prior_profs[prior_profs["session_date"] < today].sort_values("session_date").iloc[-1]
    print(f"  from {prior_row['session_date']}: POC=${prior_row['poc']:.2f}  "
          f"VAH=${prior_row['vah']:.2f}  VAL=${prior_row['val']:.2f}")

    # 2. Developing POC (cumulative from 09:30 up to each checkpoint)
    print("\n[2] Developing POC (cumulative from session open):")
    checkpoints = ["09:35", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:30"]
    for checkpoint in checkpoints:
        sub = rth[rth["ts_et"].dt.strftime("%H:%M") <= checkpoint]
        if sub.empty:
            continue
        result = compute_poc_from_bars(sub, bin_size=0.05)
        if result is None:
            continue
        poc, vah, val = result
        last_price = float(sub.iloc[-1]["c"])
        print(f"  by {checkpoint}  POC=${poc:.2f}  VAH=${vah:.2f}  VAL=${val:.2f}  "
              f"(last close=${last_price:.2f})")

    # 3. Rolling 30-minute POC (last 30 bars, updates every bar)
    print("\n[3] Rolling 30-minute POC at each checkpoint:")
    for checkpoint in checkpoints:
        sub = rth[rth["ts_et"].dt.strftime("%H:%M") <= checkpoint]
        if len(sub) < 5:
            continue
        window = sub.tail(30)
        result = compute_poc_from_bars(window, bin_size=0.05)
        if result is None:
            continue
        poc, vah, val = result
        print(f"  by {checkpoint}  rolling POC=${poc:.2f}  VAH=${vah:.2f}  VAL=${val:.2f}  "
              f"({len(window)}-bar window)")

    # 4. What was volume profile at EXACTLY 09:59 (just before the bot's signal)?
    print("\n[4] Volume profile state at 09:59 (bot's signal time):")
    sub_0959 = rth[rth["ts_et"].dt.strftime("%H:%M") <= "09:59"]
    if not sub_0959.empty:
        result = compute_poc_from_bars(sub_0959, bin_size=0.05)
        if result is not None:
            poc, vah, val = result
            last_close = float(sub_0959.iloc[-1]["c"])
            print(f"  cumulative POC: ${poc:.2f}")
            print(f"  cumulative VAH: ${vah:.2f}")
            print(f"  cumulative VAL: ${val:.2f}")
            print(f"  last close:     ${last_close:.2f}")
            print(f"  yesterday POC:  ${prior_row['poc']:.2f}")
            if abs(poc - float(prior_row['poc'])) > 0.10:
                print(f"  ⚠️  Developing POC has diverged from yesterday's POC by ${abs(poc - float(prior_row['poc'])):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
