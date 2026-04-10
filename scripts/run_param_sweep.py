"""Run grid-search on target/stop/direction for each key Experiment B config.

Loads saved reactions parquet for each (ticker, grid, window) combo, loads
the bars, filters to the same window, runs grid_search, and prints the
top-10 combinations per config ranked by expectancy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.stats.grid_search import GridSpec, grid_search

RES = RESULTS_DIR / "experiment_b"
START = "2023-01-03"
END = "2026-04-07"

CONFIGS = [
    ("SPY", "step_5.00",      "morning_0930_1115", ("09:30", "11:15"), "first_T1", True),
    ("SPY", "step_5.00",      "morning_0930_1115", ("09:30", "11:15"), "ALL", False),
    ("SPY", "step_5.00",      "rth_0930_1600",     ("09:30", "16:00"), "ALL", False),
    ("SPY", "step_1.00_t1_5", "morning_0930_1115", ("09:30", "11:15"), "ALL", False),
    ("QQQ", "step_5.00",      "morning_0930_1115", ("09:30", "11:15"), "ALL", False),
    ("QQQ", "step_10.00",     "morning_0930_1115", ("09:30", "11:15"), "first_T1", True),
]

SPEC = GridSpec(
    targets=(0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50),
    stops=(0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.00, 1.50),
    horizons=(15,),
    directions=("fade", "momentum"),
)


def run_one(
    ticker: str,
    grid_name: str,
    window_name: str,
    window: tuple[str, str],
    subset: str,
    first_only: bool,
) -> pd.DataFrame:
    tag = f"{ticker}__{grid_name}__{window_name}"
    reactions = pd.read_parquet(RES / f"reactions__{tag}.parquet")
    if first_only:
        reactions = reactions[reactions["touch_num"] == 1]
    if reactions.empty:
        return pd.DataFrame()

    bars = load_bars(ticker, START, END)
    bars_win = filter_window(bars, window[0], window[1])

    results = grid_search(bars_win, reactions, SPEC)
    results.insert(0, "config", f"{ticker}_{grid_name}_{window_name}_{subset}")
    return results


def main() -> None:
    all_results = []
    for cfg in CONFIGS:
        print(f"Grid-searching: {cfg[0]} {cfg[1]} {cfg[2]} [{cfg[4]}]")
        df = run_one(*cfg)
        if df.empty:
            print("  (empty)")
            continue
        all_results.append(df)

        # Show best (by expectancy) per direction
        print(f"  n touches: {df['n'].iloc[0]}")
        for direction in ("fade", "momentum"):
            sub = df[df["direction"] == direction]
            top = sub.nlargest(5, "expectancy")
            print(f"\n  top 5 {direction}:")
            print(top[["target", "stop", "hit", "stop_rate", "expectancy", "r_multiple"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            ))
        print()

    if all_results:
        full = pd.concat(all_results, ignore_index=True)
        full.to_csv(RES / "_grid_search.csv", index=False)

        # Global top 20 by expectancy across all configs/directions
        print("\n" + "=" * 80)
        print("GLOBAL TOP 20 BY EXPECTANCY (all configs, all directions)")
        print("=" * 80)
        top = full.nlargest(20, "expectancy")
        print(top[["config", "direction", "target", "stop", "n", "hit", "stop_rate", "expectancy", "r_multiple"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ))

        print("\n" + "=" * 80)
        print("GLOBAL TOP 20 BY R_MULTIPLE (expectancy / stop size)")
        print("=" * 80)
        top_r = full.nlargest(20, "r_multiple")
        print(top_r[["config", "direction", "target", "stop", "n", "hit", "stop_rate", "expectancy", "r_multiple"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ))


if __name__ == "__main__":
    main()
