"""Run Experiment B: whole-dollar level reactions on SPY and QQQ.

For each (ticker, grid) combination and each time window (morning vs RTH),
compute touch reactions, baseline forward moves, and summary tables.

Dumps CSVs to data/results/experiment_b/.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.experiments.b_whole_dollar import run_experiment_b
from mdq.levels.psychological import generate_grid
from mdq.levels.touches import TouchConfig
from mdq.stats.reactions import ReactionConfig


START = "2023-01-03"
END = "2026-04-07"

OUT_DIR = RESULTS_DIR / "experiment_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPY_GRIDS = {
    "step_0.50_t1_5_t2_1": generate_grid(low=300, high=800, step=0.50, tier1_step=5.0, tier2_step=1.0),
    "step_1.00_t1_5":     generate_grid(low=300, high=800, step=1.00, tier1_step=5.0),
    "step_5.00":          generate_grid(low=300, high=800, step=5.00),
}

QQQ_GRIDS = {
    "step_1.00_t1_10_t2_5": generate_grid(low=250, high=700, step=1.00, tier1_step=10.0, tier2_step=5.0),
    "step_2.50_t1_10_t2_5": generate_grid(low=250, high=700, step=2.50, tier1_step=10.0, tier2_step=5.0),
    "step_5.00":            generate_grid(low=250, high=700, step=5.00),
    "step_10.00":           generate_grid(low=250, high=700, step=10.00),
}

WINDOWS = {
    "morning_0930_1115": ("09:30", "11:15"),
    "rth_0930_1600":     ("09:30", "16:00"),
}


def run_and_save(
    ticker: str, grid_name: str, grid, window_name: str, window: tuple[str, str]
) -> None:
    t0 = time.perf_counter()
    result = run_experiment_b(
        ticker=ticker,
        start=START,
        end=END,
        grid=grid,
        time_window=window,
        touch_cfg=TouchConfig(tolerance=0.05, in_play_radius=10.0),
        reaction_cfg=ReactionConfig(
            horizons=(1, 5, 10, 15),
            target_move=0.30,
            stop_move=0.25,
            first_passage_horizon=15,
        ),
    )
    dt = time.perf_counter() - t0

    tag = f"{ticker}__{grid_name}__{window_name}"
    result.touch_summary.to_csv(OUT_DIR / f"summary__{tag}.csv", index=False)
    result.baseline_summary.to_csv(OUT_DIR / f"baseline__{tag}.csv", index=False)
    result.reactions.to_parquet(OUT_DIR / f"reactions__{tag}.parquet", index=False)

    n_touches = len(result.reactions)
    print(f"  {tag:65s}  {n_touches:>7,} touches  {dt:.1f}s")


def main() -> None:
    print(f"Experiment B: whole-dollar level reactions")
    print(f"  Date range: {START} -> {END}")
    print()

    for window_name, window in WINDOWS.items():
        print(f"\n== Window: {window_name} ==")
        for grid_name, grid in SPY_GRIDS.items():
            run_and_save("SPY", grid_name, grid, window_name, window)
        for grid_name, grid in QQQ_GRIDS.items():
            run_and_save("QQQ", grid_name, grid, window_name, window)

    print(f"\nDone. Results written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
