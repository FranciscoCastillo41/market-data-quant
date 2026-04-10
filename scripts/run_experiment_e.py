"""Run Experiment E: SPY volume-confirmed rules with train/OOS split.

Writes results to data/results/experiment_e/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.experiments.e_spy_volume import (
    STOP_MULTS,
    TARGET_MULTS,
    WINDOW,
    WINDOW_MOMENTUM,
    collect_events,
    compute_expectancy_atr,
    first_passage_grid,
    make_pseudo_events,
    measure_events_with_atr,
)
from mdq.levels.volume_profile import compute_all_profiles

OUT = RESULTS_DIR / "experiment_e"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2023-01-03"
TRAIN_END = "2025-12-31"
OOS_START = "2026-01-01"
OOS_END = "2026-04-08"


def run_split(label: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events_all_rules, pseudo_events) for a date slice."""
    print(f"\n[{label}] Loading SPY bars {start} -> {end}")
    bars = load_bars("SPY", start, end)
    profiles = compute_all_profiles(bars, bin_size=0.05)
    print(f"  bars: {len(bars):,}  sessions: {bars['session_date'].nunique()}  "
          f"profiles: {len(profiles)}")

    # Collect fade-rule events in the narrow window
    events_fade = collect_events(bars, profiles, WINDOW)
    events_mom = collect_events(bars, profiles, WINDOW_MOMENTUM)
    events_fade = events_fade[events_fade["rule"].isin(["S1", "S2", "S3"])]
    events_mom = events_mom[events_mom["rule"] == "S4"]
    events = pd.concat([events_fade, events_mom], ignore_index=True)
    print(f"  events total: {len(events):,}")
    for rule in ("S1", "S2", "S3", "S4"):
        n = (events["rule"] == rule).sum()
        print(f"    {rule}: {n:,}")

    events = measure_events_with_atr(bars, events)

    # Baseline
    pseudo_fade = make_pseudo_events(bars, WINDOW, every_nth=30)
    pseudo_mom = make_pseudo_events(bars, WINDOW_MOMENTUM, every_nth=60)
    pseudo = pd.concat([pseudo_fade, pseudo_mom], ignore_index=True)
    pseudo = measure_events_with_atr(bars, pseudo)
    print(f"  pseudo: {len(pseudo):,}")

    return bars, events, pseudo


def grid_for_events(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Run target/stop grid search, return one row per (rule, level_name, t, s)."""
    if events.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for target_mult in TARGET_MULTS:
        for stop_mult in STOP_MULTS:
            outcomes = first_passage_grid(bars, events, target_mult, stop_mult)
            events["_outcome"] = outcomes
            # Group by (rule, level_name)
            for (rule, lname), sub in events.groupby(["rule", "level_name"]):
                stats = compute_expectancy_atr(
                    sub["_outcome"], target_mult, stop_mult,
                )
                rows.append({
                    "split": label,
                    "rule": rule,
                    "level_name": lname,
                    "target_mult": target_mult,
                    "stop_mult": stop_mult,
                    **stats,
                })
            events = events.drop(columns=["_outcome"])
    return pd.DataFrame(rows)


def grid_for_baseline(bars: pd.DataFrame, pseudo: pd.DataFrame, label: str) -> pd.DataFrame:
    if pseudo.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for target_mult in TARGET_MULTS:
        for stop_mult in STOP_MULTS:
            outcomes = first_passage_grid(bars, pseudo, target_mult, stop_mult)
            stats = compute_expectancy_atr(outcomes, target_mult, stop_mult)
            rows.append({
                "split": label,
                "target_mult": target_mult,
                "stop_mult": stop_mult,
                **stats,
            })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 90)
    print(f"Experiment E — SPY volume-confirmed rules (train/OOS)")
    print("=" * 90)
    t0 = time.perf_counter()

    train_bars, train_events, train_pseudo = run_split("train", TRAIN_START, TRAIN_END)
    oos_bars, oos_events, oos_pseudo = run_split("oos", OOS_START, OOS_END)

    print(f"\n\n[grid search]")
    train_grid = grid_for_events(train_bars, train_events, "train")
    oos_grid = grid_for_events(oos_bars, oos_events, "oos")
    train_base = grid_for_baseline(train_bars, train_pseudo, "train")
    oos_base = grid_for_baseline(oos_bars, oos_pseudo, "oos")

    train_grid.to_csv(OUT / "train_grid.csv", index=False)
    oos_grid.to_csv(OUT / "oos_grid.csv", index=False)
    train_base.to_csv(OUT / "train_baseline.csv", index=False)
    oos_base.to_csv(OUT / "oos_baseline.csv", index=False)

    # Join train + OOS on (rule, level_name, target, stop) and subtract baseline
    train_grid = train_grid.rename(
        columns={"n": "n_train", "hit": "hit_train", "expectancy_atr": "exp_train"}
    )
    oos_grid = oos_grid.rename(
        columns={"n": "n_oos", "hit": "hit_oos", "expectancy_atr": "exp_oos"}
    )
    train_base_slim = train_base.rename(
        columns={"expectancy_atr": "base_train"}
    )[["target_mult", "stop_mult", "base_train"]]
    oos_base_slim = oos_base.rename(
        columns={"expectancy_atr": "base_oos"}
    )[["target_mult", "stop_mult", "base_oos"]]

    merged = train_grid.merge(
        oos_grid[["rule", "level_name", "target_mult", "stop_mult",
                  "n_oos", "hit_oos", "exp_oos"]],
        on=["rule", "level_name", "target_mult", "stop_mult"],
        how="left",
    )
    merged = merged.merge(train_base_slim, on=["target_mult", "stop_mult"], how="left")
    merged = merged.merge(oos_base_slim, on=["target_mult", "stop_mult"], how="left")
    merged["edge_train"] = merged["exp_train"] - merged["base_train"]
    merged["edge_oos"] = merged["exp_oos"] - merged["base_oos"]
    merged.to_csv(OUT / "_merged.csv", index=False)

    # Deploy candidates
    candidates = merged[
        (merged["n_train"] >= 100)
        & (merged["n_oos"] >= 10)
        & (merged["edge_train"] > 0)
        & (merged["edge_oos"] > 0)
    ].copy()
    candidates["sum_edge"] = candidates["edge_train"] + candidates["edge_oos"]
    candidates = candidates.sort_values("sum_edge", ascending=False)
    candidates.to_csv(OUT / "_candidates.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    view_cols = [
        "rule", "level_name", "target_mult", "stop_mult",
        "n_train", "hit_train", "exp_train", "base_train", "edge_train",
        "n_oos", "hit_oos", "exp_oos", "base_oos", "edge_oos",
    ]

    print("\n" + "#" * 100)
    print(f"# TOP 20 TRAIN EDGE (unfiltered)")
    print("#" * 100)
    top_train = merged[merged["n_train"] >= 100].nlargest(20, "edge_train").copy()
    for col in ("hit_train", "hit_oos", "exp_train", "exp_oos",
                "base_train", "base_oos", "edge_train", "edge_oos"):
        top_train[col] = top_train[col].apply(fmt)
    print(top_train[view_cols].to_string(index=False))

    print("\n" + "#" * 100)
    print(f"# DEPLOY CANDIDATES (n_train>=100, n_oos>=10, both edges positive)")
    print("#" * 100)
    if candidates.empty:
        print("  ** NO DEPLOY CANDIDATES **")
    else:
        cview = candidates.copy()
        for col in ("hit_train", "hit_oos", "exp_train", "exp_oos",
                    "base_train", "base_oos", "edge_train", "edge_oos"):
            cview[col] = cview[col].apply(fmt)
        print(cview[view_cols].to_string(index=False))

    dt = time.perf_counter() - t0
    print(f"\n\nExperiment E complete in {dt:.1f}s")
    print(f"Results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
