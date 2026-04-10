"""Experiment E-dynamic: compare prior-day POC vs developing POC vs rolling POC
for the S3 wick-rejection rule on SPY.

Runs the same train/OOS split as Experiment E but evaluates three different
"POC" definitions as separate level names so we can see which one has the
strongest edge.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.experiments.e_spy_volume import (
    STOP_MULTS,
    TARGET_MULTS,
    WINDOW,
    compute_expectancy_atr,
    first_passage_grid,
    make_pseudo_events,
    measure_events_with_atr,
)
from mdq.levels.dynamic_profile import compute_dynamic_profiles_per_session
from mdq.levels.dynamic_rules import evaluate_s3_dynamic
from mdq.levels.volume_profile import compute_all_profiles
from mdq.levels.volume_rules import evaluate_s3_wick_rejection

OUT = RESULTS_DIR / "experiment_e_dynamic"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2023-01-03"
TRAIN_END = "2025-12-31"
OOS_START = "2026-01-01"
OOS_END = "2026-04-08"


def collect_prior_poc_events(bars: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """S3 events against yesterday's POC (the original)."""
    frames = []
    for sd, sess in bars.groupby("session_date", sort=True):
        prior = profiles[profiles["session_date"] < sd]
        if prior.empty:
            continue
        prior_row = prior.iloc[-1]
        if pd.isna(prior_row["poc"]):
            continue
        session_win = filter_window(sess, WINDOW[0], WINDOW[1]).reset_index(drop=True)
        if session_win.empty:
            continue
        events = evaluate_s3_wick_rejection(
            session_win, [("prior_poc", float(prior_row["poc"]))]
        )
        if not events.empty:
            frames.append(events)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_dynamic_events(
    bars: pd.DataFrame,
    dyn_profiles: pd.DataFrame,
    level_col: str,
    level_name: str,
) -> pd.DataFrame:
    """S3 events against a dynamic per-bar POC level (developing or rolling)."""
    frames = []
    for sd, sess in bars.groupby("session_date", sort=True):
        session_win = filter_window(sess, WINDOW[0], WINDOW[1]).reset_index(drop=True)
        if session_win.empty:
            continue
        # Join the per-bar dynamic level onto this session's window bars
        dyn_sess = dyn_profiles[dyn_profiles["session_date"] == sd]
        if dyn_sess.empty:
            continue
        merged = session_win.merge(
            dyn_sess[["t", level_col]], on="t", how="left"
        )
        level_series = merged[level_col].to_numpy()
        events = evaluate_s3_dynamic(merged, level_series, level_name)
        if not events.empty:
            frames.append(events)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_split(label: str, start: str, end: str) -> dict:
    print(f"\n[{label}] Loading SPY bars {start} -> {end}")
    bars = load_bars("SPY", start, end)
    print(f"  {len(bars):,} bars  {bars['session_date'].nunique()} sessions")

    # Prior-day profiles for the "prior_poc" level
    profiles = compute_all_profiles(bars, bin_size=0.05)

    # Dynamic profiles (developing + rolling) — compute on RTH bars only
    rth = filter_rth(bars)
    dyn_profiles = compute_dynamic_profiles_per_session(
        rth, rolling_window=30, developing_min_bars=15
    )
    print(f"  dynamic profile rows: {len(dyn_profiles):,}")

    print("  collecting events...")
    events_prior = collect_prior_poc_events(bars, profiles)
    events_dev = collect_dynamic_events(bars, dyn_profiles, "dev_poc", "dev_poc")
    events_roll = collect_dynamic_events(bars, dyn_profiles, "roll_poc", "roll_poc")
    print(f"    prior_poc S3 events:  {len(events_prior):,}")
    print(f"    dev_poc   S3 events:  {len(events_dev):,}")
    print(f"    roll_poc  S3 events:  {len(events_roll):,}")

    events_prior = measure_events_with_atr(bars, events_prior)
    events_dev = measure_events_with_atr(bars, events_dev)
    events_roll = measure_events_with_atr(bars, events_roll)

    # Baseline
    pseudo = make_pseudo_events(bars, WINDOW, every_nth=30)
    pseudo = measure_events_with_atr(bars, pseudo)

    return {
        "bars": bars,
        "prior": events_prior,
        "dev": events_dev,
        "roll": events_roll,
        "pseudo": pseudo,
    }


def grid_search_for(bars: pd.DataFrame, events: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for t in TARGET_MULTS:
        for s in STOP_MULTS:
            outcomes = first_passage_grid(bars, events, t, s)
            stats = compute_expectancy_atr(outcomes, t, s)
            rows.append({
                "split": label,
                "target_mult": t,
                "stop_mult": s,
                **stats,
            })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 90)
    print("Experiment E-dynamic — prior POC vs developing POC vs rolling POC")
    print("=" * 90)
    t0 = time.perf_counter()

    train = run_split("train", TRAIN_START, TRAIN_END)
    oos = run_split("oos", OOS_START, OOS_END)

    # Grid search per level type per split
    results = {}
    for level_key in ("prior", "dev", "roll"):
        tr_grid = grid_search_for(train["bars"], train[level_key], "train")
        oos_grid = grid_search_for(oos["bars"], oos[level_key], "oos")
        tr_base = grid_search_for(train["bars"], train["pseudo"], "train")
        oos_base = grid_search_for(oos["bars"], oos["pseudo"], "oos")

        merged = tr_grid.rename(
            columns={"n": "n_train", "hit": "hit_train", "expectancy_atr": "exp_train"}
        ).merge(
            oos_grid[["target_mult", "stop_mult", "n", "hit", "expectancy_atr"]].rename(
                columns={"n": "n_oos", "hit": "hit_oos", "expectancy_atr": "exp_oos"}
            ),
            on=["target_mult", "stop_mult"],
        ).merge(
            tr_base[["target_mult", "stop_mult", "expectancy_atr"]].rename(
                columns={"expectancy_atr": "base_train"}
            ),
            on=["target_mult", "stop_mult"],
        ).merge(
            oos_base[["target_mult", "stop_mult", "expectancy_atr"]].rename(
                columns={"expectancy_atr": "base_oos"}
            ),
            on=["target_mult", "stop_mult"],
        )
        merged["edge_train"] = merged["exp_train"] - merged["base_train"]
        merged["edge_oos"] = merged["exp_oos"] - merged["base_oos"]
        merged["level_type"] = level_key
        results[level_key] = merged

    all_df = pd.concat(results.values(), ignore_index=True)
    all_df.to_csv(OUT / "_comparison.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    # For each level type, find the BEST (target, stop) combo by edge_train
    print("\n" + "#" * 100)
    print("# BEST GEOMETRY PER LEVEL TYPE (ranked by edge_train with n_train >= 100)")
    print("#" * 100)
    summary_rows = []
    for level_key, df in results.items():
        eligible = df[df["n_train"] >= 100]
        if eligible.empty:
            print(f"\n-- {level_key}: no combos with n_train >= 100 --")
            continue
        best = eligible.loc[eligible["edge_train"].idxmax()]
        summary_rows.append(best)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)[[
            "level_type", "target_mult", "stop_mult",
            "n_train", "hit_train", "exp_train", "base_train", "edge_train",
            "n_oos", "hit_oos", "exp_oos", "base_oos", "edge_oos",
        ]]
        for col in ("hit_train", "hit_oos", "exp_train", "exp_oos",
                    "base_train", "base_oos", "edge_train", "edge_oos"):
            summary_df[col] = summary_df[col].apply(fmt)
        print(summary_df.to_string(index=False))

    # Head-to-head at the fixed (1.0, 1.0) ATR geometry that worked for prior_poc
    print("\n" + "#" * 100)
    print("# HEAD-TO-HEAD @ target=1.00 ATR, stop=1.00 ATR")
    print("#" * 100)
    headrows = []
    for level_key, df in results.items():
        row = df[(df["target_mult"] == 1.00) & (df["stop_mult"] == 1.00)]
        if row.empty:
            continue
        headrows.append(row.iloc[0])
    if headrows:
        head_df = pd.DataFrame(headrows)[[
            "level_type", "n_train", "hit_train", "exp_train", "edge_train",
            "n_oos", "hit_oos", "exp_oos", "edge_oos",
        ]]
        for col in ("hit_train", "hit_oos", "exp_train", "exp_oos",
                    "edge_train", "edge_oos"):
            head_df[col] = head_df[col].apply(fmt)
        print(head_df.to_string(index=False))

    # Top 10 overall by edge_train (deploy candidates)
    print("\n" + "#" * 100)
    print("# TOP 10 OVERALL (any level type) BY TRAIN EDGE, n_train>=100")
    print("#" * 100)
    eligible = all_df[all_df["n_train"] >= 100]
    top10 = eligible.sort_values("edge_train", ascending=False).head(10).copy()
    for col in ("hit_train", "hit_oos", "exp_train", "exp_oos",
                "base_train", "base_oos", "edge_train", "edge_oos"):
        top10[col] = top10[col].apply(fmt)
    print(top10[[
        "level_type", "target_mult", "stop_mult",
        "n_train", "hit_train", "edge_train",
        "n_oos", "hit_oos", "edge_oos",
    ]].to_string(index=False))

    dt = time.perf_counter() - t0
    print(f"\nRan in {dt:.1f}s")
    print(f"Results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
