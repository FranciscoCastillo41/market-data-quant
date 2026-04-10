"""Experiment F: intraday HVN breakout/breakdown with volume spike.

Rule S5 (new):
    full-body bar (>=70% body of range)
    volume >= 2x trailing 20-bar average
    close above/below session VWAP (in direction of break)
    close breaks through a top-3 intraday HVN (cumulative volume profile)
    → enter at bar close, direction = momentum

Same train/OOS split + ATR-based grid as Experiment E.
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
    compute_expectancy_atr,
    first_passage_grid,
    make_pseudo_events,
    measure_events_with_atr,
)
from mdq.levels.hvn_breakout import evaluate_hvn_breakout

OUT = RESULTS_DIR / "experiment_f"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2023-01-03"
TRAIN_END = "2025-12-31"
OOS_START = "2026-01-01"
OOS_END = "2026-04-08"

# Rule requires intraday cumulative profile, so we run on RTH bars
# (not window-restricted) to have enough volume to form HVNs
WINDOW = ("09:30", "15:59")


def collect_events(bars: pd.DataFrame) -> pd.DataFrame:
    """Run hvn_breakout rule on every session's RTH bars."""
    rth = filter_rth(bars)
    if rth.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for sd, sess in rth.groupby("session_date", sort=True):
        sess = sess.sort_values("t").reset_index(drop=True)
        events = evaluate_hvn_breakout(sess)
        if not events.empty:
            frames.append(events)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_split(label: str, start: str, end: str) -> dict:
    print(f"\n[{label}] Loading SPY bars {start} -> {end}")
    bars = load_bars("SPY", start, end)
    print(f"  {len(bars):,} bars  {bars['session_date'].nunique()} sessions")

    events = collect_events(bars)
    print(f"  S5 HVN breakout events: {len(events):,}")
    if len(events) > 0:
        dir_counts = events["direction"].value_counts()
        print(f"    direction: {dict(dir_counts)}")

    events = measure_events_with_atr(bars, events)

    pseudo = make_pseudo_events(bars, WINDOW, every_nth=60)
    pseudo = measure_events_with_atr(bars, pseudo)
    print(f"  baseline pseudo: {len(pseudo):,}")

    return {"bars": bars, "events": events, "pseudo": pseudo}


def grid_for_events(bars: pd.DataFrame, events: pd.DataFrame, label: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for t in TARGET_MULTS:
        for s in STOP_MULTS:
            outcomes = first_passage_grid(bars, events, t, s)
            stats = compute_expectancy_atr(outcomes, t, s)
            rows.append({"split": label, "target_mult": t, "stop_mult": s, **stats})
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 90)
    print("Experiment F — SPY intraday HVN breakout/breakdown (S5 rule)")
    print("=" * 90)
    t0 = time.perf_counter()

    train = run_split("train", TRAIN_START, TRAIN_END)
    oos = run_split("oos", OOS_START, OOS_END)

    train_grid = grid_for_events(train["bars"], train["events"], "train")
    oos_grid = grid_for_events(oos["bars"], oos["events"], "oos")
    train_base = grid_for_events(train["bars"], train["pseudo"], "train_base")
    oos_base = grid_for_events(oos["bars"], oos["pseudo"], "oos_base")

    # Merge into a comparison frame
    merged = train_grid.rename(
        columns={"n": "n_train", "hit": "hit_train", "expectancy_atr": "exp_train"}
    ).merge(
        oos_grid[["target_mult", "stop_mult", "n", "hit", "expectancy_atr"]].rename(
            columns={"n": "n_oos", "hit": "hit_oos", "expectancy_atr": "exp_oos"}
        ),
        on=["target_mult", "stop_mult"],
    ).merge(
        train_base[["target_mult", "stop_mult", "expectancy_atr"]].rename(
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
    merged.to_csv(OUT / "_grid.csv", index=False)

    # Split by direction
    def grid_by_direction(events, pseudo, bars):
        rows = []
        for direction in ("short", "long"):
            sub = events[events["direction"] == direction]
            sub_pseudo = pseudo[pseudo["direction"] == direction]
            for t in TARGET_MULTS:
                for s in STOP_MULTS:
                    oc = first_passage_grid(bars, sub, t, s)
                    stats = compute_expectancy_atr(oc, t, s)
                    base_oc = first_passage_grid(bars, sub_pseudo, t, s)
                    base = compute_expectancy_atr(base_oc, t, s)
                    rows.append({
                        "direction": direction,
                        "target_mult": t, "stop_mult": s,
                        "n": stats["n"], "hit": stats["hit"],
                        "exp": stats["expectancy_atr"],
                        "base": base["expectancy_atr"],
                        "edge": stats["expectancy_atr"] - base["expectancy_atr"]
                            if not (np.isnan(stats["expectancy_atr"]) or np.isnan(base["expectancy_atr"]))
                            else np.nan,
                    })
        return pd.DataFrame(rows)

    train_dir = grid_by_direction(train["events"], train["pseudo"], train["bars"])
    oos_dir = grid_by_direction(oos["events"], oos["pseudo"], oos["bars"])

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    # Top 10 by train edge
    print("\n" + "#" * 100)
    print("# TOP 15 BY TRAIN EDGE (n_train >= 100)")
    print("#" * 100)
    eligible = merged[merged["n_train"] >= 100]
    top15 = eligible.sort_values("edge_train", ascending=False).head(15).copy()
    for col in ("hit_train", "hit_oos", "exp_train", "exp_oos",
                "base_train", "base_oos", "edge_train", "edge_oos"):
        top15[col] = top15[col].apply(fmt)
    print(top15[[
        "target_mult", "stop_mult",
        "n_train", "hit_train", "exp_train", "edge_train",
        "n_oos", "hit_oos", "exp_oos", "edge_oos",
    ]].to_string(index=False))

    # Deploy candidates
    print("\n" + "#" * 100)
    print("# DEPLOY CANDIDATES (n_train>=100, n_oos>=10, both edges > 0)")
    print("#" * 100)
    candidates = merged[
        (merged["n_train"] >= 100) & (merged["n_oos"] >= 10)
        & (merged["edge_train"] > 0) & (merged["edge_oos"] > 0)
    ].copy()
    if candidates.empty:
        print("  NONE")
    else:
        candidates["sum_edge"] = candidates["edge_train"] + candidates["edge_oos"]
        candidates = candidates.sort_values("sum_edge", ascending=False)
        for col in ("hit_train", "hit_oos", "exp_train", "exp_oos",
                    "base_train", "base_oos", "edge_train", "edge_oos"):
            candidates[col] = candidates[col].apply(fmt)
        print(candidates[[
            "target_mult", "stop_mult",
            "n_train", "hit_train", "edge_train",
            "n_oos", "hit_oos", "edge_oos",
        ]].to_string(index=False))

    # By direction
    print("\n" + "#" * 100)
    print("# BY DIRECTION (train) — top per direction")
    print("#" * 100)
    for direction in ("short", "long"):
        sub = train_dir[(train_dir["direction"] == direction) & (train_dir["n"] >= 50)]
        if sub.empty:
            print(f"\n  {direction}: no data")
            continue
        top = sub.nlargest(5, "edge").copy()
        print(f"\n  {direction} top 5 (train):")
        for col in ("hit", "exp", "base", "edge"):
            top[col] = top[col].apply(fmt)
        print(top[["target_mult", "stop_mult", "n", "hit", "exp", "base", "edge"]].to_string(index=False))

    print("\n" + "#" * 100)
    print("# BY DIRECTION (oos) — at train's best geometry")
    print("#" * 100)
    for direction in ("short", "long"):
        sub = oos_dir[oos_dir["direction"] == direction]
        if sub.empty:
            continue
        print(f"\n  {direction} oos:")
        disp = sub.copy()
        for col in ("hit", "exp", "base", "edge"):
            disp[col] = disp[col].apply(fmt)
        print(disp[["target_mult", "stop_mult", "n", "hit", "exp", "base", "edge"]].to_string(index=False))

    dt = time.perf_counter() - t0
    print(f"\n\nRan in {dt:.1f}s")
    print(f"Results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
