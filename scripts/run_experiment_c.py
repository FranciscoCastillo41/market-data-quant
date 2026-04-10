"""Run Experiment C: POC/VAH/VAL level reactions on SPY and QQQ.

For each ticker:
  1. Compute prior-session volume profiles (POC, VAH, VAL, H, L, C)
  2. Detect touches of those levels in the morning window
  3. Measure reactions
  4. Run grid_search at key geometries vs matched baseline
  5. Split results by level_name, confluence flag, approach, year, time-bucket

Writes CSVs to data/results/experiment_c/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.experiments.c_volume_profile import run_experiment_c
from mdq.stats.grid_search import GridSpec, grid_search

OUT_DIR = RESULTS_DIR / "experiment_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2023-01-03"
END = "2026-04-07"

WINDOW = ("09:30", "11:15")

# Test the configurations we found in Experiment B, plus Randy's original
TEST_GEOMETRIES = [
    ("momentum", 1.00, 1.50),  # QQQ whole-dollar winner
    ("momentum", 0.75, 1.00),  # tighter
    ("fade",     1.50, 1.50),  # SPY whole-dollar marginal
    ("fade",     1.00, 1.00),  # Randy-ish but wider
    ("fade",     0.30, 0.25),  # Randy's exact
]


def make_pseudo_touches(bars_win: pd.DataFrame, every_nth: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picks = bars_win.iloc[::every_nth].copy().reset_index(drop=True)
    picks["approach"] = rng.choice(["from_below", "from_above"], size=len(picks))
    picks["level"] = picks["c"]
    picks["touch_num"] = 1
    picks["open_price"] = picks.groupby("session_date")["o"].transform("first")
    picks["entry_close"] = picks["c"]
    picks["tier"] = 1
    picks["bar_idx"] = picks.index
    return picks


def run_geometry_on_subset(
    bars_win: pd.DataFrame,
    touches_subset: pd.DataFrame,
    direction: str,
    target: float,
    stop: float,
) -> dict:
    if touches_subset.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan, "expectancy": np.nan}
    spec = GridSpec(
        targets=(target,), stops=(stop,), horizons=(15,), directions=(direction,),
    )
    df = grid_search(bars_win, touches_subset, spec)
    if df.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan, "expectancy": np.nan}
    row = df.iloc[0]
    return {
        "n": int(row["n"]),
        "hit": float(row["hit"]),
        "stop_rate": float(row["stop_rate"]),
        "expectancy": float(row["expectancy"]),
    }


def main() -> None:
    all_rows: list[dict] = []

    for ticker, whole_step in (("SPY", 5.0), ("QQQ", 10.0)):
        print(f"\n{'='*80}")
        print(f"Experiment C: {ticker}  (whole-dollar confluence step = ${whole_step:.0f})")
        print(f"{'='*80}")

        t0 = time.perf_counter()
        result = run_experiment_c(
            ticker=ticker,
            start=START,
            end=END,
            time_window=WINDOW,
            bin_size=0.05,
            whole_dollar_step=whole_step,
        )
        dt = time.perf_counter() - t0
        print(f"  Ran in {dt:.1f}s")
        print(f"  Profiles computed: {len(result.profiles)}")
        print(f"  VP touches detected: {len(result.touches)}")
        first_touches = result.reactions[result.reactions["touch_num"] == 1]
        print(f"  First touches of VP level: {len(first_touches)}")

        # Save raw outputs
        result.profiles.to_parquet(OUT_DIR / f"profiles__{ticker}.parquet", index=False)
        result.reactions.to_parquet(OUT_DIR / f"reactions__{ticker}.parquet", index=False)

        bars = load_bars(ticker, START, END)
        bars_win = filter_window(bars, WINDOW[0], WINDOW[1])

        pseudo = make_pseudo_touches(bars_win)
        print(f"  Baseline pseudo-touches: {len(pseudo):,}")

        # For each geometry, run on:
        #   (level_name, confluence_flag, touch_num==1 only)
        #   matched baseline (same geometry on pseudo-touches)
        level_names = ["ALL", "poc", "vah", "val", "prior_high", "prior_low", "prior_close"]

        first_only = result.reactions[result.reactions["touch_num"] == 1].copy()

        for direction, target, stop in TEST_GEOMETRIES:
            base = run_geometry_on_subset(bars_win, pseudo, direction, target, stop)

            for lname in level_names:
                if lname == "ALL":
                    sub = first_only
                else:
                    sub = first_only[first_only["level_name"] == lname]
                for conf_flag in ("all", "confluence_true", "confluence_false"):
                    if conf_flag == "all":
                        sub2 = sub
                    elif conf_flag == "confluence_true":
                        sub2 = sub[sub["confluence"] == True]  # noqa: E712
                    else:
                        sub2 = sub[sub["confluence"] == False]  # noqa: E712

                    r = run_geometry_on_subset(bars_win, sub2, direction, target, stop)
                    edge = (
                        r["expectancy"] - base["expectancy"]
                        if not (np.isnan(r["expectancy"]) or np.isnan(base["expectancy"]))
                        else np.nan
                    )
                    hit_lift = (
                        r["hit"] - base["hit"]
                        if not (np.isnan(r["hit"]) or np.isnan(base["hit"]))
                        else np.nan
                    )
                    all_rows.append({
                        "ticker": ticker,
                        "level_name": lname,
                        "confluence": conf_flag,
                        "direction": direction,
                        "target": target,
                        "stop": stop,
                        "n": r["n"],
                        "n_base": base["n"],
                        "hit": r["hit"],
                        "base_hit": base["hit"],
                        "hit_lift": hit_lift,
                        "expectancy": r["expectancy"],
                        "base_exp": base["expectancy"],
                        "edge": edge,
                    })

    full = pd.DataFrame(all_rows)
    full.to_csv(OUT_DIR / "_summary.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    view_cols = [
        "ticker", "level_name", "confluence", "direction", "target", "stop",
        "n", "hit", "base_hit", "hit_lift", "expectancy", "base_exp", "edge",
    ]

    print("\n" + "#" * 100)
    print("# Experiment C headline results — only rows with n >= 20 and positive edge")
    print("#" * 100)
    view = full[(full["n"] >= 20) & (full["edge"] > 0)].copy()
    view = view.sort_values("edge", ascending=False)
    for col in ("hit", "base_hit", "hit_lift", "expectancy", "base_exp", "edge"):
        view[col] = view[col].apply(fmt)
    print(view[view_cols].to_string(index=False))

    print("\n" + "#" * 100)
    print("# Focus: QQQ momentum 1.00/1.50 by level_name × confluence")
    print("#" * 100)
    qm = full[
        (full["ticker"] == "QQQ")
        & (full["direction"] == "momentum")
        & (full["target"] == 1.00)
        & (full["stop"] == 1.50)
    ].copy()
    for col in ("hit", "base_hit", "hit_lift", "expectancy", "base_exp", "edge"):
        qm[col] = qm[col].apply(fmt)
    print(qm[view_cols].to_string(index=False))

    print("\n" + "#" * 100)
    print("# Focus: SPY fade 1.50/1.50 by level_name × confluence")
    print("#" * 100)
    sf = full[
        (full["ticker"] == "SPY")
        & (full["direction"] == "fade")
        & (full["target"] == 1.50)
        & (full["stop"] == 1.50)
    ].copy()
    for col in ("hit", "base_hit", "hit_lift", "expectancy", "base_exp", "edge"):
        sf[col] = sf[col].apply(fmt)
    print(sf[view_cols].to_string(index=False))

    print("\n" + "#" * 100)
    print("# Focus: SPY Randy's exact geometry (fade 0.30/0.25) by level_name × confluence")
    print("#" * 100)
    sr = full[
        (full["ticker"] == "SPY")
        & (full["direction"] == "fade")
        & (full["target"] == 0.30)
        & (full["stop"] == 0.25)
    ].copy()
    for col in ("hit", "base_hit", "hit_lift", "expectancy", "base_exp", "edge"):
        sr[col] = sr[col].apply(fmt)
    print(sr[view_cols].to_string(index=False))

    print(f"\nFull results saved: {OUT_DIR / '_summary.csv'}")


if __name__ == "__main__":
    main()
