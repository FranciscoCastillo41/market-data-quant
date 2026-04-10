"""Run Experiment D: JPM weekly level reactions.

Computes touches of prior-week volume levels, measures forward reactions
at 1/3/5 trading days, and grid-searches over target/stop/direction to find
which (level_name, direction, geometry) combinations have edge.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.experiments.d_jpm_weekly import (
    first_passage_daily,
    run_experiment_d,
)

OUT_DIR = RESULTS_DIR / "experiment_d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2023-01-03"
END = "2026-04-08"

TARGETS = (0.005, 0.010, 0.015, 0.020, 0.030)
STOPS   = (0.005, 0.010, 0.015, 0.020, 0.030)
HORIZONS = (1, 3, 5)
DIRECTIONS = ("fade", "momentum")


def make_pseudo_touches(daily: pd.DataFrame, every_nth: int = 3, seed: int = 42) -> pd.DataFrame:
    """Random pseudo-touches for baseline comparison."""
    rng = np.random.default_rng(seed)
    picks = daily.iloc[::every_nth].copy().reset_index(drop=True)
    picks["approach"] = rng.choice(["from_below", "from_above"], size=len(picks))
    picks["level"] = picks["c"]
    picks["level_name"] = "baseline"
    picks["touch_num"] = 1
    picks["prev_close"] = picks["c"]
    picks["open"] = picks["o"]
    picks["high"] = picks["h"]
    picks["low"] = picks["l"]
    picks["close"] = picks["c"]
    return picks[[
        "session_date", "level_name", "level", "approach", "touch_num",
        "open", "high", "low", "close", "prev_close",
    ]]


def run_geometry(
    daily: pd.DataFrame,
    touches: pd.DataFrame,
    direction: str,
    target: float,
    stop: float,
    horizon: int,
) -> dict:
    outcomes = first_passage_daily(daily, touches, target, stop, horizon, direction)
    if outcomes.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan, "expectancy": np.nan}
    valid = outcomes[~outcomes.isin(["no_data", "skip"])]
    if valid.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan, "expectancy": np.nan}
    n = len(valid)
    n_target = (valid == "target").sum()
    n_stop = (valid == "stop").sum()
    n_timeout = (valid == "timeout").sum()
    exp = (n_target * target - n_stop * stop) / n
    return {
        "n": n,
        "hit": n_target / n,
        "stop_rate": n_stop / n,
        "timeout_rate": n_timeout / n,
        "expectancy": exp,
    }


def main() -> int:
    print("=" * 80)
    print("Experiment D: JPM weekly-level touch reactions")
    print("=" * 80)

    t0 = time.perf_counter()
    res = run_experiment_d(START, END)
    dt = time.perf_counter() - t0
    print(f"\nRan in {dt:.1f}s")
    print(f"  profiles computed:  {len(res.profiles)}  weeks")
    print(f"  daily bars:         {len(res.daily)}")
    print(f"  touches detected:   {len(res.touches)}")
    print(f"  reactions measured: {len(res.reactions)}")

    res.profiles.to_parquet(OUT_DIR / "profiles__JPM.parquet", index=False)
    res.daily.to_parquet(OUT_DIR / "daily__JPM.parquet", index=False)
    res.touches.to_parquet(OUT_DIR / "touches__JPM.parquet", index=False)
    res.reactions.to_parquet(OUT_DIR / "reactions__JPM.parquet", index=False)

    # Grid search across level_name × direction × target × stop × horizon
    level_names = ["poc", "vah", "val", "high", "low", "close",
                   "hvn_1", "hvn_2", "lvn_1", "lvn_2"]

    # Baseline
    pseudo = make_pseudo_touches(res.daily, every_nth=3)
    print(f"\n  baseline pseudo-touches: {len(pseudo):,}")

    rows: list[dict] = []
    for lname in level_names:
        sub = res.reactions[res.reactions["level_name"] == lname]
        if sub.empty:
            continue
        first = sub[sub["touch_num"] == 1]
        for h in HORIZONS:
            for direction in DIRECTIONS:
                for t in TARGETS:
                    for s in STOPS:
                        r = run_geometry(res.daily, first, direction, t, s, h)
                        b = run_geometry(res.daily, pseudo, direction, t, s, h)
                        edge = (r["expectancy"] - b["expectancy"]
                                if not (np.isnan(r.get("expectancy", np.nan))
                                        or np.isnan(b.get("expectancy", np.nan)))
                                else np.nan)
                        rows.append({
                            "level_name": lname,
                            "direction": direction,
                            "horizon": h,
                            "target_pct": t,
                            "stop_pct": s,
                            "n": r["n"],
                            "n_base": b["n"],
                            "hit": r["hit"],
                            "base_hit": b["hit"],
                            "hit_lift": (r["hit"] - b["hit"]
                                         if not (np.isnan(r["hit"]) or np.isnan(b["hit"]))
                                         else np.nan),
                            "exp": r["expectancy"],
                            "base_exp": b["expectancy"],
                            "edge": edge,
                        })

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "_grid.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 20)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    view_cols = [
        "level_name", "direction", "horizon", "target_pct", "stop_pct",
        "n", "hit", "base_hit", "hit_lift", "exp", "base_exp", "edge",
    ]

    print("\n" + "#" * 100)
    print(f"# TOP 25 BY EDGE (min n=20, positive edge)")
    print("#" * 100)
    top = grid[(grid["n"] >= 20) & (grid["edge"] > 0)].sort_values("edge", ascending=False).head(25)
    if top.empty:
        print("(nothing)")
    else:
        for col in ("hit", "base_hit", "hit_lift", "exp", "base_exp", "edge"):
            top[col] = top[col].apply(fmt)
        print(top[view_cols].to_string(index=False))

    print("\n" + "#" * 100)
    print("# BEST GEOMETRY PER (level_name, direction) — top 1 by edge, min n=20")
    print("#" * 100)
    valid = grid[(grid["n"] >= 20)].copy()
    if not valid.empty:
        best_idx = valid.groupby(["level_name", "direction"])["edge"].idxmax()
        best = valid.loc[best_idx.dropna()].sort_values("edge", ascending=False)
        for col in ("hit", "base_hit", "hit_lift", "exp", "base_exp", "edge"):
            best[col] = best[col].apply(fmt)
        print(best[view_cols].to_string(index=False))

    print("\n" + "#" * 100)
    print("# TOUCH COUNTS BY LEVEL TYPE (raw — before first-touch filter)")
    print("#" * 100)
    counts = res.touches.groupby("level_name").size().sort_values(ascending=False)
    print(counts.to_string())

    print("\n" + "#" * 100)
    print("# FIRST-TOUCH COUNTS BY LEVEL TYPE")
    print("#" * 100)
    first_counts = res.touches[res.touches["touch_num"] == 1].groupby("level_name").size().sort_values(ascending=False)
    print(first_counts.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
