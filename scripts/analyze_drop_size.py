"""Re-analyze Experiment B reactions splitting by approach magnitude.

Reads saved reactions parquet, joins approach-magnitude from bars, buckets
touches by drop-size, and reports hit rates vs the overall baseline.

This tests Randy's core claim: "$4+ exhaustion drops into a level are the
A-grade setups, grind approaches are skip." If the data supports it, we
should see a monotonic improvement in hit rate as approach magnitude grows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.levels.approach import add_approach_magnitude, classify_drop_size

RES = RESULTS_DIR / "experiment_b"

RUNS = [
    # (ticker, grid_name, window_name, window_hhmm)
    ("SPY", "step_5.00",           "morning_0930_1115", ("09:30", "11:15")),
    ("SPY", "step_5.00",           "rth_0930_1600",     ("09:30", "16:00")),
    ("SPY", "step_1.00_t1_5",      "morning_0930_1115", ("09:30", "11:15")),
    ("QQQ", "step_5.00",           "morning_0930_1115", ("09:30", "11:15")),
    ("QQQ", "step_10.00",          "morning_0930_1115", ("09:30", "11:15")),
]

START = "2023-01-03"
END = "2026-04-07"

TARGET = 0.30
STOP = 0.25
HORIZON = 15

BINS = (0.0, 0.5, 1.0, 2.0, 4.0, float("inf"))
LABELS = ("micro", "grind", "small", "normal", "exhaustion")


def hit_rate(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "hit": np.nan, "fp_hit": np.nan, "fp_exp": np.nan,
                "mfe_mean": np.nan, "mae_mean": np.nan}
    mfe = df[f"mfe_{HORIZON}"].to_numpy()
    mae = df[f"mae_{HORIZON}"].to_numpy()
    ok = (mfe >= TARGET) & (mae < STOP)
    vc = df["fp_outcome"].value_counts(dropna=False)
    total = len(df)
    fp_hit = vc.get("target", 0) / total
    fp_exp = (vc.get("target", 0) * TARGET - vc.get("stop", 0) * STOP) / total
    return {
        "n": total,
        "hit": float(ok.mean()),
        "fp_hit": float(fp_hit),
        "fp_exp": float(fp_exp),
        "mfe_mean": float(np.nanmean(mfe)),
        "mae_mean": float(np.nanmean(mae)),
    }


def analyze_run(
    ticker: str,
    grid_name: str,
    window_name: str,
    window: tuple[str, str],
) -> pd.DataFrame:
    tag = f"{ticker}__{grid_name}__{window_name}"
    reactions_path = RES / f"reactions__{tag}.parquet"
    baseline_path = RES / f"baseline__{tag}.csv"
    reactions = pd.read_parquet(reactions_path)
    baseline = pd.read_csv(baseline_path)
    base_hit = max(
        float(baseline[baseline["horizon"] == HORIZON]["p_up_ge_target_no_stop"].iloc[0]),
        float(baseline[baseline["horizon"] == HORIZON]["p_dn_ge_target_no_stop"].iloc[0]),
    )

    bars = load_bars(ticker, START, END)
    bars_win = filter_window(bars, window[0], window[1])
    reactions = add_approach_magnitude(bars_win, reactions, lookback_minutes=30)

    rows = []
    for flavor in ("approach_from_open", "approach_swing"):
        reactions["_bucket"] = classify_drop_size(reactions[flavor], BINS, LABELS)
        for bucket in LABELS:
            sub = reactions[reactions["_bucket"] == bucket]
            first = sub[sub["touch_num"] == 1]
            row = {
                "ticker": ticker,
                "grid": grid_name,
                "window": window_name,
                "flavor": flavor,
                "bucket": bucket,
                **hit_rate(first),
                "base_hit": base_hit,
            }
            row["edge_pp"] = row["hit"] - base_hit if not np.isnan(row["hit"]) else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for ticker, grid, window_name, window in RUNS:
        all_rows.append(analyze_run(ticker, grid, window_name, window))
    out = pd.concat(all_rows, ignore_index=True)
    out_path = RES / "_drop_size_analysis.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    view = out.copy()
    for col in ("hit", "fp_hit", "fp_exp", "mfe_mean", "mae_mean", "base_hit", "edge_pp"):
        view[col] = view[col].apply(fmt)
    print(view.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
