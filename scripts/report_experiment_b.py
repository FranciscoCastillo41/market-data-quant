"""Aggregate Experiment B results into a single comparison table and print it.

For each run we pull the first-touch-Tier-1 row from the touch summary and
compare its 15-min window-max hit-rate and expectancy against the baseline's
random-bar 15-min hit-rate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mdq.config import RESULTS_DIR

RES = RESULTS_DIR / "experiment_b"


def load_pair(ticker: str, grid: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag = f"{ticker}__{grid}__{window}"
    summary = pd.read_csv(RES / f"summary__{tag}.csv")
    baseline = pd.read_csv(RES / f"baseline__{tag}.csv")
    return summary, baseline


def extract_row(summary: pd.DataFrame, group_name: str) -> dict:
    matches = summary[summary["group"] == group_name]
    if matches.empty:
        return {}
    r = matches.iloc[0]
    return {
        "n": int(r["n"]),
        "hit_15_min": r.get("hit_15_min"),
        "fp_hit": r.get("fp_hit"),
        "fp_expectancy_usd": r.get("fp_expectancy_usd"),
        "mfe_15_mean": r.get("mfe_15_mean"),
        "mae_15_mean": r.get("mae_15_mean"),
    }


def build_comparison() -> pd.DataFrame:
    runs = [
        # (ticker, grid_name, window_name)
        ("SPY", "step_0.50_t1_5_t2_1", "morning_0930_1115"),
        ("SPY", "step_0.50_t1_5_t2_1", "rth_0930_1600"),
        ("SPY", "step_1.00_t1_5",      "morning_0930_1115"),
        ("SPY", "step_1.00_t1_5",      "rth_0930_1600"),
        ("SPY", "step_5.00",           "morning_0930_1115"),
        ("SPY", "step_5.00",           "rth_0930_1600"),
        ("QQQ", "step_1.00_t1_10_t2_5", "morning_0930_1115"),
        ("QQQ", "step_1.00_t1_10_t2_5", "rth_0930_1600"),
        ("QQQ", "step_2.50_t1_10_t2_5", "morning_0930_1115"),
        ("QQQ", "step_2.50_t1_10_t2_5", "rth_0930_1600"),
        ("QQQ", "step_5.00",           "morning_0930_1115"),
        ("QQQ", "step_5.00",           "rth_0930_1600"),
        ("QQQ", "step_10.00",          "morning_0930_1115"),
        ("QQQ", "step_10.00",          "rth_0930_1600"),
    ]

    rows = []
    for ticker, grid, window in runs:
        summary, baseline = load_pair(ticker, grid, window)
        all_row = extract_row(summary, "ALL")
        t1_row = extract_row(summary, "first_touch_tier_1")
        t2_row = extract_row(summary, "first_touch_tier_2")
        t3_row = extract_row(summary, "first_touch_tier_3")

        # Baseline: 15-min horizon, prob up >= 0.30 and dn < 0.25 OR vice versa
        bl15 = baseline[baseline["horizon"] == 15]
        if not bl15.empty:
            b = bl15.iloc[0]
            # Use the max of up-direction and down-direction as the "best-case
            # random direction" baseline - fair since the touch picks a side.
            base_hit = max(
                float(b["p_up_ge_target_no_stop"]),
                float(b["p_dn_ge_target_no_stop"]),
            )
            base_mean_up = float(b["mean_max_up"])
            base_mean_dn = float(b["mean_max_dn"])
        else:
            base_hit = float("nan")
            base_mean_up = float("nan")
            base_mean_dn = float("nan")

        def row_with_tag(tag: str, src: dict) -> dict:
            return {
                "ticker": ticker,
                "grid": grid,
                "window": window,
                "bucket": tag,
                "n": src.get("n"),
                "hit_15": src.get("hit_15_min"),
                "base_hit_15": base_hit,
                "edge_pp": (src.get("hit_15_min") - base_hit) if src.get("hit_15_min") is not None else None,
                "fp_hit": src.get("fp_hit"),
                "fp_exp": src.get("fp_expectancy_usd"),
                "mfe15": src.get("mfe_15_mean"),
                "mae15": src.get("mae_15_mean"),
                "base_mean_up": base_mean_up,
                "base_mean_dn": base_mean_dn,
            }

        rows.append(row_with_tag("ALL", all_row))
        if t1_row:
            rows.append(row_with_tag("first_T1", t1_row))
        if t2_row:
            rows.append(row_with_tag("first_T2", t2_row))
        if t3_row:
            rows.append(row_with_tag("first_T3", t3_row))

    return pd.DataFrame(rows)


def main() -> None:
    df = build_comparison()
    df.to_csv(RES / "_comparison.csv", index=False)

    # Pretty print: filter to key columns
    show_cols = [
        "ticker", "grid", "window", "bucket", "n",
        "hit_15", "base_hit_15", "edge_pp",
        "fp_exp", "mfe15", "mae15",
    ]
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)

    def fmt(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int,)):
            return str(x)
        return f"{x:.4f}"

    view = df[show_cols].copy()
    for col in ("hit_15", "base_hit_15", "edge_pp", "fp_exp", "mfe15", "mae15"):
        view[col] = view[col].apply(fmt)

    print(view.to_string(index=False))
    print(f"\nFull comparison saved to: {RES / '_comparison.csv'}")


if __name__ == "__main__":
    main()
