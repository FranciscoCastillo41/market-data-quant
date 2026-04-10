"""Baseline grid search: run the same target/stop sweep on RANDOM bars in
the same time window as the touches, and compare the top results to the
touch-conditional results.

If the baseline gives similar expectancies, the apparent edge in the
touch grid search is an artifact of asymmetric target/stop geometry and
market drift — NOT a touch signal.
"""

from __future__ import annotations

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
    ("SPY", "morning_0930_1115", ("09:30", "11:15")),
    ("SPY", "rth_0930_1600",     ("09:30", "16:00")),
    ("QQQ", "morning_0930_1115", ("09:30", "11:15")),
]

SPEC = GridSpec(
    targets=(0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50),
    stops=(0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.00, 1.50),
    horizons=(15,),
    directions=("fade", "momentum"),
)


def make_pseudo_touches(bars: pd.DataFrame, every_nth: int = 10, seed: int = 42) -> pd.DataFrame:
    """Sample every Nth bar as a pseudo-touch with randomly assigned approach sign."""
    rng = np.random.default_rng(seed)
    picks = bars.iloc[::every_nth].copy().reset_index(drop=True)
    # Randomly assign approach
    approach = rng.choice(["from_below", "from_above"], size=len(picks))
    picks["approach"] = approach
    picks["level"] = picks["c"]  # dummy, not used by grid_search
    picks["touch_num"] = 1
    picks["open_price"] = picks.groupby("session_date")["o"].transform("first")
    picks["entry_close"] = picks["c"]
    picks["tier"] = 1
    picks["bar_idx"] = picks.index
    return picks


def main() -> None:
    rows = []
    for ticker, window_name, window in CONFIGS:
        print(f"\nBaseline grid search: {ticker} {window_name}")
        bars = load_bars(ticker, START, END)
        bars_win = filter_window(bars, window[0], window[1])
        pseudo = make_pseudo_touches(bars_win, every_nth=10)
        print(f"  n pseudo-touches: {len(pseudo):,}")

        df = grid_search(bars_win, pseudo, SPEC)
        df.insert(0, "config", f"{ticker}_{window_name}_BASELINE")
        rows.append(df)

        # Show top 5 per direction for the baseline
        for direction in ("fade", "momentum"):
            sub = df[df["direction"] == direction]
            top = sub.nlargest(5, "expectancy")
            print(f"\n  top 5 BASELINE {direction}:")
            print(
                top[["target", "stop", "hit", "stop_rate", "expectancy", "r_multiple"]]
                .to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )

    if rows:
        full = pd.concat(rows, ignore_index=True)
        full.to_csv(RES / "_grid_search_baseline.csv", index=False)

        # Side-by-side: load touch grid results and compute edge
        touch_results = pd.read_csv(RES / "_grid_search.csv")

        print("\n\n" + "=" * 100)
        print("TOUCH vs BASELINE edge (touch_expectancy - baseline_expectancy)")
        print("Matched on (direction, target, stop). Positive = touches beat random.")
        print("=" * 100)

        merge_keys = ["direction", "target", "stop"]
        # Normalize config naming for join
        touch_results["ticker_window"] = touch_results["config"].str.extract(
            r"(SPY|QQQ)_step_[0-9._t]+_(.+?)_"
        )[0] + "_" + touch_results["config"].str.extract(
            r"(SPY|QQQ)_step_[0-9._t]+_(.+?)_"
        )[1]
        # Simpler: just strip the subset tag
        touch_results["config_base"] = touch_results["config"].str.rsplit("_", n=1).str[0]

        for (ticker, window_name, _) in CONFIGS:
            bl_tag = f"{ticker}_{window_name}_BASELINE"
            bl = full[full["config"] == bl_tag]
            # Find all touch configs for this ticker/window
            # They're named like "SPY_step_5.00_morning_0930_1115_ALL"
            tr = touch_results[
                touch_results["config"].str.startswith(ticker)
                & touch_results["config"].str.contains(window_name)
            ]
            if tr.empty:
                continue
            for tcfg in tr["config"].unique():
                sub_t = tr[tr["config"] == tcfg]
                merged = sub_t.merge(
                    bl[merge_keys + ["expectancy", "hit", "stop_rate"]].rename(
                        columns={
                            "expectancy": "base_exp",
                            "hit": "base_hit",
                            "stop_rate": "base_stop_rate",
                        }
                    ),
                    on=merge_keys,
                    how="left",
                )
                merged["edge_exp"] = merged["expectancy"] - merged["base_exp"]
                top_edge = merged.nlargest(5, "edge_exp")
                print(f"\n-- {tcfg} --")
                print(
                    top_edge[[
                        "direction", "target", "stop", "n",
                        "hit", "base_hit",
                        "expectancy", "base_exp", "edge_exp",
                    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
                )


if __name__ == "__main__":
    main()
