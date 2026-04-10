"""Validate the QQQ $10 first-touch momentum signal with three splits:
  - year
  - approach direction (from_below -> breakout long, from_above -> breakdown short)
  - time of day (first 15 min, 15-60 min, 60+ min of window)

For each split, compute touch hit rate and expectancy at the best momentum
geometry (1.00/1.50) and compare to a matched baseline slice. Also run the
same validation for SPY $10 first-touches (filtered from the SPY $5 grid).
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

# Three geometries we want to test on every split
TEST_PARAMS = [
    ("momentum", 1.00, 1.50),  # the headline QQQ finding
    ("momentum", 0.75, 1.00),  # smaller / better risk-managed version
    ("momentum", 0.50, 1.00),  # tighter
    ("fade",     1.50, 1.50),  # SPY wide-fade finding
]


def run_single_geometry(
    bars: pd.DataFrame,
    touches: pd.DataFrame,
    direction: str,
    target: float,
    stop: float,
) -> dict:
    """Run grid_search on just one (direction, target, stop) combination."""
    if touches.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan,
                "timeout": np.nan, "expectancy": np.nan}
    spec = GridSpec(
        targets=(target,),
        stops=(stop,),
        horizons=(15,),
        directions=(direction,),
    )
    out = grid_search(bars, touches, spec)
    if out.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan,
                "timeout": np.nan, "expectancy": np.nan}
    row = out.iloc[0]
    return {
        "n": int(row["n"]),
        "hit": float(row["hit"]),
        "stop_rate": float(row["stop_rate"]),
        "timeout": float(row["timeout"]),
        "expectancy": float(row["expectancy"]),
    }


def make_pseudo_touches(bars: pd.DataFrame, every_nth: int, seed: int = 42) -> pd.DataFrame:
    """Sample every Nth bar with a random approach sign for the baseline."""
    rng = np.random.default_rng(seed)
    picks = bars.iloc[::every_nth].copy().reset_index(drop=True)
    picks["approach"] = rng.choice(["from_below", "from_above"], size=len(picks))
    picks["level"] = picks["c"]
    picks["touch_num"] = 1
    picks["open_price"] = picks.groupby("session_date")["o"].transform("first")
    picks["entry_close"] = picks["c"]
    picks["tier"] = 1
    picks["bar_idx"] = picks.index
    return picks


def add_splits(reactions: pd.DataFrame) -> pd.DataFrame:
    """Add year, minutes_since_open, time_bucket to a reactions frame."""
    out = reactions.copy()
    ts = pd.to_datetime(out["ts_et"])
    out["year"] = ts.dt.year
    # minutes since 09:30 ET
    minutes_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["min_since_open"] = minutes_of_day - (9 * 60 + 30)
    out["time_bucket"] = pd.cut(
        out["min_since_open"],
        bins=[-1, 15, 60, 10_000],
        labels=["0_15min", "15_60min", "60plus"],
    )
    return out


def validate_config(
    ticker: str,
    level_step: float,
    reactions: pd.DataFrame,
    bars_win: pd.DataFrame,
) -> pd.DataFrame:
    """Run all validations for a given ticker / level-step reactions frame."""
    reactions = add_splits(reactions)

    # Matched baseline: same window, pseudo-touches, same splits
    baseline_touches = make_pseudo_touches(bars_win, every_nth=10)
    baseline_touches = add_splits(baseline_touches)

    rows: list[dict] = []

    def add_row(split_name: str, split_value, sub_r: pd.DataFrame, sub_b: pd.DataFrame) -> None:
        for direction, target, stop in TEST_PARAMS:
            r = run_single_geometry(bars_win, sub_r, direction, target, stop)
            b = run_single_geometry(bars_win, sub_b, direction, target, stop)
            rows.append({
                "ticker": ticker,
                "step": level_step,
                "split": split_name,
                "value": split_value,
                "direction": direction,
                "target": target,
                "stop": stop,
                "n_touch": r["n"],
                "n_base": b["n"],
                "hit_touch": r["hit"],
                "hit_base": b["hit"],
                "exp_touch": r["expectancy"],
                "exp_base": b["expectancy"],
                "edge": (r["expectancy"] - b["expectancy"])
                         if not (np.isnan(r["expectancy"]) or np.isnan(b["expectancy"])) else np.nan,
                "hit_lift": (r["hit"] - b["hit"])
                             if not (np.isnan(r["hit"]) or np.isnan(b["hit"])) else np.nan,
            })

    # Overall
    add_row("ALL", "ALL", reactions, baseline_touches)

    # By year
    for year, sub in reactions.groupby("year"):
        bsub = baseline_touches[baseline_touches["year"] == year]
        add_row("year", year, sub, bsub)

    # By approach direction
    for approach, sub in reactions.groupby("approach"):
        bsub = baseline_touches[baseline_touches["approach"] == approach]
        add_row("approach", approach, sub, bsub)

    # By time of day
    for bucket, sub in reactions.groupby("time_bucket", observed=True):
        bsub = baseline_touches[baseline_touches["time_bucket"] == bucket]
        add_row("time_bucket", bucket, sub, bsub)

    return pd.DataFrame(rows)


def main() -> None:
    out_frames = []

    # --- QQQ $10 first-touch ---
    print("=" * 80)
    print("VALIDATION: QQQ $10 first-touch")
    print("=" * 80)
    qqq_reactions = pd.read_parquet(
        RES / "reactions__QQQ__step_10.00__morning_0930_1115.parquet"
    )
    qqq_reactions = qqq_reactions[qqq_reactions["touch_num"] == 1].reset_index(drop=True)
    print(f"  n first-touches: {len(qqq_reactions)}")

    qqq_bars = load_bars("QQQ", START, END)
    qqq_bars_win = filter_window(qqq_bars, "09:30", "11:15")
    out_frames.append(validate_config("QQQ", 10.0, qqq_reactions, qqq_bars_win))

    # --- SPY $10 first-touch (derived from SPY $5 reactions) ---
    print()
    print("=" * 80)
    print("VALIDATION: SPY $10 first-touch")
    print("=" * 80)
    spy_reactions = pd.read_parquet(
        RES / "reactions__SPY__step_5.00__morning_0930_1115.parquet"
    )
    # Filter to levels that are multiples of 10 AND 1st touch of that level
    is_ten = np.isclose(spy_reactions["level"] % 10.0, 0.0) | np.isclose(
        spy_reactions["level"] % 10.0, 10.0
    )
    spy10 = spy_reactions[is_ten & (spy_reactions["touch_num"] == 1)].reset_index(drop=True)
    print(f"  n first-touches: {len(spy10)}")

    spy_bars = load_bars("SPY", START, END)
    spy_bars_win = filter_window(spy_bars, "09:30", "11:15")
    out_frames.append(validate_config("SPY", 10.0, spy10, spy_bars_win))

    # --- SPY $5 first-touch Tier 1 (SPY $5-but-NOT-$10) ---
    print()
    print("=" * 80)
    print("VALIDATION: SPY $5 first-touch Tier 1 (pure $5, excluding $10)")
    print("=" * 80)
    is_five = np.isclose(spy_reactions["level"] % 5.0, 0.0) | np.isclose(
        spy_reactions["level"] % 5.0, 5.0
    )
    spy5only = spy_reactions[is_five & ~is_ten & (spy_reactions["touch_num"] == 1)].reset_index(
        drop=True
    )
    print(f"  n first-touches: {len(spy5only)}")
    out_frames.append(validate_config("SPY", 5.0, spy5only, spy_bars_win))

    full = pd.concat(out_frames, ignore_index=True)
    full.to_csv(RES / "_validation_splits.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 20)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    view_cols = [
        "ticker", "step", "split", "value",
        "direction", "target", "stop",
        "n_touch", "n_base",
        "hit_touch", "hit_base", "hit_lift",
        "exp_touch", "exp_base", "edge",
    ]

    def print_section(ticker, step, focus_geometry):
        direction, target, stop = focus_geometry
        sub = full[
            (full["ticker"] == ticker)
            & (full["step"] == step)
            & (full["direction"] == direction)
            & (full["target"] == target)
            & (full["stop"] == stop)
        ]
        print(f"\n--- {ticker} ${step:.0f} :: {direction} {target}/{stop} ---")
        view = sub[view_cols].copy()
        for col in ("hit_touch", "hit_base", "hit_lift", "exp_touch", "exp_base", "edge"):
            view[col] = view[col].apply(fmt)
        print(view.to_string(index=False))

    print("\n\n" + "#" * 100)
    print("# FOCUS: QQQ $10 momentum 1.00/1.50 across splits")
    print("#" * 100)
    print_section("QQQ", 10.0, ("momentum", 1.00, 1.50))

    print("\n\n" + "#" * 100)
    print("# FOCUS: QQQ $10 momentum 0.75/1.00 across splits (tighter geometry)")
    print("#" * 100)
    print_section("QQQ", 10.0, ("momentum", 0.75, 1.00))

    print("\n\n" + "#" * 100)
    print("# FOCUS: SPY $10 momentum 1.00/1.50 across splits (does QQQ finding transfer?)")
    print("#" * 100)
    print_section("SPY", 10.0, ("momentum", 1.00, 1.50))

    print("\n\n" + "#" * 100)
    print("# FOCUS: SPY $5 first-touch fade 1.50/1.50 across splits")
    print("#" * 100)
    print_section("SPY", 5.0, ("fade", 1.50, 1.50))

    print(f"\nFull results: {RES / '_validation_splits.csv'}")


if __name__ == "__main__":
    main()
