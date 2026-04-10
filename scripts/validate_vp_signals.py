"""Validate the two Experiment-C findings across year / approach / time splits:

  1. QQQ prior_low first-touch momentum 1.00/1.50
  2. SPY VAH first-touch fade 1.50/1.50

Same split methodology as validate_qqq_momentum.py: for each split (year,
approach direction, time-of-day bucket) recompute hit/exp vs matched
baseline and flag any year/bucket where the signal collapses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.stats.grid_search import GridSpec, grid_search

RES_C = RESULTS_DIR / "experiment_c"
START = "2023-01-03"
END = "2026-04-07"

WINDOW = ("09:30", "11:15")

# Geometries per signal
QQQ_GEOMETRIES = [
    ("momentum", 1.00, 1.50),
    ("momentum", 0.75, 1.00),
    ("momentum", 0.50, 1.00),
]
SPY_GEOMETRIES = [
    ("fade", 1.50, 1.50),
    ("fade", 1.00, 1.00),
    ("fade", 0.75, 0.75),
]


def run_single_geometry(
    bars_win: pd.DataFrame,
    touches: pd.DataFrame,
    direction: str,
    target: float,
    stop: float,
) -> dict:
    if touches.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan, "expectancy": np.nan}
    spec = GridSpec(
        targets=(target,), stops=(stop,), horizons=(15,), directions=(direction,),
    )
    out = grid_search(bars_win, touches, spec)
    if out.empty:
        return {"n": 0, "hit": np.nan, "stop_rate": np.nan, "expectancy": np.nan}
    row = out.iloc[0]
    return {
        "n": int(row["n"]),
        "hit": float(row["hit"]),
        "stop_rate": float(row["stop_rate"]),
        "expectancy": float(row["expectancy"]),
    }


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


def add_splits(reactions: pd.DataFrame) -> pd.DataFrame:
    out = reactions.copy()
    ts = pd.to_datetime(out["ts_et"])
    out["year"] = ts.dt.year
    minutes_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["min_since_open"] = minutes_of_day - (9 * 60 + 30)
    out["time_bucket"] = pd.cut(
        out["min_since_open"],
        bins=[-1, 15, 60, 10_000],
        labels=["0_15min", "15_60min", "60plus"],
    )
    return out


def validate_subset(
    label: str,
    ticker: str,
    touches: pd.DataFrame,
    bars_win: pd.DataFrame,
    geometries: list[tuple[str, float, float]],
) -> pd.DataFrame:
    touches = add_splits(touches)
    baseline = make_pseudo_touches(bars_win)
    baseline = add_splits(baseline)

    rows: list[dict] = []

    def add(direction, target, stop, split_name, split_value, sub_r, sub_b):
        r = run_single_geometry(bars_win, sub_r, direction, target, stop)
        b = run_single_geometry(bars_win, sub_b, direction, target, stop)
        rows.append({
            "label": label,
            "ticker": ticker,
            "direction": direction,
            "target": target,
            "stop": stop,
            "split": split_name,
            "value": str(split_value),
            "n_touch": r["n"],
            "n_base": b["n"],
            "hit_touch": r["hit"],
            "hit_base": b["hit"],
            "hit_lift": (r["hit"] - b["hit"]) if not (np.isnan(r["hit"]) or np.isnan(b["hit"])) else np.nan,
            "exp_touch": r["expectancy"],
            "exp_base": b["expectancy"],
            "edge": (r["expectancy"] - b["expectancy"]) if not (np.isnan(r["expectancy"]) or np.isnan(b["expectancy"])) else np.nan,
        })

    for direction, target, stop in geometries:
        add(direction, target, stop, "ALL", "ALL", touches, baseline)
        for y, sub in touches.groupby("year"):
            add(direction, target, stop, "year", y, sub, baseline[baseline["year"] == y])
        for app, sub in touches.groupby("approach"):
            add(direction, target, stop, "approach", app, sub, baseline[baseline["approach"] == app])
        for b_, sub in touches.groupby("time_bucket", observed=True):
            add(direction, target, stop, "time", b_, sub, baseline[baseline["time_bucket"] == b_])

    return pd.DataFrame(rows)


def main() -> None:
    results: list[pd.DataFrame] = []

    # --- QQQ prior_low first-touch ---
    print("=" * 80)
    print("QQQ prior_low first-touch momentum")
    print("=" * 80)
    qqq = pd.read_parquet(RES_C / "reactions__QQQ.parquet")
    qqq_pl = qqq[(qqq["level_name"] == "prior_low") & (qqq["touch_num"] == 1)].reset_index(drop=True)
    print(f"  total first-touches: {len(qqq_pl)}")

    qqq_bars = load_bars("QQQ", START, END)
    qqq_bars_win = filter_window(qqq_bars, WINDOW[0], WINDOW[1])

    # All prior_low first-touches
    results.append(
        validate_subset("QQQ_prior_low_ALL", "QQQ", qqq_pl, qqq_bars_win, QQQ_GEOMETRIES)
    )

    # Only non-confluence (where edge was strongest in Experiment C)
    qqq_pl_nc = qqq_pl[qqq_pl["confluence"] == False].reset_index(drop=True)  # noqa: E712
    print(f"  non-confluence first-touches: {len(qqq_pl_nc)}")
    results.append(
        validate_subset("QQQ_prior_low_NC", "QQQ", qqq_pl_nc, qqq_bars_win, QQQ_GEOMETRIES)
    )

    # --- SPY VAH first-touch ---
    print()
    print("=" * 80)
    print("SPY VAH first-touch fade")
    print("=" * 80)
    spy = pd.read_parquet(RES_C / "reactions__SPY.parquet")
    spy_vah = spy[(spy["level_name"] == "vah") & (spy["touch_num"] == 1)].reset_index(drop=True)
    print(f"  total first-touches: {len(spy_vah)}")

    spy_bars = load_bars("SPY", START, END)
    spy_bars_win = filter_window(spy_bars, WINDOW[0], WINDOW[1])

    results.append(
        validate_subset("SPY_vah_ALL", "SPY", spy_vah, spy_bars_win, SPY_GEOMETRIES)
    )

    spy_vah_c = spy_vah[spy_vah["confluence"] == True].reset_index(drop=True)  # noqa: E712
    print(f"  confluence-true first-touches: {len(spy_vah_c)}")
    results.append(
        validate_subset("SPY_vah_C", "SPY", spy_vah_c, spy_bars_win, SPY_GEOMETRIES)
    )

    full = pd.concat(results, ignore_index=True)
    full.to_csv(RES_C / "_validation_splits.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 25)

    def fmt(x, p=4):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        return f"{x:.{p}f}"

    view_cols = [
        "label", "direction", "target", "stop", "split", "value",
        "n_touch", "n_base",
        "hit_touch", "hit_base", "hit_lift",
        "exp_touch", "exp_base", "edge",
    ]

    def print_block(label: str, direction: str, target: float, stop: float):
        sub = full[
            (full["label"] == label)
            & (full["direction"] == direction)
            & (full["target"] == target)
            & (full["stop"] == stop)
        ].copy()
        print(f"\n--- {label} :: {direction} {target}/{stop} ---")
        view = sub[view_cols].copy()
        for col in ("hit_touch", "hit_base", "hit_lift", "exp_touch", "exp_base", "edge"):
            view[col] = view[col].apply(fmt)
        print(view.to_string(index=False))

    print("\n" + "#" * 100)
    print("# FOCUS 1: QQQ prior_low ALL, momentum 1.00/1.50 (THE FLAGSHIP)")
    print("#" * 100)
    print_block("QQQ_prior_low_ALL", "momentum", 1.00, 1.50)

    print("\n" + "#" * 100)
    print("# FOCUS 2: QQQ prior_low NON-CONFLUENCE, momentum 1.00/1.50 (cleanest subset)")
    print("#" * 100)
    print_block("QQQ_prior_low_NC", "momentum", 1.00, 1.50)

    print("\n" + "#" * 100)
    print("# FOCUS 3: QQQ prior_low NON-CONFLUENCE, momentum 0.75/1.00 (tighter)")
    print("#" * 100)
    print_block("QQQ_prior_low_NC", "momentum", 0.75, 1.00)

    print("\n" + "#" * 100)
    print("# FOCUS 4: SPY VAH ALL, fade 1.50/1.50")
    print("#" * 100)
    print_block("SPY_vah_ALL", "fade", 1.50, 1.50)

    print("\n" + "#" * 100)
    print("# FOCUS 5: SPY VAH CONFLUENCE, fade 1.50/1.50")
    print("#" * 100)
    print_block("SPY_vah_C", "fade", 1.50, 1.50)

    print("\n" + "#" * 100)
    print("# FOCUS 6: SPY VAH ALL, fade 1.00/1.00 (tighter)")
    print("#" * 100)
    print_block("SPY_vah_ALL", "fade", 1.00, 1.00)

    print(f"\nFull results saved: {RES_C / '_validation_splits.csv'}")


if __name__ == "__main__":
    main()
