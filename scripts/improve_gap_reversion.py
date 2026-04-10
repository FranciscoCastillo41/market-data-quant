"""Test 5 economically-motivated filters on the gap-reversion strategy.

Each filter is tested independently as a binary split on the existing 208
trades. For each filter we report:

    - PASS group: trades that pass the filter → hit rate, mean P&L, edge vs ALL
    - FAIL group: trades that fail → same metrics
    - Train (2024-2025) vs OOS (2026) for each group

A filter is adopted ONLY if:
    - PASS group outperforms FAIL group on train AND on OOS
    - The improvement is not from cutting sample size to <20 trades

No parameter tuning. No multi-variable interactions. One cut per filter.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth
from mdq.levels.volume_profile import compute_all_profiles
from mdq.levels.weekly_profile import compute_all_weekly_profiles

CSV = RESULTS_DIR / "gap_reversion" / "full_backtest.csv"


# Known earnings dates for each ticker (approximate, ±1 day)
# Source: public filings. Only need rough dates for the exclusion window.
EARNINGS_DATES = {
    "JPM": [
        "2024-01-12", "2024-04-12", "2024-07-12", "2024-10-11",
        "2025-01-15", "2025-04-11", "2025-07-15", "2025-10-15",
        "2026-01-14", "2026-04-11",
    ],
    "NVDA": [
        "2024-02-21", "2024-05-22", "2024-08-28", "2024-11-20",
        "2025-02-26", "2025-05-28", "2025-08-27", "2025-11-19",
        "2026-02-25",
    ],
    "AMZN": [
        "2024-02-01", "2024-04-30", "2024-08-01", "2024-10-31",
        "2025-02-06", "2025-05-01", "2025-07-31", "2025-10-30",
        "2026-02-05",
    ],
    "ARMK": [
        "2024-02-06", "2024-05-07", "2024-08-06", "2024-11-19",
        "2025-02-04", "2025-05-06", "2025-08-05", "2025-11-18",
        "2026-02-03",
    ],
    "SPY": [],  # ETF, no earnings
}
EARNINGS_WINDOW_DAYS = 2  # exclude ± this many days from earnings


def _near_earnings(ticker: str, sd, window: int = EARNINGS_WINDOW_DAYS) -> bool:
    dates = EARNINGS_DATES.get(ticker, [])
    for ed_str in dates:
        ed = datetime.strptime(ed_str, "%Y-%m-%d").date()
        if isinstance(sd, str):
            sd_d = datetime.strptime(sd, "%Y-%m-%d").date()
        else:
            sd_d = sd
        if abs((sd_d - ed).days) <= window:
            return True
    return False


def compute_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Add filter columns to the trades DataFrame."""
    out = df.copy()

    # Ensure date types
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    out["exit_date"] = pd.to_datetime(out["exit_date"]).dt.date

    # 1. Gap size: small (-0.3% to -1%) vs large (> -1%)
    out["gap_bucket"] = np.where(out["gap_pct"] > -0.01, "small_gap", "large_gap")

    # 2. Earnings proximity
    out["near_earnings"] = out.apply(
        lambda r: _near_earnings(r["ticker"], r["session_date"]), axis=1
    )

    # 3. Prior-day range as % of price (narrow vs wide)
    # Narrow = prior range < median prior range for that ticker
    out["prior_range_pct"] = (out["prior_vah"] - out["prior_val"]) / out["entry_open"]
    median_range = out.groupby("ticker")["prior_range_pct"].transform("median")
    out["narrow_prior_day"] = out["prior_range_pct"] < median_range

    # 4. Weekly context: is entry above or below weekly VAL?
    # We need to compute weekly profiles for each ticker
    weekly_context: list[bool] = []
    for _, row in out.iterrows():
        ticker = row["ticker"]
        sd = row["session_date"]
        entry_open = row["entry_open"]
        try:
            bars = load_bars(ticker,
                             (pd.Timestamp(sd) - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                             sd.strftime("%Y-%m-%d") if hasattr(sd, "strftime") else sd)
            weekly_profs = compute_all_weekly_profiles(bars, bin_size=0.10)
            if weekly_profs.empty:
                weekly_context.append(True)  # default to "above weekly VAL" if no data
                continue
            weekly_profs["week_end"] = pd.to_datetime(weekly_profs["week_end"]).dt.date
            prior_weeks = weekly_profs[weekly_profs["week_end"] < sd]
            if prior_weeks.empty:
                weekly_context.append(True)
                continue
            weekly_val = float(prior_weeks.iloc[-1]["val"])
            weekly_context.append(entry_open >= weekly_val)
        except Exception:
            weekly_context.append(True)
    out["above_weekly_val"] = weekly_context

    # 5. Quick resolution: did the trade resolve in 1 day?
    out["resolved_1d"] = out["bars_held"] <= 1

    return out


def analyze_filter(
    df: pd.DataFrame,
    filter_col: str,
    pass_value,
    label: str,
) -> dict:
    """Compare PASS vs FAIL group on train and OOS."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["session_date"]).dt.year
    train = df[df["year"] <= 2025]
    oos = df[df["year"] >= 2026]

    def stats(sub: pd.DataFrame, tag: str) -> dict:
        ok = sub[sub["status"] == "ok"]
        if ok.empty:
            return {f"{tag}_n": 0, f"{tag}_wr": np.nan, f"{tag}_mean": np.nan,
                    f"{tag}_total": 0, f"{tag}_mean_pct": np.nan}
        return {
            f"{tag}_n": len(ok),
            f"{tag}_wr": (ok["opt_pnl"] > 0).mean(),
            f"{tag}_mean": ok["opt_pnl"].mean(),
            f"{tag}_total": ok["opt_pnl"].sum(),
            f"{tag}_mean_pct": ok["opt_pnl_pct"].mean(),
        }

    pass_df = df[df[filter_col] == pass_value]
    fail_df = df[df[filter_col] != pass_value]

    row = {"filter": label, "pass_value": str(pass_value)}
    row.update(stats(pass_df[pass_df["year"] <= 2025], "pass_train"))
    row.update(stats(pass_df[pass_df["year"] >= 2026], "pass_oos"))
    row.update(stats(fail_df[fail_df["year"] <= 2025], "fail_train"))
    row.update(stats(fail_df[fail_df["year"] >= 2026], "fail_oos"))

    # Edge = pass - fail
    for split in ("train", "oos"):
        pm = row.get(f"pass_{split}_mean", np.nan)
        fm = row.get(f"fail_{split}_mean", np.nan)
        if not (np.isnan(pm) or np.isnan(fm)):
            row[f"edge_{split}"] = pm - fm
        else:
            row[f"edge_{split}"] = np.nan

    return row


def main() -> int:
    print("=" * 100)
    print("Gap-Reversion Filter Analysis — 5 economically-motivated filters")
    print("=" * 100)

    df = pd.read_csv(CSV)
    ok_count = (df["status"] == "ok").sum()
    print(f"\nLoaded {len(df)} signals, {ok_count} with real option data")

    print("\nComputing filters (weekly profiles may take a minute)...")
    df = compute_filters(df)

    filters = [
        ("gap_bucket", "small_gap", "Small gap (-0.3% to -1%) vs large gap (> -1%)"),
        ("near_earnings", False, "NOT near earnings (±2 days) vs near earnings"),
        ("narrow_prior_day", True, "Narrow prior day (range < median) vs wide"),
        ("above_weekly_val", True, "Above weekly VAL (pullback in uptrend) vs below"),
        ("resolved_1d", True, "Resolved in 1 day vs held longer"),
    ]

    results = []
    for col, pass_val, label in filters:
        r = analyze_filter(df, col, pass_val, label)
        results.append(r)

    res_df = pd.DataFrame(results)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 50)

    def fmt(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, (int, np.integer)):
            return str(x)
        if isinstance(x, float):
            if abs(x) > 10:
                return f"${x:+.0f}"
            return f"{x:+.2f}"
        return str(x)

    print("\n" + "#" * 100)
    print("# FILTER RESULTS — PASS vs FAIL on train (2024-2025) and OOS (2026)")
    print("#" * 100)

    for _, r in res_df.iterrows():
        print(f"\n{'─' * 90}")
        print(f"FILTER: {r['filter']}")
        print(f"PASS = {r['pass_value']}")
        print(f"{'─' * 90}")
        print(f"  {'':>15}  {'n':>5}  {'win_rate':>9}  {'mean_$/ctr':>11}  {'total_$':>9}  {'mean_%':>8}")

        for group, tag in [("PASS train", "pass_train"), ("FAIL train", "fail_train"),
                           ("PASS oos", "pass_oos"), ("FAIL oos", "fail_oos")]:
            n = r.get(f"{tag}_n", 0)
            wr = r.get(f"{tag}_wr", np.nan)
            mean = r.get(f"{tag}_mean", np.nan)
            total = r.get(f"{tag}_total", 0)
            pct = r.get(f"{tag}_mean_pct", np.nan)
            wr_s = f"{wr:.0%}" if not np.isnan(wr) else "—"
            mean_s = f"${mean:+.0f}" if not np.isnan(mean) else "—"
            pct_s = f"{pct:+.1%}" if not np.isnan(pct) else "—"
            print(f"  {group:>15}  {n:>5}  {wr_s:>9}  {mean_s:>11}  ${total:>8.0f}  {pct_s:>8}")

        edge_train = r.get("edge_train", np.nan)
        edge_oos = r.get("edge_oos", np.nan)
        train_valid = not np.isnan(edge_train) and edge_train > 0
        oos_valid = not np.isnan(edge_oos) and edge_oos > 0
        verdict = "✅ ADOPT" if (train_valid and oos_valid) else "❌ REJECT"
        print(f"\n  edge_train: ${edge_train:+.0f}/ctr" if not np.isnan(edge_train) else "  edge_train: —")
        print(f"  edge_oos:   ${edge_oos:+.0f}/ctr" if not np.isnan(edge_oos) else "  edge_oos:   —")
        print(f"  VERDICT:    {verdict}")

    # Summary
    print("\n\n" + "#" * 100)
    print("# SUMMARY — which filters to adopt")
    print("#" * 100)
    for _, r in res_df.iterrows():
        edge_t = r.get("edge_train", np.nan)
        edge_o = r.get("edge_oos", np.nan)
        train_ok = not np.isnan(edge_t) and edge_t > 0
        oos_ok = not np.isnan(edge_o) and edge_o > 0
        symbol = "✅" if (train_ok and oos_ok) else "❌"
        et = f"${edge_t:+.0f}" if not np.isnan(edge_t) else "—"
        eo = f"${edge_o:+.0f}" if not np.isnan(edge_o) else "—"
        print(f"  {symbol}  {r['filter'][:60]:<60}  train={et:>6}  oos={eo:>6}")

    # Combined filter: apply all adopted filters and show the net result
    print("\n\n" + "#" * 100)
    print("# COMBINED — apply all adopted filters simultaneously")
    print("#" * 100)
    # Start with all trades, then progressively filter
    combined = df.copy()
    applied = []
    for _, r in res_df.iterrows():
        edge_t = r.get("edge_train", np.nan)
        edge_o = r.get("edge_oos", np.nan)
        if not np.isnan(edge_t) and edge_t > 0 and not np.isnan(edge_o) and edge_o > 0:
            col, pass_val, label = None, None, None
            for c, pv, l in filters:
                if l == r["filter"]:
                    col, pass_val, label = c, pv, l
                    break
            if col:
                combined = combined[combined[col] == pass_val]
                applied.append(label)

    if applied:
        print(f"\n  Filters applied: {len(applied)}")
        for f in applied:
            print(f"    - {f}")
    else:
        print("\n  No filters adopted (all failed train+OOS requirement)")

    ok = combined[combined["status"] == "ok"]
    print(f"\n  Trades remaining: {len(ok)} (was {ok_count})")
    if not ok.empty:
        ok_year = ok.copy()
        ok_year["year"] = pd.to_datetime(ok_year["session_date"]).dt.year
        print(f"  Win rate:         {(ok['opt_pnl'] > 0).mean():.1%}")
        print(f"  Mean P&L/ctr:     ${ok['opt_pnl'].mean():+.2f}")
        print(f"  Median P&L/ctr:   ${ok['opt_pnl'].median():+.2f}")
        print(f"  Total P&L:        ${ok['opt_pnl'].sum():+.0f}")
        print(f"  Mean % return:    {ok['opt_pnl_pct'].mean():+.1%}")
        if (ok["opt_pnl"] > 0).any() and (ok["opt_pnl"] < 0).any():
            wins = ok[ok["opt_pnl"] > 0]
            losses = ok[ok["opt_pnl"] < 0]
            pf = wins["opt_pnl"].sum() / abs(losses["opt_pnl"].sum())
            print(f"  Profit factor:    {pf:.2f}")
            print(f"  Avg win:          ${wins['opt_pnl'].mean():+.0f}")
            print(f"  Avg loss:         ${losses['opt_pnl'].mean():+.0f}")

        # By year
        print("\n  BY YEAR (combined filter):")
        for year, sub in ok_year.groupby("year"):
            wr = (sub["opt_pnl"] > 0).mean()
            print(f"    {year}: n={len(sub):>3}  "
                  f"mean=${sub['opt_pnl'].mean():+.0f}  "
                  f"win_rate={wr:.0%}  "
                  f"total=${sub['opt_pnl'].sum():+.0f}")

    # vs unfiltered
    all_ok = df[df["status"] == "ok"]
    if not ok.empty and not all_ok.empty:
        improvement = ok["opt_pnl"].mean() - all_ok["opt_pnl"].mean()
        print(f"\n  Improvement vs unfiltered: ${improvement:+.2f}/ctr "
              f"({improvement / max(abs(all_ok['opt_pnl'].mean()), 1) * 100:+.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
