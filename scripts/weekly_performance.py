"""Weekly performance breakdown of the Tier 1 real-options backtest.

Reads tier1_real_options_backtest.csv (produced by real_option_backtest.py)
and produces:
  1. Trade-by-trade listing (most recent first) with a clean format
  2. Week-by-week aggregation showing cadence and P&L variance
  3. Last 4-week rolling summary (what you'd see if you'd been paper trading)

Usage:
    poetry run python3 scripts/weekly_performance.py
    poetry run python3 scripts/weekly_performance.py --n-recent 20
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR

CSV = RESULTS_DIR / "tier1_real_options_backtest.csv"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-recent", type=int, default=20,
                   help="How many most-recent trades to show in detail")
    p.add_argument("--weeks", type=int, default=0,
                   help="If >0, only show the last N weeks")
    args = p.parse_args()

    if not CSV.exists():
        print(f"Missing {CSV}. Run scripts/real_option_backtest.py first.")
        return 1

    df = pd.read_csv(CSV)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful trades in the CSV.")
        return 0

    ok["session_date"] = pd.to_datetime(ok["session_date"])
    ok["year"] = ok["session_date"].dt.year
    ok["iso_week"] = ok["session_date"].dt.isocalendar().week
    ok["year_week"] = ok["session_date"].dt.strftime("%G-W%V")
    ok = ok.sort_values("session_date", ascending=False).reset_index(drop=True)

    if args.weeks > 0:
        cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=args.weeks)
        ok = ok[ok["session_date"] >= cutoff]
        print(f"Showing last {args.weeks} weeks ({len(ok)} trades)\n")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 30)

    # ================================================================
    # 1. Trade-by-trade detail (most recent N)
    # ================================================================
    print("=" * 90)
    print(f"MOST RECENT {min(args.n_recent, len(ok))} TRADES")
    print("=" * 90)

    recent = ok.head(args.n_recent).copy()
    recent["date"] = recent["session_date"].dt.strftime("%Y-%m-%d (%a)")
    recent["entry_time"] = (
        pd.to_datetime(recent["entry_ts"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    recent["held_min"] = recent["bars_held"].astype(int)
    recent["entry_qqq"] = recent["entry_qqq"].map(lambda x: f"${x:.2f}")
    recent["strike"] = recent["strike"].astype(int).map(lambda x: f"${x}")
    recent["entry_opt"] = recent["entry_opt"].map(lambda x: f"${x:.2f}")
    recent["exit_opt"]  = recent["exit_opt"].map(lambda x: f"${x:.2f}")
    recent["pnl"] = recent["pnl_contract"].map(lambda x: f"${x:+.0f}")

    disp_cols = ["date", "entry_time", "entry_qqq", "strike", "sim_outcome",
                 "held_min", "entry_opt", "exit_opt", "pnl"]
    print(recent[disp_cols].to_string(index=False))

    # Summary line for recent slice
    recent_pnl = ok.head(args.n_recent)["pnl_contract"]
    recent_wr = (recent_pnl > 0).mean()
    print(f"\nRecent {len(recent_pnl)} trades: ")
    print(f"  total P&L:  ${recent_pnl.sum():+.0f}")
    print(f"  mean P&L:   ${recent_pnl.mean():+.2f}/trade")
    print(f"  win rate:   {recent_wr:.1%}")

    # ================================================================
    # 2. Week-by-week aggregation
    # ================================================================
    print("\n" + "=" * 90)
    print("WEEK-BY-WEEK P&L (per contract)")
    print("=" * 90)

    weekly = ok.groupby("year_week").agg(
        n=("pnl_contract", "size"),
        total_pnl=("pnl_contract", "sum"),
        mean_pnl=("pnl_contract", "mean"),
        wins=("pnl_contract", lambda x: (x > 0).sum()),
        week_start=("session_date", "min"),
    ).reset_index()
    weekly["win_rate"] = weekly["wins"] / weekly["n"]
    weekly = weekly.sort_values("week_start", ascending=False).reset_index(drop=True)
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")

    def fmt_money(x):
        return f"${x:+.0f}"

    disp_weekly = weekly[["year_week", "week_start", "n", "wins", "win_rate", "mean_pnl", "total_pnl"]].copy()
    disp_weekly["win_rate"] = disp_weekly["win_rate"].map(lambda x: f"{x:.0%}")
    disp_weekly["mean_pnl"] = disp_weekly["mean_pnl"].map(fmt_money)
    disp_weekly["total_pnl"] = disp_weekly["total_pnl"].map(fmt_money)
    disp_weekly.columns = ["week", "starting", "n", "W", "WR", "avg/trade", "total"]

    print(disp_weekly.head(30).to_string(index=False))
    print(f"\n(showing most recent {min(30, len(disp_weekly))} weeks out of {len(weekly)})")

    # ================================================================
    # 3. Last 4 weeks rolling
    # ================================================================
    print("\n" + "=" * 90)
    print("LAST 4 WEEKS — ROLLING VIEW")
    print("=" * 90)

    four_weeks_ago = ok["session_date"].max() - pd.Timedelta(weeks=4)
    last4 = ok[ok["session_date"] >= four_weeks_ago]
    if last4.empty:
        print("No trades in last 4 weeks of the dataset.")
    else:
        print(f"  date range:  {last4['session_date'].min().date()} -> {last4['session_date'].max().date()}")
        print(f"  trades:      {len(last4)}")
        print(f"  wins:        {(last4['pnl_contract'] > 0).sum()}")
        print(f"  win rate:    {(last4['pnl_contract'] > 0).mean():.1%}")
        print(f"  mean P&L:    ${last4['pnl_contract'].mean():+.2f}/trade")
        print(f"  total P&L:   ${last4['pnl_contract'].sum():+.0f}")
        print(f"  best trade:  ${last4['pnl_contract'].max():+.0f}")
        print(f"  worst trade: ${last4['pnl_contract'].min():+.0f}")

        # Run equity curve
        last4 = last4.sort_values("session_date").copy()
        last4["cumulative"] = last4["pnl_contract"].cumsum()
        print("\n  equity curve (chronological):")
        for _, row in last4.iterrows():
            date_str = row["session_date"].strftime("%Y-%m-%d %a")
            print(f"    {date_str}  {row['sim_outcome']:>8}  ${row['pnl_contract']:+6.0f}  cum=${row['cumulative']:+6.0f}")

    # ================================================================
    # 4. Big picture
    # ================================================================
    print("\n" + "=" * 90)
    print("FULL HISTORY (for context)")
    print("=" * 90)
    print(f"  trades with real option data:  {len(ok)}")
    print(f"  date range:                    {ok['session_date'].min().date()} -> {ok['session_date'].max().date()}")
    print(f"  mean P&L / contract:           ${ok['pnl_contract'].mean():+.2f}")
    print(f"  median P&L / contract:         ${ok['pnl_contract'].median():+.2f}")
    print(f"  win rate:                      {(ok['pnl_contract'] > 0).mean():.1%}")
    print(f"  total P&L (1 contract):        ${ok['pnl_contract'].sum():+.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
