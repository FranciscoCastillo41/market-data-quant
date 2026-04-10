"""Reverse-engineer Francisco's 5 winning swing trades against volume profile.

For each trade:
  1. Pull 1-min bars for the ticker around entry date
  2. Compute prior-day volume profile (POC, VAH, VAL, H, L)
  3. Show where the entry happened relative to those levels
  4. Compute intraday volume context at entry time
  5. Look for common patterns across all 5 trades
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from mdq.data.bars import download_range_async, load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.levels.volume_profile import compute_session_profile
from mdq.levels.weekly_profile import _build_volume_at_price, compute_weekly_profile


TRADES = [
    {
        "ticker": "JPM", "direction": "call", "strike": 295,
        "entry_date": "2026-02-24", "exit_date": "2026-02-25",
        "entry_price_opt": 7.46, "exit_price_opt": 9.40,
        "pnl_pct": 0.26,
    },
    {
        "ticker": "NVDA", "direction": "call", "strike": 177.5,
        "entry_date": "2026-03-03", "exit_date": "2026-03-04",
        "entry_price_opt": 7.15, "exit_price_opt": 8.64,
        "pnl_pct": 0.21,
    },
    {
        "ticker": "AMZN", "direction": "call", "strike": 207.5,
        "entry_date": "2026-03-24", "exit_date": "2026-03-25",
        "entry_price_opt": 4.70, "exit_price_opt": 7.65,
        "pnl_pct": 0.63,
    },
    {
        "ticker": "JPM", "direction": "call", "strike": 290,
        "entry_date": "2026-04-02", "exit_date": "2026-04-08",
        "entry_price_opt": 11.85, "exit_price_opt": 21.80,
        "pnl_pct": 0.84,
    },
    {
        "ticker": "ARMK", "direction": "call", "strike": 42,
        "entry_date": "2026-04-02", "exit_date": "2026-04-08",
        "entry_price_opt": 1.44, "exit_price_opt": 2.05,
        "pnl_pct": 0.42,
    },
]


def _download_ticker_range(ticker: str, start: str, end: str) -> None:
    asyncio.run(
        download_range_async(ticker=ticker, start=start, end=end, overwrite=False)
    )


def analyze_one_trade(trade: dict) -> dict:
    ticker = trade["ticker"]
    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]

    # Pull 2 weeks of data around the trade for context
    start = (pd.Timestamp(entry_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = exit_date

    print(f"\n  [{ticker}] Pulling bars {start} -> {end}...")
    _download_ticker_range(ticker, start, end)

    try:
        bars = load_bars(ticker, start, end)
    except FileNotFoundError as e:
        print(f"    FAIL: {e}")
        return {"error": str(e)}

    rth = filter_rth(bars)
    if rth.empty:
        return {"error": "no RTH bars"}

    entry_d = datetime.strptime(entry_date, "%Y-%m-%d").date()
    exit_d = datetime.strptime(exit_date, "%Y-%m-%d").date()

    # --- Prior-day volume profile ---
    prior_sessions = rth[rth["session_date"] < entry_d]
    if prior_sessions.empty:
        return {"error": "no prior sessions"}

    # Find the last trading day before entry
    prior_dates = sorted(prior_sessions["session_date"].unique())
    prior_date = max(prior_dates)
    prior_bars = prior_sessions[prior_sessions["session_date"] == prior_date]
    prof = compute_session_profile(prior_bars)
    if prof is None:
        return {"error": "profile computation failed"}

    # --- Prior WEEK profile ---
    # Get the full week before entry
    week_start = (pd.Timestamp(entry_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    week_bars = rth[(rth["session_date"] >= datetime.strptime(week_start, "%Y-%m-%d").date())
                    & (rth["session_date"] < entry_d)]
    weekly_prof = compute_weekly_profile(week_bars, bin_size=0.10) if not week_bars.empty else None

    # --- Entry day bars ---
    entry_bars = rth[rth["session_date"] == entry_d]
    if entry_bars.empty:
        # Try next day
        next_dates = sorted(d for d in rth["session_date"].unique() if d >= entry_d)
        if next_dates:
            entry_d = next_dates[0]
            entry_bars = rth[rth["session_date"] == entry_d]
    if entry_bars.empty:
        return {"error": "no entry day bars"}

    entry_open = float(entry_bars.iloc[0]["o"])
    entry_high = float(entry_bars["h"].max())
    entry_low = float(entry_bars["l"].min())
    entry_close = float(entry_bars.iloc[-1]["c"])

    # --- Exit day bars ---
    exit_bars = rth[rth["session_date"] == exit_d]
    exit_close = float(exit_bars.iloc[-1]["c"]) if not exit_bars.empty else None

    # --- Position relative to levels ---
    poc = prof.poc
    vah = prof.vah
    val = prof.val
    prior_high = prof.high
    prior_low = prof.low
    prior_close_px = prof.close

    # Where did the entry-day open relative to levels?
    entry_vs_poc = entry_open - poc
    entry_vs_vah = entry_open - vah
    entry_vs_val = entry_open - val
    entry_vs_prior_low = entry_open - prior_low
    entry_vs_prior_high = entry_open - prior_high

    # Was entry-day open below VAL (buying the dip)?
    below_val = entry_open < val
    below_poc = entry_open < poc
    near_prior_low = abs(entry_open - prior_low) < (prior_high - prior_low) * 0.15

    # Entry day range relative to prior day
    prior_range = prior_high - prior_low
    entry_range = entry_high - entry_low
    range_ratio = entry_range / prior_range if prior_range > 0 else 0

    # How far did underlying move entry -> exit?
    if exit_close is not None:
        underlying_move = exit_close - entry_open
        underlying_pct = underlying_move / entry_open
    else:
        underlying_move = None
        underlying_pct = None

    result = {
        "ticker": ticker,
        "entry_date": str(entry_d),
        "exit_date": str(exit_d),
        "hold_days": (exit_d - entry_d).days if isinstance(exit_d, date) and isinstance(entry_d, date) else None,
        "prior_date": str(prior_date),
        "prior_poc": poc,
        "prior_vah": vah,
        "prior_val": val,
        "prior_high": prior_high,
        "prior_low": prior_low,
        "prior_close": prior_close_px,
        "prior_range": prior_range,
        "entry_open": entry_open,
        "entry_close": entry_close,
        "entry_high": entry_high,
        "entry_low": entry_low,
        "exit_close": exit_close,
        "entry_vs_poc": entry_vs_poc,
        "entry_vs_val": entry_vs_val,
        "entry_vs_vah": entry_vs_vah,
        "entry_vs_prior_low": entry_vs_prior_low,
        "below_val": below_val,
        "below_poc": below_poc,
        "near_prior_low": near_prior_low,
        "underlying_pct": underlying_pct,
        "opt_pnl_pct": trade["pnl_pct"],
        "strike": trade["strike"],
    }

    # Weekly context
    if weekly_prof is not None:
        result["weekly_poc"] = weekly_prof.poc
        result["weekly_vah"] = weekly_prof.vah
        result["weekly_val"] = weekly_prof.val
        result["entry_vs_weekly_poc"] = entry_open - weekly_prof.poc
        result["below_weekly_val"] = entry_open < weekly_prof.val
    else:
        result["weekly_poc"] = None
        result["weekly_vah"] = None
        result["weekly_val"] = None

    return result


def main() -> int:
    print("=" * 100)
    print("Reverse-engineering Francisco's 5 winning swing trades")
    print("=" * 100)

    results = []
    for trade in TRADES:
        print(f"\n{'─' * 80}")
        print(f"Trade: {trade['ticker']} {trade['direction'].upper()} ${trade['strike']}  "
              f"{trade['entry_date']} -> {trade['exit_date']}  +{trade['pnl_pct']:.0%}")
        print(f"{'─' * 80}")
        r = analyze_one_trade(trade)
        results.append(r)

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        print(f"\n  Prior-day ({r['prior_date']}) volume profile:")
        print(f"    high  = ${r['prior_high']:.2f}")
        print(f"    VAH   = ${r['prior_vah']:.2f}")
        print(f"    POC   = ${r['prior_poc']:.2f}")
        print(f"    VAL   = ${r['prior_val']:.2f}")
        print(f"    low   = ${r['prior_low']:.2f}")
        print(f"    close = ${r['prior_close']:.2f}")
        print(f"    range = ${r['prior_range']:.2f}")

        print(f"\n  Entry day ({r['entry_date']}):")
        print(f"    open  = ${r['entry_open']:.2f}")
        print(f"    close = ${r['entry_close']:.2f}")
        print(f"    high  = ${r['entry_high']:.2f}")
        print(f"    low   = ${r['entry_low']:.2f}")

        print(f"\n  Position vs levels:")
        print(f"    entry vs POC:        {r['entry_vs_poc']:+.2f}  "
              f"({'BELOW' if r['below_poc'] else 'ABOVE'} POC)")
        print(f"    entry vs VAL:        {r['entry_vs_val']:+.2f}  "
              f"({'BELOW' if r['below_val'] else 'ABOVE'} VAL)")
        print(f"    entry vs prior_low:  {r['entry_vs_prior_low']:+.2f}  "
              f"({'NEAR' if r['near_prior_low'] else 'far from'} prior low)")

        if r.get("weekly_poc") is not None:
            print(f"\n  Weekly context:")
            print(f"    weekly POC = ${r['weekly_poc']:.2f}  "
                  f"(entry {r.get('entry_vs_weekly_poc', 0):+.2f})")
            print(f"    weekly VAL = ${r['weekly_val']:.2f}  "
                  f"({'BELOW' if r.get('below_weekly_val') else 'ABOVE'})")

        if r["underlying_pct"] is not None:
            print(f"\n  Underlying move: {r['underlying_pct']:+.2%}")

    # ================================================================
    # CROSS-TRADE PATTERN ANALYSIS
    # ================================================================
    print("\n\n" + "=" * 100)
    print("CROSS-TRADE PATTERN ANALYSIS")
    print("=" * 100)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid trades to analyze")
        return 1

    df = pd.DataFrame(valid)

    print("\nCommon features across all 5 trades:")
    print(f"  All calls (bullish): YES — every trade is a call")
    print(f"\n  Entry vs prior-day POC:")
    for _, r in df.iterrows():
        print(f"    {r['ticker']:>5} {r['entry_date']}: entry ${r['entry_open']:.2f}  "
              f"vs POC ${r['prior_poc']:.2f}  ({r['entry_vs_poc']:+.2f}  "
              f"{'BELOW' if r['below_poc'] else 'ABOVE'})")

    print(f"\n  Entry vs prior-day VAL:")
    for _, r in df.iterrows():
        print(f"    {r['ticker']:>5} {r['entry_date']}: entry ${r['entry_open']:.2f}  "
              f"vs VAL ${r['prior_val']:.2f}  ({r['entry_vs_val']:+.2f}  "
              f"{'BELOW' if r['below_val'] else 'ABOVE'})")

    print(f"\n  Entry near prior-day low?:")
    for _, r in df.iterrows():
        pct_from_low = (r['entry_open'] - r['prior_low']) / r['prior_range'] * 100
        print(f"    {r['ticker']:>5} {r['entry_date']}: entry ${r['entry_open']:.2f}  "
              f"vs low ${r['prior_low']:.2f}  ({pct_from_low:+.1f}% of prior range)")

    print(f"\n  Entry vs prior-day close:")
    for _, r in df.iterrows():
        gap = r['entry_open'] - r['prior_close']
        gap_pct = gap / r['prior_close'] * 100
        print(f"    {r['ticker']:>5} {r['entry_date']}: open ${r['entry_open']:.2f}  "
              f"vs prior close ${r['prior_close']:.2f}  (gap {gap_pct:+.2f}%)")

    # Weekly context
    print(f"\n  Entry vs weekly POC:")
    for _, r in df.iterrows():
        if r.get("weekly_poc") is not None:
            dist = r['entry_open'] - r['weekly_poc']
            print(f"    {r['ticker']:>5} {r['entry_date']}: entry ${r['entry_open']:.2f}  "
                  f"vs weekly POC ${r['weekly_poc']:.2f}  ({dist:+.2f})")

    # Summary stats
    print(f"\n\n  SUMMARY STATISTICS:")
    print(f"  {'':>5}  {'below_poc':>10}  {'below_val':>10}  {'near_low':>10}  {'opt_pnl':>10}")
    for _, r in df.iterrows():
        print(f"  {r['ticker']:>5}  {str(r['below_poc']):>10}  {str(r['below_val']):>10}  "
              f"{str(r['near_prior_low']):>10}  {r['opt_pnl_pct']:>+10.0%}")

    n_below_poc = df["below_poc"].sum()
    n_below_val = df["below_val"].sum()
    n_near_low = df["near_prior_low"].sum()
    print(f"\n  Count below POC:    {n_below_poc} / {len(df)}")
    print(f"  Count below VAL:    {n_below_val} / {len(df)}")
    print(f"  Count near low:     {n_near_low} / {len(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
