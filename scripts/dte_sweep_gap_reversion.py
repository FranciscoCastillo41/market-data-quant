"""DTE sweep for the gap-reversion CZ strategy with real Alpaca option fills.

For each of the 208 historical signals, we test the CZ intraday entry/exit
(buy at 30-min low, sell on cum_delta exhaustion) using real option bars
across multiple DTE buckets:

    0DTE:       same-day expiry
    this_fri:   next Friday from signal date (1-5 DTE depending on weekday)
    next_fri:   Friday after next (6-12 DTE)
    monthly:    3rd Friday of the month (15-45 DTE)

For each DTE we fetch actual ATM call bars from Alpaca and compute:
    - Entry price = option bar close at the CZ entry bar
    - Exit price = option bar close at the CZ exit bar
    - P&L per contract in real dollars
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth
from mdq.live.alpaca import AlpacaClient

OUT = RESULTS_DIR / "gap_reversion"

# Load the intraday timing results (has entry_idx, exit_idx per signal)
TIMING_CSV = OUT / "intraday_timing.csv"
SIGNALS_CSV = OUT / "full_backtest.csv"


def _next_friday(d: date) -> date:
    days_ahead = 4 - d.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


def _next_next_friday(d: date) -> date:
    return _next_friday(_next_friday(d) + timedelta(days=1))


def _monthly_expiry(d: date) -> date:
    """3rd Friday of the current month. If already past, use next month's."""
    year, month = d.year, d.month
    # Find 3rd Friday
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    if third_friday <= d:
        # Use next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        first_day = date(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
    return third_friday


def _build_occ_call(underlying: str, exp: date, strike: float) -> str:
    yy = exp.strftime("%y")
    mm = exp.strftime("%m")
    dd = exp.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}C{strike_int:08d}"


def _fetch_day_option_bars(client: AlpacaClient, symbol: str, sd: date) -> pd.DataFrame | None:
    """Fetch all option bars for a symbol on a given date. Returns DataFrame or None."""
    start = datetime(sd.year, sd.month, sd.day, 9, 25)
    end = datetime(sd.year, sd.month, sd.day, 16, 5)
    try:
        bars = client.get_option_bars(symbol, start, end)
    except Exception:
        return None
    if not bars:
        return None
    opt_df = pd.DataFrame(bars)
    opt_df["t_dt"] = pd.to_datetime(opt_df["t"])
    if opt_df["t_dt"].dt.tz is not None:
        opt_df["t_dt"] = opt_df["t_dt"].dt.tz_localize(None)
    return opt_df


def _lookup_close_at(opt_df: pd.DataFrame, target_ts) -> float | None:
    """Find the bar closest to target_ts and return its close price.

    Handles both UTC and ET timestamps by converting everything to UTC.
    """
    if opt_df is None or opt_df.empty:
        return None
    import pytz
    ET = pytz.timezone("America/New_York")
    UTC = pytz.utc

    # Normalize target to UTC
    if hasattr(target_ts, "tzinfo") and target_ts.tzinfo is not None:
        target_utc = target_ts.astimezone(UTC).replace(tzinfo=None)
    else:
        # Assume ET if naive
        target_utc = ET.localize(target_ts).astimezone(UTC).replace(tzinfo=None)

    # Normalize opt_df timestamps to UTC naive
    t_col = opt_df["t_dt"]
    if t_col.dt.tz is not None:
        t_naive = t_col.dt.tz_convert(UTC).dt.tz_localize(None)
    else:
        t_naive = t_col

    dists = abs((t_naive - target_utc).dt.total_seconds())
    closest_idx = dists.idxmin()
    if dists.loc[closest_idx] > 300:
        return None
    return float(opt_df.loc[closest_idx, "c"])


def main() -> int:
    print("=" * 100)
    print("DTE Sweep — Gap-Reversion CZ Strategy with Real Options")
    print("=" * 100)

    # Load the CZ intraday timing results
    timing = pd.read_csv(TIMING_CSV)
    cz = timing[(timing["entry_rule"] == "C") & (timing["exit_rule"] == "Z")
                & (timing["status"] == "ok")].copy()
    print(f"CZ trades with valid timing: {len(cz)}")

    # Load original signals for metadata
    signals = pd.read_csv(SIGNALS_CSV)
    signals["session_date"] = pd.to_datetime(signals["session_date"]).dt.date

    client = AlpacaClient()
    t0 = time.perf_counter()

    dte_labels = ["0dte", "this_fri", "next_fri", "monthly"]
    rows: list[dict] = []

    for i, row in cz.iterrows():
        ticker = row["ticker"]
        sd = datetime.strptime(str(row["session_date"]), "%Y-%m-%d").date()
        entry_idx = int(row["entry_idx"])
        exit_idx = int(row["exit_idx"])
        entry_price_under = row["entry_price"]
        exit_price_under = row["exit_price"]
        bars_held = int(row["bars_held"])

        # Load 1-min bars to get timestamps
        try:
            bars = load_bars(ticker, str(sd), str(sd))
        except Exception:
            continue
        rth = filter_rth(bars).reset_index(drop=True)
        if entry_idx >= len(rth) or exit_idx >= len(rth):
            continue

        entry_ts = rth.iloc[entry_idx]["ts_et"]
        exit_ts = rth.iloc[exit_idx]["ts_et"]

        # Convert to naive datetime for Alpaca API
        if hasattr(entry_ts, "to_pydatetime"):
            entry_dt = entry_ts.to_pydatetime().replace(tzinfo=None)
            exit_dt = exit_ts.to_pydatetime().replace(tzinfo=None)
        else:
            entry_dt = entry_ts.replace(tzinfo=None) if hasattr(entry_ts, "replace") else entry_ts
            exit_dt = exit_ts.replace(tzinfo=None) if hasattr(exit_ts, "replace") else exit_ts

        atm_strike = round(entry_price_under)

        # Compute expiry dates for each DTE bucket
        expiries = {
            "0dte": sd,
            "this_fri": _next_friday(sd),
            "next_fri": _next_next_friday(sd),
            "monthly": _monthly_expiry(sd),
        }

        for dte_label, expiry in expiries.items():
            base_strike = round(entry_price_under)
            symbol = _build_occ_call(ticker, expiry, base_strike)
            dte_days = (expiry - sd).days

            # Fetch all option bars for this contract on signal day
            opt_df = _fetch_day_option_bars(client, symbol, sd)
            if opt_df is None or (opt_df is not None and opt_df.empty):
                # Try ±1 strike
                for alt in (base_strike - 1, base_strike + 1):
                    alt_sym = _build_occ_call(ticker, expiry, alt)
                    opt_df = _fetch_day_option_bars(client, alt_sym, sd)
                    if opt_df is not None and not opt_df.empty:
                        symbol = alt_sym
                        base_strike = alt
                        break

            if opt_df is None or opt_df.empty:
                rows.append({
                    "ticker": ticker, "session_date": str(sd),
                    "dte_label": dte_label, "dte_days": dte_days,
                    "status": "no_entry_bar",
                })
                continue

            entry_opt = _lookup_close_at(opt_df, entry_dt)
            if entry_opt is None:
                rows.append({
                    "ticker": ticker, "session_date": str(sd),
                    "dte_label": dte_label, "dte_days": dte_days,
                    "status": "no_entry_bar",
                })
                continue

            exit_opt = _lookup_close_at(opt_df, exit_dt)
            if exit_opt is None:
                rows.append({
                    "ticker": ticker, "session_date": str(sd),
                    "dte_label": dte_label, "dte_days": dte_days,
                    "status": "no_exit_bar", "symbol": symbol,
                })
                continue

            pnl = (exit_opt - entry_opt) * 100
            pnl_pct = (exit_opt - entry_opt) / entry_opt if entry_opt > 0 else 0

            rows.append({
                "ticker": ticker,
                "session_date": str(sd),
                "dte_label": dte_label,
                "dte_days": dte_days,
                "status": "ok",
                "symbol": symbol,
                "expiry": str(expiry),
                "strike": base_strike,
                "entry_opt": entry_opt,
                "exit_opt": exit_opt,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "bars_held": bars_held,
                "underlying_pct": row["underlying_pct"],
            })

        done = sum(1 for r in rows if r.get("status") == "ok")
        total_attempts = len(rows)
        if (total_attempts) % 80 == 0 or i == len(cz) - 1:
            dt = time.perf_counter() - t0
            print(f"  processed signal {i+1}/{len(cz)}  ok={done}  elapsed={dt:.0f}s")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "dte_sweep_cz.csv", index=False)

    # Analysis
    ok = results[results["status"] == "ok"].copy()
    print(f"\nTotal fills: {len(ok)} / {len(results)}")

    pd.set_option("display.width", 220)

    # Summary per DTE
    print("\n" + "#" * 100)
    print("# RESULTS BY DTE — Real Alpaca Option Fills")
    print("#" * 100)

    summary = ok.groupby("dte_label").agg(
        n=("pnl", "size"),
        mean_pnl=("pnl", "mean"),
        median_pnl=("pnl", "median"),
        total_pnl=("pnl", "sum"),
        mean_pct=("pnl_pct", "mean"),
        median_pct=("pnl_pct", "median"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_premium=("entry_opt", "mean"),
    )
    # Profit factor
    for dte in summary.index:
        sub = ok[ok["dte_label"] == dte]
        wins = sub[sub["pnl"] > 0]["pnl"].sum()
        losses = abs(sub[sub["pnl"] < 0]["pnl"].sum())
        summary.loc[dte, "profit_factor"] = wins / losses if losses > 0 else float("inf")
        summary.loc[dte, "avg_win"] = sub[sub["pnl"] > 0]["pnl"].mean() if (sub["pnl"] > 0).any() else 0
        summary.loc[dte, "avg_loss"] = sub[sub["pnl"] < 0]["pnl"].mean() if (sub["pnl"] < 0).any() else 0

    # Order by DTE
    order = ["0dte", "this_fri", "next_fri", "monthly"]
    summary = summary.reindex([o for o in order if o in summary.index])

    for dte in summary.index:
        r = summary.loc[dte]
        print(f"\n  {dte:>10}  (avg premium ${r['avg_premium']:.2f})")
        print(f"    n:              {int(r['n'])}")
        print(f"    win rate:       {r['win_rate']:.1%}")
        print(f"    mean P&L/ctr:   ${r['mean_pnl']:+.0f}")
        print(f"    median P&L/ctr: ${r['median_pnl']:+.0f}")
        print(f"    mean % return:  {r['mean_pct']:+.1%}")
        print(f"    median % return:{r['median_pct']:+.1%}")
        print(f"    avg win:        ${r['avg_win']:+.0f}")
        print(f"    avg loss:       ${r['avg_loss']:+.0f}")
        print(f"    profit factor:  {r['profit_factor']:.2f}")
        print(f"    total P&L:      ${r['total_pnl']:+.0f}")

    # Ranked
    print("\n" + "#" * 100)
    print("# RANKED BY MEAN % RETURN")
    print("#" * 100)
    ranked = summary.sort_values("mean_pct", ascending=False)
    print(f"\n  {'DTE':>10}  {'n':>4}  {'WR':>5}  {'mean%':>8}  {'med%':>8}  "
          f"{'$/ctr':>8}  {'PF':>5}  {'total$':>9}  {'avg_prem':>9}")
    print("  " + "-" * 80)
    for dte in ranked.index:
        r = ranked.loc[dte]
        print(f"  {dte:>10}  {int(r['n']):>4}  {r['win_rate']:>4.0%}  "
              f"{r['mean_pct']:>+7.1%}  {r['median_pct']:>+7.1%}  "
              f"${r['mean_pnl']:>+7.0f}  {r['profit_factor']:>5.2f}  "
              f"${r['total_pnl']:>+8.0f}  ${r['avg_premium']:>8.2f}")

    # By ticker for the winner
    winner_dte = ranked.index[0]
    print(f"\n" + "#" * 100)
    print(f"# WINNER ({winner_dte}) — BY TICKER")
    print("#" * 100)
    winner_data = ok[ok["dte_label"] == winner_dte]
    for ticker in sorted(winner_data["ticker"].unique()):
        sub = winner_data[winner_data["ticker"] == ticker]
        wr = (sub["pnl"] > 0).mean()
        print(f"  {ticker:>5}: n={len(sub):>3}  mean=${sub['pnl'].mean():+.0f}  "
              f"mean%={sub['pnl_pct'].mean():+.1%}  WR={wr:.0%}  total=${sub['pnl'].sum():+.0f}")

    # Year split for winner
    print(f"\n  BY YEAR ({winner_dte}):")
    winner_data = winner_data.copy()
    winner_data["year"] = pd.to_datetime(winner_data["session_date"]).dt.year
    for year, sub in winner_data.groupby("year"):
        wr = (sub["pnl"] > 0).mean()
        print(f"    {year}: n={len(sub):>3}  mean=${sub['pnl'].mean():+.0f}  "
              f"mean%={sub['pnl_pct'].mean():+.1%}  WR={wr:.0%}  total=${sub['pnl'].sum():+.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
