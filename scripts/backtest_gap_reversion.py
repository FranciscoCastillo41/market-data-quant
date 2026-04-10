"""Backtest Francisco's gap-down-below-VAL mean-reversion call strategy.

Rule:
    TRIGGER: stock opens below yesterday's VAL AND gap >= 0.3% below prior close
    ENTRY:   buy ATM call at open
    TARGET:  underlying reverts to prior-day POC
    STOP:    underlying drops 2x the gap from open (i.e. gap doubles)
    TIMEOUT: 5 trading days max hold
    EXIT:    sell call at the trigger (target/stop/timeout)

Backtested on SPY, JPM, NVDA, AMZN, ARMK — 2024-01-01 to 2026-04-08.
(Starting 2024 because Alpaca options data begins Feb 2024.)

Phase 1: underlying signals + forward P&L on daily bars
Phase 2: real Alpaca option bars for every signal → actual call P&L
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import download_range_async, load_bars
from mdq.data.calendar import filter_rth
from mdq.levels.volume_profile import compute_all_profiles
from mdq.live.alpaca import AlpacaClient

OUT = RESULTS_DIR / "gap_reversion"
OUT.mkdir(parents=True, exist_ok=True)

START = "2024-01-01"
END = "2026-04-08"
TICKERS = ["SPY", "JPM", "NVDA", "AMZN", "ARMK"]

MIN_GAP_PCT = 0.003  # 0.3% minimum gap down
MAX_HOLD_DAYS = 5


def _build_occ_call(underlying: str, exp_date: date, strike: float) -> str:
    yy = exp_date.strftime("%y")
    mm = exp_date.strftime("%m")
    dd = exp_date.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}C{strike_int:08d}"


def _next_friday(d: date) -> date:
    """Return the next Friday on or after d."""
    days_ahead = 4 - d.weekday()  # Friday = 4
    if days_ahead < 0:
        days_ahead += 7
    if days_ahead == 0:
        days_ahead = 7  # if it's Friday, use NEXT Friday for weeklies
    return d + timedelta(days=days_ahead)


def run_underlying_backtest(ticker: str) -> pd.DataFrame:
    """Phase 1: find all gap-down-below-VAL signals and simulate on daily bars."""
    print(f"\n  [{ticker}] Loading bars...")
    try:
        bars = load_bars(ticker, START, END)
    except FileNotFoundError:
        print(f"  [{ticker}] No cached bars. Downloading...")
        asyncio.run(download_range_async(ticker, START, END, overwrite=False))
        bars = load_bars(ticker, START, END)

    profiles = compute_all_profiles(bars, bin_size=0.05)
    rth = filter_rth(bars)

    # Build daily OHLCV
    daily = rth.groupby("session_date").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"),
        c=("c", "last"), v=("v", "sum"),
    ).reset_index().sort_values("session_date").reset_index(drop=True)
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.date

    profiles["session_date"] = pd.to_datetime(profiles["session_date"]).dt.date

    signals: list[dict] = []

    for i in range(1, len(daily)):
        today = daily.iloc[i]
        sd = today["session_date"]

        # Find prior-day profile
        prior_profs = profiles[profiles["session_date"] < sd]
        if prior_profs.empty:
            continue
        prior = prior_profs.iloc[-1]
        prior_close = float(daily.iloc[i - 1]["c"])
        prior_poc = float(prior["poc"])
        prior_val = float(prior["val"])
        prior_vah = float(prior["vah"])

        entry_open = float(today["o"])

        # TRIGGER: open below VAL AND gap >= MIN_GAP_PCT
        if entry_open >= prior_val:
            continue
        gap_pct = (entry_open - prior_close) / prior_close
        if gap_pct > -MIN_GAP_PCT:
            continue

        # TARGET: revert to POC
        target_px = prior_poc
        # STOP: 2x the gap from open (further down)
        gap_dollars = prior_close - entry_open
        stop_px = entry_open - gap_dollars  # drop another gap_dollars

        # Forward daily simulation (up to MAX_HOLD_DAYS)
        end_idx = min(i + 1 + MAX_HOLD_DAYS, len(daily))
        outcome = "timeout"
        exit_px = entry_open
        exit_date = sd
        bars_held = 0

        for j in range(i + 1, end_idx):
            fwd = daily.iloc[j]
            bars_held = j - i
            # Check target (high reaches POC)
            if fwd["h"] >= target_px:
                outcome = "target"
                exit_px = target_px
                exit_date = fwd["session_date"]
                break
            # Check stop (low breaks stop)
            if fwd["l"] <= stop_px:
                outcome = "stop"
                exit_px = stop_px
                exit_date = fwd["session_date"]
                break
            exit_px = float(fwd["c"])
            exit_date = fwd["session_date"]

        if outcome == "timeout" and end_idx > i + 1:
            bars_held = end_idx - i - 1
            exit_px = float(daily.iloc[end_idx - 1]["c"])
            exit_date = daily.iloc[end_idx - 1]["session_date"]

        underlying_pnl = exit_px - entry_open
        underlying_pct = underlying_pnl / entry_open

        signals.append({
            "ticker": ticker,
            "session_date": sd,
            "entry_open": entry_open,
            "prior_close": prior_close,
            "prior_poc": prior_poc,
            "prior_val": prior_val,
            "prior_vah": prior_vah,
            "gap_pct": gap_pct,
            "target_px": target_px,
            "stop_px": stop_px,
            "outcome": outcome,
            "exit_px": exit_px,
            "exit_date": exit_date,
            "bars_held": bars_held,
            "underlying_pnl": underlying_pnl,
            "underlying_pct": underlying_pct,
        })

    return pd.DataFrame(signals)


def fetch_option_pnl(
    client: AlpacaClient,
    ticker: str,
    signal_date: date,
    entry_open: float,
    exit_date: date,
    outcome: str,
    exit_px: float,
) -> dict:
    """Phase 2: fetch real ATM call option bars from Alpaca and compute P&L."""
    atm_strike = round(entry_open)
    # Use next Friday expiry (weekly options)
    expiry = _next_friday(signal_date)
    # Make sure expiry is after exit_date
    while expiry < exit_date:
        expiry = _next_friday(expiry + timedelta(days=1))

    symbol = _build_occ_call(ticker, expiry, atm_strike)

    # Fetch bars from signal_date open through exit_date close
    start_dt = datetime(signal_date.year, signal_date.month, signal_date.day, 9, 30)
    end_dt = datetime(exit_date.year, exit_date.month, exit_date.day, 16, 5)

    try:
        opt_bars = client.get_option_bars(symbol, start_dt, end_dt)
    except Exception:
        opt_bars = []

    # Fallback: try ±1 strike
    if not opt_bars:
        for alt in (atm_strike - 1, atm_strike + 1, atm_strike - 0.5, atm_strike + 0.5):
            alt_sym = _build_occ_call(ticker, expiry, alt)
            try:
                alt_bars = client.get_option_bars(alt_sym, start_dt, end_dt)
            except Exception:
                alt_bars = []
            if alt_bars:
                symbol = alt_sym
                opt_bars = alt_bars
                atm_strike = alt
                break

    if not opt_bars:
        return {"status": "no_bars", "symbol": symbol}

    opt_df = pd.DataFrame(opt_bars)
    opt_df["t_dt"] = pd.to_datetime(opt_df["t"]).dt.tz_localize(None)

    # Entry = first bar of signal_date (open price)
    signal_day_bars = opt_df[opt_df["t_dt"].dt.date == signal_date]
    if signal_day_bars.empty:
        return {"status": "no_entry_bar", "symbol": symbol}
    entry_opt = float(signal_day_bars.iloc[0]["o"])  # open of first bar = market open price

    # Exit = last bar of exit_date (close price) for timeout,
    #        or the bar closest to when target/stop hit for target/stop
    exit_day_bars = opt_df[opt_df["t_dt"].dt.date == exit_date]
    if exit_day_bars.empty:
        # Fall back to last available bar
        exit_opt = float(opt_df.iloc[-1]["c"])
    else:
        exit_opt = float(exit_day_bars.iloc[-1]["c"])

    pnl = (exit_opt - entry_opt) * 100
    pnl_pct = (exit_opt - entry_opt) / entry_opt if entry_opt > 0 else 0

    return {
        "status": "ok",
        "symbol": symbol,
        "expiry": expiry,
        "strike": atm_strike,
        "entry_opt": entry_opt,
        "exit_opt": exit_opt,
        "opt_pnl": pnl,
        "opt_pnl_pct": pnl_pct,
    }


def main() -> int:
    print("=" * 100)
    print("Gap-Down-Below-VAL Mean Reversion Call Strategy — Full Backtest")
    print(f"Tickers: {TICKERS}  Period: {START} -> {END}")
    print("=" * 100)

    # Phase 1: Underlying signals
    print("\n" + "─" * 50)
    print("PHASE 1: Underlying signal detection + daily simulation")
    print("─" * 50)
    t0 = time.perf_counter()

    all_signals: list[pd.DataFrame] = []
    for ticker in TICKERS:
        df = run_underlying_backtest(ticker)
        print(f"  [{ticker}] {len(df)} signals found")
        all_signals.append(df)

    signals = pd.concat(all_signals, ignore_index=True)
    signals = signals.sort_values("session_date").reset_index(drop=True)
    dt1 = time.perf_counter() - t0
    print(f"\nTotal signals: {len(signals)} across {len(TICKERS)} tickers ({dt1:.1f}s)")

    # Phase 1 summary
    print("\n  UNDERLYING RESULTS:")
    vc = signals["outcome"].value_counts()
    for outcome, count in vc.items():
        pct = count / len(signals)
        mean_pnl = signals[signals["outcome"] == outcome]["underlying_pct"].mean()
        print(f"    {outcome:>8}: {count:>4} ({pct:>5.1%})  mean underlying return: {mean_pnl:+.2%}")
    print(f"    overall mean return: {signals['underlying_pct'].mean():+.2%}")
    print(f"    overall win rate:    {(signals['underlying_pnl'] > 0).mean():.1%}")

    # By ticker
    print("\n  BY TICKER:")
    for ticker in TICKERS:
        sub = signals[signals["ticker"] == ticker]
        if sub.empty:
            print(f"    {ticker}: 0 signals")
            continue
        wr = (sub["underlying_pnl"] > 0).mean()
        print(f"    {ticker}: {len(sub):>3} signals  "
              f"mean={sub['underlying_pct'].mean():+.2%}  "
              f"win_rate={wr:.0%}  "
              f"targets={int((sub['outcome']=='target').sum())}  "
              f"stops={int((sub['outcome']=='stop').sum())}  "
              f"timeouts={int((sub['outcome']=='timeout').sum())}")

    # Phase 2: Real options
    print("\n" + "─" * 50)
    print("PHASE 2: Real Alpaca option fills")
    print("─" * 50)
    t0 = time.perf_counter()

    client = AlpacaClient()
    opt_results: list[dict] = []

    for i, row in signals.iterrows():
        sd = row["session_date"]
        if isinstance(sd, str):
            sd = datetime.strptime(sd, "%Y-%m-%d").date()
        exit_d = row["exit_date"]
        if isinstance(exit_d, str):
            exit_d = datetime.strptime(exit_d, "%Y-%m-%d").date()

        result = fetch_option_pnl(
            client,
            row["ticker"],
            sd,
            row["entry_open"],
            exit_d,
            row["outcome"],
            row["exit_px"],
        )
        opt_results.append(result)

        if (i + 1) % 20 == 0 or (i + 1) == len(signals):
            ok = sum(1 for r in opt_results if r.get("status") == "ok")
            dt = time.perf_counter() - t0
            print(f"  processed {i+1:>4}/{len(signals)}  ok={ok}  elapsed={dt:.0f}s")

    opt_df = pd.DataFrame(opt_results)
    signals = pd.concat([signals.reset_index(drop=True), opt_df.reset_index(drop=True)], axis=1)
    signals.to_csv(OUT / "full_backtest.csv", index=False)

    # Phase 2 summary
    ok = signals[signals["status"] == "ok"].copy()
    print(f"\nSignals with real option data: {len(ok)} / {len(signals)}")

    if ok.empty:
        print("No option data available. Check Alpaca subscription.")
        return 0

    print(f"\n{'='*100}")
    print("REAL OPTIONS P&L SUMMARY")
    print(f"{'='*100}")
    print(f"\n  Mean option P&L / contract:  ${ok['opt_pnl'].mean():+.2f}")
    print(f"  Median:                      ${ok['opt_pnl'].median():+.2f}")
    print(f"  Total:                       ${ok['opt_pnl'].sum():+.2f}")
    print(f"  Win rate:                    {(ok['opt_pnl'] > 0).mean():.1%}")
    print(f"  Mean % return on premium:    {ok['opt_pnl_pct'].mean():+.1%}")

    if (ok["opt_pnl"] > 0).any() and (ok["opt_pnl"] < 0).any():
        wins = ok[ok["opt_pnl"] > 0]
        losses = ok[ok["opt_pnl"] < 0]
        pf = wins["opt_pnl"].sum() / abs(losses["opt_pnl"].sum())
        print(f"  Avg win:                     ${wins['opt_pnl'].mean():+.2f}")
        print(f"  Avg loss:                    ${losses['opt_pnl'].mean():+.2f}")
        print(f"  Profit factor:               {pf:.2f}")

    # By ticker
    print(f"\n  BY TICKER (real options):")
    for ticker in TICKERS:
        sub = ok[ok["ticker"] == ticker]
        if sub.empty:
            print(f"    {ticker}: 0 trades with option data")
            continue
        wr = (sub["opt_pnl"] > 0).mean()
        print(f"    {ticker}: {len(sub):>3} trades  "
              f"mean=${sub['opt_pnl'].mean():+.0f}/ctr  "
              f"mean_pct={sub['opt_pnl_pct'].mean():+.1%}  "
              f"win_rate={wr:.0%}  "
              f"total=${sub['opt_pnl'].sum():+.0f}")

    # By outcome
    print(f"\n  BY UNDERLYING OUTCOME (option P&L):")
    for outcome in ("target", "stop", "timeout"):
        sub = ok[ok["outcome"] == outcome]
        if sub.empty:
            continue
        print(f"    {outcome:>8}: n={len(sub):>3}  "
              f"mean_opt=${sub['opt_pnl'].mean():+.0f}  "
              f"mean_pct={sub['opt_pnl_pct'].mean():+.1%}")

    # Recent trades
    print(f"\n  MOST RECENT 15 TRADES:")
    recent = ok.sort_values("session_date", ascending=False).head(15)
    print(f"  {'date':>12}  {'ticker':>6}  {'gap':>7}  {'outcome':>8}  "
          f"{'held':>4}  {'strike':>7}  {'entry':>7}  {'exit':>7}  {'pnl':>8}  {'pct':>7}")
    print("  " + "-" * 90)
    for _, r in recent.iterrows():
        print(f"  {str(r['session_date']):>12}  {r['ticker']:>6}  "
              f"{r['gap_pct']:>+6.1%}  {r['outcome']:>8}  "
              f"{int(r['bars_held']):>4}d  "
              f"${r.get('strike', 0):>6.0f}  "
              f"${r.get('entry_opt', 0):>6.2f}  ${r.get('exit_opt', 0):>6.2f}  "
              f"${r['opt_pnl']:>+7.0f}  {r['opt_pnl_pct']:>+6.1%}")

    print(f"\nFull results: {OUT / 'full_backtest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
