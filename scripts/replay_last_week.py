"""Replay the gap-reversion CZ strategy for every day of last week (and this week).

For each trading day from 2026-03-31 to 2026-04-09:
  1. Check if SPY qualifies (open < prior VAL, gap >= 0.3%, above weekly VAL)
  2. If yes: simulate Entry C (dip-buy) + Exit Z (cum_delta exhaustion)
  3. Fetch real Alpaca 0DTE option bars and compute actual P&L
  4. Print a clean day-by-day report
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth
from mdq.levels.weekly_profile import compute_all_weekly_profiles
from mdq.levels.volume_profile import compute_all_profiles
from mdq.live.alpaca import AlpacaClient
from mdq.live.gap_runner import (
    CUM_DELTA_DECLINE_BARS,
    MAX_DIP_WAIT_BARS,
    MIN_GAP_PCT,
    _buy_pressure,
)

START_DATE = date(2026, 2, 1)   # 2 months back
END_DATE = date(2026, 4, 9)    # Today


def _build_occ_call(underlying: str, exp_date: date, strike: float) -> str:
    yy = exp_date.strftime("%y")
    mm = exp_date.strftime("%m")
    dd = exp_date.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}C{strike_int:08d}"


def _load_profiles() -> pd.DataFrame:
    path = RESULTS_DIR / "experiment_c" / "profiles__SPY.parquet"
    profs = pd.read_parquet(path)
    profs["session_date"] = pd.to_datetime(profs["session_date"]).dt.date
    return profs.sort_values("session_date")


def _get_weekly_val(bars_all: pd.DataFrame, target_date: date) -> float | None:
    try:
        cutoff = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        sub = bars_all[bars_all["session_date"] <= target_date]
        weekly = compute_all_weekly_profiles(sub, bin_size=0.10)
        if weekly.empty:
            return None
        weekly["week_end"] = pd.to_datetime(weekly["week_end"]).dt.date
        prior = weekly[weekly["week_end"] < target_date]
        if prior.empty:
            return None
        return float(prior.iloc[-1]["val"])
    except Exception:
        return None


def simulate_cz_day(
    rth: pd.DataFrame,
    prior_close: float,
    prior_val: float,
    prior_poc: float,
    weekly_val: float | None,
) -> dict:
    """Full CZ simulation on one day's RTH bars."""
    if rth.empty:
        return {"status": "no_bars"}

    open_price = float(rth.iloc[0]["o"])
    gap_pct = (open_price - prior_close) / prior_close

    result = {
        "open": open_price,
        "gap_pct": gap_pct,
        "prior_val": prior_val,
        "prior_poc": prior_poc,
        "prior_close": prior_close,
        "weekly_val": weekly_val,
    }

    # Filter checks
    if open_price >= prior_val:
        result["status"] = "no_signal"
        result["reason"] = f"open ${open_price:.2f} >= VAL ${prior_val:.2f}"
        return result
    if gap_pct > -MIN_GAP_PCT:
        result["status"] = "no_signal"
        result["reason"] = f"gap {gap_pct:+.2%} < 0.3%"
        return result
    if weekly_val is not None and open_price < weekly_val:
        result["status"] = "no_signal"
        result["reason"] = f"below weekly VAL ${weekly_val:.2f}"
        return result

    gap_dollars = prior_close - open_price
    stop_px = open_price - gap_dollars
    target_px = prior_poc

    result["signal"] = True
    result["gap_dollars"] = gap_dollars
    result["stop_px"] = stop_px
    result["target_px"] = target_px

    # Entry C: find session low in first 30 bars, wait for bounce
    session_low = float("inf")
    entry_idx = None
    for i in range(min(MAX_DIP_WAIT_BARS, len(rth))):
        bar = rth.iloc[i]
        if bar["l"] < session_low:
            session_low = bar["l"]
        if i >= 5 and bar["c"] > session_low + 0.10:
            entry_idx = i
            break
    if entry_idx is None and len(rth) >= MAX_DIP_WAIT_BARS:
        last_bar = rth.iloc[MAX_DIP_WAIT_BARS - 1]
        if last_bar["c"] > session_low + 0.05:
            entry_idx = MAX_DIP_WAIT_BARS - 1

    if entry_idx is None:
        result["status"] = "no_entry"
        result["reason"] = "no bounce in 30 bars"
        return result

    entry_price = float(rth.iloc[entry_idx]["c"])
    entry_ts = rth.iloc[entry_idx]["ts_et"]
    result["entry_idx"] = entry_idx
    result["entry_price"] = entry_price
    result["entry_ts"] = entry_ts
    result["session_low"] = session_low

    # Exit Z: cum_delta exhaustion
    cum_delta = 0.0
    peak = 0.0
    decline_count = 0
    prev_cd = 0.0
    exit_idx = None
    exit_reason = "timeout"

    max_bars = min(entry_idx + 180, len(rth))
    for i in range(entry_idx + 1, max_bars):
        bar = rth.iloc[i]
        bp = _buy_pressure(bar["o"], bar["h"], bar["l"], bar["c"])
        bar_delta = (2 * bp - 1) * bar["v"]
        cum_delta += bar_delta

        if cum_delta > peak:
            peak = cum_delta
            decline_count = 0
        elif cum_delta < prev_cd:
            decline_count += 1
        else:
            decline_count = 0
        prev_cd = cum_delta

        if bar["l"] <= stop_px:
            exit_idx = i
            exit_reason = "stop"
            break
        if bar["h"] >= target_px:
            exit_idx = i
            exit_reason = "target"
            break
        if decline_count >= CUM_DELTA_DECLINE_BARS and (i - entry_idx) >= 10:
            exit_idx = i
            exit_reason = "delta_exhaustion"
            break

    if exit_idx is None:
        exit_idx = max_bars - 1
        exit_reason = "timeout"

    exit_price = float(rth.iloc[exit_idx]["c"])
    exit_ts = rth.iloc[exit_idx]["ts_et"]
    bars_held = exit_idx - entry_idx
    pnl = exit_price - entry_price
    pnl_pct = pnl / entry_price

    result["status"] = "traded"
    result["exit_idx"] = exit_idx
    result["exit_price"] = exit_price
    result["exit_ts"] = exit_ts
    result["exit_reason"] = exit_reason
    result["bars_held"] = bars_held
    result["underlying_pnl"] = pnl
    result["underlying_pct"] = pnl_pct
    return result


def fetch_option_pnl(client: AlpacaClient, sd: date, entry_ts, exit_ts,
                     entry_price: float) -> dict:
    """Fetch real 0DTE call option P&L."""
    atm_strike = round(entry_price)
    symbol = _build_occ_call("SPY", sd, atm_strike)

    entry_dt = entry_ts.to_pydatetime().replace(tzinfo=None) if hasattr(entry_ts, "to_pydatetime") else entry_ts
    exit_dt = exit_ts.to_pydatetime().replace(tzinfo=None) if hasattr(exit_ts, "to_pydatetime") else exit_ts

    start = datetime(sd.year, sd.month, sd.day, 9, 25)
    end = datetime(sd.year, sd.month, sd.day, 16, 5)

    try:
        bars = client.get_option_bars(symbol, start, end)
    except Exception:
        bars = []

    if not bars:
        for alt in (atm_strike - 1, atm_strike + 1):
            alt_sym = _build_occ_call("SPY", sd, alt)
            try:
                alt_bars = client.get_option_bars(alt_sym, start, end)
            except Exception:
                alt_bars = []
            if alt_bars:
                symbol = alt_sym
                bars = alt_bars
                break

    if not bars:
        return {"opt_status": "no_bars", "symbol": symbol}

    import pytz
    UTC = pytz.utc
    ET = pytz.timezone("America/New_York")

    opt_df = pd.DataFrame(bars)
    opt_df["t_dt"] = pd.to_datetime(opt_df["t"])
    if opt_df["t_dt"].dt.tz is not None:
        opt_df["t_utc"] = opt_df["t_dt"].dt.tz_convert(UTC).dt.tz_localize(None)
    else:
        opt_df["t_utc"] = opt_df["t_dt"]

    def lookup(ts):
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts_utc = ts.astimezone(UTC).replace(tzinfo=None)
        else:
            ts_utc = ET.localize(ts).astimezone(UTC).replace(tzinfo=None)
        dists = abs((opt_df["t_utc"] - ts_utc).dt.total_seconds())
        idx = dists.idxmin()
        if dists.loc[idx] > 300:
            return None
        return float(opt_df.loc[idx, "c"])

    entry_opt = lookup(entry_dt)
    exit_opt = lookup(exit_dt)

    if entry_opt is None:
        return {"opt_status": "no_entry_match", "symbol": symbol}
    if exit_opt is None:
        return {"opt_status": "no_exit_match", "symbol": symbol}

    pnl = (exit_opt - entry_opt) * 100
    pnl_pct = (exit_opt - entry_opt) / entry_opt if entry_opt > 0 else 0

    return {
        "opt_status": "ok",
        "symbol": symbol,
        "entry_opt": entry_opt,
        "exit_opt": exit_opt,
        "opt_pnl": pnl,
        "opt_pnl_pct": pnl_pct,
    }


def main() -> int:
    print("=" * 100)
    print(f"Last week + this week replay: {START_DATE} to {END_DATE}")
    print("=" * 100)

    profiles = _load_profiles()

    # Load all SPY bars for weekly VAL computation
    bars_all = load_bars("SPY", "2026-01-01", END_DATE.isoformat())
    bars_all_rth = filter_rth(bars_all)

    client = AlpacaClient()

    # Generate trading days
    current = START_DATE
    days: list[date] = []
    while current <= END_DATE:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)

    print(f"\nTrading days to replay: {len(days)}")

    cumulative_pnl = 0.0
    trades_taken = 0
    wins = 0

    for sd in days:
        print(f"\n{'─' * 90}")
        print(f"  {sd.strftime('%Y-%m-%d %A')}")
        print(f"{'─' * 90}")

        # Get prior-day profile
        prior_profs = profiles[profiles["session_date"] < sd]
        if prior_profs.empty:
            print("  no prior profile — skip")
            continue
        prior = prior_profs.iloc[-1]
        prior_close = float(prior["close"])
        prior_val = float(prior["val"])
        prior_poc = float(prior["poc"])

        # Weekly VAL
        weekly_val = _get_weekly_val(bars_all_rth, sd)

        # Load today's bars
        try:
            day_bars = load_bars("SPY", sd.isoformat(), sd.isoformat())
        except Exception:
            print("  no bars available")
            continue
        rth = filter_rth(day_bars).reset_index(drop=True)
        if rth.empty:
            print("  no RTH bars (holiday?)")
            continue

        # Simulate
        sim = simulate_cz_day(rth, prior_close, prior_val, prior_poc, weekly_val)

        open_px = sim.get("open", 0)
        gap = sim.get("gap_pct", 0)
        print(f"  open=${open_px:.2f}  prior_close=${prior_close:.2f}  gap={gap:+.2%}")
        print(f"  VAL=${prior_val:.2f}  POC=${prior_poc:.2f}  weekly_VAL=${weekly_val:.2f}" if weekly_val else
              f"  VAL=${prior_val:.2f}  POC=${prior_poc:.2f}")

        if sim.get("status") == "no_signal":
            print(f"  ❌ NO SIGNAL: {sim.get('reason')}")
            continue
        if sim.get("status") == "no_entry":
            print(f"  ❌ SIGNAL but NO ENTRY: {sim.get('reason')}")
            continue
        if sim.get("status") != "traded":
            print(f"  ❌ status={sim.get('status')}")
            continue

        entry_ts = sim["entry_ts"]
        exit_ts = sim["exit_ts"]
        entry_p = sim["entry_price"]
        exit_p = sim["exit_price"]
        reason = sim["exit_reason"]
        held = sim["bars_held"]
        u_pnl = sim["underlying_pnl"]
        u_pct = sim["underlying_pct"]

        entry_time = entry_ts.strftime("%H:%M") if hasattr(entry_ts, "strftime") else str(entry_ts)
        exit_time = exit_ts.strftime("%H:%M") if hasattr(exit_ts, "strftime") else str(exit_ts)

        print(f"  ✅ SIGNAL FIRED")
        print(f"     entry: ${entry_p:.2f} @ {entry_time}  (session low ${sim.get('session_low', 0):.2f})")
        print(f"     exit:  ${exit_p:.2f} @ {exit_time}  ({reason}, {held} bars)")
        print(f"     underlying P&L: ${u_pnl:+.2f} ({u_pct:+.2%})")

        # Fetch real option P&L
        opt = fetch_option_pnl(client, sd, entry_ts, exit_ts, entry_p)
        if opt.get("opt_status") == "ok":
            print(f"     OPTION: {opt['symbol']}")
            print(f"       entry: ${opt['entry_opt']:.2f}  exit: ${opt['exit_opt']:.2f}")
            print(f"       P&L:   ${opt['opt_pnl']:+.0f}/contract  ({opt['opt_pnl_pct']:+.1%})")
            cumulative_pnl += opt["opt_pnl"]
            trades_taken += 1
            if opt["opt_pnl"] > 0:
                wins += 1
        else:
            print(f"     OPTION: {opt.get('opt_status')} ({opt.get('symbol', '?')})")
            # Still count underlying
            cumulative_pnl += u_pnl * 100  # rough proxy
            trades_taken += 1
            if u_pnl > 0:
                wins += 1

    print(f"\n\n{'=' * 100}")
    print(f"SUMMARY: {START_DATE} to {END_DATE}")
    print(f"{'=' * 100}")
    print(f"  Trading days:     {len(days)}")
    print(f"  Trades taken:     {trades_taken}")
    print(f"  Wins:             {wins}")
    print(f"  Win rate:         {wins/trades_taken:.0%}" if trades_taken > 0 else "  Win rate: —")
    print(f"  Cumulative P&L:   ${cumulative_pnl:+.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
