"""Replay the Tier 1 signal against a given date's QQQ bars + real option prices.

Usage:
    poetry run python3 scripts/replay_today.py                  # today
    poetry run python3 scripts/replay_today.py 2026-04-07       # specific date

Pipeline:
  1. Pull the target date's QQQ 1-min bars from Massive (fresh, not cached)
  2. Load prior session's prior_low from the Experiment C profile cache
  3. Detect first-touch of prior_low in the 09:30-11:15 ET window
  4. Apply Tier 1 filters (non-confluence with $10 whole-dollar, approach=from_above)
  5. For each signal: pick nearest ATM 0DTE put strike, build OCC symbol
  6. Fetch Alpaca historical 1-min bars for that exact contract
  7. Simulate the trade: entry at touch-bar close, target -$1.00 QQQ, stop +$1.50,
     15-min timeout
  8. Report: would we have fired? Would we have won? What's the option P&L?

No orders are placed. This is pure end-of-day validation.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.data.bars import download_range_async, load_bars
from mdq.data.calendar import filter_window
from mdq.levels.touches import TouchConfig
from mdq.levels.volume_profile import detect_vp_touches_session
from mdq.live.alpaca import AlpacaClient

ET = pytz.timezone("America/New_York")
WINDOW = ("09:30", "11:15")


def _build_occ_put_symbol(underlying: str, expiration: date, strike: float) -> str:
    """Build OCC format put symbol. E.g. QQQ260408P00606000 for QQQ $606 put."""
    yy = expiration.strftime("%y")
    mm = expiration.strftime("%m")
    dd = expiration.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}P{strike_int:08d}"


def _fetch_todays_qqq_bars(today: date) -> pd.DataFrame:
    """Pull today's QQQ 1-min bars from Massive, bypassing the monthly cache."""
    asyncio.run(
        download_range_async(
            ticker="QQQ",
            start=today.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
            overwrite=True,
        )
    )
    # Now load — the downloader wrote to the month file; load_bars will clip
    df = load_bars(
        "QQQ",
        today.strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
    )
    return df


def _get_prior_low(today: date) -> tuple[date, float] | None:
    """Load yesterday's prior_low from the Experiment C profile cache for QQQ."""
    path = RESULTS_DIR / "experiment_c" / "profiles__QQQ.parquet"
    if not path.exists():
        print(f"  profile cache not found at {path}")
        return None
    profs = pd.read_parquet(path)
    profs = profs.sort_values("session_date")
    prior = profs[profs["session_date"] < today]
    if prior.empty:
        return None
    last = prior.iloc[-1]
    return last["session_date"], float(last["low"])


def _is_non_confluence(level: float, whole_dollar_step: float = 10.0, radius: float = 0.50) -> bool:
    nearest = round(level / whole_dollar_step) * whole_dollar_step
    return abs(level - nearest) > radius


def _simulate_trade_on_underlying(
    bars_win: pd.DataFrame,
    entry_idx: int,
    target_move: float = 1.00,
    stop_move: float = 1.50,
    horizon: int = 15,
    direction: str = "short",
) -> dict:
    """Race target vs stop on forward 1-min QQQ bars. Returns outcome dict."""
    start = entry_idx + 1
    end = min(start + horizon, len(bars_win))
    if start >= end:
        return {"outcome": "no_forward_bars"}

    entry_price = float(bars_win.iloc[entry_idx]["c"])
    fwd = bars_win.iloc[start:end]

    if direction == "short":
        target_px = entry_price - target_move
        stop_px = entry_price + stop_move
        for i, (_, row) in enumerate(fwd.iterrows()):
            if row["l"] <= target_px:
                return {
                    "outcome": "target",
                    "bars_held": i + 1,
                    "entry_price": entry_price,
                    "exit_price": target_px,
                    "exit_ts": row["ts_et"],
                    "pnl_underlying": target_move,
                }
            if row["h"] >= stop_px:
                return {
                    "outcome": "stop",
                    "bars_held": i + 1,
                    "entry_price": entry_price,
                    "exit_price": stop_px,
                    "exit_ts": row["ts_et"],
                    "pnl_underlying": -stop_move,
                }
    last = fwd.iloc[-1]
    return {
        "outcome": "timeout",
        "bars_held": horizon,
        "entry_price": entry_price,
        "exit_price": float(last["c"]),
        "exit_ts": last["ts_et"],
        "pnl_underlying": entry_price - float(last["c"]) if direction == "short" else float(last["c"]) - entry_price,
    }


def _parse_date_arg() -> date:
    if len(sys.argv) > 1:
        return datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    return date.today()


def main() -> int:
    today = _parse_date_arg()
    print(f"=" * 70)
    print(f"Tier 1 signal replay for {today}")
    print(f"=" * 70)

    # 1. Pull today's QQQ bars
    print("\n[1/6] Pulling today's QQQ 1-min bars from Massive...")
    try:
        bars = _fetch_todays_qqq_bars(today)
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1
    print(f"  got {len(bars)} bars, range {bars['ts_et'].min()} -> {bars['ts_et'].max()}")

    # 2. Load prior_low
    print("\n[2/6] Loading yesterday's QQQ prior_low...")
    pl_res = _get_prior_low(today)
    if pl_res is None:
        print("  FAIL: no prior_low in cache. Run run_experiment_c.py first.")
        return 1
    prior_date, prior_low = pl_res
    print(f"  prior_low from {prior_date}: ${prior_low:.2f}")

    # 3. Non-confluence filter
    nc = _is_non_confluence(prior_low, whole_dollar_step=10.0, radius=0.50)
    print(f"  non-confluence (> $0.50 from nearest $10): {nc}")
    if not nc:
        print("  NOTE: this prior_low is near a $10 whole dollar, so the non-confluence")
        print("        Tier 1 signal would NOT fire today. Showing touch anyway for diagnostics.")

    # 4. Detect touches in the morning window
    print("\n[3/6] Detecting touches in morning window 09:30-11:15 ET...")
    bars_win = filter_window(bars, WINDOW[0], WINDOW[1]).reset_index(drop=True)
    print(f"  bars in window: {len(bars_win)}")

    if bars_win.empty:
        print("  FAIL: no bars in window. Market may not have opened yet?")
        return 1

    touches = detect_vp_touches_session(
        bars_win,
        named_levels=[("prior_low", prior_low)],
        cfg=TouchConfig(tolerance=0.05, in_play_radius=10.0),
    )
    print(f"  total touches detected: {len(touches)}")
    if touches.empty:
        print("  NO SIGNAL today: prior_low was never touched in the morning window.")
        return 0

    first = touches[touches["touch_num"] == 1]
    if first.empty:
        print("  no first-touches (shouldn't happen but handling it)")
        return 0

    ft = first.iloc[0]
    print(f"  FIRST TOUCH at {ft['ts_et']}")
    print(f"    bar_idx (within window): {int(ft['bar_idx'])}")
    print(f"    touch level: ${ft['level']:.2f}")
    print(f"    approach: {ft['approach']}")
    print(f"    entry_close: ${ft['entry_close']:.2f}")

    # Tier 1 requires approach=from_above (breakdown short). Log but continue either way.
    if ft["approach"] != "from_above":
        print(f"  NOTE: approach is {ft['approach']}, not 'from_above'. Tier 1 would not fire.")
        tier1_fires = False
    else:
        tier1_fires = nc  # must also be non-confluence
    print(f"\n  TIER 1 SIGNAL FIRES: {tier1_fires}")

    # 5. Simulate on underlying
    print("\n[4/6] Simulating on underlying (target -$1.00, stop +$1.50, 15 min, short)...")
    sim = _simulate_trade_on_underlying(
        bars_win,
        entry_idx=int(ft["bar_idx"]),
        target_move=1.00,
        stop_move=1.50,
        horizon=15,
        direction="short",
    )
    print(f"  outcome      : {sim['outcome']}")
    print(f"  bars_held    : {sim.get('bars_held', '—')}")
    print(f"  entry QQQ    : ${sim.get('entry_price', 0):.2f}")
    print(f"  exit QQQ     : ${sim.get('exit_price', 0):.2f}")
    print(f"  exit ts      : {sim.get('exit_ts', '—')}")
    print(f"  pnl (points) : ${sim.get('pnl_underlying', 0):+.2f}")

    # 6. Pull Alpaca option bars for the corresponding ATM put
    print("\n[5/6] Fetching Alpaca historical bars for the ATM 0DTE put...")
    entry_qqq = sim.get("entry_price")
    if entry_qqq is None:
        print("  skipped: no entry price")
        return 0
    atm_strike = round(entry_qqq)
    symbol = _build_occ_put_symbol("QQQ", today, atm_strike)
    print(f"  ATM strike: ${atm_strike}  symbol: {symbol}")

    entry_ts = ft["ts_et"]
    if hasattr(entry_ts, "to_pydatetime"):
        entry_dt = entry_ts.to_pydatetime()
    else:
        entry_dt = entry_ts
    start_fetch = entry_dt - timedelta(minutes=1)
    end_fetch = entry_dt + timedelta(minutes=20)

    try:
        client = AlpacaClient()
        opt_bars = client.get_option_bars(symbol, start_fetch, end_fetch)
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1
    print(f"  got {len(opt_bars)} option bars")

    if not opt_bars:
        # Try adjacent strikes in case ATM rounded wrong
        for alt in (atm_strike - 1, atm_strike + 1):
            alt_sym = _build_occ_put_symbol("QQQ", today, alt)
            try:
                alt_bars = client.get_option_bars(alt_sym, start_fetch, end_fetch)
            except Exception:
                alt_bars = []
            if alt_bars:
                print(f"  fallback strike ${alt} ({alt_sym}) returned {len(alt_bars)} bars")
                symbol = alt_sym
                opt_bars = alt_bars
                atm_strike = alt
                break

    if not opt_bars:
        print("  no option bars available — contract may have had no trades, or")
        print("  the paper account doesn't have historical option data access.")
        return 0

    opt_df = pd.DataFrame(opt_bars)
    opt_df["t"] = pd.to_datetime(opt_df["t"]).dt.tz_convert(ET)
    print("\n  First 5 option bars:")
    print(opt_df[["t", "o", "h", "l", "c", "v"]].head().to_string(index=False))
    print("\n  Last 5 option bars:")
    print(opt_df[["t", "o", "h", "l", "c", "v"]].tail().to_string(index=False))

    # 7. Simulate the option trade
    print("\n[6/6] Simulating option trade on Alpaca data...")
    entry_dt_aware = entry_dt if entry_dt.tzinfo else ET.localize(entry_dt)
    entry_bar = opt_df[opt_df["t"] >= entry_dt_aware]
    if entry_bar.empty:
        print("  no option bar at/after entry timestamp")
        return 0
    entry_opt_price = float(entry_bar.iloc[0]["c"])  # approximation: close of entry minute
    print(f"  approx option entry price (close of minute {entry_bar.iloc[0]['t']}): ${entry_opt_price:.2f}")

    # Find exit bar matching our underlying-sim exit time
    exit_ts = sim.get("exit_ts")
    if exit_ts is None:
        print("  no exit ts from underlying sim")
        return 0
    if hasattr(exit_ts, "to_pydatetime"):
        exit_dt = exit_ts.to_pydatetime()
    else:
        exit_dt = exit_ts

    exit_bars = opt_df[opt_df["t"] >= exit_dt]
    if exit_bars.empty:
        exit_opt_price = float(opt_df.iloc[-1]["c"])
        print(f"  (no bar at exit ts, using last available: ${exit_opt_price:.2f})")
    else:
        exit_opt_price = float(exit_bars.iloc[0]["c"])
        print(f"  approx option exit price: ${exit_opt_price:.2f}")

    pnl_per_contract = (exit_opt_price - entry_opt_price) * 100  # options are 100x
    print(f"\n  === SIMULATED TRADE RESULT ===")
    print(f"  contract     : {symbol}")
    print(f"  entry price  : ${entry_opt_price:.2f} (= ${entry_opt_price*100:.0f} per contract)")
    print(f"  exit price   : ${exit_opt_price:.2f} (= ${exit_opt_price*100:.0f} per contract)")
    print(f"  P&L / ctr    : ${pnl_per_contract:+.0f}")
    print(f"  (backtest EV would predict +$5/contract on average)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
