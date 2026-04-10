"""Real-options backtest of the Tier 1 signal using Alpaca historical option bars.

Pipeline:
  1. Load all historical Tier 1 signals from Experiment C reactions parquet
     (QQQ, prior_low, first touch, non-confluence, from_above)
  2. For each signal:
     a. Reload that session's QQQ bars, find touch bar
     b. Simulate target/stop race on UNDERLYING with Tier 1 geometry
        (target -$1.00, stop +$1.50, horizon 15 bars, short direction)
     c. Pick nearest ATM strike, build OCC put symbol for that day's 0DTE
     d. Fetch Alpaca historical 1-min option bars for that contract
     e. Record entry price (close of touch bar) and exit price
        (close of bar where target/stop/timeout triggered)
     f. Compute option P&L per contract: (exit - entry) * 100
  3. Aggregate: per-trade stats, win rate, expectancy, breakdown by year / outcome

This replaces the theoretical "underlying P&L -> option P&L" approximation with
actual option fills at actual minute resolution.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.live.alpaca import AlpacaClient

ET = pytz.timezone("America/New_York")
RES_C = RESULTS_DIR / "experiment_c"
OUT = RESULTS_DIR / "tier1_real_options_backtest.csv"


def build_occ_put(underlying: str, exp_date: date, strike: float) -> str:
    """OCC format: e.g. QQQ260407P00585000."""
    yy = exp_date.strftime("%y")
    mm = exp_date.strftime("%m")
    dd = exp_date.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}P{strike_int:08d}"


def simulate_short_race(
    bars_win: pd.DataFrame,
    entry_idx: int,
    target: float = 1.00,
    stop: float = 1.50,
    horizon: int = 15,
) -> dict:
    """Race target vs stop on forward 1-min underlying bars for a short entry.

    Returns dict with outcome, bars_held, exit_idx, exit_ts, exit_price.
    """
    entry_price = float(bars_win.iloc[entry_idx]["c"])
    start = entry_idx + 1
    end = min(start + horizon, len(bars_win))
    if start >= end:
        return {"outcome": "no_forward_bars", "entry_price": entry_price}

    target_px = entry_price - target
    stop_px = entry_price + stop

    fwd = bars_win.iloc[start:end]
    for i, (_, row) in enumerate(fwd.iterrows()):
        hit_target = row["l"] <= target_px
        hit_stop = row["h"] >= stop_px
        # Conservative: if both hit in same bar, stop wins
        if hit_stop:
            return {
                "outcome": "stop",
                "bars_held": i + 1,
                "exit_idx": start + i,
                "exit_ts": row["ts_et"],
                "entry_price": entry_price,
                "exit_price": float(row["c"]),
            }
        if hit_target:
            return {
                "outcome": "target",
                "bars_held": i + 1,
                "exit_idx": start + i,
                "exit_ts": row["ts_et"],
                "entry_price": entry_price,
                "exit_price": float(row["c"]),
            }

    last = fwd.iloc[-1]
    last_idx = start + len(fwd) - 1
    return {
        "outcome": "timeout",
        "bars_held": len(fwd),
        "exit_idx": last_idx,
        "exit_ts": last["ts_et"],
        "entry_price": entry_price,
        "exit_price": float(last["c"]),
    }


def _find_opt_bar_at_or_after(opt_df: pd.DataFrame, target_ts) -> tuple[float, pd.Timestamp] | None:
    """Return (close_price, bar_ts) of the first option bar at or after target_ts."""
    after = opt_df[opt_df["t"] >= target_ts]
    if after.empty:
        return None
    row = after.iloc[0]
    return float(row["c"]), row["t"]


def _normalize_session_date(sd) -> date:
    """Convert whatever is in session_date column to a date."""
    if isinstance(sd, date) and not isinstance(sd, datetime):
        return sd
    if isinstance(sd, datetime):
        return sd.date()
    if hasattr(sd, "date") and callable(sd.date):
        return sd.date()
    if isinstance(sd, str):
        return datetime.strptime(sd, "%Y-%m-%d").date()
    return sd


def main() -> int:
    print("=" * 70)
    print("Tier 1 real-options backtest (QQQ prior_low non-confluence breakdown)")
    print("=" * 70)

    reactions = pd.read_parquet(RES_C / "reactions__QQQ.parquet")
    t1 = reactions[
        (reactions["level_name"] == "prior_low")
        & (reactions["touch_num"] == 1)
        & (reactions["confluence"] == False)  # noqa: E712
        & (reactions["approach"] == "from_above")
    ].copy()
    t1 = t1.sort_values("session_date").reset_index(drop=True)
    print(f"\nTotal Tier 1 signals in history: {len(t1)}")

    if t1.empty:
        print("No signals to process.")
        return 0

    client = AlpacaClient()
    results: list[dict] = []
    t0 = time.perf_counter()

    for i, row in t1.iterrows():
        sd = _normalize_session_date(row["session_date"])
        sd_str = sd.strftime("%Y-%m-%d")

        # Load session bars (from parquet cache, fast)
        try:
            bars = load_bars("QQQ", sd_str, sd_str)
        except Exception as e:
            results.append({
                "session_date": sd_str, "status": "no_bars", "error": str(e)[:80],
            })
            continue

        bars_win = filter_window(bars, "09:30", "11:15").reset_index(drop=True)
        if bars_win.empty:
            results.append({"session_date": sd_str, "status": "empty_window"})
            continue

        # Find touch bar via unix ms match
        touch_t = int(row["t"])
        matching = bars_win[bars_win["t"] == touch_t]
        if matching.empty:
            results.append({"session_date": sd_str, "status": "touch_bar_missing"})
            continue
        entry_idx = int(matching.index[0])
        entry_close_under = float(matching.iloc[0]["c"])
        entry_ts = matching.iloc[0]["ts_et"]

        # Simulate underlying race
        sim = simulate_short_race(bars_win, entry_idx)
        exit_ts = sim.get("exit_ts")
        exit_close_under = sim.get("exit_price")
        outcome = sim.get("outcome")

        # ATM strike selection
        strike = round(entry_close_under)
        symbol = build_occ_put("QQQ", sd, strike)

        # Fetch Alpaca option bars from 1 min before entry to 5 min after exit
        fetch_start = entry_ts - pd.Timedelta(minutes=1)
        if exit_ts is not None:
            fetch_end = exit_ts + pd.Timedelta(minutes=5)
        else:
            fetch_end = entry_ts + pd.Timedelta(minutes=20)

        try:
            opt_bars = client.get_option_bars(
                symbol,
                fetch_start.to_pydatetime() if hasattr(fetch_start, "to_pydatetime") else fetch_start,
                fetch_end.to_pydatetime() if hasattr(fetch_end, "to_pydatetime") else fetch_end,
            )
        except Exception as e:
            # Try ±1 strike fallback (sometimes round() goes the wrong way)
            alt_results = None
            for alt_strike in (strike - 1, strike + 1):
                alt_sym = build_occ_put("QQQ", sd, alt_strike)
                try:
                    alt_bars = client.get_option_bars(alt_sym, fetch_start, fetch_end)
                    if alt_bars:
                        alt_results = (alt_sym, alt_bars)
                        break
                except Exception:
                    pass
            if alt_results is None:
                results.append({
                    "session_date": sd_str, "symbol": symbol, "status": "api_error",
                    "error": str(e)[:80],
                })
                continue
            symbol, opt_bars = alt_results

        if not opt_bars:
            # ±1 strike fallback if first fetch returned empty
            alt_found = None
            for alt_strike in (strike - 1, strike + 1):
                alt_sym = build_occ_put("QQQ", sd, alt_strike)
                try:
                    alt_bars = client.get_option_bars(alt_sym, fetch_start, fetch_end)
                except Exception:
                    alt_bars = []
                if alt_bars:
                    alt_found = (alt_sym, alt_bars)
                    break
            if alt_found is None:
                results.append({
                    "session_date": sd_str, "symbol": symbol, "status": "no_option_bars",
                })
                continue
            symbol, opt_bars = alt_found

        opt_df = pd.DataFrame(opt_bars)
        opt_df["t"] = pd.to_datetime(opt_df["t"]).dt.tz_convert(ET)

        # Entry option price = close of option bar at or after entry_ts
        entry_opt_res = _find_opt_bar_at_or_after(opt_df, entry_ts)
        if entry_opt_res is None:
            results.append({
                "session_date": sd_str, "symbol": symbol, "status": "no_entry_opt_bar",
            })
            continue
        entry_opt, entry_opt_ts = entry_opt_res

        # Exit option price = close of option bar at or after exit_ts
        if exit_ts is None:
            results.append({
                "session_date": sd_str, "symbol": symbol, "status": "no_exit_ts",
            })
            continue
        exit_opt_res = _find_opt_bar_at_or_after(opt_df, exit_ts)
        if exit_opt_res is None:
            # Fall back to last available bar
            last_row = opt_df.iloc[-1]
            exit_opt = float(last_row["c"])
            exit_opt_ts = last_row["t"]
        else:
            exit_opt, exit_opt_ts = exit_opt_res

        pnl_contract = (exit_opt - entry_opt) * 100
        pnl_pct = (exit_opt - entry_opt) / entry_opt if entry_opt > 0 else np.nan

        results.append({
            "session_date": sd_str,
            "symbol": symbol,
            "status": "ok",
            "sim_outcome": outcome,
            "bars_held": sim.get("bars_held"),
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "entry_qqq": entry_close_under,
            "exit_qqq": exit_close_under,
            "strike": strike,
            "entry_opt": entry_opt,
            "exit_opt": exit_opt,
            "pnl_contract": pnl_contract,
            "pnl_pct": pnl_pct,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(t1):
            dt = time.perf_counter() - t0
            ok_so_far = sum(1 for r in results if r.get("status") == "ok")
            print(f"  processed {i+1:>3}/{len(t1)}  ok={ok_so_far}  elapsed={dt:.0f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")

    # -------- Summary --------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    status_counts = df["status"].value_counts()
    print(f"\nStatus counts:")
    for s, c in status_counts.items():
        print(f"  {s}: {c}")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("\nNo successful sims. Check Alpaca historical options coverage.")
        return 0

    print(f"\nSuccessful sims: {len(ok)}")
    print(f"\nP&L per contract (on real Alpaca option bars):")
    print(f"  mean:    ${ok['pnl_contract'].mean():+.2f}")
    print(f"  median:  ${ok['pnl_contract'].median():+.2f}")
    print(f"  std:     ${ok['pnl_contract'].std():.2f}")
    print(f"  min:     ${ok['pnl_contract'].min():+.2f}")
    print(f"  max:     ${ok['pnl_contract'].max():+.2f}")
    print(f"  total:   ${ok['pnl_contract'].sum():+.2f}  (across {len(ok)} trades, 1 contract each)")

    wins = ok[ok["pnl_contract"] > 0]
    losses = ok[ok["pnl_contract"] < 0]
    print(f"\nWin/loss:")
    print(f"  win rate:   {len(wins) / len(ok):.1%}")
    if len(wins) > 0:
        print(f"  avg win:    ${wins['pnl_contract'].mean():+.2f}")
    if len(losses) > 0:
        print(f"  avg loss:   ${losses['pnl_contract'].mean():+.2f}")
    if len(wins) > 0 and len(losses) > 0:
        pf = wins['pnl_contract'].sum() / abs(losses['pnl_contract'].sum())
        print(f"  profit factor: {pf:.2f}")

    print("\nBy year:")
    ok["year"] = pd.to_datetime(ok["session_date"]).dt.year
    by_year = ok.groupby("year").agg(
        n=("pnl_contract", "size"),
        mean_pnl=("pnl_contract", "mean"),
        total_pnl=("pnl_contract", "sum"),
        win_rate=("pnl_contract", lambda x: (x > 0).mean()),
    )
    print(by_year.to_string(float_format=lambda x: f"{x:+.2f}"))

    print("\nBy underlying outcome:")
    by_outcome = ok.groupby("sim_outcome").agg(
        n=("pnl_contract", "size"),
        mean_pnl=("pnl_contract", "mean"),
        total_pnl=("pnl_contract", "sum"),
    )
    print(by_outcome.to_string(float_format=lambda x: f"{x:+.2f}"))

    ann = ok["pnl_contract"].mean() * 66
    print(f"\nAnnualized estimate (66 trades/yr × mean P&L):  ${ann:+.0f}/yr per contract")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
