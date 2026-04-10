"""Test multiple DTE + strike selection combinations on the Tier 1 signal history.

For each of 106 historical Tier 1 signals, we fetch real Alpaca option bars
for the Cartesian product of:

    DTE: 0, 1, 2, 3, 5 business days ahead
    Strike: ATM, 1-OTM (below spot), 1-ITM (above spot)

We then simulate the same target/stop race and record P&L per contract.
The result tells us which (DTE, strike) combo captures the edge best.

All fills are at 1-min bar closes — realistic slippage haircut of $0.05/trade
(entry + exit) can be subtracted post-hoc for comparison.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.live.alpaca import AlpacaClient

ET = pytz.timezone("America/New_York")
RES_C = RESULTS_DIR / "experiment_c"
OUT = RESULTS_DIR / "dte_strike_sweep.csv"

DTE_OFFSETS = [0, 1, 2, 3, 5]  # business days ahead
STRIKE_OFFSETS = {
    "ATM":    0,    # round(spot)
    "1OTM":  -1,    # put OTM = strike below spot
    "1ITM":  +1,    # put ITM = strike above spot
}


def build_occ_put(underlying: str, exp_date: date, strike: float) -> str:
    yy = exp_date.strftime("%y")
    mm = exp_date.strftime("%m")
    dd = exp_date.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}P{strike_int:08d}"


def next_business_days(start: date, n: int) -> date:
    """Return the business day that is n business days after start.
    n=0 means same day. Uses pandas BDay which skips weekends (not holidays)."""
    if n == 0:
        return start
    return (pd.Timestamp(start) + pd.offsets.BDay(n)).date()


def simulate_short_race(
    bars_win: pd.DataFrame,
    entry_idx: int,
    target: float = 1.00,
    stop: float = 1.50,
    horizon: int = 15,
) -> dict:
    entry_price = float(bars_win.iloc[entry_idx]["c"])
    start = entry_idx + 1
    end = min(start + horizon, len(bars_win))
    if start >= end:
        return {"outcome": "no_forward_bars", "entry_price": entry_price}
    target_px = entry_price - target
    stop_px = entry_price + stop
    fwd = bars_win.iloc[start:end]
    for i, (_, row) in enumerate(fwd.iterrows()):
        if row["h"] >= stop_px:
            return {"outcome": "stop", "bars_held": i + 1, "exit_ts": row["ts_et"],
                    "entry_price": entry_price, "exit_price": float(row["c"])}
        if row["l"] <= target_px:
            return {"outcome": "target", "bars_held": i + 1, "exit_ts": row["ts_et"],
                    "entry_price": entry_price, "exit_price": float(row["c"])}
    last = fwd.iloc[-1]
    return {"outcome": "timeout", "bars_held": len(fwd), "exit_ts": last["ts_et"],
            "entry_price": entry_price, "exit_price": float(last["c"])}


def _find_bar_at_or_after(opt_df: pd.DataFrame, target_ts) -> float | None:
    after = opt_df[opt_df["t"] >= target_ts]
    if after.empty:
        return None
    return float(after.iloc[0]["c"])


def _normalize_session_date(sd) -> date:
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
    print("=" * 80)
    print("DTE x Strike sweep — QQQ Tier 1 signal")
    print("=" * 80)

    reactions = pd.read_parquet(RES_C / "reactions__QQQ.parquet")
    t1 = reactions[
        (reactions["level_name"] == "prior_low")
        & (reactions["touch_num"] == 1)
        & (reactions["confluence"] == False)  # noqa: E712
        & (reactions["approach"] == "from_above")
    ].sort_values("session_date").reset_index(drop=True)
    print(f"Total Tier 1 signals: {len(t1)}")

    client = AlpacaClient()
    results: list[dict] = []
    t0 = time.perf_counter()

    for i, row in t1.iterrows():
        sd = _normalize_session_date(row["session_date"])
        sd_str = sd.strftime("%Y-%m-%d")

        # Load bars + find touch bar
        try:
            bars = load_bars("QQQ", sd_str, sd_str)
        except Exception:
            continue
        bars_win = filter_window(bars, "09:30", "11:15").reset_index(drop=True)
        if bars_win.empty:
            continue
        touch_t = int(row["t"])
        matching = bars_win[bars_win["t"] == touch_t]
        if matching.empty:
            continue
        entry_idx = int(matching.index[0])
        entry_close_under = float(matching.iloc[0]["c"])
        entry_ts = matching.iloc[0]["ts_et"]

        sim = simulate_short_race(bars_win, entry_idx)
        exit_ts = sim.get("exit_ts")
        if exit_ts is None:
            continue

        atm_strike = round(entry_close_under)

        for dte in DTE_OFFSETS:
            exp_date = next_business_days(sd, dte)
            for strike_label, offset in STRIKE_OFFSETS.items():
                strike = atm_strike + offset
                symbol = build_occ_put("QQQ", exp_date, strike)

                fetch_start = entry_ts - pd.Timedelta(minutes=1)
                fetch_end = exit_ts + pd.Timedelta(minutes=5)
                try:
                    opt_bars = client.get_option_bars(
                        symbol,
                        fetch_start.to_pydatetime() if hasattr(fetch_start, "to_pydatetime") else fetch_start,
                        fetch_end.to_pydatetime() if hasattr(fetch_end, "to_pydatetime") else fetch_end,
                    )
                except Exception:
                    opt_bars = []

                if not opt_bars:
                    results.append({
                        "session_date": sd_str, "dte": dte, "strike_label": strike_label,
                        "symbol": symbol, "status": "no_bars",
                    })
                    continue

                opt_df = pd.DataFrame(opt_bars)
                opt_df["t"] = pd.to_datetime(opt_df["t"]).dt.tz_convert(ET)
                entry_opt = _find_bar_at_or_after(opt_df, entry_ts)
                exit_opt = _find_bar_at_or_after(opt_df, exit_ts)
                if entry_opt is None:
                    results.append({
                        "session_date": sd_str, "dte": dte, "strike_label": strike_label,
                        "symbol": symbol, "status": "no_entry_bar",
                    })
                    continue
                if exit_opt is None:
                    exit_opt = float(opt_df.iloc[-1]["c"])

                pnl_contract = (exit_opt - entry_opt) * 100
                results.append({
                    "session_date": sd_str,
                    "dte": dte,
                    "strike_label": strike_label,
                    "symbol": symbol,
                    "strike": strike,
                    "expiration": exp_date,
                    "status": "ok",
                    "sim_outcome": sim["outcome"],
                    "bars_held": sim.get("bars_held"),
                    "entry_qqq": entry_close_under,
                    "exit_qqq": sim["exit_price"],
                    "entry_opt": entry_opt,
                    "exit_opt": exit_opt,
                    "pnl_contract": pnl_contract,
                })

        if (i + 1) % 20 == 0 or (i + 1) == len(t1):
            dt = time.perf_counter() - t0
            print(f"  processed {i+1:>3}/{len(t1)} signals  elapsed={dt:.0f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")

    # ================== Summary ==================
    print("\n" + "=" * 80)
    print("SUMMARY — P&L per contract by (DTE, strike)")
    print("=" * 80)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful sims.")
        return 0

    grouped = (
        ok.groupby(["dte", "strike_label"])
        .agg(
            n=("pnl_contract", "size"),
            mean=("pnl_contract", "mean"),
            median=("pnl_contract", "median"),
            total=("pnl_contract", "sum"),
            std=("pnl_contract", "std"),
            win_rate=("pnl_contract", lambda x: (x > 0).mean()),
            avg_win=("pnl_contract", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            avg_loss=("pnl_contract", lambda x: x[x < 0].mean() if (x < 0).any() else 0),
        )
        .reset_index()
    )

    def pf(row):
        if row["avg_loss"] == 0:
            return float("inf")
        avg_win = row["avg_win"]
        wr = row["win_rate"]
        if wr == 0:
            return 0.0
        return (avg_win * wr) / (abs(row["avg_loss"]) * (1 - wr))

    grouped["profit_factor"] = grouped.apply(pf, axis=1)
    grouped = grouped.sort_values(["dte", "strike_label"]).reset_index(drop=True)

    display = grouped.copy()
    for col in ("mean", "median", "total", "std", "avg_win", "avg_loss"):
        display[col] = display[col].map(lambda x: f"${x:+.0f}")
    display["win_rate"] = display["win_rate"].map(lambda x: f"{x:.0%}")
    display["profit_factor"] = display["profit_factor"].map(lambda x: f"{x:.2f}")
    print(display.to_string(index=False))

    print("\n" + "=" * 80)
    print("RANKED BY MEAN P&L / CONTRACT")
    print("=" * 80)
    ranked = grouped.sort_values("mean", ascending=False).copy()
    for col in ("mean", "median", "total", "std", "avg_win", "avg_loss"):
        ranked[col] = ranked[col].map(lambda x: f"${x:+.0f}")
    ranked["win_rate"] = ranked["win_rate"].map(lambda x: f"{x:.0%}")
    ranked["profit_factor"] = ranked["profit_factor"].map(lambda x: f"{x:.2f}")
    print(ranked[["dte", "strike_label", "n", "win_rate", "mean", "median",
                  "avg_win", "avg_loss", "profit_factor", "total"]].to_string(index=False))

    # Breakdown by outcome for the top config
    print("\n" + "=" * 80)
    print("OUTCOME BREAKDOWN FOR EACH (DTE, strike)")
    print("=" * 80)
    by_outcome = (
        ok.groupby(["dte", "strike_label", "sim_outcome"])
        .agg(n=("pnl_contract", "size"), mean_pnl=("pnl_contract", "mean"))
        .reset_index()
    )
    pivot_mean = by_outcome.pivot_table(
        index=["dte", "strike_label"], columns="sim_outcome",
        values="mean_pnl", aggfunc="first",
    ).fillna(0)
    pivot_n = by_outcome.pivot_table(
        index=["dte", "strike_label"], columns="sim_outcome",
        values="n", aggfunc="first",
    ).fillna(0).astype(int)

    combined = pd.DataFrame(index=pivot_mean.index)
    for col in pivot_mean.columns:
        combined[f"{col} (n)"] = pivot_n[col]
        combined[f"{col} ($)"] = pivot_mean[col].map(lambda x: f"${x:+.0f}")
    print(combined.to_string())

    # Annualized comparison with realistic slippage haircut
    print("\n" + "=" * 80)
    print("ANNUALIZED P&L (66 signals/yr) — with realistic slippage haircut")
    print("=" * 80)
    SLIPPAGE_PER_TRADE = 10  # $5 entry + $5 exit, conservative for non-0DTE
    ZERO_DTE_SLIPPAGE = 5    # 0DTE QQQ tends to be tighter on ATM
    out_rows = []
    for _, row in grouped.iterrows():
        slip = ZERO_DTE_SLIPPAGE if row["dte"] == 0 else SLIPPAGE_PER_TRADE
        net_mean = row["mean"] - slip
        annual = net_mean * 66
        out_rows.append({
            "dte": row["dte"],
            "strike": row["strike_label"],
            "gross_mean": f"${row['mean']:+.2f}",
            "slippage": f"-${slip}",
            "net_mean": f"${net_mean:+.2f}",
            "annual_1ctr": f"${annual:+.0f}",
        })
    print(pd.DataFrame(out_rows).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
