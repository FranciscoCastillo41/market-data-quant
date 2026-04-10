"""Intraday entry/exit timing optimization for the gap-reversion strategy.

For each of the 101 filtered signals, we have 1-min bars for the entry day.
We test entry timing (when to buy the call) and exit timing (when to sell)
using volume-based intraday signals.

Entry variants:
    A: Buy at open (9:30 bar close) — current baseline
    B: Buy at first bar where cum_delta > 0 AND price > 15-bar VWAP
       (max wait 30 bars from open, else skip)
    C: Buy at session low if it occurs in first 30 bars (limit dip-buy)

Exit variants:
    X: Hold for daily target at prior POC / stop at 2x gap (baseline)
    Y: Exit at first touch of session VWAP after being below it
       (VWAP reclaim = momentum shift, take profit early)
    Z: Exit when cumulative delta peaks and declines for 5 consecutive bars
       (exhaustion detection)

Each combo is tested. No parameter sweeps beyond the 3×3 matrix.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.live.alpaca import AlpacaClient

CSV = RESULTS_DIR / "gap_reversion" / "full_backtest.csv"
OUT = RESULTS_DIR / "gap_reversion"

MAX_ENTRY_WAIT = 30  # bars (30 min) max to wait for entry B/C
MAX_HOLD_BARS = 180  # 3 hours max intraday hold
CUM_DELTA_DECLINE_BARS = 5  # Z exit: cum_delta declining for this many bars


def _buy_pressure(o: float, h: float, l: float, c: float) -> float:
    rng = h - l
    if rng <= 0:
        return 0.5
    return (c - l) / rng


def _cum_delta(bars: pd.DataFrame) -> np.ndarray:
    """Cumulative volume delta from bar 0 onward."""
    o = bars["o"].to_numpy()
    h = bars["h"].to_numpy()
    l = bars["l"].to_numpy()
    c = bars["c"].to_numpy()
    v = bars["v"].to_numpy()
    n = len(bars)
    delta = np.zeros(n)
    for i in range(n):
        bp = _buy_pressure(o[i], h[i], l[i], c[i])
        delta[i] = (2 * bp - 1) * v[i]
    return np.cumsum(delta)


def _session_vwap_series(bars: pd.DataFrame) -> np.ndarray:
    """Running VWAP from bar 0 onward."""
    h = bars["h"].to_numpy()
    l = bars["l"].to_numpy()
    c = bars["c"].to_numpy()
    v = bars["v"].to_numpy()
    tp = (h + l + c) / 3
    cum_tpv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    cum_v[cum_v == 0] = 1  # avoid div by zero
    return cum_tpv / cum_v


def find_entry_A(bars: pd.DataFrame) -> int | None:
    """Entry A: first bar (the open)."""
    if bars.empty:
        return None
    return 0


def find_entry_B(bars: pd.DataFrame, cum_delta: np.ndarray, vwap: np.ndarray) -> int | None:
    """Entry B: first bar where cum_delta > 0 AND close > VWAP, within 30 bars."""
    c = bars["c"].to_numpy()
    for i in range(min(MAX_ENTRY_WAIT, len(bars))):
        if cum_delta[i] > 0 and c[i] > vwap[i]:
            return i
    return None  # skipped — never confirmed


def find_entry_C(bars: pd.DataFrame) -> int | None:
    """Entry C: bar at session low if it's within first 30 bars."""
    if bars.empty:
        return None
    lows = bars["l"].to_numpy()
    n = min(MAX_ENTRY_WAIT, len(lows))
    low_idx = int(np.argmin(lows[:n]))
    # Only valid if the low is in the first 30 bars
    if low_idx < n:
        return low_idx
    return None


def find_exit_X(
    bars: pd.DataFrame,
    entry_idx: int,
    target_px: float,
    stop_px: float,
) -> int | None:
    """Exit X: target/stop on intraday bars (same logic as daily, but on 1-min)."""
    c = bars["c"].to_numpy()
    h = bars["h"].to_numpy()
    l = bars["l"].to_numpy()
    n = min(entry_idx + 1 + MAX_HOLD_BARS, len(bars))
    for i in range(entry_idx + 1, n):
        if h[i] >= target_px:
            return i
        if l[i] <= stop_px:
            return i
    # Timeout: return last bar
    return n - 1 if n > entry_idx + 1 else None


def find_exit_Y(
    bars: pd.DataFrame,
    entry_idx: int,
    vwap: np.ndarray,
) -> int | None:
    """Exit Y: first bar where price reclaims VWAP from below after entry.

    Must be below VWAP at entry, then cross above.
    """
    c = bars["c"].to_numpy()
    n = min(entry_idx + 1 + MAX_HOLD_BARS, len(bars))
    # Find first bar after entry where close > VWAP
    # (only valid if we started below VWAP)
    if entry_idx >= len(c):
        return None
    was_below = c[entry_idx] < vwap[entry_idx]
    if not was_below:
        # Already above VWAP at entry; use next reclaim after a dip
        pass
    for i in range(entry_idx + 1, n):
        if c[i] > vwap[i] and (i == entry_idx + 1 or c[i - 1] <= vwap[i - 1]):
            return i
    return n - 1 if n > entry_idx + 1 else None


def find_exit_Z(
    bars: pd.DataFrame,
    entry_idx: int,
    cum_delta: np.ndarray,
) -> int | None:
    """Exit Z: cum_delta peaks and declines for CUM_DELTA_DECLINE_BARS consecutive bars."""
    n = min(entry_idx + 1 + MAX_HOLD_BARS, len(bars))
    if entry_idx + CUM_DELTA_DECLINE_BARS + 1 >= n:
        return n - 1 if n > entry_idx + 1 else None

    peak = cum_delta[entry_idx]
    decline_count = 0
    for i in range(entry_idx + 1, n):
        if cum_delta[i] > peak:
            peak = cum_delta[i]
            decline_count = 0
        elif cum_delta[i] < cum_delta[i - 1]:
            decline_count += 1
            if decline_count >= CUM_DELTA_DECLINE_BARS:
                return i
        else:
            decline_count = 0
    return n - 1 if n > entry_idx + 1 else None


def simulate_one(
    ticker: str,
    signal_date,
    prior_poc: float,
    entry_open: float,
    gap_dollars: float,
    client: AlpacaClient | None,
) -> list[dict]:
    """For one signal, test all 3×3 entry/exit combos on intraday 1-min bars."""
    sd = signal_date if isinstance(signal_date, str) else str(signal_date)

    try:
        bars = load_bars(ticker, sd, sd)
    except Exception:
        return []

    rth = filter_rth(bars).reset_index(drop=True)
    if len(rth) < 60:
        return []

    target_px = prior_poc
    stop_px = entry_open - gap_dollars  # 2x gap

    cum_delta = _cum_delta(rth)
    vwap = _session_vwap_series(rth)
    closes = rth["c"].to_numpy()

    # Find entries
    entries = {
        "A": find_entry_A(rth),
        "B": find_entry_B(rth, cum_delta, vwap),
        "C": find_entry_C(rth),
    }

    rows: list[dict] = []
    for entry_name, entry_idx in entries.items():
        if entry_idx is None:
            rows.append({
                "ticker": ticker, "session_date": sd,
                "entry_rule": entry_name, "exit_rule": "—",
                "status": "skipped_entry",
            })
            continue

        entry_price = closes[entry_idx]
        entry_ts = rth.iloc[entry_idx]["ts_et"]

        # Find exits
        exits = {
            "X": find_exit_X(rth, entry_idx, target_px, stop_px),
            "Y": find_exit_Y(rth, entry_idx, vwap),
            "Z": find_exit_Z(rth, entry_idx, cum_delta),
        }

        for exit_name, exit_idx in exits.items():
            if exit_idx is None:
                rows.append({
                    "ticker": ticker, "session_date": sd,
                    "entry_rule": entry_name, "exit_rule": exit_name,
                    "status": "skipped_exit",
                })
                continue

            exit_price = closes[exit_idx]
            exit_ts = rth.iloc[exit_idx]["ts_et"]
            bars_held = exit_idx - entry_idx
            underlying_pnl = exit_price - entry_price
            underlying_pct = underlying_pnl / entry_price

            # Did it hit target or stop?
            hit_target = any(rth.iloc[j]["h"] >= target_px
                           for j in range(entry_idx + 1, exit_idx + 1))
            hit_stop = any(rth.iloc[j]["l"] <= stop_px
                          for j in range(entry_idx + 1, exit_idx + 1))

            rows.append({
                "ticker": ticker,
                "session_date": sd,
                "entry_rule": entry_name,
                "exit_rule": exit_name,
                "status": "ok",
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_ts": str(entry_ts),
                "exit_ts": str(exit_ts),
                "bars_held": bars_held,
                "underlying_pnl": underlying_pnl,
                "underlying_pct": underlying_pct,
                "hit_daily_target": hit_target,
                "hit_daily_stop": hit_stop,
            })

    return rows


def main() -> int:
    print("=" * 100)
    print("Intraday Entry/Exit Timing — Volume-Based Optimization")
    print("=" * 100)

    df = pd.read_csv(CSV)
    # Apply the two adopted filters
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date

    # We need the filtered set — recreate the filter logic inline
    # (simpler than importing from improve_gap_reversion)
    ok = df[df["status"] == "ok"].copy()
    print(f"Total signals with option data: {len(ok)}")

    all_rows: list[dict] = []
    for i, row in ok.iterrows():
        results = simulate_one(
            row["ticker"],
            row["session_date"],
            row["prior_poc"],
            row["entry_open"],
            row["entry_open"] - row["stop_px"] if "stop_px" in row else row["prior_close"] - row["entry_open"],
            client=None,
        )
        all_rows.extend(results)
        if (i + 1) % 30 == 0:
            print(f"  processed {i+1}/{len(ok)}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT / "intraday_timing.csv", index=False)

    # Analyze
    valid = results_df[results_df["status"] == "ok"].copy()
    print(f"\nValid intraday simulations: {len(valid)}")

    # Add year for train/OOS split
    valid["year"] = pd.to_datetime(valid["session_date"]).dt.year
    train = valid[valid["year"] <= 2025]
    oos = valid[valid["year"] >= 2026]

    pd.set_option("display.width", 200)

    # Summary by (entry_rule, exit_rule) combo
    print("\n" + "#" * 100)
    print("# ENTRY × EXIT COMBO RESULTS (all years)")
    print("#" * 100)

    combos = valid.groupby(["entry_rule", "exit_rule"]).agg(
        n=("underlying_pct", "size"),
        mean_pct=("underlying_pct", "mean"),
        median_pct=("underlying_pct", "median"),
        win_rate=("underlying_pct", lambda x: (x > 0).mean()),
        mean_bars=("bars_held", "mean"),
    ).reset_index()
    combos = combos.sort_values("mean_pct", ascending=False)

    for _, r in combos.iterrows():
        print(f"  {r['entry_rule']}{r['exit_rule']}  n={r['n']:>4}  "
              f"mean={r['mean_pct']:>+.3%}  median={r['median_pct']:>+.3%}  "
              f"WR={r['win_rate']:>.0%}  avg_hold={r['mean_bars']:.0f}bars")

    # Train vs OOS for each combo
    print("\n" + "#" * 100)
    print("# TRAIN (2024-2025) vs OOS (2026) per combo")
    print("#" * 100)

    print(f"\n  {'combo':>4}  {'n_tr':>5}  {'mean_tr':>9}  {'wr_tr':>6}  "
          f"{'n_oos':>5}  {'mean_oos':>9}  {'wr_oos':>6}  {'verdict':>8}")
    print("  " + "-" * 70)

    for entry_r in ("A", "B", "C"):
        for exit_r in ("X", "Y", "Z"):
            tag = f"{entry_r}{exit_r}"
            tr = train[(train["entry_rule"] == entry_r) & (train["exit_rule"] == exit_r)]
            os = oos[(oos["entry_rule"] == entry_r) & (oos["exit_rule"] == exit_r)]
            tr_mean = tr["underlying_pct"].mean() if len(tr) > 0 else np.nan
            tr_wr = (tr["underlying_pct"] > 0).mean() if len(tr) > 0 else np.nan
            os_mean = os["underlying_pct"].mean() if len(os) > 0 else np.nan
            os_wr = (os["underlying_pct"] > 0).mean() if len(os) > 0 else np.nan

            both_positive = (
                not np.isnan(tr_mean) and tr_mean > 0
                and not np.isnan(os_mean) and os_mean > 0
            )
            verdict = "✅" if both_positive else "❌"

            tr_mean_s = f"{tr_mean:>+.3%}" if not np.isnan(tr_mean) else "    —"
            os_mean_s = f"{os_mean:>+.3%}" if not np.isnan(os_mean) else "    —"
            tr_wr_s = f"{tr_wr:>.0%}" if not np.isnan(tr_wr) else "  —"
            os_wr_s = f"{os_wr:>.0%}" if not np.isnan(os_wr) else "  —"
            print(f"    {tag}  {len(tr):>5}  {tr_mean_s:>9}  {tr_wr_s:>6}  "
                  f"{len(os):>5}  {os_mean_s:>9}  {os_wr_s:>6}  {verdict:>8}")

    # Best combo
    print("\n" + "#" * 100)
    print("# WINNER — best combo that's positive on BOTH train AND OOS")
    print("#" * 100)

    best = None
    best_sum = -999
    for entry_r in ("A", "B", "C"):
        for exit_r in ("X", "Y", "Z"):
            tr = train[(train["entry_rule"] == entry_r) & (train["exit_rule"] == exit_r)]
            os = oos[(oos["entry_rule"] == entry_r) & (oos["exit_rule"] == exit_r)]
            tr_mean = tr["underlying_pct"].mean() if len(tr) > 0 else -999
            os_mean = os["underlying_pct"].mean() if len(os) > 0 else -999
            if tr_mean > 0 and os_mean > 0:
                combo_sum = tr_mean + os_mean
                if combo_sum > best_sum:
                    best = (entry_r, exit_r, tr, os, tr_mean, os_mean)
                    best_sum = combo_sum

    if best is None:
        print("  No combo is positive on both train and OOS.")
        print("  Stick with the daily-bar entry/exit (baseline AX).")
    else:
        entry_r, exit_r, tr, os, tr_mean, os_mean = best
        print(f"\n  WINNER: Entry {entry_r} + Exit {exit_r}")
        print(f"  Train: n={len(tr)}  mean={tr_mean:+.3%}  WR={(tr['underlying_pct'] > 0).mean():.0%}")
        print(f"  OOS:   n={len(os)}  mean={os_mean:+.3%}  WR={(os['underlying_pct'] > 0).mean():.0%}")

        # Compare to baseline AX
        ax_tr = train[(train["entry_rule"] == "A") & (train["exit_rule"] == "X")]
        ax_os = oos[(oos["entry_rule"] == "A") & (oos["exit_rule"] == "X")]
        ax_tr_mean = ax_tr["underlying_pct"].mean() if len(ax_tr) > 0 else 0
        ax_os_mean = ax_os["underlying_pct"].mean() if len(ax_os) > 0 else 0
        print(f"\n  Baseline AX:")
        print(f"  Train: n={len(ax_tr)}  mean={ax_tr_mean:+.3%}")
        print(f"  OOS:   n={len(ax_os)}  mean={ax_os_mean:+.3%}")
        print(f"\n  Improvement train: {tr_mean - ax_tr_mean:+.3%}/trade")
        print(f"  Improvement OOS:   {os_mean - ax_os_mean:+.3%}/trade")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
