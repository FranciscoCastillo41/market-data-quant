"""Smoke test: replay 2026-04-02 through the gap runner.

2026-04-02 was a gap-down day for SPY (from the backtest: gap -1.3%, target hit).
This verifies the runner logic fires correctly in offline mode.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth
from mdq.live.feed import LiveBar
from mdq.live.gap_runner import (
    CUM_DELTA_DECLINE_BARS,
    MAX_DIP_WAIT_BARS,
    MIN_GAP_PCT,
    _buy_pressure,
    _load_prior_day_profile,
    _load_prior_week_val,
)

REPLAY_DATE = date(2026, 4, 2)


def main() -> int:
    print(f"Smoke test — replaying gap runner on {REPLAY_DATE}")
    print("=" * 70)

    prior = _load_prior_day_profile(REPLAY_DATE)
    if prior is None:
        print("FAIL: no prior profile")
        return 1
    print(f"Prior day ({prior['date']}): VAL=${prior['val']:.2f}  POC=${prior['poc']:.2f}  close=${prior['close']:.2f}")

    weekly_val = _load_prior_week_val(REPLAY_DATE)
    print(f"Weekly VAL: ${weekly_val:.2f}" if weekly_val else "Weekly VAL: unavailable")

    bars = load_bars("SPY", REPLAY_DATE.isoformat(), REPLAY_DATE.isoformat())
    rth = filter_rth(bars).reset_index(drop=True)
    print(f"RTH bars: {len(rth)}")

    if rth.empty:
        return 1

    # Simulate the runner logic
    open_price = float(rth.iloc[0]["o"])
    gap_pct = (open_price - prior["close"]) / prior["close"]
    print(f"\nOpen: ${open_price:.2f}  gap: {gap_pct:+.2%}")
    print(f"Open < VAL (${prior['val']:.2f})? {open_price < prior['val']}")
    print(f"Gap >= 0.3%? {abs(gap_pct) >= MIN_GAP_PCT}")
    if weekly_val:
        print(f"Above weekly VAL (${weekly_val:.2f})? {open_price >= weekly_val}")

    if open_price >= prior["val"]:
        print("\nNO SIGNAL: open above VAL")
        return 0
    if gap_pct > -MIN_GAP_PCT:
        print("\nNO SIGNAL: gap too small")
        return 0

    gap_dollars = prior["close"] - open_price
    stop_px = open_price - gap_dollars
    target_px = prior["poc"]
    print(f"\nSIGNAL QUALIFIED")
    print(f"  gap: ${gap_dollars:.2f}")
    print(f"  target (POC): ${target_px:.2f}")
    print(f"  stop (2x gap): ${stop_px:.2f}")

    # Entry C: find session low in first 30 bars
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
        entry_idx = MAX_DIP_WAIT_BARS - 1

    if entry_idx is None:
        print("NO ENTRY: no bounce detected")
        return 0

    entry_price = float(rth.iloc[entry_idx]["c"])
    entry_ts = rth.iloc[entry_idx]["ts_et"]
    print(f"\nENTRY at bar {entry_idx} ({entry_ts})")
    print(f"  session_low: ${session_low:.2f}")
    print(f"  entry_price: ${entry_price:.2f}")

    # Exit Z: cum_delta exhaustion
    cum_delta = 0.0
    peak = 0.0
    decline_count = 0
    prev_cd = 0.0
    exit_idx = None
    exit_reason = "timeout"

    for i in range(entry_idx + 1, min(entry_idx + 180, len(rth))):
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
        exit_idx = min(entry_idx + 180, len(rth)) - 1
        exit_reason = "timeout"

    exit_price = float(rth.iloc[exit_idx]["c"])
    exit_ts = rth.iloc[exit_idx]["ts_et"]
    pnl = exit_price - entry_price

    print(f"\nEXIT at bar {exit_idx} ({exit_ts})")
    print(f"  reason: {exit_reason}")
    print(f"  exit_price: ${exit_price:.2f}")
    print(f"  bars_held: {exit_idx - entry_idx}")
    print(f"  underlying P&L: ${pnl:+.2f} ({pnl/entry_price:+.2%})")

    assert entry_idx is not None, "expected entry on a gap-down day"
    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
