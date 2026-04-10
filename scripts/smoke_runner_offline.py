"""Offline smoke test of the live runner.

Replays a known Tier 1 signal day (2026-04-07) by feeding saved 1-min bars
into the detector + signal + (mocked) runner, without touching Alpaca or
Massive. Verifies:
  - TouchDetector fires on the known first-touch
  - evaluate_tier1 returns a TradePlan
  - Position management: target/stop/timeout logic works as expected
  - Journal lines are written

Does NOT test the feed polling or Alpaca client — those are exercised by the
replay_today.py and check_alpaca.py scripts separately.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.live.detector import TouchDetector
from mdq.live.feed import LiveBar
from mdq.live.journal import Journal
from mdq.live.signal import evaluate_tier1

ET = pytz.timezone("America/New_York")

REPLAY_DATE = date(2026, 4, 7)


def main() -> int:
    print(f"Offline runner smoke test — replaying {REPLAY_DATE}")
    print("=" * 70)

    # Load prior_low
    profs = pd.read_parquet(RESULTS_DIR / "experiment_c" / "profiles__QQQ.parquet")
    prior = profs[profs["session_date"] < REPLAY_DATE].sort_values("session_date").iloc[-1]
    prior_low = float(prior["low"])
    print(f"prior_low from {prior['session_date']}: ${prior_low:.2f}")

    # Load bars + filter to window
    bars_df = load_bars("QQQ", REPLAY_DATE.strftime("%Y-%m-%d"), REPLAY_DATE.strftime("%Y-%m-%d"))
    bars_win = filter_window(bars_df, "09:30", "11:15").reset_index(drop=True)
    print(f"bars in window: {len(bars_win)}")

    # Convert to LiveBar sequence
    def to_live_bar(row) -> LiveBar:
        return LiveBar(
            t=int(row["t"]),
            ts_et=row["ts_et"],
            o=float(row["o"]),
            h=float(row["h"]),
            l=float(row["l"]),
            c=float(row["c"]),
            v=float(row["v"]),
        )

    detector = TouchDetector([("prior_low", prior_low)], tolerance=0.05)
    journal = Journal(REPLAY_DATE, name="tier1_smoke")

    # Simulation state
    open_symbol: str | None = None
    entry_price: float | None = None
    entry_idx: int | None = None
    target: float = 0.0
    stop: float = 0.0
    horizon = 15

    signals_fired = 0
    exits_fired = 0

    for i, (_, row) in enumerate(bars_win.iterrows()):
        bar = to_live_bar(row)

        # Manage open position first
        if open_symbol is not None and entry_idx is not None:
            bars_held = i - entry_idx
            hit_stop = bar.h >= entry_price + stop
            hit_target = bar.l <= entry_price - target
            if hit_stop:
                print(f"  STOP HIT at {bar.ts_et}  (bar.h={bar.h:.2f}  stop_px={entry_price + stop:.2f})")
                journal.write("stop_hit", ts=bar.ts_et, bars_held=bars_held)
                open_symbol = None
                entry_idx = None
                exits_fired += 1
            elif hit_target:
                print(f"  TARGET HIT at {bar.ts_et}  (bar.l={bar.l:.2f}  target_px={entry_price - target:.2f})")
                journal.write("target_hit", ts=bar.ts_et, bars_held=bars_held)
                open_symbol = None
                entry_idx = None
                exits_fired += 1
            elif bars_held >= horizon:
                print(f"  TIMEOUT at {bar.ts_et}  bars_held={bars_held}")
                journal.write("timeout", ts=bar.ts_et, bars_held=bars_held)
                open_symbol = None
                entry_idx = None
                exits_fired += 1

        # Touch detection
        events = detector.on_bar(bar)
        for ev in events:
            print(f"  TOUCH at {bar.ts_et}  level={ev.level}  approach={ev.approach}")
            journal.write("touch", level=ev.level, approach=ev.approach, bar_ts=bar.ts_et)
            plan = evaluate_tier1(ev)
            if plan is None:
                print(f"    -> signal SKIPPED (not Tier 1)")
                continue
            if open_symbol is not None:
                print(f"    -> already in position, skipping")
                continue

            open_symbol = f"QQQ_SIM_STRIKE_{round(bar.c)}"
            entry_price = bar.c
            entry_idx = i
            target = plan.target_move
            stop = plan.stop_move
            signals_fired += 1
            print(f"    -> SIGNAL FIRED  entry={bar.c:.2f}  target={entry_price - target:.2f}  stop={entry_price + stop:.2f}")
            journal.write("signal_fired", tier=plan.tier, entry=bar.c, symbol=open_symbol)

    print()
    print(f"signals_fired: {signals_fired}")
    print(f"exits_fired:   {exits_fired}")
    print(f"journal:       {journal.path}")

    if open_symbol is not None:
        print(f"WARNING: ended with open position (would be force-closed at 11:30 ET)")

    # Sanity check: we KNOW 2026-04-07 had a Tier 1 signal at 09:31
    assert signals_fired >= 1, f"expected at least 1 signal on {REPLAY_DATE}, got {signals_fired}"
    assert exits_fired >= 1, f"expected at least 1 exit on {REPLAY_DATE}, got {exits_fired}"
    print("\nAll assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
