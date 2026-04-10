"""Offline smoke test of the SPY S3 POC detector.

Replays 2026-02-10 (known-good: 3 S3 POC longs, all targets) by feeding
saved 1-min bars through the detector and asserting the expected signals fire.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mdq.data.bars import load_bars
from mdq.data.calendar import filter_window
from mdq.levels.volume_profile import compute_all_profiles
from mdq.live.feed import LiveBar
from mdq.live.spy_signal import SpyS3PocDetector

REPLAY_DATE = date(2026, 2, 10)


def main() -> int:
    print(f"Offline smoke test — replaying {REPLAY_DATE}")
    print("=" * 70)

    # Load profiles to get prior POC
    bars_all = load_bars("SPY", "2026-02-01", "2026-02-28")
    profiles = compute_all_profiles(bars_all, bin_size=0.05)
    prior = profiles[profiles["session_date"] < REPLAY_DATE].sort_values("session_date").iloc[-1]
    prior_poc = float(prior["poc"])
    print(f"prior POC from {prior['session_date']}: ${prior_poc:.2f}")

    bars_day = load_bars("SPY", REPLAY_DATE.isoformat(), REPLAY_DATE.isoformat())
    bars_win = filter_window(bars_day, "09:30", "11:30").reset_index(drop=True)
    print(f"bars in window: {len(bars_win)}")

    # Need extra bars BEFORE the window for ATR warmup
    # Just feed everything from 09:00 onward
    bars_ext = filter_window(bars_day, "09:00", "11:30").reset_index(drop=True)

    detector = SpyS3PocDetector(level_name="poc", level=prior_poc)

    signals = []
    for _, row in bars_ext.iterrows():
        lb = LiveBar(
            t=int(row["t"]),
            ts_et=row["ts_et"],
            o=float(row["o"]),
            h=float(row["h"]),
            l=float(row["l"]),
            c=float(row["c"]),
            v=float(row["v"]),
        )
        sig = detector.on_bar(lb)
        if sig is not None:
            print(f"  SIGNAL at {sig.bar.ts_et}  dir={sig.direction}  "
                  f"atr={sig.atr:.3f}  entry={sig.bar.c:.2f}")
            signals.append(sig)

    print(f"\nTotal signals on {REPLAY_DATE}: {len(signals)}")
    assert len(signals) >= 1, f"expected >=1 signal on {REPLAY_DATE}, got {len(signals)}"
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
