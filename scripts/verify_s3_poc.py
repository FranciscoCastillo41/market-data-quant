"""Verify the S3 POC signal on 2026 OOS data.

Prints every individual trade with dates, entry/exit, outcome. Lets us eyeball
whether the 83% OOS hit rate is real or a bug artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.data.bars import load_bars
from mdq.experiments.e_spy_volume import (
    WINDOW,
    collect_events,
    first_passage_grid,
    measure_events_with_atr,
)
from mdq.levels.volume_profile import compute_all_profiles

OOS_START = "2026-01-01"
OOS_END = "2026-04-08"


def main() -> int:
    print("Loading OOS SPY bars...")
    bars = load_bars("SPY", OOS_START, OOS_END)
    profiles = compute_all_profiles(bars, bin_size=0.05)

    events = collect_events(bars, profiles, WINDOW)
    events = events[(events["rule"] == "S3") & (events["level_name"] == "poc")]
    events = measure_events_with_atr(bars, events)
    events = events.reset_index(drop=True)
    print(f"\nS3 POC events in OOS: {len(events)}")

    if events.empty:
        return 0

    outcomes = first_passage_grid(bars, events, target_mult=1.00, stop_mult=1.00)
    events["outcome"] = outcomes

    print("\nMonthly breakdown:")
    events["month"] = pd.to_datetime(events["ts_et"]).dt.to_period("M")
    by_month = events.groupby("month").agg(
        n=("outcome", "size"),
        targets=("outcome", lambda x: (x == "target").sum()),
        stops=("outcome", lambda x: (x == "stop").sum()),
        timeouts=("outcome", lambda x: (x == "timeout").sum()),
    )
    by_month["hit_rate"] = by_month["targets"] / by_month["n"]
    print(by_month.to_string())

    print("\n\nAll S3 POC trades (OOS):")
    disp = events[[
        "session_date", "ts_et", "level", "direction", "approach",
        "entry_price", "atr", "outcome",
    ]].copy()
    disp["atr"] = disp["atr"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    disp["entry_price"] = disp["entry_price"].map(lambda x: f"${x:.2f}")
    disp["level"] = disp["level"].map(lambda x: f"${x:.2f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 30)
    print(disp.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
