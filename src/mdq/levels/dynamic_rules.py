"""S3-style wick rejection rule evaluated against dynamic (per-bar) levels.

The original S3 evaluator (in volume_rules.py) takes a fixed set of levels
for the whole session. This version takes a per-bar level timeseries so the
level can move as the session progresses (developing POC, rolling POC, etc).

Rule semantics are identical to S3:
    - bar touches the current (per-bar) level
    - rejection wick >= 50% of range in the correct direction
    - bar closes on the rejection side
    - next bar confirms by closing further in the rejection direction
    - fire on bar i+1, direction = fade
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.levels.volume_rules import (
    TOL,
    WICK_THRESHOLD,
    _approach,
    _touched,
)


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "session_date", "ts_et", "t", "bar_idx", "level", "level_name",
        "approach", "rule", "direction", "entry_ts", "entry_price",
    ])


def evaluate_s3_dynamic(
    session_bars: pd.DataFrame,
    level_series: np.ndarray,
    level_name: str,
) -> pd.DataFrame:
    """S3 wick-rejection fade against a per-bar level timeseries.

    `level_series` must have the same length as `session_bars` and contain
    the level value at each bar (NaN before warmup). The rule uses the
    level at bar i when evaluating bar i.

    Enters at bar i+1 close (after follow-through confirmation).
    """
    if session_bars.empty or len(level_series) != len(session_bars):
        return _empty_events_df()

    bars = session_bars.reset_index(drop=True)
    n = len(bars)
    if n < 3:
        return _empty_events_df()

    o = bars["o"].to_numpy()
    h = bars["h"].to_numpy()
    l = bars["l"].to_numpy()
    c = bars["c"].to_numpy()
    ts_et = bars["ts_et"].to_numpy()
    t = bars["t"].to_numpy()
    session_date = bars.iloc[0]["session_date"]

    rows: list[dict] = []
    for i in range(1, n - 1):
        level = level_series[i]
        if np.isnan(level):
            continue
        if not _touched(h[i], l[i], level):
            continue
        prev_c = c[i - 1]
        approach = _approach(prev_c, level)
        if approach == "at":
            continue

        bar_range = h[i] - l[i]
        if bar_range <= 0:
            continue
        body_hi = max(o[i], c[i])
        body_lo = min(o[i], c[i])
        upper_wick = h[i] - body_hi
        lower_wick = body_lo - l[i]

        if approach == "from_below":
            if upper_wick / bar_range < WICK_THRESHOLD:
                continue
            if c[i] >= level:
                continue
            if c[i + 1] >= c[i]:
                continue
            direction = "short"
        else:
            if lower_wick / bar_range < WICK_THRESHOLD:
                continue
            if c[i] <= level:
                continue
            if c[i + 1] <= c[i]:
                continue
            direction = "long"

        rows.append({
            "session_date": session_date,
            "ts_et": ts_et[i + 1],
            "t": int(t[i + 1]),
            "bar_idx": i + 1,
            "level": float(level),
            "level_name": level_name,
            "approach": approach,
            "rule": "S3",
            "direction": direction,
            "entry_ts": ts_et[i + 1],
            "entry_price": float(c[i + 1]),
        })

    return pd.DataFrame(rows) if rows else _empty_events_df()
