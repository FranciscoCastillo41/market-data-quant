"""Trading calendar helpers.

US equities session bucketing, RTH filtering, and session-date assignment.
We use America/New_York for all session math; clients can convert to PT for
display.
"""

from __future__ import annotations

import pandas as pd

from mdq.config import ET_TZ


def to_et(df: pd.DataFrame, ts_col: str = "t") -> pd.DataFrame:
    """Add a tz-aware ET timestamp column `ts_et` from a unix-ms column.

    Returns a new DataFrame (does not mutate input).
    """
    out = df.copy()
    out["ts_et"] = pd.to_datetime(out[ts_col], unit="ms", utc=True).dt.tz_convert(ET_TZ)
    return out


def add_session_date(df: pd.DataFrame, ts_col: str = "ts_et") -> pd.DataFrame:
    """Add a `session_date` column = the ET calendar date of each bar.

    For equities, session date = calendar date in ET since no cross-midnight.
    """
    out = df.copy()
    out["session_date"] = out[ts_col].dt.tz_convert(ET_TZ).dt.date
    return out


def filter_rth(df: pd.DataFrame, ts_col: str = "ts_et") -> pd.DataFrame:
    """Keep only bars inside regular trading hours (09:30:00 - 15:59:00 ET).

    Bars are left-labeled (bar at 09:30 covers 09:30-09:31). RTH range is
    inclusive of 09:30 and exclusive of 16:00 so the last kept bar is 15:59.
    """
    ts = df[ts_col].dt.tz_convert(ET_TZ)
    t = ts.dt.time
    mask = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())
    return df[mask].copy()


def filter_window(
    df: pd.DataFrame,
    start_hhmm: str,
    end_hhmm: str,
    ts_col: str = "ts_et",
) -> pd.DataFrame:
    """Keep bars with ts in [start_hhmm, end_hhmm) ET."""
    ts = df[ts_col].dt.tz_convert(ET_TZ)
    t = ts.dt.time
    start = pd.Timestamp(start_hhmm).time()
    end = pd.Timestamp(end_hhmm).time()
    mask = (t >= start) & (t < end)
    return df[mask].copy()
