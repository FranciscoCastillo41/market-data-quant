"""Volume-confirmed level touch rules for Experiment E.

Four mechanical rule sets that turn visual "level rejection / breakout"
intuition into testable triggers. Each rule takes a DataFrame of 1-min bars
(already filtered to a time window and sorted by session, then by t) plus a
set of (level_name, level_price) pairs, and returns a DataFrame of triggered
entry events.

Each row in the output has:
    session_date, ts_et, t, bar_idx, level, level_name, approach,
    rule, direction, entry_ts, entry_price

We deliberately keep the rule evaluators pure — no I/O, no grid search, no
baseline math. That happens in the experiment orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TOL = 0.05  # dollars; "touched" if low <= level + TOL and high >= level - TOL
VOLUME_WINDOW = 20  # trailing bars for avg volume baseline
VOLUME_MULT = 2.0   # "spike" threshold (multiples of trailing average)
WICK_THRESHOLD = 0.50  # wick must be >= this fraction of bar range
MULTI_TOUCH_WINDOW = 10  # bars within which multi-touch counts for S2
MULTI_TOUCH_MULT = 1.5   # cumulative vol threshold for S2


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "session_date", "ts_et", "t", "bar_idx", "level", "level_name",
        "approach", "rule", "direction", "entry_ts", "entry_price",
    ])


def _touched(bar_h: float, bar_l: float, level: float) -> bool:
    return (bar_l <= level + TOL) and (bar_h >= level - TOL)


def _approach(prev_close: float, level: float) -> str:
    if prev_close < level:
        return "from_below"
    if prev_close > level:
        return "from_above"
    return "at"


def _rule_direction(rule: str, approach: str) -> str | None:
    """Map rule + approach to trade direction.

    Fade rules short when touched from below (resistance), long when touched
    from above (support). Momentum rules do the opposite.
    """
    if rule in ("S1", "S2", "S3"):
        if approach == "from_below":
            return "short"
        if approach == "from_above":
            return "long"
        return None
    if rule == "S4":
        if approach == "from_below":
            return "long"
        if approach == "from_above":
            return "short"
        return None
    return None


@dataclass
class _SessionArrays:
    """Per-session numpy arrays we use repeatedly."""
    ts_et: np.ndarray
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    session_date: object


def _prep_session(session_bars: pd.DataFrame) -> _SessionArrays:
    return _SessionArrays(
        ts_et=session_bars["ts_et"].to_numpy(),
        t=session_bars["t"].to_numpy(),
        o=session_bars["o"].to_numpy(),
        h=session_bars["h"].to_numpy(),
        l=session_bars["l"].to_numpy(),
        c=session_bars["c"].to_numpy(),
        v=session_bars["v"].to_numpy(),
        session_date=session_bars.iloc[0]["session_date"],
    )


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean ignoring the current value (look-ahead safe)."""
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window)
        if start == i:
            continue
        out[i] = values[start:i].mean()
    return out


def evaluate_s1_volume_spike_rejection(
    session_bars: pd.DataFrame,
    named_levels: list[tuple[str, float]],
) -> pd.DataFrame:
    """S1 — volume-spike rejection fade.

    On each bar, for each level:
      - touch condition
      - volume >= VOLUME_MULT * trailing VOLUME_WINDOW-bar avg
      - close on the rejection side (short if from_below, long if from_above)
    Enter at THIS bar's close.
    """
    if session_bars.empty or not named_levels:
        return _empty_events_df()

    sess = _prep_session(session_bars)
    n = len(sess.c)
    if n < 2:
        return _empty_events_df()
    vol_mean = _trailing_mean(sess.v, VOLUME_WINDOW)

    rows: list[dict] = []
    for i in range(1, n):  # start at 1 so we have a prev_close
        prev_c = sess.c[i - 1]
        if np.isnan(vol_mean[i]) or vol_mean[i] <= 0:
            continue
        if sess.v[i] < VOLUME_MULT * vol_mean[i]:
            continue
        for name, level in named_levels:
            if not _touched(sess.h[i], sess.l[i], level):
                continue
            approach = _approach(prev_c, level)
            if approach == "at":
                continue
            # Rejection side check
            if approach == "from_below" and sess.c[i] >= level:
                continue  # closed above resistance — not rejection
            if approach == "from_above" and sess.c[i] <= level:
                continue  # closed below support — not rejection
            direction = _rule_direction("S1", approach)
            if direction is None:
                continue
            rows.append({
                "session_date": sess.session_date,
                "ts_et": sess.ts_et[i],
                "t": int(sess.t[i]),
                "bar_idx": i,
                "level": float(level),
                "level_name": name,
                "approach": approach,
                "rule": "S1",
                "direction": direction,
                "entry_ts": sess.ts_et[i],
                "entry_price": float(sess.c[i]),
            })
    return pd.DataFrame(rows) if rows else _empty_events_df()


def evaluate_s2_multi_touch_absorption(
    session_bars: pd.DataFrame,
    named_levels: list[tuple[str, float]],
) -> pd.DataFrame:
    """S2 — multi-touch absorption fade.

    Trigger fires on the 2nd (or 3rd) touch of the same level within the last
    MULTI_TOUCH_WINDOW bars. Cumulative volume across touches must be >=
    MULTI_TOUCH_MULT * trailing average. Direction = fade the level.
    """
    if session_bars.empty or not named_levels:
        return _empty_events_df()

    sess = _prep_session(session_bars)
    n = len(sess.c)
    if n < 2:
        return _empty_events_df()
    vol_mean = _trailing_mean(sess.v, VOLUME_WINDOW)

    # Track per-level touch history: list of (bar_idx, volume)
    history: dict[str, list[tuple[int, float]]] = {name: [] for name, _ in named_levels}
    rows: list[dict] = []

    for i in range(1, n):
        prev_c = sess.c[i - 1]
        for name, level in named_levels:
            if not _touched(sess.h[i], sess.l[i], level):
                continue
            history[name].append((i, float(sess.v[i])))
            # Trim to only touches within MULTI_TOUCH_WINDOW bars
            history[name] = [
                (bi, bv) for (bi, bv) in history[name]
                if i - bi <= MULTI_TOUCH_WINDOW
            ]
            touches_in_window = history[name]
            if len(touches_in_window) < 2:
                continue  # need at least 2 touches

            # Cumulative volume check
            cum_v = sum(bv for _, bv in touches_in_window)
            if np.isnan(vol_mean[i]) or vol_mean[i] <= 0:
                continue
            if cum_v < MULTI_TOUCH_MULT * vol_mean[i] * len(touches_in_window):
                continue

            approach = _approach(prev_c, level)
            if approach == "at":
                continue
            # Rejection-side check based on current close
            if approach == "from_below" and sess.c[i] >= level:
                continue
            if approach == "from_above" and sess.c[i] <= level:
                continue
            direction = _rule_direction("S2", approach)
            if direction is None:
                continue
            rows.append({
                "session_date": sess.session_date,
                "ts_et": sess.ts_et[i],
                "t": int(sess.t[i]),
                "bar_idx": i,
                "level": float(level),
                "level_name": name,
                "approach": approach,
                "rule": "S2",
                "direction": direction,
                "entry_ts": sess.ts_et[i],
                "entry_price": float(sess.c[i]),
            })
            # Prevent re-fires on the same level within the window
            history[name] = []

    return pd.DataFrame(rows) if rows else _empty_events_df()


def evaluate_s3_wick_rejection(
    session_bars: pd.DataFrame,
    named_levels: list[tuple[str, float]],
) -> pd.DataFrame:
    """S3 — wick rejection with follow-through fade.

    Trigger on bar i + 1 if bar i:
      - touched the level
      - has a wick on the rejection side >= WICK_THRESHOLD of bar range
      - closed on the rejection side
    AND bar i+1 closes further in the rejection direction.
    Entry is at bar i+1 close.
    """
    if session_bars.empty or not named_levels:
        return _empty_events_df()

    sess = _prep_session(session_bars)
    n = len(sess.c)
    if n < 3:
        return _empty_events_df()

    rows: list[dict] = []
    for i in range(1, n - 1):  # need prev_c and next bar
        prev_c = sess.c[i - 1]
        bar_range = sess.h[i] - sess.l[i]
        if bar_range <= 0:
            continue

        body_hi = max(sess.o[i], sess.c[i])
        body_lo = min(sess.o[i], sess.c[i])
        upper_wick = sess.h[i] - body_hi
        lower_wick = body_lo - sess.l[i]

        for name, level in named_levels:
            if not _touched(sess.h[i], sess.l[i], level):
                continue
            approach = _approach(prev_c, level)
            if approach == "at":
                continue

            if approach == "from_below":
                # Rejecting resistance: upper wick, close < level, next close < this close
                if upper_wick / bar_range < WICK_THRESHOLD:
                    continue
                if sess.c[i] >= level:
                    continue
                if sess.c[i + 1] >= sess.c[i]:
                    continue
                direction = "short"
            elif approach == "from_above":
                if lower_wick / bar_range < WICK_THRESHOLD:
                    continue
                if sess.c[i] <= level:
                    continue
                if sess.c[i + 1] <= sess.c[i]:
                    continue
                direction = "long"
            else:
                continue

            rows.append({
                "session_date": sess.session_date,
                "ts_et": sess.ts_et[i + 1],
                "t": int(sess.t[i + 1]),
                "bar_idx": i + 1,
                "level": float(level),
                "level_name": name,
                "approach": approach,
                "rule": "S3",
                "direction": direction,
                "entry_ts": sess.ts_et[i + 1],
                "entry_price": float(sess.c[i + 1]),
            })
    return pd.DataFrame(rows) if rows else _empty_events_df()


def evaluate_s4_volume_breakout(
    session_bars: pd.DataFrame,
    named_levels: list[tuple[str, float]],
) -> pd.DataFrame:
    """S4 — volume-confirmed breakout momentum.

    Trigger:
      - bar touched the level
      - bar close is through the level (above for resistance, below for support)
      - bar volume >= VOLUME_MULT * trailing avg
      - direction = momentum (go with the break)
    Entry at bar close.
    """
    if session_bars.empty or not named_levels:
        return _empty_events_df()

    sess = _prep_session(session_bars)
    n = len(sess.c)
    if n < 2:
        return _empty_events_df()
    vol_mean = _trailing_mean(sess.v, VOLUME_WINDOW)

    rows: list[dict] = []
    for i in range(1, n):
        prev_c = sess.c[i - 1]
        if np.isnan(vol_mean[i]) or vol_mean[i] <= 0:
            continue
        if sess.v[i] < VOLUME_MULT * vol_mean[i]:
            continue
        for name, level in named_levels:
            if not _touched(sess.h[i], sess.l[i], level):
                continue
            approach = _approach(prev_c, level)
            if approach == "at":
                continue
            # Break-through condition
            if approach == "from_below" and sess.c[i] <= level:
                continue  # didn't break above
            if approach == "from_above" and sess.c[i] >= level:
                continue  # didn't break below
            direction = _rule_direction("S4", approach)
            if direction is None:
                continue
            rows.append({
                "session_date": sess.session_date,
                "ts_et": sess.ts_et[i],
                "t": int(sess.t[i]),
                "bar_idx": i,
                "level": float(level),
                "level_name": name,
                "approach": approach,
                "rule": "S4",
                "direction": direction,
                "entry_ts": sess.ts_et[i],
                "entry_price": float(sess.c[i]),
            })
    return pd.DataFrame(rows) if rows else _empty_events_df()


def evaluate_all_rules(
    session_bars: pd.DataFrame,
    named_levels: list[tuple[str, float]],
) -> pd.DataFrame:
    frames = [
        evaluate_s1_volume_spike_rejection(session_bars, named_levels),
        evaluate_s2_multi_touch_absorption(session_bars, named_levels),
        evaluate_s3_wick_rejection(session_bars, named_levels),
        evaluate_s4_volume_breakout(session_bars, named_levels),
    ]
    return pd.concat(frames, ignore_index=True)
