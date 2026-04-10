"""Dynamic HVN breakdown/breakout rule — the 10:18 SPY pattern generalized.

Trigger (SHORT side, mirror for LONG):
    bar is full-body DOWN (body >= 0.70 of range)
    bar volume >= VOL_MULT x trailing 20-bar average
    bar close is below session VWAP
    bar close is below a top-K intraday HVN (one of the volume shelves)
    the HVN has been "established" (>= 10 bars of history in the cumulative profile)
    → enter SHORT at bar close

The long mirror: full-body up, above VWAP, above HVN, same established rule.

This is a momentum-through-level rule, not a rejection-at-level rule.
Captures institutional liquidation (or impulse buying) through a volume shelf
that has been holding as support/resistance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.levels.weekly_profile import _build_volume_at_price

VOL_MULT = 2.0
BODY_PCT = 0.70
TOP_K = 3
MIN_SEPARATION_BINS = 4
MIN_HVN_BARS = 10
BIN_SIZE = 0.05
VOL_AVG_WINDOW = 20


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "session_date", "ts_et", "t", "bar_idx", "level", "level_name",
        "approach", "rule", "direction", "entry_ts", "entry_price",
    ])


def _top_k_hvns(
    lows: np.ndarray,
    highs: np.ndarray,
    vols: np.ndarray,
    k: int,
    bin_size: float,
    min_separation: int,
) -> list[float]:
    """Top-k HVN prices from cumulative bar arrays."""
    if len(lows) == 0:
        return []
    centers, volumes, _, _ = _build_volume_at_price(lows, highs, vols, bin_size)
    if volumes.size == 0 or volumes.sum() == 0:
        return []
    order = np.argsort(-volumes)
    picked: list[int] = []
    for idx in order:
        if volumes[idx] == 0:
            break
        if all(abs(idx - p) >= min_separation for p in picked):
            picked.append(int(idx))
            if len(picked) >= k:
                break
    return sorted([float(centers[i]) for i in picked])


def _session_vwap(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray
) -> float:
    if v.sum() == 0:
        return float(c[-1]) if len(c) > 0 else 0.0
    tp = (h + l + c) / 3
    return float((tp * v).sum() / v.sum())


def _body_fraction(o: float, h: float, l: float, c: float) -> tuple[float, str]:
    """Return (body_fraction_of_range, direction). Direction is 'up' or 'down'."""
    rng = h - l
    if rng <= 0:
        return 0.0, "flat"
    body = abs(c - o)
    direction = "up" if c > o else "down" if c < o else "flat"
    return body / rng, direction


def evaluate_hvn_breakout(session_bars: pd.DataFrame) -> pd.DataFrame:
    """Evaluate HVN breakdown/breakout events on a single session's bars.

    `session_bars` should be RTH-filtered, sorted by t.
    """
    if session_bars.empty:
        return _empty_events_df()

    bars = session_bars.reset_index(drop=True)
    n = len(bars)
    if n < MIN_HVN_BARS + 5:
        return _empty_events_df()

    o = bars["o"].to_numpy()
    h = bars["h"].to_numpy()
    l = bars["l"].to_numpy()
    c = bars["c"].to_numpy()
    v = bars["v"].to_numpy()
    ts_et = bars["ts_et"].to_numpy()
    t = bars["t"].to_numpy()
    session_date = bars.iloc[0]["session_date"]

    rows: list[dict] = []

    for i in range(MIN_HVN_BARS + 5, n):
        # 1. Body and direction
        body_pct, direction = _body_fraction(o[i], h[i], l[i], c[i])
        if body_pct < BODY_PCT or direction == "flat":
            continue

        # 2. Volume spike
        vol_avg = v[max(0, i - VOL_AVG_WINDOW):i].mean()
        if vol_avg <= 0 or v[i] < VOL_MULT * vol_avg:
            continue

        # 3. VWAP check
        vwap = _session_vwap(o[:i + 1], h[:i + 1], l[:i + 1], c[:i + 1], v[:i + 1])
        if direction == "down" and c[i] >= vwap:
            continue
        if direction == "up" and c[i] <= vwap:
            continue

        # 4. Top-K HVNs from cumulative profile up to the PREVIOUS bar
        #    (don't include current bar's volume to avoid fitting the break itself)
        hvns = _top_k_hvns(
            l[:i], h[:i], v[:i],
            k=TOP_K, bin_size=BIN_SIZE, min_separation=MIN_SEPARATION_BINS,
        )
        if not hvns:
            continue

        # 5. Find an HVN the bar is breaking through
        broken_hvn: float | None = None
        if direction == "down":
            # price is dropping; find HVNs above close but <= prev_close
            # (price was at or above HVN prior bar, closed below it this bar)
            prev_c = c[i - 1]
            candidates = [p for p in hvns if prev_c >= p - BIN_SIZE and c[i] < p - BIN_SIZE]
            if candidates:
                broken_hvn = max(candidates)  # highest HVN broken
        else:  # up
            prev_c = c[i - 1]
            candidates = [p for p in hvns if prev_c <= p + BIN_SIZE and c[i] > p + BIN_SIZE]
            if candidates:
                broken_hvn = min(candidates)

        if broken_hvn is None:
            continue

        rows.append({
            "session_date": session_date,
            "ts_et": ts_et[i],
            "t": int(t[i]),
            "bar_idx": i,
            "level": float(broken_hvn),
            "level_name": "hvn_breakout",
            "approach": "from_above" if direction == "down" else "from_below",
            "rule": "S5",
            "direction": "short" if direction == "down" else "long",
            "entry_ts": ts_et[i],
            "entry_price": float(c[i]),
        })

    return pd.DataFrame(rows) if rows else _empty_events_df()
