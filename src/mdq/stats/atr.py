"""ATR (Average True Range) computation from 1-min bars.

Used to size target/stop as a fraction of realized volatility rather than
fixed dollar amounts. This lets the same rule work across different price
regimes without re-tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(bars: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Return a per-bar ATR array (same length as bars).

    Uses the standard Wilder-style true range:
        TR_i = max(h - l, |h - prev_close|, |l - prev_close|)
    Then ATR = simple moving average of TR over `window` bars. We use SMA
    (not exponential) because it's easier to reason about and the difference
    is tiny over 20 bars.

    `bars` must have columns h, l, c and be sorted by time.
    """
    if bars.empty:
        return np.array([])

    h = bars["h"].to_numpy()
    l = bars["l"].to_numpy()
    c = bars["c"].to_numpy()
    n = len(h)

    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        prev_c = c[i - 1]
        tr[i] = max(h[i] - l[i], abs(h[i] - prev_c), abs(l[i] - prev_c))

    atr = np.full(n, np.nan)
    for i in range(window - 1, n):
        atr[i] = tr[i - window + 1:i + 1].mean()
    return atr
