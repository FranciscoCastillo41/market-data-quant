"""Dynamic (intra-session) volume profile: developing POC and rolling POC.

Developing POC:
    Recomputed at each bar from all RTH bars seen so far today. Starts with
    the first bar after 09:30 ET. Can fluctuate wildly in the first 15 min
    before enough volume accumulates.

Rolling POC:
    Recomputed at each bar from the trailing N bars. Responds faster to
    regime shifts than developing POC. We use N=30 (30 minutes) by default.

Both return a per-bar timeseries of POC/VAH/VAL values so the touch detector
can check "is THIS bar touching the THEN-current POC" without look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mdq.levels.weekly_profile import _build_volume_at_price, _value_area


def _compute_poc(
    lows: np.ndarray,
    highs: np.ndarray,
    vols: np.ndarray,
    bin_size: float = 0.05,
) -> tuple[float, float, float] | None:
    """Compute (poc, vah, val) from bar arrays. Returns None if insufficient data."""
    if len(lows) == 0:
        return None
    centers, volumes, _, _ = _build_volume_at_price(lows, highs, vols, bin_size)
    if volumes.size == 0 or volumes.sum() == 0:
        return None
    poc_idx = int(np.argmax(volumes))
    val_idx, vah_idx = _value_area(volumes, poc_idx, 0.70)
    return float(centers[poc_idx]), float(centers[vah_idx]), float(centers[val_idx])


def compute_developing_poc(
    rth_bars: pd.DataFrame,
    bin_size: float = 0.05,
    min_bars: int = 15,
) -> pd.DataFrame:
    """Return a DataFrame aligned with rth_bars containing developing POC at each bar.

    For bar i, the POC is computed from rth_bars[0..i] inclusive. Before
    min_bars have accumulated, POC is NaN (too noisy to be useful).
    """
    if rth_bars.empty:
        return pd.DataFrame(columns=["t", "dev_poc", "dev_vah", "dev_val"])

    lows = rth_bars["l"].to_numpy()
    highs = rth_bars["h"].to_numpy()
    vols = rth_bars["v"].to_numpy()
    n = len(rth_bars)

    poc_arr = np.full(n, np.nan)
    vah_arr = np.full(n, np.nan)
    val_arr = np.full(n, np.nan)

    for i in range(min_bars - 1, n):
        result = _compute_poc(
            lows[: i + 1], highs[: i + 1], vols[: i + 1], bin_size
        )
        if result is None:
            continue
        poc_arr[i], vah_arr[i], val_arr[i] = result

    return pd.DataFrame({
        "t": rth_bars["t"].to_numpy(),
        "dev_poc": poc_arr,
        "dev_vah": vah_arr,
        "dev_val": val_arr,
    })


def compute_rolling_poc(
    rth_bars: pd.DataFrame,
    window_bars: int = 30,
    bin_size: float = 0.05,
) -> pd.DataFrame:
    """Return a DataFrame aligned with rth_bars containing rolling-window POC.

    For bar i, the POC is computed from rth_bars[max(0, i - window_bars + 1) .. i].
    Before window_bars have accumulated, uses whatever's available (degrades
    gracefully to developing POC at the start of the session).
    """
    if rth_bars.empty:
        return pd.DataFrame(columns=["t", "roll_poc", "roll_vah", "roll_val"])

    lows = rth_bars["l"].to_numpy()
    highs = rth_bars["h"].to_numpy()
    vols = rth_bars["v"].to_numpy()
    n = len(rth_bars)

    poc_arr = np.full(n, np.nan)
    vah_arr = np.full(n, np.nan)
    val_arr = np.full(n, np.nan)

    # Need at least 5 bars for anything meaningful
    min_start = 5
    for i in range(min_start - 1, n):
        start = max(0, i - window_bars + 1)
        result = _compute_poc(
            lows[start : i + 1], highs[start : i + 1], vols[start : i + 1], bin_size
        )
        if result is None:
            continue
        poc_arr[i], vah_arr[i], val_arr[i] = result

    return pd.DataFrame({
        "t": rth_bars["t"].to_numpy(),
        "roll_poc": poc_arr,
        "roll_vah": vah_arr,
        "roll_val": val_arr,
    })


def compute_dynamic_profiles_per_session(
    bars: pd.DataFrame,
    rolling_window: int = 30,
    developing_min_bars: int = 15,
    bin_size: float = 0.05,
) -> pd.DataFrame:
    """For a multi-session bar frame (already RTH-filtered), compute both
    developing and rolling POC timeseries per session.

    Returns a DataFrame with columns:
        session_date, t, dev_poc, dev_vah, dev_val, roll_poc, roll_vah, roll_val
    """
    if bars.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for sd, sess in bars.groupby("session_date", sort=True):
        sess = sess.sort_values("t").reset_index(drop=True)
        dev = compute_developing_poc(sess, bin_size=bin_size, min_bars=developing_min_bars)
        roll = compute_rolling_poc(sess, window_bars=rolling_window, bin_size=bin_size)
        combined = dev.merge(roll, on="t")
        combined["session_date"] = sd
        frames.append(combined)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result[["session_date", "t", "dev_poc", "dev_vah", "dev_val",
                   "roll_poc", "roll_vah", "roll_val"]]
