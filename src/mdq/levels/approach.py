"""Approach magnitude: how far did price travel *into* the level?

Randy's playbook keys its A/B/C grade partly on "drop size" — the distance
price covered in the fade direction before reaching the level. A $4+ drop into
a level is "exhaustion" (A-grade), $1–2 is "normal" (B-grade), a grind is
skip.

We compute two flavors:
    approach_from_open   = |level - session_open| in the fade direction
    approach_swing_30m   = max travel from prior 30-min swing extreme to level

Both are signed so that positive = "price came into the level from the
expected direction" (resistance touch from below: positive = rally size;
support touch from above: positive = drop size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_approach_magnitude(
    bars: pd.DataFrame,
    reactions: pd.DataFrame,
    lookback_minutes: int = 30,
) -> pd.DataFrame:
    """Append approach_from_open and approach_swing columns to a reactions frame.

    The reactions/touches frame must contain session_date, t, level, open_price,
    approach. `bars` must be the full 1-min bars frame with ts_et and session_date.
    """
    if reactions.empty:
        out = reactions.copy()
        out["approach_from_open"] = np.nan
        out["approach_swing"] = np.nan
        return out

    bars_sorted = bars.sort_values(["session_date", "t"]).reset_index(drop=True)

    # Pre-compute per-session arrays for speed
    session_arrays: dict = {}
    for sd, g in bars_sorted.groupby("session_date", sort=True):
        session_arrays[sd] = {
            "t": g["t"].to_numpy(),
            "h": g["h"].to_numpy(),
            "l": g["l"].to_numpy(),
            "o": float(g.iloc[0]["o"]),
        }

    n = len(reactions)
    from_open = np.full(n, np.nan)
    swing = np.full(n, np.nan)

    t_vals = reactions["t"].to_numpy()
    sessions = reactions["session_date"].to_numpy()
    levels = reactions["level"].to_numpy()
    approaches = reactions["approach"].to_numpy()

    # 1 minute = 60_000 ms
    lookback_ms = lookback_minutes * 60_000

    for i in range(n):
        arr = session_arrays.get(sessions[i])
        if arr is None:
            continue
        ts_arr = arr["t"]
        bi = int(np.searchsorted(ts_arr, t_vals[i]))
        if bi >= len(ts_arr) or ts_arr[bi] != t_vals[i]:
            continue

        open_px = arr["o"]
        lvl = float(levels[i])
        approach = approaches[i]

        if approach == "from_below":
            # Resistance: price rallied UP to the level
            from_open[i] = lvl - open_px  # positive if rallied from below open
            # swing: max of (level - min low) over prior window
            lookback_start_ms = ts_arr[bi] - lookback_ms
            j = int(np.searchsorted(ts_arr, lookback_start_ms))
            prior_lows = arr["l"][j:bi + 1]
            if prior_lows.size > 0:
                swing[i] = lvl - float(prior_lows.min())
        elif approach == "from_above":
            # Support: price fell DOWN to the level
            from_open[i] = open_px - lvl  # positive if dropped from above open
            lookback_start_ms = ts_arr[bi] - lookback_ms
            j = int(np.searchsorted(ts_arr, lookback_start_ms))
            prior_highs = arr["h"][j:bi + 1]
            if prior_highs.size > 0:
                swing[i] = float(prior_highs.max()) - lvl
        # else: "at" - leave nan

    out = reactions.copy()
    out["approach_from_open"] = from_open
    out["approach_swing"] = swing
    return out


def classify_drop_size(
    approach_magnitude: pd.Series | np.ndarray,
    bins: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, float("inf")),
    labels: tuple[str, ...] = ("grind", "small", "normal", "exhaustion"),
) -> pd.Series:
    """Bucket approach magnitudes into Randy-style categories.

    Defaults map: [0,1) grind, [1,2) small, [2,4) normal, [4,inf) exhaustion.
    """
    vals = pd.Series(approach_magnitude).astype(float)
    return pd.cut(vals, bins=list(bins), labels=list(labels), right=False)
