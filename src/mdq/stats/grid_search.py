"""Grid search target / stop / direction / horizon for first-passage expectancy.

Given a touches frame and the underlying bars, re-compute first-passage
outcomes for arbitrary (target, stop, horizon, direction) combinations.

`direction`:
    'fade'     — trade back away from the level (Randy's side)
    'momentum' — trade through the level (breakout side)

Vectorized by pre-extracting per-session high/low arrays once, then iterating
touches. For ~1k touches × ~100 param combos this is milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GridSpec:
    targets: tuple[float, ...] = (0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
    stops:   tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.00)
    horizons: tuple[int, ...] = (15,)
    directions: tuple[str, ...] = ("fade", "momentum")


def _extract_forward_arrays(
    bars: pd.DataFrame,
    touches: pd.DataFrame,
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-touch forward high / low arrays and entry prices, approach sign.

    Shape (n_touches, max_horizon). NaN-filled where forward bars run short.
    approach_sign: +1 for from_below (resistance, fade=short), -1 for from_above.
    """
    n = len(touches)
    fwd_h = np.full((n, max_horizon), np.nan)
    fwd_l = np.full((n, max_horizon), np.nan)
    entry = np.full(n, np.nan)
    sign = np.zeros(n, dtype=int)  # +1 from_below, -1 from_above, 0 skip

    # Pre-extract per-session arrays
    sess_arr: dict = {}
    for sd, g in bars.groupby("session_date", sort=True):
        sess_arr[sd] = {
            "t": g["t"].to_numpy(),
            "h": g["h"].to_numpy(),
            "l": g["l"].to_numpy(),
            "c": g["c"].to_numpy(),
        }

    t_vals = touches["t"].to_numpy()
    sessions = touches["session_date"].to_numpy()
    approaches = touches["approach"].to_numpy()

    for i in range(n):
        arr = sess_arr.get(sessions[i])
        if arr is None:
            continue
        ts = arr["t"]
        bi = int(np.searchsorted(ts, t_vals[i]))
        if bi >= len(ts) or ts[bi] != t_vals[i]:
            continue
        start = bi + 1
        end = min(start + max_horizon, len(ts))
        k = end - start
        if k <= 0:
            continue
        fwd_h[i, :k] = arr["h"][start:end]
        fwd_l[i, :k] = arr["l"][start:end]
        entry[i] = arr["c"][bi]
        if approaches[i] == "from_below":
            sign[i] = 1
        elif approaches[i] == "from_above":
            sign[i] = -1

    return fwd_h, fwd_l, entry, sign


def _first_passage(
    fwd_h: np.ndarray,
    fwd_l: np.ndarray,
    entry: np.ndarray,
    sign: np.ndarray,  # +1 from_below, -1 from_above
    target: float,
    stop: float,
    horizon: int,
    direction: str,
) -> np.ndarray:
    """Return outcome code for each touch for one (t,s,h,d) combination.

    Outcome codes:
        +1 = target hit first
        -1 = stop hit first
         0 = timeout
        NaN = invalid (no forward bars or 'at' approach)
    """
    n = fwd_h.shape[0]
    out = np.full(n, np.nan)

    for i in range(n):
        if sign[i] == 0 or np.isnan(entry[i]):
            continue

        # Determine trade direction:
        # fade short for from_below, fade long for from_above
        # momentum long for from_below, momentum short for from_above
        is_short_trade = (
            (direction == "fade" and sign[i] == 1)
            or (direction == "momentum" and sign[i] == -1)
        )

        if is_short_trade:
            target_px = entry[i] - target
            stop_px = entry[i] + stop
            hit_t = fwd_l[i, :horizon] <= target_px
            hit_s = fwd_h[i, :horizon] >= stop_px
        else:
            target_px = entry[i] + target
            stop_px = entry[i] - stop
            hit_t = fwd_h[i, :horizon] >= target_px
            hit_s = fwd_l[i, :horizon] <= stop_px

        # Treat nan-forward bars as no-hit
        hit_t = np.where(np.isnan(fwd_l[i, :horizon]) | np.isnan(fwd_h[i, :horizon]), False, hit_t)
        hit_s = np.where(np.isnan(fwd_l[i, :horizon]) | np.isnan(fwd_h[i, :horizon]), False, hit_s)

        idx_t = int(np.argmax(hit_t)) if hit_t.any() else -1
        idx_s = int(np.argmax(hit_s)) if hit_s.any() else -1

        if idx_t == -1 and idx_s == -1:
            out[i] = 0  # timeout
        elif idx_t == -1:
            out[i] = -1
        elif idx_s == -1:
            out[i] = 1
        else:
            # Conservative: if same bar, stop wins
            out[i] = 1 if idx_t < idx_s else -1

    return out


def grid_search(
    bars: pd.DataFrame,
    touches: pd.DataFrame,
    spec: GridSpec = GridSpec(),
) -> pd.DataFrame:
    """Run the full grid search. Returns a DataFrame of summary rows."""
    if touches.empty:
        return pd.DataFrame()

    max_h = max(spec.horizons)
    fwd_h, fwd_l, entry, sign = _extract_forward_arrays(bars, touches, max_h)

    rows: list[dict] = []
    for direction in spec.directions:
        for h in spec.horizons:
            for t_move in spec.targets:
                for s_move in spec.stops:
                    outcomes = _first_passage(
                        fwd_h, fwd_l, entry, sign, t_move, s_move, h, direction
                    )
                    valid = ~np.isnan(outcomes)
                    n_valid = int(valid.sum())
                    if n_valid == 0:
                        continue
                    n_target = int((outcomes == 1).sum())
                    n_stop = int((outcomes == -1).sum())
                    n_timeout = int((outcomes == 0).sum())
                    hit = n_target / n_valid
                    stp = n_stop / n_valid
                    tmo = n_timeout / n_valid
                    exp = (n_target * t_move - n_stop * s_move) / n_valid
                    rows.append({
                        "direction": direction,
                        "horizon": h,
                        "target": t_move,
                        "stop": s_move,
                        "n": n_valid,
                        "hit": hit,
                        "stop_rate": stp,
                        "timeout": tmo,
                        "expectancy": exp,
                        "r_multiple": exp / s_move,
                    })
    return pd.DataFrame(rows)
