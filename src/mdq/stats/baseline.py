"""Baseline / null distribution for level-reaction tests.

The question: *is the forward-return distribution at level touches different
from the forward-return distribution at randomly selected bars in the same
time window?*

Approach: take the same bars used for touch detection, and for every eligible
bar compute the same forward metrics as a "touch" would see. Treat the bar's
close as entry, and measure forward MFE/MAE in BOTH directions over the
horizons. This gives the unconditional distribution of forward moves in the
window.

Then the level-touch distribution can be compared against it (mean difference,
quantile shifts, KS statistic, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mdq.stats.reactions import ReactionConfig


@dataclass(frozen=True)
class BaselineResult:
    """Container for baseline forward-move distribution stats."""

    n: int
    horizons: tuple[int, ...]
    mfe_up: dict[int, np.ndarray]
    mfe_dn: dict[int, np.ndarray]
    entries: np.ndarray


def compute_baseline(
    bars: pd.DataFrame,
    cfg: ReactionConfig = ReactionConfig(),
    subsample: int | None = None,
    rng_seed: int = 42,
) -> BaselineResult:
    """Compute per-bar forward max-up and max-down moves over each horizon.

    Args:
        bars: multi-session 1-min bars frame (already filtered to the time
              window of interest, e.g. morning window).
        cfg:  ReactionConfig for horizons etc.
        subsample: if given, sample this many bars uniformly at random across
                   all sessions (useful for very large frames).
        rng_seed: RNG seed for reproducible subsampling.
    """
    if bars.empty:
        return BaselineResult(
            n=0, horizons=cfg.horizons, mfe_up={}, mfe_dn={}, entries=np.array([]),
        )

    horizons = cfg.horizons
    max_h = max(horizons)

    session_groups = dict(list(bars.groupby("session_date", sort=True)))
    session_arrays: dict = {}
    for sd, g in session_groups.items():
        session_arrays[sd] = {
            "h": g["h"].to_numpy(),
            "l": g["l"].to_numpy(),
            "c": g["c"].to_numpy(),
        }

    mfe_up = {h: [] for h in horizons}
    mfe_dn = {h: [] for h in horizons}
    entries: list[float] = []

    for sd, arr in session_arrays.items():
        highs = arr["h"]
        lows = arr["l"]
        closes = arr["c"]
        n_bars = len(closes)
        if n_bars < 2:
            continue

        # For each starting bar i, compute rolling max high and min low over
        # [i+1, i+h]. Using pandas rolling on reversed arrays is fine and fast.
        for h in horizons:
            # Build arrays of length n_bars where index i holds max of highs[i+1:i+1+h]
            # Simple implementation: use np.lib.stride_tricks sliding windows where possible.
            if n_bars - 1 < h:
                # Not enough forward bars for this horizon in this session
                continue
            # We'll compute for i in 0 .. n_bars-1-h
            valid_end = n_bars - h  # exclusive
            if valid_end <= 0:
                continue
            # Sliding max/min over windows of size h starting at i+1
            # Use pandas rolling: compute rolling max on highs[1:], then take first (n_bars - h) values
            shifted_h = highs[1:]
            shifted_l = lows[1:]
            roll_max = pd.Series(shifted_h).rolling(window=h, min_periods=1).max().to_numpy()
            roll_min = pd.Series(shifted_l).rolling(window=h, min_periods=1).min().to_numpy()
            # For window size h at position i in shifted_h, the window covers
            # shifted_h[i-h+1:i+1]. We want shifted_h[i:i+h], which corresponds
            # to position i+h-1 in the rolling output. So the forward max for
            # entry bar i is roll_max[i + h - 1] (if i + h - 1 < len(shifted_h)).
            max_i = min(valid_end, len(shifted_h) - h + 1)
            idx = np.arange(max_i) + (h - 1)
            fwd_max = roll_max[idx]
            fwd_min = roll_min[idx]
            entry_closes = closes[:max_i]

            mfe_up[h].extend((fwd_max - entry_closes).tolist())
            mfe_dn[h].extend((entry_closes - fwd_min).tolist())

        # Entries used once per bar at the max-horizon eligibility
        valid_end = n_bars - max_h
        if valid_end > 0:
            entries.extend(closes[:valid_end].tolist())

    result = BaselineResult(
        n=len(entries),
        horizons=horizons,
        mfe_up={h: np.array(v) for h, v in mfe_up.items()},
        mfe_dn={h: np.array(v) for h, v in mfe_dn.items()},
        entries=np.array(entries),
    )

    if subsample is not None and result.n > subsample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(result.n, size=subsample, replace=False)
        result = BaselineResult(
            n=subsample,
            horizons=horizons,
            mfe_up={h: v[idx] for h, v in result.mfe_up.items()},
            mfe_dn={h: v[idx] for h, v in result.mfe_dn.items()},
            entries=result.entries[idx],
        )

    return result


def baseline_summary(
    result: BaselineResult,
    target_move: float = 0.30,
    stop_move: float = 0.25,
) -> pd.DataFrame:
    """Summary table of the baseline distribution.

    Reports, for each horizon, the fraction of random bars where forward
    move in either direction would have hit `target_move` before (naively)
    `stop_move` in the opposite direction. This is the 'random entry'
    reference for the hit-rate of true touches.
    """
    rows = []
    for h in result.horizons:
        up = result.mfe_up[h]
        dn = result.mfe_dn[h]
        if up.size == 0:
            continue
        rows.append({
            "horizon": h,
            "n": up.size,
            "mean_max_up": up.mean(),
            "mean_max_dn": dn.mean(),
            "p_up_ge_target": (up >= target_move).mean(),
            "p_dn_ge_target": (dn >= target_move).mean(),
            "p_up_ge_target_no_stop": ((up >= target_move) & (dn < stop_move)).mean(),
            "p_dn_ge_target_no_stop": ((dn >= target_move) & (up < stop_move)).mean(),
        })
    return pd.DataFrame(rows)
