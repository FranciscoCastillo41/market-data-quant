"""Touch detection: when does price reach a level?

Given a DataFrame of 1-min bars for a single session, and a grid of candidate
levels, find each bar/level pair where the bar's high-low range intersects
the level's tolerance band, annotate with touch ordinal (1st / 2nd / 3rd ...)
per level per session, approach direction, and session metadata.

Vectorised with numpy where possible to stay fast on 800+ sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mdq.levels.psychological import LevelGrid


@dataclass(frozen=True)
class TouchConfig:
    """Parameters for touch detection."""

    tolerance: float = 0.05
    """Half-width of the touch band around each level, in dollars."""

    in_play_radius: float = 10.0
    """A level is 'in play' for a session if it is within this many dollars
    of the session's open price."""

    approach_lookback: int = 1
    """Number of prior bars used to determine approach direction."""


def detect_touches_session(
    session_bars: pd.DataFrame,
    grid: LevelGrid,
    cfg: TouchConfig = TouchConfig(),
) -> pd.DataFrame:
    """Detect touches for one session of 1-min bars.

    Expects session_bars sorted by `t` ascending, already filtered to the
    time window of interest (e.g. RTH or morning window).

    Returns a DataFrame with one row per (bar, level) touch event, columns:
        session_date, ts_et, t, bar_idx, level, tier, approach,
        touch_num, open_price, entry_close
    """
    if session_bars.empty:
        return _empty_touches_frame()

    bars = session_bars.reset_index(drop=True)
    open_price = float(bars.iloc[0]["o"])
    session_date = bars.iloc[0]["session_date"]

    # Restrict grid to levels within in_play_radius of the open
    in_play = grid.in_range(
        low=open_price - cfg.in_play_radius,
        high=open_price + cfg.in_play_radius,
    )
    if in_play.prices.size == 0:
        return _empty_touches_frame()

    highs = bars["h"].to_numpy()
    lows = bars["l"].to_numpy()
    closes = bars["c"].to_numpy()
    n_bars = len(bars)

    tol = cfg.tolerance
    levels = in_play.prices  # (n_lvl,)
    tiers = in_play.tiers
    n_lvl = len(levels)

    # Broadcast: touch matrix shape (n_bars, n_lvl)
    # A bar touches level L iff low <= L + tol and high >= L - tol.
    touch_mat = (
        (lows[:, None] <= levels[None, :] + tol)
        & (highs[:, None] >= levels[None, :] - tol)
    )

    if not touch_mat.any():
        return _empty_touches_frame()

    # Compute touch ordinals (1st/2nd/3rd) per level via cumulative sum per column
    touch_ord = np.where(touch_mat, touch_mat.cumsum(axis=0), 0)

    # Extract (bar_idx, lvl_idx) pairs where touch happened
    bar_idx_arr, lvl_idx_arr = np.where(touch_mat)

    if bar_idx_arr.size == 0:
        return _empty_touches_frame()

    # Approach direction: use prior bar close (or open if first bar)
    approaches: list[str] = []
    for bi, li in zip(bar_idx_arr, lvl_idx_arr):
        lvl = levels[li]
        if bi == 0:
            ref = open_price
        else:
            ref = closes[bi - cfg.approach_lookback] if bi - cfg.approach_lookback >= 0 else open_price
        if ref < lvl:
            approaches.append("from_below")  # resistance
        elif ref > lvl:
            approaches.append("from_above")  # support
        else:
            approaches.append("at")  # already at level

    out = pd.DataFrame({
        "session_date": session_date,
        "ts_et": bars["ts_et"].to_numpy()[bar_idx_arr],
        "t": bars["t"].to_numpy()[bar_idx_arr],
        "bar_idx": bar_idx_arr.astype(int),
        "level": levels[lvl_idx_arr],
        "tier": tiers[lvl_idx_arr],
        "approach": approaches,
        "touch_num": touch_ord[bar_idx_arr, lvl_idx_arr].astype(int),
        "open_price": open_price,
        "entry_close": closes[bar_idx_arr],
    })

    return out


def detect_touches(
    bars: pd.DataFrame,
    grid: LevelGrid,
    cfg: TouchConfig = TouchConfig(),
) -> pd.DataFrame:
    """Detect touches across all sessions in a multi-session bars frame.

    `bars` must have columns: t, o, h, l, c, v, ts_et, session_date.
    """
    frames = []
    for _, session in bars.groupby("session_date", sort=True):
        frames.append(detect_touches_session(session, grid, cfg))
    if not frames:
        return _empty_touches_frame()
    return pd.concat(frames, ignore_index=True)


def _empty_touches_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "session_date", "ts_et", "t", "bar_idx", "level", "tier",
            "approach", "touch_num", "open_price", "entry_close",
        ]
    )
