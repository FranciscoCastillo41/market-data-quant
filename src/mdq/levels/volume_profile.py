"""Volume profile levels: POC, VAH, VAL from prior-session RTH bars.

Volume profile from 1-min bar data is an approximation — we don't have
tick-level trades, so we distribute each bar's volume uniformly across
its [low, high] range in fixed price bins. This is the standard approach
when all you have is bar data.

For each RTH session (09:30-16:00 ET) we compute:
    - POC: the price bin with the highest volume
    - VAH: top of the 70%-value-area
    - VAL: bottom of the 70%-value-area
    - high/low/close of the session (free bonus levels)

These get used as pre-known levels on the NEXT session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mdq.data.calendar import filter_rth
from mdq.levels.touches import TouchConfig


@dataclass(frozen=True)
class SessionProfile:
    """Volume profile summary for one RTH session."""

    session_date: object  # date
    poc: float
    vah: float
    val: float
    high: float
    low: float
    close: float
    total_volume: float


def compute_session_profile(
    session_bars: pd.DataFrame,
    bin_size: float = 0.05,
    value_area_pct: float = 0.70,
) -> SessionProfile | None:
    """Compute POC / VAH / VAL / OHLC for one session's bars.

    `session_bars` must be a single session's 1-min RTH bars with columns
    h, l, v, c, session_date. Returns None on empty input.
    """
    if session_bars.empty:
        return None

    lows = session_bars["l"].to_numpy()
    highs = session_bars["h"].to_numpy()
    vols = session_bars["v"].to_numpy()

    lo = float(lows.min())
    hi = float(highs.max())
    lo_bin = np.floor(lo / bin_size) * bin_size
    hi_bin = np.ceil(hi / bin_size) * bin_size
    n_bins = int(round((hi_bin - lo_bin) / bin_size)) + 1
    if n_bins <= 0:
        return None

    bin_centers = lo_bin + (np.arange(n_bins) + 0.5) * bin_size
    volumes = np.zeros(n_bins)

    # Distribute each bar's volume uniformly across the bins it spans
    for i in range(len(lows)):
        bar_lo = lows[i]
        bar_hi = highs[i]
        bar_v = vols[i]
        if bar_v <= 0:
            continue
        lo_idx = int(np.floor((bar_lo - lo_bin) / bin_size))
        hi_idx = int(np.floor((bar_hi - lo_bin) / bin_size))
        lo_idx = max(0, lo_idx)
        hi_idx = min(n_bins - 1, hi_idx)
        n_spanned = hi_idx - lo_idx + 1
        if n_spanned <= 0:
            continue
        volumes[lo_idx:hi_idx + 1] += bar_v / n_spanned

    total_vol = float(volumes.sum())
    if total_vol <= 0:
        return None

    poc_idx = int(np.argmax(volumes))
    poc = float(bin_centers[poc_idx])

    # Single-step value area expansion from POC outward toward whichever side
    # has the higher bin volume at the expansion boundary
    target_vol = total_vol * value_area_pct
    va_low_idx = poc_idx
    va_high_idx = poc_idx
    va_vol = float(volumes[poc_idx])

    while va_vol < target_vol:
        can_down = va_low_idx > 0
        can_up = va_high_idx < n_bins - 1
        if not can_down and not can_up:
            break
        if not can_down:
            va_high_idx += 1
            va_vol += float(volumes[va_high_idx])
        elif not can_up:
            va_low_idx -= 1
            va_vol += float(volumes[va_low_idx])
        else:
            up_vol = float(volumes[va_high_idx + 1])
            down_vol = float(volumes[va_low_idx - 1])
            if up_vol >= down_vol:
                va_high_idx += 1
                va_vol += up_vol
            else:
                va_low_idx -= 1
                va_vol += down_vol

    vah = float(bin_centers[va_high_idx])
    val = float(bin_centers[va_low_idx])

    return SessionProfile(
        session_date=session_bars.iloc[0]["session_date"],
        poc=round(poc, 2),
        vah=round(vah, 2),
        val=round(val, 2),
        high=float(highs.max()),
        low=float(lows.min()),
        close=float(session_bars.iloc[-1]["c"]),
        total_volume=total_vol,
    )


def compute_all_profiles(
    bars: pd.DataFrame,
    bin_size: float = 0.05,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """Compute per-session volume profiles for a multi-session bar frame.

    `bars` may include extended hours; we filter to RTH internally.
    """
    rth = filter_rth(bars)
    rows: list[dict] = []
    for _, sess in rth.groupby("session_date", sort=True):
        p = compute_session_profile(sess, bin_size=bin_size, value_area_pct=value_area_pct)
        if p is not None:
            rows.append({
                "session_date": p.session_date,
                "poc": p.poc,
                "vah": p.vah,
                "val": p.val,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "total_volume": p.total_volume,
            })
    return pd.DataFrame(rows)


def _empty_vp_touches_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "session_date", "ts_et", "t", "bar_idx", "level", "level_name",
            "tier", "approach", "touch_num", "open_price", "entry_close",
        ]
    )


def detect_vp_touches_session(
    session_bars: pd.DataFrame,
    named_levels: list[tuple[str, float]],
    cfg: TouchConfig = TouchConfig(),
) -> pd.DataFrame:
    """Detect touches of a custom set of (name, price) levels in one session."""
    if session_bars.empty or not named_levels:
        return _empty_vp_touches_frame()

    bars = session_bars.reset_index(drop=True)
    open_price = float(bars.iloc[0]["o"])
    session_date = bars.iloc[0]["session_date"]

    in_play = [
        (name, p) for (name, p) in named_levels
        if (p is not None) and (not pd.isna(p)) and abs(p - open_price) <= cfg.in_play_radius
    ]
    if not in_play:
        return _empty_vp_touches_frame()

    names = np.array([n for n, _ in in_play])
    prices = np.array([float(p) for _, p in in_play])

    highs = bars["h"].to_numpy()
    lows = bars["l"].to_numpy()
    closes = bars["c"].to_numpy()

    tol = cfg.tolerance
    touch_mat = (
        (lows[:, None] <= prices[None, :] + tol)
        & (highs[:, None] >= prices[None, :] - tol)
    )

    if not touch_mat.any():
        return _empty_vp_touches_frame()

    touch_ord = np.where(touch_mat, touch_mat.cumsum(axis=0), 0)
    bar_idx_arr, lvl_idx_arr = np.where(touch_mat)

    approaches: list[str] = []
    for bi, li in zip(bar_idx_arr, lvl_idx_arr):
        lvl = prices[li]
        ref = open_price if bi == 0 else closes[bi - 1]
        if ref < lvl:
            approaches.append("from_below")
        elif ref > lvl:
            approaches.append("from_above")
        else:
            approaches.append("at")

    return pd.DataFrame({
        "session_date": session_date,
        "ts_et": bars["ts_et"].to_numpy()[bar_idx_arr],
        "t": bars["t"].to_numpy()[bar_idx_arr],
        "bar_idx": bar_idx_arr.astype(int),
        "level": prices[lvl_idx_arr],
        "level_name": names[lvl_idx_arr],
        "tier": 1,
        "approach": approaches,
        "touch_num": touch_ord[bar_idx_arr, lvl_idx_arr].astype(int),
        "open_price": open_price,
        "entry_close": closes[bar_idx_arr],
    })


def detect_vp_touches(
    bars_win: pd.DataFrame,
    profiles: pd.DataFrame,
    cfg: TouchConfig = TouchConfig(),
) -> pd.DataFrame:
    """Detect touches for all sessions, using the PRIOR session's profile.

    Handles weekends / holidays by taking the most recent prior session that
    has a profile.
    """
    if bars_win.empty or profiles.empty:
        return _empty_vp_touches_frame()

    profiles_by_date = {
        row["session_date"]: row for _, row in profiles.iterrows()
    }
    sorted_profile_dates = sorted(profiles_by_date.keys())

    session_dates = sorted(bars_win["session_date"].unique())
    out_frames: list[pd.DataFrame] = []

    for sd in session_dates:
        # Find most recent prior profile date strictly before sd
        prior_date = None
        for pd_ in reversed(sorted_profile_dates):
            if pd_ < sd:
                prior_date = pd_
                break
        if prior_date is None:
            continue
        prior = profiles_by_date[prior_date]

        named = [
            ("poc", prior["poc"]),
            ("vah", prior["vah"]),
            ("val", prior["val"]),
            ("prior_high", prior["high"]),
            ("prior_low", prior["low"]),
            ("prior_close", prior["close"]),
        ]

        sess_bars = bars_win[bars_win["session_date"] == sd]
        touches = detect_vp_touches_session(sess_bars, named, cfg)
        if not touches.empty:
            out_frames.append(touches)

    if not out_frames:
        return _empty_vp_touches_frame()
    return pd.concat(out_frames, ignore_index=True)


def add_confluence_flag(
    touches: pd.DataFrame,
    whole_dollar_step: float,
    confluence_radius: float = 0.50,
) -> pd.DataFrame:
    """Flag VP touches that sit within `confluence_radius` of a whole-dollar.

    Returns a copy with a new 'confluence' boolean column and a
    'nearest_whole' column.
    """
    out = touches.copy()
    if out.empty:
        out["confluence"] = False
        out["nearest_whole"] = np.nan
        return out

    lvl = out["level"].to_numpy()
    nearest = np.round(lvl / whole_dollar_step) * whole_dollar_step
    dist = np.abs(lvl - nearest)
    out["nearest_whole"] = nearest
    out["confluence"] = dist <= confluence_radius
    return out
