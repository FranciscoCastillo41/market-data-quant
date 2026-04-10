"""Weekly volume profile: aggregate RTH bars across Mon-Fri into a single
profile, plus high-volume-node (HVN) and low-volume-node (LVN) detection.

Unlike the daily volume_profile used for QQQ, this aggregates the full
trading week and exposes multiple levels per week:

    poc, vah, val    (standard)
    h, l, c          (week high/low/close)
    hvn_1, hvn_2     (top 2 high-volume nodes excluding POC)
    lvn_1, lvn_2     (top 2 low-volume nodes — volume gaps)

An HVN is a local maximum in the volume-at-price distribution — a price
shelf where the market traded a lot of contracts. An LVN is a local
minimum — a "volume gap" where price moved through quickly. Both are
structural levels that institutional traders watch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mdq.data.calendar import filter_rth


@dataclass(frozen=True)
class WeeklyProfile:
    """Volume profile summary for one trading week."""

    week_start: object  # Monday of the week (date)
    week_end: object    # Friday of the week (date)
    poc: float
    vah: float
    val: float
    high: float
    low: float
    close: float
    hvn_1: float | None
    hvn_2: float | None
    lvn_1: float | None
    lvn_2: float | None
    total_volume: float


def _build_volume_at_price(
    lows: np.ndarray,
    highs: np.ndarray,
    vols: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Distribute bar volume uniformly across price bins.

    Returns (bin_centers, volumes, lo_bin, hi_bin).
    """
    lo = float(lows.min())
    hi = float(highs.max())
    lo_bin = np.floor(lo / bin_size) * bin_size
    hi_bin = np.ceil(hi / bin_size) * bin_size
    n_bins = int(round((hi_bin - lo_bin) / bin_size)) + 1
    if n_bins <= 0:
        return np.array([]), np.array([]), lo_bin, hi_bin

    bin_centers = lo_bin + (np.arange(n_bins) + 0.5) * bin_size
    volumes = np.zeros(n_bins)

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

    return bin_centers, volumes, lo_bin, hi_bin


def _value_area(
    volumes: np.ndarray,
    poc_idx: int,
    pct: float = 0.70,
) -> tuple[int, int]:
    """Return (val_idx, vah_idx) of the value area around poc."""
    total = volumes.sum()
    target = total * pct
    va_lo = poc_idx
    va_hi = poc_idx
    va_vol = float(volumes[poc_idx])

    while va_vol < target:
        can_down = va_lo > 0
        can_up = va_hi < len(volumes) - 1
        if not can_down and not can_up:
            break
        if not can_down:
            va_hi += 1
            va_vol += volumes[va_hi]
        elif not can_up:
            va_lo -= 1
            va_vol += volumes[va_lo]
        else:
            up_vol = volumes[va_hi + 1]
            down_vol = volumes[va_lo - 1]
            if up_vol >= down_vol:
                va_hi += 1
                va_vol += up_vol
            else:
                va_lo -= 1
                va_vol += down_vol
    return va_lo, va_hi


def _find_hvns(
    volumes: np.ndarray,
    bin_centers: np.ndarray,
    poc_idx: int,
    min_separation_bins: int = 4,
    top_n: int = 2,
) -> list[float]:
    """Find top N local maxima of the volume distribution, excluding the POC.

    A local max is a bin that is strictly greater than its immediate neighbors
    AND separated from other picked peaks by `min_separation_bins`.
    """
    n = len(volumes)
    if n < 3:
        return []

    candidates: list[tuple[float, int]] = []
    for i in range(1, n - 1):
        if i == poc_idx:
            continue
        if volumes[i] > volumes[i - 1] and volumes[i] > volumes[i + 1]:
            candidates.append((float(volumes[i]), i))

    candidates.sort(reverse=True)
    picked: list[int] = []
    for vol, i in candidates:
        if all(abs(i - p) >= min_separation_bins for p in picked):
            picked.append(i)
            if len(picked) >= top_n:
                break

    return [float(bin_centers[i]) for i in picked]


def _find_lvns(
    volumes: np.ndarray,
    bin_centers: np.ndarray,
    val_idx: int,
    vah_idx: int,
    min_separation_bins: int = 4,
    top_n: int = 2,
) -> list[float]:
    """Find top N local minima (volume gaps) inside the trading range.

    We only look *inside* [val_idx, vah_idx] because LVNs outside the value
    area are just tails (not interesting structurally).
    """
    if vah_idx - val_idx < 3:
        return []

    candidates: list[tuple[float, int]] = []
    for i in range(val_idx + 1, vah_idx):
        if volumes[i] < volumes[i - 1] and volumes[i] < volumes[i + 1]:
            candidates.append((float(volumes[i]), i))

    # Ascending — lowest volume first
    candidates.sort()
    picked: list[int] = []
    for vol, i in candidates:
        if all(abs(i - p) >= min_separation_bins for p in picked):
            picked.append(i)
            if len(picked) >= top_n:
                break

    return [float(bin_centers[i]) for i in picked]


def compute_weekly_profile(
    week_bars: pd.DataFrame,
    bin_size: float = 0.10,
    value_area_pct: float = 0.70,
) -> WeeklyProfile | None:
    """Compute a WeeklyProfile from a full week of RTH bars."""
    if week_bars.empty:
        return None

    lows = week_bars["l"].to_numpy()
    highs = week_bars["h"].to_numpy()
    vols = week_bars["v"].to_numpy()

    bin_centers, volumes, lo_bin, hi_bin = _build_volume_at_price(
        lows, highs, vols, bin_size
    )
    if volumes.size == 0 or volumes.sum() == 0:
        return None

    poc_idx = int(np.argmax(volumes))
    poc = float(bin_centers[poc_idx])

    val_idx, vah_idx = _value_area(volumes, poc_idx, value_area_pct)
    val = float(bin_centers[val_idx])
    vah = float(bin_centers[vah_idx])

    hvns = _find_hvns(volumes, bin_centers, poc_idx, min_separation_bins=4, top_n=2)
    lvns = _find_lvns(volumes, bin_centers, val_idx, vah_idx, min_separation_bins=4, top_n=2)

    session_dates = sorted(pd.to_datetime(week_bars["session_date"]).dt.date.unique())

    return WeeklyProfile(
        week_start=session_dates[0],
        week_end=session_dates[-1],
        poc=round(poc, 2),
        vah=round(vah, 2),
        val=round(val, 2),
        high=float(highs.max()),
        low=float(lows.min()),
        close=float(week_bars.iloc[-1]["c"]),
        hvn_1=round(hvns[0], 2) if len(hvns) > 0 else None,
        hvn_2=round(hvns[1], 2) if len(hvns) > 1 else None,
        lvn_1=round(lvns[0], 2) if len(lvns) > 0 else None,
        lvn_2=round(lvns[1], 2) if len(lvns) > 1 else None,
        total_volume=float(volumes.sum()),
    )


def compute_all_weekly_profiles(
    bars: pd.DataFrame,
    bin_size: float = 0.10,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """Compute a weekly profile for every trading week in the bar frame.

    Groups by ISO week (year + week number), filters to RTH, computes profile.
    Returns a DataFrame with one row per week.
    """
    rth = filter_rth(bars)
    if rth.empty:
        return pd.DataFrame()

    rth = rth.copy()
    rth["iso_year"] = pd.to_datetime(rth["session_date"]).dt.isocalendar().year
    rth["iso_week"] = pd.to_datetime(rth["session_date"]).dt.isocalendar().week

    rows: list[dict] = []
    for (y, w), week_bars in rth.groupby(["iso_year", "iso_week"], sort=True):
        prof = compute_weekly_profile(week_bars, bin_size=bin_size, value_area_pct=value_area_pct)
        if prof is None:
            continue
        rows.append({
            "iso_year": int(y),
            "iso_week": int(w),
            "week_start": prof.week_start,
            "week_end": prof.week_end,
            "poc": prof.poc,
            "vah": prof.vah,
            "val": prof.val,
            "high": prof.high,
            "low": prof.low,
            "close": prof.close,
            "hvn_1": prof.hvn_1,
            "hvn_2": prof.hvn_2,
            "lvn_1": prof.lvn_1,
            "lvn_2": prof.lvn_2,
            "total_volume": prof.total_volume,
        })

    return pd.DataFrame(rows)
