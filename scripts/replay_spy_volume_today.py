"""Rich volume-context replay of today's SPY session.

Walks through every RTH bar in order and at each minute prints:
  - Current price, session VWAP, distance from VWAP
  - Top 5 volume-at-price bins (intra-session histogram so far)
  - Yesterday's POC/VAH/VAL for reference
  - Volume of current bar vs rolling 20-bar average
  - Bar shape: body vs wicks, rejection direction
  - Which HVNs are "in play" (above/below current price, within $3)
  - Any S3-style rejection candidates fired in the last bar

Narrative format: designed so you can eyeball what happened around 09:59
when you made your manual trade, and see what a smarter multi-HVN bot
would have seen at each step.
"""

from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd

from mdq.data.bars import download_range_async, load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.levels.volume_profile import compute_all_profiles
from mdq.levels.weekly_profile import _build_volume_at_price


def _find_top_k_hvns(
    lows: np.ndarray,
    highs: np.ndarray,
    vols: np.ndarray,
    k: int = 5,
    bin_size: float = 0.05,
    min_separation: int = 4,
) -> list[tuple[float, float]]:
    """Return up to k HVNs as (price, volume) tuples, sorted by volume desc.

    Uses simple peak-picking with minimum separation between peaks so we
    don't return adjacent bins.
    """
    if len(lows) == 0:
        return []
    centers, volumes, _, _ = _build_volume_at_price(lows, highs, vols, bin_size)
    if volumes.size == 0 or volumes.sum() == 0:
        return []

    # Rank bins by volume desc, skip adjacent (within min_separation bins)
    order = np.argsort(-volumes)
    picked_idx: list[int] = []
    for idx in order:
        if volumes[idx] == 0:
            break
        if all(abs(idx - p) >= min_separation for p in picked_idx):
            picked_idx.append(int(idx))
            if len(picked_idx) >= k:
                break

    return [(float(centers[i]), float(volumes[i])) for i in picked_idx]


def _session_vwap(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> float:
    """Session VWAP = sum(typical_price * volume) / sum(volume)."""
    if v.sum() == 0:
        return float(c[-1])
    tp = (h + l + c) / 3
    return float((tp * v).sum() / v.sum())


def _bar_shape(o: float, h: float, l: float, c: float) -> str:
    """Describe bar shape in one short string."""
    rng = h - l
    if rng <= 0:
        return "flat"
    body_hi = max(o, c)
    body_lo = min(o, c)
    upper_wick = (h - body_hi) / rng
    lower_wick = (body_lo - l) / rng
    body_pct = (body_hi - body_lo) / rng
    direction = "▲" if c > o else ("▼" if c < o else "=")
    tag = direction
    if upper_wick >= 0.5:
        tag += " UW"  # upper wick dominant (rejection of highs)
    if lower_wick >= 0.5:
        tag += " LW"  # lower wick dominant (rejection of lows)
    if body_pct >= 0.8:
        tag += " body"  # full body, momentum
    return tag


def main() -> int:
    today = date.today()
    print("=" * 100)
    print(f"SPY intraday volume replay — {today}")
    print("=" * 100)

    print("\nPulling latest SPY bars from Massive...")
    asyncio.run(
        download_range_async(
            ticker="SPY",
            start=today.isoformat(),
            end=today.isoformat(),
            overwrite=True,
        )
    )
    bars = load_bars("SPY", today.isoformat(), today.isoformat())

    # Window: 09:30 to end of available data
    bars_win = filter_window(bars, "09:30", "16:00").reset_index(drop=True)
    if bars_win.empty:
        print("No RTH bars yet.")
        return 1
    print(f"RTH bars: {len(bars_win)}  range: {bars_win['ts_et'].min()} -> {bars_win['ts_et'].max()}")

    # Load yesterday's profile for reference
    prior_profs = pd.read_parquet("data/results/experiment_c/profiles__SPY.parquet")
    prior_profs["session_date"] = pd.to_datetime(prior_profs["session_date"]).dt.date
    prior_row = prior_profs[prior_profs["session_date"] < today].sort_values("session_date").iloc[-1]
    prior_poc = float(prior_row["poc"])
    prior_vah = float(prior_row["vah"])
    prior_val = float(prior_row["val"])
    prior_high = float(prior_row["high"])
    prior_low = float(prior_row["low"])

    print(f"\nYesterday's ({prior_row['session_date']}) reference levels:")
    print(f"  prior_high = ${prior_high:.2f}")
    print(f"  prior_vah  = ${prior_vah:.2f}")
    print(f"  prior_poc  = ${prior_poc:.2f}")
    print(f"  prior_val  = ${prior_val:.2f}")
    print(f"  prior_low  = ${prior_low:.2f}")

    # Walk forward bar by bar
    print("\n" + "=" * 100)
    print("BAR-BY-BAR REPLAY")
    print("=" * 100)
    print(f"{'time':>6}  {'close':>8}  {'vwap':>8}  {'±vwap':>7}  "
          f"{'barV/avg':>9}  {'shape':>9}  top-3 HVNs (price × volume)")
    print("-" * 100)

    lows = bars_win["l"].to_numpy()
    highs = bars_win["h"].to_numpy()
    vols = bars_win["v"].to_numpy()
    opens = bars_win["o"].to_numpy()
    closes = bars_win["c"].to_numpy()
    times = bars_win["ts_et"].to_numpy()

    for i in range(len(bars_win)):
        # Cumulative profile through bar i
        hvns = _find_top_k_hvns(
            lows[: i + 1], highs[: i + 1], vols[: i + 1],
            k=3, bin_size=0.05, min_separation=4,
        )
        vwap = _session_vwap(
            opens[: i + 1], highs[: i + 1], lows[: i + 1], closes[: i + 1], vols[: i + 1]
        )

        # Volume vs 20-bar rolling average
        if i >= 20:
            vol_avg = vols[i - 20:i].mean()
            vol_ratio = vols[i] / vol_avg if vol_avg > 0 else float("nan")
        else:
            vol_ratio = float("nan")

        shape = _bar_shape(opens[i], highs[i], lows[i], closes[i])

        # Format HVNs
        hvn_str = "  ".join(
            f"${p:.2f}({v/1e3:.0f}k)" for p, v in hvns
        )

        ts_str = pd.Timestamp(times[i]).strftime("%H:%M")
        price = closes[i]
        vwap_dist = price - vwap
        vol_ratio_str = f"{vol_ratio:>6.1f}x" if not np.isnan(vol_ratio) else "    —"

        # Mark the critical minute with a flag
        marker = " ◀" if ts_str == "09:59" else ""

        print(f"{ts_str:>6}  ${price:>6.2f}  ${vwap:>6.2f}  "
              f"{vwap_dist:>+6.3f}  {vol_ratio_str:>9}  {shape:>9}  {hvn_str}{marker}")

        # Only print through 10:30 to keep it readable
        if ts_str >= "10:30":
            remaining = len(bars_win) - i - 1
            if remaining > 0:
                print(f"\n  ... ({remaining} more bars not shown; session continues to "
                      f"{pd.Timestamp(times[-1]).strftime('%H:%M')})")
            break

    # Final state snapshot
    print("\n" + "=" * 100)
    print("FINAL STATE (all RTH bars so far)")
    print("=" * 100)
    final_hvns = _find_top_k_hvns(lows, highs, vols, k=5, bin_size=0.05, min_separation=4)
    final_vwap = _session_vwap(opens, highs, lows, closes, vols)
    last_price = closes[-1]
    print(f"\nLast price:    ${last_price:.2f}")
    print(f"Session VWAP:  ${final_vwap:.2f}  ({last_price - final_vwap:+.2f} from VWAP)")
    print(f"\nTop 5 intraday HVNs:")
    for i, (p, v) in enumerate(final_hvns, 1):
        relative = "above" if p > last_price else "below" if p < last_price else "at"
        dist = abs(p - last_price)
        print(f"  {i}. ${p:.2f}  volume={v/1e3:.0f}k  ({relative} last, dist=${dist:.2f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
