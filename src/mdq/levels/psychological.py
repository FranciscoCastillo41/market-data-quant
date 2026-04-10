"""Psychological price levels: whole-dollar / half-dollar grids.

A "level grid" is just an arithmetic sequence of prices at a fixed step.
We generate one grid per ticker per step-size and classify each level
into a tier by whether it's a multiple of a larger step (Tier 1 = magic
levels, Tier 2 = second-order, Tier 3 = fractional).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LevelGrid:
    """An ordered set of prices with per-level tier classification."""

    step: float
    prices: np.ndarray  # shape (n,)
    tiers: np.ndarray   # shape (n,) of int (1,2,3)

    def in_range(self, low: float, high: float) -> "LevelGrid":
        """Return a sub-grid containing only prices in [low, high]."""
        mask = (self.prices >= low) & (self.prices <= high)
        return LevelGrid(
            step=self.step,
            prices=self.prices[mask],
            tiers=self.tiers[mask],
        )


def _is_multiple(value: float, step: float, eps: float = 1e-6) -> bool:
    """True if value is an integer multiple of step (within float tolerance)."""
    ratio = value / step
    return abs(ratio - round(ratio)) < eps


def generate_grid(
    low: float,
    high: float,
    step: float,
    tier1_step: float | None = None,
    tier2_step: float | None = None,
) -> LevelGrid:
    """Generate a LevelGrid from low to high at step resolution.

    Tier assignment:
        tier 1: price is a multiple of tier1_step  (e.g. $5 for SPY)
        tier 2: price is a multiple of tier2_step but not tier1 (e.g. $1)
        tier 3: everything else in the grid (e.g. $0.50)

    If tier1_step / tier2_step are None, the grid is all tier 1.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if high < low:
        raise ValueError("high must be >= low")

    # Align low to the next multiple of step >= low
    start = np.ceil(low / step) * step
    n = int(np.floor((high - start) / step)) + 1
    prices = np.round(start + np.arange(n) * step, 6)

    tiers = np.full(n, 1, dtype=int)
    if tier1_step is not None and tier2_step is not None:
        t = np.ones(n, dtype=int) * 3
        for i, p in enumerate(prices):
            if _is_multiple(p, tier1_step):
                t[i] = 1
            elif _is_multiple(p, tier2_step):
                t[i] = 2
        tiers = t
    elif tier1_step is not None:
        # Only tier 1 vs tier 3
        t = np.where(
            np.array([_is_multiple(p, tier1_step) for p in prices]), 1, 3
        )
        tiers = t.astype(int)

    return LevelGrid(step=step, prices=prices, tiers=tiers)


# Convenience presets
def spy_grid(low: float, high: float) -> LevelGrid:
    """SPY standard grid: $0.50 step, T1 = $5, T2 = $1."""
    return generate_grid(
        low=low, high=high, step=0.50, tier1_step=5.0, tier2_step=1.0
    )


def qqq_grid(low: float, high: float, step: float = 1.0) -> LevelGrid:
    """QQQ grid with configurable step (for empirical discovery).

    Default T1 = $10, T2 = $5.
    """
    return generate_grid(
        low=low, high=high, step=step, tier1_step=10.0, tier2_step=5.0
    )
