"""Stateful first-touch detector for live bars.

Holds the set of target levels for today's session and watches incoming bars
for the first bar that touches each level. Emits a TouchEvent on first-touch
only; subsequent touches are silently tracked but not re-emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mdq.live.feed import LiveBar


@dataclass(frozen=True)
class TouchEvent:
    level_name: str
    level: float
    bar: LiveBar
    approach: Literal["from_below", "from_above", "at"]
    touch_num: int  # always 1 when emitted (first-touch)


class TouchDetector:
    """Tracks first-touches of a fixed set of (name, price) levels.

    Usage:
        det = TouchDetector([("prior_low", 578.40)], tolerance=0.05)
        for bar in feed:
            for event in det.on_bar(bar):
                ...
    """

    def __init__(
        self,
        levels: list[tuple[str, float]],
        tolerance: float = 0.05,
    ):
        self.levels = levels
        self.tolerance = tolerance
        self._touched: set[str] = set()
        self._prev_close: float | None = None

    def on_bar(self, bar: LiveBar) -> list[TouchEvent]:
        """Feed a newly closed bar. Returns list of TouchEvents (usually 0 or 1)."""
        events: list[TouchEvent] = []
        for name, level in self.levels:
            if name in self._touched:
                continue
            if bar.l <= level + self.tolerance and bar.h >= level - self.tolerance:
                if self._prev_close is None:
                    approach: Literal["from_below", "from_above", "at"] = "at"
                elif self._prev_close < level:
                    approach = "from_below"
                elif self._prev_close > level:
                    approach = "from_above"
                else:
                    approach = "at"
                events.append(
                    TouchEvent(
                        level_name=name,
                        level=level,
                        bar=bar,
                        approach=approach,
                        touch_num=1,
                    )
                )
                self._touched.add(name)
        self._prev_close = bar.c
        return events
