"""SPY S3 POC wick-rejection signal for live paper trading.

Stateful single-session detector that watches 1-min bars as they close and
fires an event when the S3 rule triggers on yesterday's POC level. Also
maintains rolling ATR so the runner can size the target/stop per signal.

Rule (from Experiment E, validated on 2026 OOS, 22/23 targets Jan-Feb):
    bar_i touches level (prior-day POC)
    bar_i upper wick >= 50% of range (for from_below approach) or
    bar_i lower wick >= 50% of range (for from_above approach)
    bar_i close is on the rejection side of the level
    bar_{i+1} closes further in the rejection direction
    => fire on bar_{i+1} close, direction = fade

Target = 1.0 x ATR(20). Stop = 1.0 x ATR(20). Max hold = 15 bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mdq.live.feed import LiveBar

TOL = 0.05
WICK_THRESHOLD = 0.50
ATR_WINDOW = 20
TARGET_ATR_MULT = 1.00
STOP_ATR_MULT = 1.00
HORIZON_BARS = 15


@dataclass(frozen=True)
class SpySignal:
    level_name: str
    level: float
    bar: LiveBar
    direction: Literal["short", "long"]
    atr: float
    target_move: float
    stop_move: float


class SpyS3PocDetector:
    """Stateful detector for the S3 POC wick-rejection fade on SPY.

    Feed bars one at a time via on_bar(). Returns a SpySignal when the rule
    fires (which requires two consecutive bars — the rejection bar and the
    follow-through bar).

    State held:
      - rolling high/low/close for ATR computation
      - the most recent "candidate" rejection bar awaiting follow-through
    """

    def __init__(self, level_name: str, level: float):
        self.level_name = level_name
        self.level = level
        self._bars: list[LiveBar] = []
        self._pending: dict | None = None
        self._fired_for_this_level = False

    def on_bar(self, bar: LiveBar) -> SpySignal | None:
        self._bars.append(bar)
        if len(self._bars) < ATR_WINDOW + 2:
            return None

        atr = self._compute_atr()
        if atr is None or atr <= 0:
            return None

        signal: SpySignal | None = None

        # If there's a pending candidate from the previous bar, check follow-through
        if self._pending is not None:
            pending_bar: LiveBar = self._pending["bar"]
            direction = self._pending["direction"]

            if direction == "short" and bar.c < pending_bar.c:
                signal = SpySignal(
                    level_name=self.level_name,
                    level=self.level,
                    bar=bar,
                    direction="short",
                    atr=atr,
                    target_move=atr * TARGET_ATR_MULT,
                    stop_move=atr * STOP_ATR_MULT,
                )
            elif direction == "long" and bar.c > pending_bar.c:
                signal = SpySignal(
                    level_name=self.level_name,
                    level=self.level,
                    bar=bar,
                    direction="long",
                    atr=atr,
                    target_move=atr * TARGET_ATR_MULT,
                    stop_move=atr * STOP_ATR_MULT,
                )
            self._pending = None  # consume candidate either way

        # Now check whether THIS bar is itself a new rejection candidate
        prev_bar = self._bars[-2]  # bar before the current
        candidate = self._is_rejection_candidate(bar, prev_bar)
        if candidate is not None:
            self._pending = candidate

        return signal

    def _compute_atr(self) -> float | None:
        """Simple SMA-based ATR over last ATR_WINDOW bars."""
        if len(self._bars) < ATR_WINDOW + 1:
            return None
        recent = self._bars[-(ATR_WINDOW + 1):]
        trs: list[float] = []
        for i in range(1, len(recent)):
            h = recent[i].h
            l = recent[i].l
            prev_c = recent[i - 1].c
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        if not trs:
            return None
        return sum(trs) / len(trs)

    def _is_rejection_candidate(self, bar: LiveBar, prev_bar: LiveBar) -> dict | None:
        """Does `bar` satisfy the S3 rejection wick criteria vs the level?"""
        level = self.level

        # Touch check
        if bar.l > level + TOL or bar.h < level - TOL:
            return None

        # Approach based on previous close
        if prev_bar.c < level:
            approach = "from_below"
        elif prev_bar.c > level:
            approach = "from_above"
        else:
            return None

        bar_range = bar.h - bar.l
        if bar_range <= 0:
            return None

        body_hi = max(bar.o, bar.c)
        body_lo = min(bar.o, bar.c)
        upper_wick = bar.h - body_hi
        lower_wick = body_lo - bar.l

        if approach == "from_below":
            # resistance test: upper wick >= threshold, close below level
            if upper_wick / bar_range < WICK_THRESHOLD:
                return None
            if bar.c >= level:
                return None
            return {"bar": bar, "direction": "short"}
        else:
            # support test: lower wick >= threshold, close above level
            if lower_wick / bar_range < WICK_THRESHOLD:
                return None
            if bar.c <= level:
                return None
            return {"bar": bar, "direction": "long"}
