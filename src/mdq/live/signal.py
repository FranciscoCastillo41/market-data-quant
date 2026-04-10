"""Tier 1 signal rules.

Pure functions that take a TouchEvent + context and decide:
  - Does this fire the Tier 1 trade?
  - What direction, what strike type, what target/stop?

Separated from the runner so we can unit-test rules in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mdq.live.detector import TouchEvent


@dataclass(frozen=True)
class TradePlan:
    tier: int
    side: Literal["short_put", "long_put", "long_call"]  # for this iteration, we trade puts
    contract_type: Literal["put", "call"]
    target_move: float  # underlying dollars in our favor
    stop_move: float    # underlying dollars against us
    horizon_bars: int   # minutes to hold max
    rationale: str


def is_non_confluence(level: float, whole_dollar_step: float = 10.0, radius: float = 0.50) -> bool:
    nearest = round(level / whole_dollar_step) * whole_dollar_step
    return abs(level - nearest) > radius


def evaluate_tier1(event: TouchEvent) -> TradePlan | None:
    """Tier 1 rule: QQQ prior_low, first-touch, non-confluence, from_above.

    Direction is always SHORT (buy puts) since the setup is a breakdown of
    yesterday's low. Target −$1.00 on QQQ, stop +$1.50, 15-min max hold.
    """
    if event.level_name != "prior_low":
        return None
    if event.approach != "from_above":
        return None
    if not is_non_confluence(event.level, whole_dollar_step=10.0, radius=0.50):
        return None
    return TradePlan(
        tier=1,
        side="long_put",
        contract_type="put",
        target_move=1.00,
        stop_move=1.50,
        horizon_bars=15,
        rationale="Tier 1: QQQ prior_low first-touch non-confluence from_above breakdown",
    )
