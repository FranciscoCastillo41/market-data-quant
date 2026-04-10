"""Risk / kill-switch checks for the live runner.

Centralized so the runner loop doesn't have risk logic scattered around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mdq.config import PROJECT_ROOT


KILL_FILE = PROJECT_ROOT / "data" / "KILL"


@dataclass
class RiskState:
    max_open_positions: int = 1
    max_trades_per_session: int = 3
    max_daily_loss_usd: float = 300.0

    open_positions: int = 0
    trades_today: int = 0
    realized_pnl_today: float = 0.0
    locked_out: bool = False
    lockout_reason: str = ""

    def can_open_new(self) -> tuple[bool, str]:
        if self.locked_out:
            return False, f"locked_out: {self.lockout_reason}"
        if KILL_FILE.exists():
            self.locked_out = True
            self.lockout_reason = f"kill file exists at {KILL_FILE}"
            return False, self.lockout_reason
        if self.open_positions >= self.max_open_positions:
            return False, f"max_open_positions={self.max_open_positions} reached"
        if self.trades_today >= self.max_trades_per_session:
            return False, f"max_trades_per_session={self.max_trades_per_session} reached"
        if self.realized_pnl_today <= -self.max_daily_loss_usd:
            self.locked_out = True
            self.lockout_reason = f"daily loss {self.realized_pnl_today:.2f} ≤ -{self.max_daily_loss_usd}"
            return False, self.lockout_reason
        return True, "ok"

    def on_entry(self) -> None:
        self.open_positions += 1
        self.trades_today += 1

    def on_exit(self, realized_pnl: float) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.realized_pnl_today += realized_pnl
