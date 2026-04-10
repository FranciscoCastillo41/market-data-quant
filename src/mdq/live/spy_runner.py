"""SPY paper trading runner for S3 POC wick-rejection signal.

Parallel to the QQQ runner. Watches SPY 1-min bars in the 09:30-11:30 ET
window, fires the S3 detector against yesterday's POC, enters an ATM 0DTE
call (long support) or put (short resistance), manages target/stop/timeout.

Key differences from QQQ runner:
  - Level = POC (not prior_low)
  - Direction can be either long OR short (not just short)
  - Target/stop are ATR-based, not fixed dollars
  - Multiple signals per session allowed (S3 can fire multiple times)
  - Journal + risk under separate namespace so it doesn't mix with QQQ
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Literal

import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.live.alpaca import AlpacaClient
from mdq.live.feed import LiveBar, LiveBarFeed
from mdq.live.journal import Journal
from mdq.live.risk import RiskState
from mdq.live.spy_signal import (
    HORIZON_BARS,
    SpyS3PocDetector,
    SpySignal,
)

ET = pytz.timezone("America/New_York")
RES_C = RESULTS_DIR / "experiment_c"

WINDOW_OPEN = (9, 30)
WINDOW_CLOSE = (11, 30)
HARD_CLOSE = (11, 45)


def _build_occ(underlying: str, exp_date: date, strike: float, kind: Literal["C", "P"]) -> str:
    yy = exp_date.strftime("%y")
    mm = exp_date.strftime("%m")
    dd = exp_date.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}{kind}{strike_int:08d}"


def _load_prior_poc(target_date: date) -> tuple[date, float] | None:
    path = RES_C / "profiles__SPY.parquet"
    if not path.exists():
        return None
    profs = pd.read_parquet(path).sort_values("session_date")
    profs["session_date"] = pd.to_datetime(profs["session_date"]).dt.date
    prior = profs[profs["session_date"] < target_date]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return row["session_date"], float(row["poc"])


class SpyLiveRunner:
    def __init__(self, contracts: int = 1, dry_run: bool = False):
        self.contracts = contracts
        self.dry_run = dry_run
        self.today = datetime.now(tz=ET).date()
        self.journal = Journal(self.today, name="spy_s3_poc")
        self.risk = RiskState(max_trades_per_session=5)  # allow more — S3 can fire multiple times
        self.alpaca = AlpacaClient() if not dry_run else None

        self.detector: SpyS3PocDetector | None = None
        self.feed: LiveBarFeed | None = None

        self.open_symbol: str | None = None
        self.open_direction: Literal["short", "long"] | None = None
        self.open_entry_price_under: float | None = None
        self.open_target: float = 0.0
        self.open_stop: float = 0.0
        self.open_bars_held: int = 0
        self.open_entry_ts: datetime | None = None

    def log(self, event: str, **fields) -> None:
        print(f"  [{datetime.now(tz=ET).strftime('%H:%M:%S')}] SPY {event}  {fields}")
        self.journal.write(event, **fields)

    def _today_time(self, hour: int, minute: int) -> datetime:
        return ET.localize(datetime(self.today.year, self.today.month, self.today.day, hour, minute))

    async def run(self) -> None:
        window_open = self._today_time(*WINDOW_OPEN)
        window_close = self._today_time(*WINDOW_CLOSE)
        hard_close = self._today_time(*HARD_CLOSE)

        self.log("session_start", today=self.today.isoformat(),
                 rule="S3_POC", dry_run=self.dry_run, contracts=self.contracts)

        pl_res = _load_prior_poc(self.today)
        if pl_res is None:
            self.log("abort", reason="no_prior_poc_profile")
            return
        prior_date, prior_poc = pl_res
        self.log("prior_poc_loaded", prior_date=prior_date, prior_poc=prior_poc)

        self.detector = SpyS3PocDetector(level_name="poc", level=prior_poc)

        if self.alpaca:
            try:
                acct = self.alpaca.get_account()
                self.log("account", cash=acct.cash, portfolio_value=acct.portfolio_value)
            except Exception as e:
                self.log("account_error", error=str(e))

        now = datetime.now(tz=ET)
        if now < window_open:
            wait_s = (window_open - now).total_seconds()
            self.log("waiting_for_open", seconds=wait_s)
            await asyncio.sleep(max(0, wait_s))

        self.feed = LiveBarFeed("SPY", poll_interval_s=15.0)
        self.log("feed_started", until=hard_close.isoformat())

        try:
            async for bar in self.feed.iter_bars(self.today, hard_close):
                if bar.ts_et < window_open:
                    continue

                if self.open_symbol is not None:
                    await self._manage_position(bar, hard_close)

                if bar.ts_et >= window_close:
                    continue

                signal = self.detector.on_bar(bar)
                if signal is None:
                    continue

                self.log("signal_fired",
                         direction=signal.direction,
                         level=signal.level,
                         atr=signal.atr,
                         target=signal.target_move,
                         stop=signal.stop_move,
                         bar_close=bar.c,
                         bar_ts=bar.ts_et)

                ok, reason = self.risk.can_open_new()
                if not ok:
                    self.log("signal_blocked", reason=reason)
                    continue
                if self.open_symbol is not None:
                    self.log("signal_blocked", reason="position_open")
                    continue

                await self._enter(signal, bar)

            if self.open_symbol is not None:
                self.log("feed_ended_with_open_position", symbol=self.open_symbol)
                await self._force_exit()
        finally:
            if self.feed:
                await self.feed.close()
            self.log("session_end", trades=self.risk.trades_today,
                     realized_pnl=self.risk.realized_pnl_today)

    async def _enter(self, signal: SpySignal, bar: LiveBar) -> None:
        atm_strike = round(bar.c)
        kind: Literal["C", "P"] = "P" if signal.direction == "short" else "C"
        symbol = _build_occ("SPY", self.today, atm_strike, kind)

        self.log("entry_planned", symbol=symbol, strike=atm_strike, kind=kind,
                 entry_under=bar.c, target_move=signal.target_move,
                 stop_move=signal.stop_move)

        if self.dry_run:
            self.log("entry_dry_run", symbol=symbol)
        else:
            try:
                order = self.alpaca.buy_option_market(symbol, qty=self.contracts)
                self.log("entry_order_submitted", order_id=order.id,
                         status=order.status, symbol=symbol)
            except Exception as e:
                self.log("entry_order_error", error=str(e), symbol=symbol)
                return

        self.risk.on_entry()
        self.open_symbol = symbol
        self.open_direction = signal.direction
        self.open_entry_price_under = bar.c
        self.open_target = signal.target_move
        self.open_stop = signal.stop_move
        self.open_bars_held = 0
        self.open_entry_ts = bar.ts_et

    async def _manage_position(self, bar: LiveBar, hard_close: datetime) -> None:
        if self.open_entry_price_under is None or self.open_direction is None:
            return
        self.open_bars_held += 1

        entry = self.open_entry_price_under
        if self.open_direction == "short":
            target_px = entry - self.open_target
            stop_px = entry + self.open_stop
            hit_target = bar.l <= target_px
            hit_stop = bar.h >= stop_px
        else:
            target_px = entry + self.open_target
            stop_px = entry - self.open_stop
            hit_target = bar.h >= target_px
            hit_stop = bar.l <= stop_px

        if hit_stop:
            self.log("stop_hit", stop_px=stop_px, bars_held=self.open_bars_held)
            await self._exit(bar, reason="stop")
            return
        if hit_target:
            self.log("target_hit", target_px=target_px, bars_held=self.open_bars_held)
            await self._exit(bar, reason="target")
            return
        if self.open_bars_held >= HORIZON_BARS:
            self.log("timeout_exit", bars_held=self.open_bars_held)
            await self._exit(bar, reason="timeout")
            return
        if bar.ts_et >= hard_close:
            self.log("hard_close_exit", bars_held=self.open_bars_held)
            await self._exit(bar, reason="hard_close")

    async def _exit(self, bar: LiveBar, reason: str) -> None:
        if self.open_symbol is None:
            return
        symbol = self.open_symbol
        realized = 0.0

        if self.dry_run:
            self.log("exit_dry_run", symbol=symbol, reason=reason)
        else:
            try:
                order = self.alpaca.sell_option_market(symbol, qty=self.contracts)
                self.log("exit_order_submitted", order_id=order.id,
                         status=order.status, reason=reason)
                try:
                    pos = self.alpaca.get_position(symbol)
                    realized = pos.unrealized_pl if pos and pos.unrealized_pl is not None else 0.0
                except Exception:
                    realized = 0.0
            except Exception as e:
                self.log("exit_order_error", error=str(e), symbol=symbol)

        self.risk.on_exit(realized)
        self.log("position_closed", symbol=symbol, reason=reason,
                 realized_pnl=realized, bars_held=self.open_bars_held,
                 entry_under=self.open_entry_price_under, exit_under=bar.c)

        self.open_symbol = None
        self.open_direction = None
        self.open_entry_price_under = None
        self.open_target = 0.0
        self.open_stop = 0.0
        self.open_bars_held = 0

    async def _force_exit(self) -> None:
        if self.open_symbol is None:
            return
        symbol = self.open_symbol
        self.log("force_exit", symbol=symbol)
        if not self.dry_run:
            try:
                self.alpaca.close_position(symbol)
            except Exception as e:
                self.log("force_exit_error", error=str(e))
        self.risk.on_exit(0.0)
        self.open_symbol = None
        self.open_direction = None
