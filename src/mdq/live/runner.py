"""Live paper trading runner for Tier 1 signal.

Main event loop:
  1. At start: load yesterday's prior_low from Experiment C profiles
  2. Start LiveBarFeed polling Massive for today's QQQ bars
  3. Feed closed bars into TouchDetector
  4. On touch event, evaluate Tier 1 signal rules
  5. If signal fires and risk allows: pick ATM 0DTE put, submit buy order
  6. Monitor open position: race target vs stop vs 15-min timeout
  7. Close position on exit condition
  8. Journal everything to JSONL

Hard stops:
  - Window closes at 11:15 ET (detection)
  - Session closes at 11:30 ET (forced exit of any open position)
  - Kill file at data/KILL halts new entries immediately
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.live.alpaca import AlpacaClient, OrderResult, Position
from mdq.live.detector import TouchDetector, TouchEvent
from mdq.live.feed import LiveBar, LiveBarFeed
from mdq.live.journal import Journal
from mdq.live.risk import RiskState
from mdq.live.signal import TradePlan, evaluate_tier1

ET = pytz.timezone("America/New_York")
RES_C = RESULTS_DIR / "experiment_c"

WINDOW_OPEN = (9, 30)   # 09:30 ET
WINDOW_CLOSE = (11, 15) # 11:15 ET — no new signals after this
HARD_CLOSE = (11, 30)   # 11:30 ET — force close any open position


def _build_occ_put(underlying: str, exp_date: date, strike: float) -> str:
    yy = exp_date.strftime("%y")
    mm = exp_date.strftime("%m")
    dd = exp_date.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}P{strike_int:08d}"


def _load_prior_low(target_date: date, ticker: str = "QQQ") -> tuple[date, float] | None:
    """Load the most recent prior-session low from the Experiment C profile cache."""
    path = RES_C / f"profiles__{ticker}.parquet"
    if not path.exists():
        return None
    profs = pd.read_parquet(path).sort_values("session_date")
    prior = profs[profs["session_date"] < target_date]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return row["session_date"], float(row["low"])


class LiveRunner:
    def __init__(
        self,
        ticker: str = "QQQ",
        contracts: int = 1,
        dry_run: bool = False,
    ):
        self.ticker = ticker
        self.contracts = contracts
        self.dry_run = dry_run

        self.today = datetime.now(tz=ET).date()
        self.journal = Journal(self.today, name="tier1")
        self.risk = RiskState()
        self.alpaca = AlpacaClient() if not dry_run else None

        self.detector: TouchDetector | None = None
        self.feed: LiveBarFeed | None = None

        # Open position tracking
        self.open_symbol: str | None = None
        self.open_entry_price_under: float | None = None
        self.open_entry_ts: datetime | None = None
        self.open_plan: TradePlan | None = None
        self.open_bars_held: int = 0

    def log(self, event: str, **fields) -> None:
        print(f"  [{datetime.now(tz=ET).strftime('%H:%M:%S')}] {event}  {fields}")
        self.journal.write(event, **fields)

    def _today_time(self, hour: int, minute: int) -> datetime:
        return ET.localize(datetime(self.today.year, self.today.month, self.today.day, hour, minute))

    async def run(self) -> None:
        window_open = self._today_time(*WINDOW_OPEN)
        window_close = self._today_time(*WINDOW_CLOSE)
        hard_close = self._today_time(*HARD_CLOSE)

        self.log("session_start", today=self.today.isoformat(), dry_run=self.dry_run,
                 contracts=self.contracts)

        # Load prior-day low
        pl_res = _load_prior_low(self.today, self.ticker)
        if pl_res is None:
            self.log("abort", reason="no_prior_low_profile")
            return
        prior_date, prior_low = pl_res
        self.log("prior_low_loaded", prior_date=prior_date, prior_low=prior_low)

        # Initialize detector for Tier 1 level
        self.detector = TouchDetector(
            levels=[("prior_low", prior_low)],
            tolerance=0.05,
        )

        # Account snapshot
        if self.alpaca:
            try:
                acct = self.alpaca.get_account()
                self.log("account", cash=acct.cash, portfolio_value=acct.portfolio_value,
                         options_level=acct.options_trading_level)
            except Exception as e:
                self.log("account_error", error=str(e))

        # Wait until window opens
        now = datetime.now(tz=ET)
        if now < window_open:
            wait_s = (window_open - now).total_seconds()
            self.log("waiting_for_open", seconds=wait_s, open_at=window_open.isoformat())
            await asyncio.sleep(max(0, wait_s))

        # Start feed
        self.feed = LiveBarFeed(self.ticker, poll_interval_s=15.0)
        self.log("feed_started", until=hard_close.isoformat())

        try:
            async for bar in self.feed.iter_bars(self.today, hard_close):
                # Outside detection window? Still monitor open position.
                now = datetime.now(tz=ET)
                if bar.ts_et < window_open:
                    continue

                # Monitor open position on every bar
                if self.open_symbol is not None:
                    await self._manage_position(bar, hard_close)

                # Detection logic only inside window
                if bar.ts_et >= window_close:
                    # Detection window closed — only manage existing position
                    continue

                # Feed bar to detector
                events = self.detector.on_bar(bar)
                for ev in events:
                    self.log("touch", level=ev.level, level_name=ev.level_name,
                             approach=ev.approach, bar_close=bar.c, bar_ts=bar.ts_et)
                    plan = evaluate_tier1(ev)
                    if plan is None:
                        self.log("signal_skipped", reason="not_tier1",
                                 approach=ev.approach, level=ev.level)
                        continue
                    ok, reason = self.risk.can_open_new()
                    if not ok:
                        self.log("signal_blocked", reason=reason)
                        continue
                    if self.open_symbol is not None:
                        self.log("signal_blocked", reason="already_in_position")
                        continue
                    await self._enter(ev, plan, bar)

                # Hard-close force exit
                if bar.ts_et >= hard_close and self.open_symbol is not None:
                    self.log("hard_close_reached", ts=bar.ts_et)
                    await self._exit_position(bar, reason="hard_close")

            # Feed loop ended. If still in position, force close.
            if self.open_symbol is not None:
                self.log("feed_ended_with_open_position", symbol=self.open_symbol)
                await self._force_exit_current()

        finally:
            if self.feed:
                await self.feed.close()
            self.log("session_end", realized_pnl=self.risk.realized_pnl_today,
                     trades=self.risk.trades_today)

    async def _enter(self, event: TouchEvent, plan: TradePlan, bar: LiveBar) -> None:
        """Pick ATM 0DTE put, submit buy, record entry state."""
        atm_strike = round(bar.c)
        symbol = _build_occ_put(self.ticker, self.today, atm_strike)
        self.log("entry_planned", tier=plan.tier, symbol=symbol, strike=atm_strike,
                 entry_close_under=bar.c, touch_level=event.level)

        if self.dry_run:
            self.log("entry_dry_run", symbol=symbol)
        else:
            try:
                order = self.alpaca.buy_option_market(symbol, qty=self.contracts)
                self.log("entry_order_submitted", order_id=order.id, status=order.status,
                         symbol=order.symbol)
            except Exception as e:
                self.log("entry_order_error", error=str(e), symbol=symbol)
                return

        self.risk.on_entry()
        self.open_symbol = symbol
        self.open_entry_price_under = bar.c
        self.open_entry_ts = bar.ts_et
        self.open_plan = plan
        self.open_bars_held = 0

    async def _manage_position(self, bar: LiveBar, hard_close: datetime) -> None:
        """Called on every new bar while a position is open.

        Races target vs stop on the UNDERLYING and exits when either hits or
        after horizon_bars elapse.
        """
        if self.open_plan is None or self.open_entry_price_under is None:
            return
        self.open_bars_held += 1

        entry = self.open_entry_price_under
        target_px = entry - self.open_plan.target_move  # short
        stop_px = entry + self.open_plan.stop_move

        hit_target = bar.l <= target_px
        hit_stop = bar.h >= stop_px

        if hit_stop:
            self.log("stop_hit", bar_high=bar.h, stop_px=stop_px, bars_held=self.open_bars_held)
            await self._exit_position(bar, reason="stop")
            return

        if hit_target:
            self.log("target_hit", bar_low=bar.l, target_px=target_px, bars_held=self.open_bars_held)
            await self._exit_position(bar, reason="target")
            return

        if self.open_bars_held >= self.open_plan.horizon_bars:
            self.log("timeout_exit", bars_held=self.open_bars_held)
            await self._exit_position(bar, reason="timeout")
            return

        if bar.ts_et >= hard_close:
            self.log("hard_close_exit", bars_held=self.open_bars_held)
            await self._exit_position(bar, reason="hard_close")

    async def _exit_position(self, bar: LiveBar, reason: str) -> None:
        """Close the open position via market sell order."""
        if self.open_symbol is None:
            return

        symbol = self.open_symbol
        if self.dry_run:
            self.log("exit_dry_run", symbol=symbol, reason=reason)
            realized = 0.0
        else:
            try:
                order = self.alpaca.sell_option_market(symbol, qty=self.contracts)
                self.log("exit_order_submitted", order_id=order.id, status=order.status,
                         reason=reason)
                # Try to read realized P&L from position close
                try:
                    pos = self.alpaca.get_position(symbol)
                    realized = pos.unrealized_pl if pos and pos.unrealized_pl is not None else 0.0
                except Exception:
                    realized = 0.0
            except Exception as e:
                self.log("exit_order_error", error=str(e), symbol=symbol)
                realized = 0.0

        self.risk.on_exit(realized)
        self.log("position_closed", symbol=symbol, reason=reason,
                 realized_pnl=realized, bars_held=self.open_bars_held,
                 entry_under=self.open_entry_price_under, exit_under=bar.c)

        self.open_symbol = None
        self.open_entry_price_under = None
        self.open_entry_ts = None
        self.open_plan = None
        self.open_bars_held = 0

    async def _force_exit_current(self) -> None:
        """Force-close position regardless of bar data (end-of-session)."""
        if self.open_symbol is None:
            return
        symbol = self.open_symbol
        self.log("force_exit", symbol=symbol)
        if self.dry_run:
            self.log("force_exit_dry_run", symbol=symbol)
        else:
            try:
                self.alpaca.close_position(symbol)
            except Exception as e:
                self.log("force_exit_error", error=str(e))
        self.risk.on_exit(0.0)
        self.open_symbol = None
        self.open_plan = None
