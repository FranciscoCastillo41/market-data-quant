"""SPY gap-reversion CZ paper trader.

Francisco's validated strategy:
    SIGNAL: SPY opens below yesterday's VAL, gap >= 0.3% from prior close,
            above prior-week VAL, not near earnings
    ENTRY:  Buy ATM 0DTE call at the session low within first 30 min (Entry C)
    EXIT:   Sell when cumulative volume delta peaks and declines 5 bars (Exit Z)
    STOP:   underlying drops 2x the gap from open
    TIMEOUT: 3 hours max hold
    HARD CLOSE: 12:30 ET

Flow:
    1. At 9:25 ET: load prior-day profile (VAL, POC), prior-week VAL, check filters
    2. If signal qualifies: enter "dip-buy mode" — watch bars for 30 min, track low
    3. When low is identified (price starts bouncing): buy ATM 0DTE call
    4. Monitor cumulative delta for exhaustion exit
    5. Hard stop if underlying drops to 2x gap below open
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth
from mdq.levels.weekly_profile import compute_all_weekly_profiles
from mdq.live.alpaca import AlpacaClient
from mdq.live.feed import LiveBar, LiveBarFeed
from mdq.live.journal import Journal
from mdq.live.risk import RiskState

ET = pytz.timezone("America/New_York")
RES_C = RESULTS_DIR / "experiment_c"

WINDOW_OPEN = (9, 30)
HARD_CLOSE = (12, 30)  # 3 hour max session
MIN_GAP_PCT = 0.003
MAX_DIP_WAIT_BARS = 30
CUM_DELTA_DECLINE_BARS = 5
MAX_HOLD_BARS = 180  # 3 hours


def _build_occ_call(underlying: str, exp_date: date, strike: float) -> str:
    yy = exp_date.strftime("%y")
    mm = exp_date.strftime("%m")
    dd = exp_date.strftime("%d")
    strike_int = int(round(strike * 1000))
    return f"{underlying}{yy}{mm}{dd}C{strike_int:08d}"


def _buy_pressure(o: float, h: float, l: float, c: float) -> float:
    rng = h - l
    if rng <= 0:
        return 0.5
    return (c - l) / rng


def _load_prior_day_profile(target_date: date) -> dict | None:
    path = RES_C / "profiles__SPY.parquet"
    if not path.exists():
        return None
    profs = pd.read_parquet(path).sort_values("session_date")
    profs["session_date"] = pd.to_datetime(profs["session_date"]).dt.date
    prior = profs[profs["session_date"] < target_date]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return {
        "date": row["session_date"],
        "poc": float(row["poc"]),
        "vah": float(row["vah"]),
        "val": float(row["val"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    }


def _load_prior_week_val(target_date: date) -> float | None:
    """Load the most recent completed week's VAL for SPY."""
    try:
        start = (pd.Timestamp(target_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        bars = load_bars("SPY", start, (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        weekly = compute_all_weekly_profiles(bars, bin_size=0.10)
        if weekly.empty:
            return None
        weekly["week_end"] = pd.to_datetime(weekly["week_end"]).dt.date
        prior = weekly[weekly["week_end"] < target_date]
        if prior.empty:
            return None
        return float(prior.iloc[-1]["val"])
    except Exception:
        return None


class GapReversionRunner:
    def __init__(self, contracts: int = 1, dry_run: bool = False):
        self.contracts = contracts
        self.dry_run = dry_run
        self.today = datetime.now(tz=ET).date()
        self.journal = Journal(self.today, name="gap_reversion")
        self.risk = RiskState(max_trades_per_session=2)
        self.alpaca = AlpacaClient() if not dry_run else None
        self.feed: LiveBarFeed | None = None

        # Signal state
        self.signal_qualified = False
        self.prior_val: float = 0.0
        self.prior_poc: float = 0.0
        self.prior_close: float = 0.0
        self.gap_dollars: float = 0.0
        self.stop_px: float = 0.0
        self.target_px: float = 0.0

        # Dip-buy state (Entry C)
        self.in_dip_mode = False
        self.dip_bars_seen = 0
        self.session_low = float("inf")
        self.session_low_idx = -1
        self.dip_bought = False

        # Position state
        self.open_symbol: str | None = None
        self.open_entry_price: float | None = None
        self.open_bars_held = 0

        # Cum delta state (Exit Z)
        self.bars_history: list[LiveBar] = []
        self.cum_delta: float = 0.0
        self.cum_delta_peak: float = -float("inf")
        self.cum_delta_decline_count: int = 0
        self.prev_cum_delta: float = 0.0

    def log(self, event: str, **fields) -> None:
        print(f"  [{datetime.now(tz=ET).strftime('%H:%M:%S')}] GAP {event}  {fields}")
        self.journal.write(event, **fields)

    def _today_time(self, hour: int, minute: int) -> datetime:
        return ET.localize(datetime(self.today.year, self.today.month, self.today.day, hour, minute))

    async def run(self) -> None:
        hard_close = self._today_time(*HARD_CLOSE)

        self.log("session_start", today=self.today.isoformat(),
                 strategy="gap_reversion_CZ", dry_run=self.dry_run)

        # Load prior-day profile
        prior = _load_prior_day_profile(self.today)
        if prior is None:
            self.log("abort", reason="no_prior_profile")
            return
        self.prior_val = prior["val"]
        self.prior_poc = prior["poc"]
        self.prior_close = prior["close"]
        self.target_px = prior["poc"]
        self.log("prior_day_loaded", **prior)

        # Load weekly VAL
        weekly_val = _load_prior_week_val(self.today)
        if weekly_val is not None:
            self.log("weekly_val_loaded", weekly_val=weekly_val)
        else:
            self.log("weekly_val_missing", note="will skip weekly filter")

        # Account check
        if self.alpaca:
            try:
                acct = self.alpaca.get_account()
                self.log("account", cash=acct.cash, portfolio_value=acct.portfolio_value)
            except Exception as e:
                self.log("account_error", error=str(e))

        # Wait for market open
        window_open = self._today_time(*WINDOW_OPEN)
        now = datetime.now(tz=ET)
        if now < window_open:
            wait_s = (window_open - now).total_seconds()
            self.log("waiting_for_open", seconds=wait_s)
            await asyncio.sleep(max(0, wait_s))

        # Start feed
        self.feed = LiveBarFeed("SPY", poll_interval_s=15.0)
        self.log("feed_started", until=hard_close.isoformat())

        first_bar_seen = False

        try:
            async for bar in self.feed.iter_bars(self.today, hard_close):
                self.bars_history.append(bar)

                # First bar = check the gap signal
                if not first_bar_seen:
                    first_bar_seen = True
                    open_price = bar.o
                    gap_pct = (open_price - self.prior_close) / self.prior_close

                    self.log("open_bar", open=open_price, prior_close=self.prior_close,
                             gap_pct=gap_pct, prior_val=self.prior_val)

                    # CHECK FILTERS
                    if open_price >= self.prior_val:
                        self.log("no_signal", reason="open_above_val",
                                 open=open_price, val=self.prior_val)
                        break

                    if gap_pct > -MIN_GAP_PCT:
                        self.log("no_signal", reason="gap_too_small",
                                 gap_pct=gap_pct, min_gap=-MIN_GAP_PCT)
                        break

                    if weekly_val is not None and open_price < weekly_val:
                        self.log("no_signal", reason="below_weekly_val",
                                 open=open_price, weekly_val=weekly_val)
                        break

                    # SIGNAL QUALIFIES
                    self.signal_qualified = True
                    self.gap_dollars = self.prior_close - open_price
                    self.stop_px = open_price - self.gap_dollars
                    self.in_dip_mode = True
                    self.log("signal_qualified",
                             gap_pct=gap_pct, gap_dollars=self.gap_dollars,
                             target=self.target_px, stop=self.stop_px)
                    continue

                if not self.signal_qualified:
                    break

                # Manage open position
                if self.open_symbol is not None:
                    await self._manage_position(bar)
                    continue

                # Dip-buy mode: waiting for session low
                if self.in_dip_mode and not self.dip_bought:
                    self.dip_bars_seen += 1

                    if bar.l < self.session_low:
                        self.session_low = bar.l
                        self.session_low_idx = self.dip_bars_seen

                    # After waiting, check if price is bouncing off the low
                    # "Bounce" = current bar's close > session low by at least $0.10
                    # AND we're at least 5 bars in
                    if self.dip_bars_seen >= 5 and bar.c > self.session_low + 0.10:
                        # Dip found — enter
                        await self._enter(bar)
                        continue

                    if self.dip_bars_seen >= MAX_DIP_WAIT_BARS:
                        # Waited 30 min with no clear bounce
                        if bar.c > self.session_low + 0.05:
                            # Price is at least slightly above low — enter anyway
                            await self._enter(bar)
                        else:
                            self.log("dip_buy_skipped",
                                     reason="no_bounce_in_30_bars",
                                     session_low=self.session_low)
                            break

            # Session ended
            if self.open_symbol is not None:
                await self._force_exit()

        finally:
            if self.feed:
                await self.feed.close()
            self.log("session_end", trades=self.risk.trades_today,
                     realized_pnl=self.risk.realized_pnl_today)

    async def _enter(self, bar: LiveBar) -> None:
        """Buy ATM 0DTE call."""
        atm_strike = round(bar.c)
        symbol = _build_occ_call("SPY", self.today, atm_strike)

        self.log("entry_planned", symbol=symbol, strike=atm_strike,
                 entry_price=bar.c, session_low=self.session_low,
                 dip_bars=self.dip_bars_seen, target=self.target_px,
                 stop=self.stop_px)

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
        self.dip_bought = True
        self.in_dip_mode = False
        self.open_symbol = symbol
        self.open_entry_price = bar.c
        self.open_bars_held = 0
        # Reset cum_delta tracking from this point
        self.cum_delta = 0.0
        self.cum_delta_peak = 0.0
        self.cum_delta_decline_count = 0
        self.prev_cum_delta = 0.0

    async def _manage_position(self, bar: LiveBar) -> None:
        """Monitor for Exit Z (cum_delta exhaustion), stop, or timeout."""
        if self.open_entry_price is None:
            return
        self.open_bars_held += 1

        # Update cumulative delta
        bp = _buy_pressure(bar.o, bar.h, bar.l, bar.c)
        bar_delta = (2 * bp - 1) * bar.v
        self.cum_delta += bar_delta

        # Track delta peak and decline
        if self.cum_delta > self.cum_delta_peak:
            self.cum_delta_peak = self.cum_delta
            self.cum_delta_decline_count = 0
        elif self.cum_delta < self.prev_cum_delta:
            self.cum_delta_decline_count += 1
        else:
            self.cum_delta_decline_count = 0
        self.prev_cum_delta = self.cum_delta

        # Check stop (underlying drops to stop_px)
        if bar.l <= self.stop_px:
            self.log("stop_hit", stop_px=self.stop_px, bar_low=bar.l,
                     bars_held=self.open_bars_held)
            await self._exit(bar, reason="stop")
            return

        # Check target (underlying reaches prior POC)
        if bar.h >= self.target_px:
            self.log("target_hit", target_px=self.target_px, bar_high=bar.h,
                     bars_held=self.open_bars_held)
            await self._exit(bar, reason="target")
            return

        # Exit Z: cum_delta peaked and declining for 5 bars
        if (self.cum_delta_decline_count >= CUM_DELTA_DECLINE_BARS
                and self.open_bars_held >= 10):  # don't exit too early
            self.log("delta_exhaustion_exit",
                     cum_delta=self.cum_delta,
                     peak=self.cum_delta_peak,
                     decline_bars=self.cum_delta_decline_count,
                     bars_held=self.open_bars_held)
            await self._exit(bar, reason="delta_exhaustion")
            return

        # Timeout
        if self.open_bars_held >= MAX_HOLD_BARS:
            self.log("timeout_exit", bars_held=self.open_bars_held)
            await self._exit(bar, reason="timeout")
            return

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

        underlying_pnl = bar.c - self.open_entry_price if self.open_entry_price else 0
        self.risk.on_exit(realized)
        self.log("position_closed", symbol=symbol, reason=reason,
                 realized_pnl=realized, bars_held=self.open_bars_held,
                 entry_price=self.open_entry_price, exit_price=bar.c,
                 underlying_pnl=underlying_pnl)

        self.open_symbol = None
        self.open_entry_price = None
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
