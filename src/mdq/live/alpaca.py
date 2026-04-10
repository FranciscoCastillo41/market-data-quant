"""Thin wrapper around alpaca-py for paper options trading.

Goals of this wrapper:
  - Own the two Alpaca SDK clients (trading + option market data)
  - Expose only the 5-6 methods our runner actually needs
  - Convert SDK response objects into plain dicts/dataclasses so nothing
    downstream imports from alpaca.* (isolates us from SDK churn)
  - Make dry-run + unit-test stubbing trivial later

No streaming, no websockets, no complex order types. REST + market orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType as AlpacaContractType
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
)

from mdq.config import AlpacaSettings


@dataclass(frozen=True)
class AccountInfo:
    account_number: str
    status: str
    buying_power: float
    cash: float
    portfolio_value: float
    options_trading_level: int | None
    options_buying_power: float | None


@dataclass(frozen=True)
class OptionContract:
    """One option contract snapshot with quote + greeks."""

    symbol: str           # OCC format, e.g. QQQ240419P00470000
    underlying: str
    contract_type: Literal["call", "put"]
    strike: float
    expiration: date
    bid: float | None
    ask: float | None
    bid_size: int | None
    ask_size: int | None
    last: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    iv: float | None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        if self.bid <= 0 or self.ask <= 0:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass(frozen=True)
class OrderResult:
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    status: str
    submitted_at: datetime | None
    filled_at: datetime | None
    filled_avg_price: float | None


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float | None
    unrealized_pl: float | None


class AlpacaClient:
    """Minimal paper-trading client for single-leg long options.

    Construct with `AlpacaClient()` — credentials are loaded from env.
    """

    def __init__(self, settings: AlpacaSettings | None = None):
        self.settings = settings or AlpacaSettings.from_env()
        self._trading = TradingClient(
            api_key=self.settings.api_key,
            secret_key=self.settings.api_secret,
            paper=self.settings.paper,
        )
        # Option historical/snapshot client doesn't take paper flag —
        # it uses the same data base URL regardless.
        self._option_data = OptionHistoricalDataClient(
            api_key=self.settings.api_key,
            secret_key=self.settings.api_secret,
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        a = self._trading.get_account()
        return AccountInfo(
            account_number=a.account_number,
            status=str(a.status),
            buying_power=float(a.buying_power),
            cash=float(a.cash),
            portfolio_value=float(a.portfolio_value),
            options_trading_level=getattr(a, "options_trading_level", None),
            options_buying_power=(
                float(a.options_buying_power)
                if getattr(a, "options_buying_power", None) is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    def get_chain_snapshot(
        self,
        underlying: str,
        contract_type: Literal["call", "put"],
        expiration: date,
        strike_min: float | None = None,
        strike_max: float | None = None,
    ) -> list[OptionContract]:
        """Fetch snapshot of contracts for one underlying / type / expiration.

        Returns a list of OptionContract with quotes + greeks for each
        strike in [strike_min, strike_max]. If bounds are None, no filter.
        """
        req = OptionChainRequest(
            underlying_symbol=underlying,
            type=AlpacaContractType.CALL if contract_type == "call" else AlpacaContractType.PUT,
            expiration_date=expiration,
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
        )
        snapshots = self._option_data.get_option_chain(req)
        # snapshots is a dict: {symbol: OptionSnapshot}
        out: list[OptionContract] = []
        for symbol, snap in snapshots.items():
            parsed = _parse_occ_symbol(symbol)
            if parsed is None:
                continue
            under, exp, c_type, strike = parsed
            quote = snap.latest_quote
            trade = snap.latest_trade
            greeks = snap.greeks
            out.append(
                OptionContract(
                    symbol=symbol,
                    underlying=under,
                    contract_type=c_type,
                    strike=strike,
                    expiration=exp,
                    bid=float(quote.bid_price) if quote and quote.bid_price is not None else None,
                    ask=float(quote.ask_price) if quote and quote.ask_price is not None else None,
                    bid_size=int(quote.bid_size) if quote and quote.bid_size is not None else None,
                    ask_size=int(quote.ask_size) if quote and quote.ask_size is not None else None,
                    last=float(trade.price) if trade and trade.price is not None else None,
                    delta=float(greeks.delta) if greeks and greeks.delta is not None else None,
                    gamma=float(greeks.gamma) if greeks and greeks.gamma is not None else None,
                    theta=float(greeks.theta) if greeks and greeks.theta is not None else None,
                    vega=float(greeks.vega) if greeks and greeks.vega is not None else None,
                    iv=float(snap.implied_volatility) if snap.implied_volatility is not None else None,
                )
            )
        return sorted(out, key=lambda c: c.strike)

    def get_option_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Fetch 1-minute historical bars for one option contract symbol.

        Works for already-expired same-day contracts (useful for end-of-day
        signal replays).
        """
        req = OptionBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        bars_resp = self._option_data.get_option_bars(req)
        # Response is a BarSet-like dict keyed by symbol
        raw = bars_resp.data.get(symbol, []) if hasattr(bars_resp, "data") else []
        out: list[dict] = []
        for b in raw:
            out.append({
                "t": b.timestamp,
                "o": float(b.open),
                "h": float(b.high),
                "l": float(b.low),
                "c": float(b.close),
                "v": int(b.volume) if b.volume is not None else 0,
                "vw": float(b.vwap) if b.vwap is not None else None,
                "n": int(b.trade_count) if b.trade_count is not None else None,
            })
        return out

    def list_contracts(
        self,
        underlying: str,
        expiration: date | None = None,
    ) -> list[dict]:
        """List available contracts (no quotes/greeks). Useful for discovery."""
        req_kwargs = {"underlying_symbols": [underlying]}
        if expiration is not None:
            req_kwargs["expiration_date"] = expiration
        req = GetOptionContractsRequest(**req_kwargs)
        resp = self._trading.get_option_contracts(req)
        return [
            {
                "symbol": c.symbol,
                "underlying": c.underlying_symbol,
                "type": str(c.type),
                "strike": float(c.strike_price),
                "expiration": c.expiration_date,
            }
            for c in resp.option_contracts
        ]

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def buy_option_market(self, symbol: str, qty: int) -> OrderResult:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading.submit_order(req)
        return _order_to_result(order)

    def sell_option_market(self, symbol: str, qty: int) -> OrderResult:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading.submit_order(req)
        return _order_to_result(order)

    def get_order(self, order_id: str) -> OrderResult:
        return _order_to_result(self._trading.get_order_by_id(order_id))

    def cancel_all_orders(self) -> None:
        self._trading.cancel_orders()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Position | None:
        try:
            p = self._trading.get_open_position(symbol)
        except Exception:
            return None
        return Position(
            symbol=p.symbol,
            qty=int(float(p.qty)),
            avg_entry_price=float(p.avg_entry_price),
            current_price=float(p.current_price) if p.current_price is not None else None,
            unrealized_pl=float(p.unrealized_pl) if p.unrealized_pl is not None else None,
        )

    def list_positions(self) -> list[Position]:
        out: list[Position] = []
        for p in self._trading.get_all_positions():
            out.append(
                Position(
                    symbol=p.symbol,
                    qty=int(float(p.qty)),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price) if p.current_price is not None else None,
                    unrealized_pl=float(p.unrealized_pl) if p.unrealized_pl is not None else None,
                )
            )
        return out

    def close_position(self, symbol: str) -> OrderResult | None:
        """Market-close a position. Returns the order, or None if nothing to close."""
        pos = self.get_position(symbol)
        if pos is None or pos.qty == 0:
            return None
        return self.sell_option_market(symbol, qty=pos.qty)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _parse_occ_symbol(symbol: str) -> tuple[str, date, Literal["call", "put"], float] | None:
    """Parse OCC option symbol, e.g. 'QQQ240419P00470000'.

    Returns (underlying, expiration, contract_type, strike_price) or None
    if parsing fails.
    """
    if len(symbol) < 15:
        return None
    # OCC: underlying is variable-length, then 6-digit YYMMDD, 1 char C/P, 8-digit strike*1000
    try:
        strike_str = symbol[-8:]
        c_char = symbol[-9]
        date_str = symbol[-15:-9]
        underlying = symbol[:-15]

        strike = int(strike_str) / 1000.0
        year = 2000 + int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        exp = date(year, month, day)
        c_type: Literal["call", "put"] = "call" if c_char == "C" else "put"
        if c_char not in ("C", "P"):
            return None
        return underlying, exp, c_type, strike
    except (ValueError, IndexError):
        return None


def _order_to_result(order) -> OrderResult:
    return OrderResult(
        id=str(order.id),
        symbol=order.symbol,
        side="buy" if order.side == OrderSide.BUY else "sell",
        qty=int(float(order.qty)),
        status=str(order.status),
        submitted_at=order.submitted_at,
        filled_at=order.filled_at,
        filled_avg_price=(
            float(order.filled_avg_price) if order.filled_avg_price is not None else None
        ),
    )
