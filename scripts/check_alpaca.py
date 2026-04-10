"""Alpaca paper trading sanity check.

Does FOUR things, no orders placed:
  1. Confirm we can auth and read the paper account.
  2. Fetch and print QQQ 0DTE put chain snapshot around the current price.
  3. Print the 5 strikes closest to ATM with bid/ask/delta/IV.
  4. Summarize observed spread (which is the real-world slippage we care about).

Exits non-zero if anything fails, so you can wire this into a pre-market cron
check later.
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from mdq.data.bars import load_bars
from mdq.live.alpaca import AlpacaClient


def _nearest_qqq_price() -> float | None:
    """Pick a reasonable reference price for QQQ.

    Uses the most recent cached 1-min bar close if available.
    """
    try:
        today = date.today()
        start = (pd.Timestamp(today) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        bars = load_bars("QQQ", start, today.strftime("%Y-%m-%d"))
        if bars.empty:
            return None
        return float(bars.iloc[-1]["c"])
    except Exception:
        return None


def main() -> int:
    print("=" * 70)
    print("Alpaca paper trading sanity check")
    print("=" * 70)

    try:
        client = AlpacaClient()
    except Exception as e:
        print(f"FAIL: could not construct AlpacaClient: {e}")
        return 1

    # 1. Account
    print("\n[1/4] Account info")
    try:
        acct = client.get_account()
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1
    print(f"  account_number      : {acct.account_number}")
    print(f"  status              : {acct.status}")
    print(f"  cash                : ${acct.cash:>12,.2f}")
    print(f"  portfolio_value     : ${acct.portfolio_value:>12,.2f}")
    print(f"  buying_power        : ${acct.buying_power:>12,.2f}")
    print(f"  options_level       : {acct.options_trading_level}")
    print(f"  options_bp          : {acct.options_buying_power}")

    if acct.options_trading_level is None or acct.options_trading_level < 2:
        print(
            "  WARN: options trading level is below 2 — "
            "long calls/puts may be rejected. Check dashboard."
        )

    # 2. Reference QQQ price
    print("\n[2/4] Reference QQQ price")
    qqq_ref = _nearest_qqq_price()
    if qqq_ref is None:
        print("  No cached QQQ bars; chain fetch will not be strike-filtered.")
        strike_lo = None
        strike_hi = None
    else:
        print(f"  last cached QQQ close: ${qqq_ref:.2f}")
        strike_lo = round(qqq_ref - 5)
        strike_hi = round(qqq_ref + 5)
        print(f"  will fetch puts in strike range [{strike_lo}, {strike_hi}]")

    # 3. Fetch today's 0DTE put chain
    print("\n[3/4] Fetching QQQ 0DTE put chain snapshot")
    today = date.today()
    try:
        puts = client.get_chain_snapshot(
            underlying="QQQ",
            contract_type="put",
            expiration=today,
            strike_min=strike_lo,
            strike_max=strike_hi,
        )
    except Exception as e:
        print(f"  FAIL: {e}")
        print(
            "\n  Possible causes:\n"
            "   - Market is closed and there's no 0DTE today (QQQ has daily expirations\n"
            "     Mon-Fri, so this should only fail on weekends/holidays).\n"
            "   - Your paper account does not have the option data subscription.\n"
            "   - Options Trading Level is not enabled.\n"
        )
        return 1

    print(f"  fetched {len(puts)} put contracts for expiration {today}")

    if not puts:
        print("  No contracts returned. Likely weekend / holiday / no 0DTE today.")
        print("  Try again on a weekday morning.")
        return 0

    # 4. Show the 5 closest-to-ATM strikes with bid/ask/delta/IV
    print("\n[4/4] ATM strikes — bid/ask/delta/IV/spread")
    if qqq_ref is not None:
        puts_sorted = sorted(puts, key=lambda c: abs(c.strike - qqq_ref))[:5]
        puts_sorted = sorted(puts_sorted, key=lambda c: c.strike)
    else:
        puts_sorted = puts[:5]

    def _fmt(v, width=8, prec=4):
        if v is None:
            return "—".rjust(width)
        return f"{v:>{width}.{prec}f}"

    header = (
        f"  {'strike':>7}  {'bid':>8}  {'ask':>8}  {'spread':>8}  "
        f"{'mid':>8}  {'delta':>8}  {'iv':>8}  symbol"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    spreads: list[float] = []
    for c in puts_sorted:
        sp = c.spread
        if sp is not None:
            spreads.append(sp)
        print(
            f"  {c.strike:>7.0f}  "
            f"{_fmt(c.bid)}  {_fmt(c.ask)}  {_fmt(sp)}  "
            f"{_fmt(c.mid)}  {_fmt(c.delta)}  {_fmt(c.iv)}  {c.symbol}"
        )

    if spreads:
        avg_spread = sum(spreads) / len(spreads)
        print(f"\n  avg bid-ask spread on shown strikes: ${avg_spread:.3f}")
        if avg_spread > 0.10:
            print("  NOTE: spread is wide — slippage will eat a lot of edge on 0DTE.")
        elif avg_spread > 0.05:
            print("  spread is moderate — in line with expectations for 0DTE QQQ.")
        else:
            print("  spread is tight — nice execution conditions.")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
