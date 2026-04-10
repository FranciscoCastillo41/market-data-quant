"""Entry point: run the Tier 1 live paper trader for today's session.

Usage:
    poetry run python3 scripts/run_live_paper.py              # real paper trading
    poetry run python3 scripts/run_live_paper.py --dry-run    # no orders placed
"""

from __future__ import annotations

import argparse
import asyncio

from mdq.live.runner import LiveRunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="run the full loop but do not place any orders")
    p.add_argument("--ticker", default="QQQ")
    p.add_argument("--contracts", type=int, default=1)
    args = p.parse_args()

    runner = LiveRunner(
        ticker=args.ticker,
        contracts=args.contracts,
        dry_run=args.dry_run,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
