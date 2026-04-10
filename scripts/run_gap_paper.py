"""Entry point: run the SPY gap-reversion CZ paper trader.

Usage:
    PYTHONPATH=src caffeinate -i python3 scripts/run_gap_paper.py
    PYTHONPATH=src python3 scripts/run_gap_paper.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio

from mdq.live.gap_runner import GapReversionRunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--contracts", type=int, default=1)
    args = p.parse_args()

    runner = GapReversionRunner(contracts=args.contracts, dry_run=args.dry_run)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
