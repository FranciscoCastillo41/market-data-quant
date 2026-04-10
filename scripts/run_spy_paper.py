"""Entry point: run the SPY S3 POC paper trader for today's session.

Usage:
    PYTHONPATH=src python3 scripts/run_spy_paper.py
    PYTHONPATH=src python3 scripts/run_spy_paper.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio

from mdq.live.spy_runner import SpyLiveRunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--contracts", type=int, default=1)
    args = p.parse_args()

    runner = SpyLiveRunner(contracts=args.contracts, dry_run=args.dry_run)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
