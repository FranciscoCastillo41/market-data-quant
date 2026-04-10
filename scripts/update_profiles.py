"""Nightly profile refresh: append today's QQQ volume profile to the cache.

Run this AFTER market close (4 PM ET or later) so we have the full RTH
session to compute the profile from. Writes to:

    data/results/experiment_c/profiles__QQQ.parquet

The runner reads from that file every morning to get yesterday's prior_low.
Without this script running daily, the runner uses stale data.

Usage:
    poetry run python3 scripts/update_profiles.py                 # today
    poetry run python3 scripts/update_profiles.py --date 2026-04-08
    poetry run python3 scripts/update_profiles.py --ticker QQQ SPY
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import download_range_async, load_bars
from mdq.levels.volume_profile import compute_session_profile
from mdq.data.calendar import filter_rth

OUT_DIR = RESULTS_DIR / "experiment_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _profile_path(ticker: str) -> Path:
    return OUT_DIR / f"profiles__{ticker}.parquet"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--ticker", nargs="+", default=["QQQ"], help="Tickers to update")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if the date is already in the cache")
    return p.parse_args()


def update_one(ticker: str, target: date, force: bool) -> bool:
    """Update profile cache for one ticker/date. Returns True if added/updated."""
    path = _profile_path(ticker)

    # Load existing cache (if any)
    if path.exists():
        existing = pd.read_parquet(path)
        existing["session_date"] = pd.to_datetime(existing["session_date"]).dt.date
        if target in set(existing["session_date"]) and not force:
            print(f"  [{ticker}] {target} already in cache, skipping (use --force to recompute)")
            return False
    else:
        existing = pd.DataFrame(columns=[
            "session_date", "poc", "vah", "val", "high", "low", "close", "total_volume"
        ])

    # Pull fresh bars for target date (bypass cache if forcing)
    d_str = target.strftime("%Y-%m-%d")
    print(f"  [{ticker}] pulling bars for {d_str} from Massive...")
    asyncio.run(
        download_range_async(
            ticker=ticker,
            start=d_str,
            end=d_str,
            overwrite=True,
        )
    )

    try:
        bars = load_bars(ticker, d_str, d_str)
    except FileNotFoundError as e:
        print(f"  [{ticker}] FAIL: {e}")
        return False

    if bars.empty:
        print(f"  [{ticker}] no bars returned (market closed? holiday? weekend?)")
        return False

    # Filter to RTH and compute volume profile
    rth = filter_rth(bars)
    if rth.empty:
        print(f"  [{ticker}] no RTH bars (got extended hours only — market was closed)")
        return False

    prof = compute_session_profile(rth)
    if prof is None:
        print(f"  [{ticker}] profile computation returned None")
        return False

    row = {
        "session_date": prof.session_date,
        "poc": prof.poc,
        "vah": prof.vah,
        "val": prof.val,
        "high": prof.high,
        "low": prof.low,
        "close": prof.close,
        "total_volume": prof.total_volume,
    }
    print(f"  [{ticker}] {target}  poc={prof.poc:.2f}  vah={prof.vah:.2f}  "
          f"val={prof.val:.2f}  high={prof.high:.2f}  low={prof.low:.2f}")

    # Append or update
    existing = existing[existing["session_date"] != target]
    new = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    new = new.sort_values("session_date").reset_index(drop=True)
    new.to_parquet(path, index=False)
    print(f"  [{ticker}] wrote {len(new)} rows to {path}")
    return True


def main() -> int:
    args = _parse_args()
    target = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else date.today()
    )

    print("=" * 70)
    print(f"Profile update for {target}")
    print("=" * 70)

    # Skip weekends early
    if target.weekday() >= 5:
        print(f"{target} is a weekend ({target.strftime('%A')}). No RTH data. Exiting.")
        return 0

    updated = 0
    for ticker in args.ticker:
        if update_one(ticker, target, force=args.force):
            updated += 1

    print(f"\nDone. {updated}/{len(args.ticker)} tickers updated.")
    return 0 if updated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
