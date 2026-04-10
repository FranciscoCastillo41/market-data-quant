"""Download SPY + QQQ 1-min bars for the research window.

Default: 2023-01-01 through yesterday. Cached per (ticker, month) as parquet.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from mdq.data.bars import download_many_tickers, load_bars


def parse_args() -> argparse.Namespace:
    yesterday = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=yesterday)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Pulling {args.tickers} 1-min bars {args.start} -> {args.end} "
        f"(concurrency={args.concurrency}, overwrite={args.overwrite})"
    )

    t0 = time.perf_counter()
    paths_by_ticker = download_many_tickers(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
        max_concurrency=args.concurrency,
    )
    dl_s = time.perf_counter() - t0
    total_months = sum(len(v) for v in paths_by_ticker.values())
    print(f"\nDownloaded {total_months} month files in {dl_s:.2f}s "
          f"({total_months / max(dl_s, 1e-9):.1f} files/s)")

    print("\nVerifying cache by loading each ticker...")
    for ticker in args.tickers:
        t0 = time.perf_counter()
        df = load_bars(ticker, args.start, args.end)
        load_s = time.perf_counter() - t0
        print(
            f"  {ticker}: {len(df):>10,} bars   "
            f"{df['session_date'].nunique():>4} sessions   "
            f"{df['ts_et'].min().date()} -> {df['ts_et'].max().date()}   "
            f"(loaded in {load_s:.2f}s)"
        )


if __name__ == "__main__":
    main()
