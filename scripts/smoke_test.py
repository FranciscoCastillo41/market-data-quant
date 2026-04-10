"""Smoke test: pull 1 week of SPY 1-min bars and report timing + shape."""

from __future__ import annotations

import time

from mdq.data.bars import download_range, load_bars


def main() -> None:
    t0 = time.perf_counter()
    paths = download_range("SPY", "2024-01-02", "2024-01-12", overwrite=True)
    dl_s = time.perf_counter() - t0
    print(f"Downloaded {len(paths)} month file(s) in {dl_s:.2f}s")

    t0 = time.perf_counter()
    df = load_bars("SPY", "2024-01-02", "2024-01-12")
    ld_s = time.perf_counter() - t0
    print(f"Loaded {len(df):,} bars in {ld_s:.2f}s")

    print("\nColumns:", list(df.columns))
    print("\nHead:")
    print(df.head(3))
    print("\nTail:")
    print(df.tail(3))
    print(f"\nUnique session dates: {df['session_date'].nunique()}")
    print(f"Bar range: {df['ts_et'].min()}  ->  {df['ts_et'].max()}")


if __name__ == "__main__":
    main()
