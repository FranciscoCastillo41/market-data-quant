"""Async bar fetching with parquet caching.

One parquet file per (ticker, year-month). Idempotent — skips existing files
unless overwrite=True. Downloads run concurrently via the MassiveClient's
internal semaphore.

Reads (load_bars) remain synchronous: parquet IO is fast and local.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from mdq.config import RAW_DIR
from mdq.data.calendar import add_session_date, to_et
from mdq.data.massive import MassiveClient

_EXPECTED_COLS = ["t", "o", "h", "l", "c", "v", "vw", "n"]


def _month_path(ticker: str, year: int, month: int) -> Path:
    return RAW_DIR / ticker / f"{year:04d}-{month:02d}.parquet"


def _month_range(start: str, end: str) -> list[tuple[int, int]]:
    start_ts = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end)
    out: list[tuple[int, int]] = []
    cur = start_ts
    while cur <= end_ts:
        out.append((cur.year, cur.month))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


async def _download_one_month(
    client: MassiveClient,
    ticker: str,
    year: int,
    month: int,
    overwrite: bool,
) -> Path:
    out_path = _month_path(ticker, year, month)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from_date = f"{year:04d}-{month:02d}-01"
    to_date = (
        (pd.Timestamp(from_date) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    )

    rows = await client.fetch_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="minute",
        from_date=from_date,
        to_date=to_date,
    )

    if not rows:
        df = pd.DataFrame(columns=_EXPECTED_COLS)
    else:
        df = pd.DataFrame(rows)
        # Ensure consistent column set and order (API may omit vw/n)
        for col in _EXPECTED_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[_EXPECTED_COLS].sort_values("t").reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    return out_path


async def download_range_async(
    ticker: str,
    start: str,
    end: str,
    overwrite: bool = False,
    max_concurrency: int = 16,
) -> list[Path]:
    """Concurrently download a date range month-by-month.

    `start`/`end` are YYYY-MM-DD (inclusive). Concurrency is bounded by the
    client's semaphore.
    """
    months = _month_range(start, end)
    async with MassiveClient(max_concurrency=max_concurrency) as client:
        tasks = [
            _download_one_month(client, ticker, y, m, overwrite)
            for (y, m) in months
        ]
        return await asyncio.gather(*tasks)


def download_range(
    ticker: str,
    start: str,
    end: str,
    overwrite: bool = False,
    max_concurrency: int = 16,
) -> list[Path]:
    """Sync wrapper around download_range_async for scripts and notebooks."""
    return asyncio.run(
        download_range_async(ticker, start, end, overwrite, max_concurrency)
    )


async def download_many_tickers_async(
    tickers: list[str],
    start: str,
    end: str,
    overwrite: bool = False,
    max_concurrency: int = 16,
) -> dict[str, list[Path]]:
    """Download multiple tickers concurrently under a single shared client.

    This maximizes throughput by letting all month requests across all tickers
    compete for the same semaphore pool.
    """
    months = _month_range(start, end)
    async with MassiveClient(max_concurrency=max_concurrency) as client:
        coros = {
            ticker: [
                _download_one_month(client, ticker, y, m, overwrite)
                for (y, m) in months
            ]
            for ticker in tickers
        }
        flat_tasks: list = []
        index: list[tuple[str, int]] = []
        for ticker, task_list in coros.items():
            for i, t in enumerate(task_list):
                flat_tasks.append(t)
                index.append((ticker, i))

        results = await asyncio.gather(*flat_tasks)

        out: dict[str, list[Path]] = {t: [None] * len(months) for t in tickers}  # type: ignore
        for (ticker, i), path in zip(index, results):
            out[ticker][i] = path
        return out


def download_many_tickers(
    tickers: list[str],
    start: str,
    end: str,
    overwrite: bool = False,
    max_concurrency: int = 16,
) -> dict[str, list[Path]]:
    """Sync wrapper."""
    return asyncio.run(
        download_many_tickers_async(
            tickers, start, end, overwrite, max_concurrency
        )
    )


def load_bars(
    ticker: str,
    start: str,
    end: str,
    tz_aware: bool = True,
) -> pd.DataFrame:
    """Load cached bars for a date range.

    Returns a DataFrame with columns:
        t (ms), o, h, l, c, v, vw, n
    If tz_aware: adds ts_et (tz-aware datetime in ET) and session_date.
    """
    months = _month_range(start, end)
    frames: list[pd.DataFrame] = []
    for (y, m) in months:
        p = _month_path(ticker, y, m)
        if p.exists():
            frames.append(pd.read_parquet(p))

    if not frames:
        raise FileNotFoundError(
            f"No cached bars for {ticker} in {start}..{end}. "
            "Run download_range first."
        )

    df = pd.concat(frames, ignore_index=True)
    # Clip to exact date range (inclusive of end date)
    start_ms = int(
        pd.Timestamp(start).tz_localize("America/New_York").timestamp() * 1000
    )
    end_ms = int(
        (pd.Timestamp(end) + pd.Timedelta(days=1))
        .tz_localize("America/New_York")
        .timestamp()
        * 1000
    )
    df = df[(df["t"] >= start_ms) & (df["t"] < end_ms)].reset_index(drop=True)

    if tz_aware:
        df = to_et(df)
        df = add_session_date(df)

    return df
