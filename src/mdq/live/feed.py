"""Real-time 1-minute bar polling from Massive.

Strategy: poll Massive for today's bars once per minute and yield any newly
closed bars since the last poll. This is NOT a websocket — it's REST polling.
Simpler, more reliable, and totally fine at 1-minute cadence.

Each poll fetches the full day's bars so far (cheap on Massive), dedupes
against what we've already seen, and yields only bars that are definitively
closed (their timestamp + 60s is in the past).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import AsyncIterator

import pandas as pd
import pytz

from mdq.data.calendar import add_session_date, to_et
from mdq.data.massive import MassiveClient

ET = pytz.timezone("America/New_York")


@dataclass(frozen=True)
class LiveBar:
    t: int           # unix ms
    ts_et: datetime  # tz-aware ET
    o: float
    h: float
    l: float
    c: float
    v: float


class LiveBarFeed:
    """Yields newly closed 1-minute bars for a ticker via Massive REST polling."""

    def __init__(
        self,
        ticker: str,
        poll_interval_s: float = 15.0,
    ):
        self.ticker = ticker
        self.poll_interval_s = poll_interval_s
        self._client = MassiveClient()
        self._seen_t: set[int] = set()

    async def close(self) -> None:
        await self._client.close()

    async def _fetch_today(self, target_date: date) -> list[LiveBar]:
        """Fetch all of today's bars so far (one round trip)."""
        d_str = target_date.strftime("%Y-%m-%d")
        rows = await self._client.fetch_aggs(
            ticker=self.ticker,
            multiplier=1,
            timespan="minute",
            from_date=d_str,
            to_date=d_str,
        )
        if not rows:
            return []
        df = pd.DataFrame(rows)
        df = df.sort_values("t").reset_index(drop=True)
        df = to_et(df)
        df = add_session_date(df)

        out: list[LiveBar] = []
        for _, r in df.iterrows():
            out.append(
                LiveBar(
                    t=int(r["t"]),
                    ts_et=r["ts_et"],
                    o=float(r["o"]),
                    h=float(r["h"]),
                    l=float(r["l"]),
                    c=float(r["c"]),
                    v=float(r["v"]),
                )
            )
        return out

    async def iter_bars(
        self,
        target_date: date,
        until_ts_et: datetime,
    ) -> AsyncIterator[LiveBar]:
        """Yield newly closed bars until `until_ts_et` is reached.

        A bar is considered closed when its timestamp + 60s is in the past.
        """
        while True:
            now_et = datetime.now(tz=ET)
            if now_et >= until_ts_et:
                return

            try:
                bars = await self._fetch_today(target_date)
            except Exception as e:
                print(f"  [feed] fetch error: {e}, retrying in {self.poll_interval_s}s")
                await asyncio.sleep(self.poll_interval_s)
                continue

            # Only yield bars whose close is confirmed (bar timestamp + 60s < now)
            cutoff_ms = int((now_et - timedelta(seconds=60)).timestamp() * 1000)
            new_bars: list[LiveBar] = []
            for b in bars:
                if b.t in self._seen_t:
                    continue
                if b.t > cutoff_ms:
                    continue  # bar not yet closed
                new_bars.append(b)
                self._seen_t.add(b.t)

            for b in new_bars:
                yield b

            await asyncio.sleep(self.poll_interval_s)
