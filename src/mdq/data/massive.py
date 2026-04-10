"""Massive.com (ex-Polygon) async REST client for aggregate bars.

Thin wrapper around httpx.AsyncClient. Handles auth, pagination via `next_url`,
and retries on transient errors. Returns raw JSON rows (list of dicts) so the
caller can hand them straight to pandas without per-row Python object overhead.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from mdq.config import Settings


class MassiveClient:
    """Async REST client for Massive aggregate bars.

    Use as an async context manager:
        async with MassiveClient() as client:
            rows = await client.fetch_aggs("SPY", 1, "minute", "2024-01-01", "2024-01-31")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        max_concurrency: int = 16,
    ):
        self.settings = settings or Settings.from_env()
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.settings.api_base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
            limits=httpx.Limits(
                max_connections=max_concurrency * 2,
                max_keepalive_connections=max_concurrency * 2,
            ),
        )

    async def __aenter__(self) -> "MassiveClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_aggs(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> list[dict[str, Any]]:
        """Fetch all aggregate rows across pages for a single range.

        Returns a list of raw JSON dicts with keys {t,o,h,l,c,v,vw,n}.
        """
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params: dict[str, Any] = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
            "apiKey": self.settings.api_key,
        }

        all_rows: list[dict[str, Any]] = []
        url: str = path
        query: dict | None = params

        while True:
            data = await self._get_with_retries(url, query)
            query = None  # next_url has its own query

            results = data.get("results") or []
            all_rows.extend(results)

            next_url = data.get("next_url")
            if not next_url:
                break
            sep = "&" if "?" in next_url else "?"
            url = f"{next_url}{sep}apiKey={self.settings.api_key}"

        return all_rows

    async def _get_with_retries(self, url: str, params: dict | None) -> dict:
        last_exc: Exception | None = None
        async with self._sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await self._client.get(url, params=params)
                    if resp.status_code == 429:
                        await asyncio.sleep(2**attempt)
                        continue
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    last_exc = e
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"Request failed after {self.max_retries} retries: {last_exc}")
