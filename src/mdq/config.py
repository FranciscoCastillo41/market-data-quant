"""Configuration: env loading, paths, and shared constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results"

for _d in (RAW_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_base_url: str = "https://api.massive.com"

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key found. Set MASSIVE_API_KEY (or POLYGON_API_KEY) in .env."
            )
        return cls(api_key=key)


@dataclass(frozen=True)
class AlpacaSettings:
    """Alpaca paper trading credentials. Paper mode is hardcoded — we never
    want this flippable from config alone."""

    api_key: str
    api_secret: str
    paper: bool = True

    @classmethod
    def from_env(cls) -> "AlpacaSettings":
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in .env."
            )
        if not key.startswith("PK"):
            raise RuntimeError(
                f"ALPACA_API_KEY does not look like a paper key "
                f"(should start with 'PK', got '{key[:4]}...'). "
                f"Regenerate from the paper dashboard."
            )
        return cls(api_key=key, api_secret=secret, paper=True)


# Shared constants
ET_TZ = "America/New_York"
PT_TZ = "America/Los_Angeles"

# Regular trading hours in ET
RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"

# Randy's morning window in ET (06:30-08:15 PST = 09:30-11:15 ET)
MORNING_OPEN = "09:30"
MORNING_CLOSE = "11:15"
