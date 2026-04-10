"""Append-only trade journal as JSONL.

Every event (signal fired, order submitted, fill received, exit, end-of-session)
is recorded as a JSON line with a timestamp. Simple, greppable, crash-resistant.

Why JSONL instead of parquet: append-only writes are atomic per line, no risk
of corrupting a parquet file if the runner crashes mid-write.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mdq.config import RESULTS_DIR


def _default_encoder(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class Journal:
    """Append-only JSONL event log."""

    def __init__(self, session_date: date, name: str = "tier1"):
        self.dir = RESULTS_DIR / "live" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{session_date.strftime('%Y-%m-%d')}.jsonl"

    def write(self, event_type: str, **fields: Any) -> None:
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            **fields,
        }
        line = json.dumps(record, default=_default_encoder)
        with self.path.open("a") as f:
            f.write(line + "\n")
