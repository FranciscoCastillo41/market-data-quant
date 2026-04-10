"""Post-session report for the Tier 1 paper trader.

Reads the day's JSONL journal and prints a clean one-page summary:
  - Did the signal fire?
  - Trade details (entry, exit, fill prices, P&L)
  - vs backtest expectation
  - Running totals across all paper sessions to date

Usage:
    poetry run python3 scripts/session_report.py                  # today
    poetry run python3 scripts/session_report.py --date 2026-04-09
    poetry run python3 scripts/session_report.py --all            # full history
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from mdq.config import RESULTS_DIR

JOURNAL_DIR = RESULTS_DIR / "live" / "tier1"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--all", action="store_true", help="Show full paper trading history")
    return p.parse_args()


def load_journal(session_date: date) -> list[dict]:
    path = JOURNAL_DIR / f"{session_date.strftime('%Y-%m-%d')}.jsonl"
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def extract_trade(events: list[dict]) -> dict | None:
    """Find entry + exit + outcome from a session's events."""
    entry = None
    exit_ = None
    closed = None
    for e in events:
        if e["event"] == "entry_planned":
            entry = e
        elif e["event"] in ("target_hit", "stop_hit", "timeout_exit", "hard_close_exit"):
            exit_ = e
        elif e["event"] == "position_closed":
            closed = e
    if entry is None:
        return None
    return {"entry": entry, "exit": exit_, "closed": closed}


def report_one_session(session_date: date) -> dict:
    events = load_journal(session_date)
    out = {"session_date": session_date, "n_events": len(events), "fired": False,
           "pnl": 0.0, "outcome": None}

    if not events:
        return out

    # Find touches & signals
    touches = [e for e in events if e["event"] == "touch"]
    signals = [e for e in events if e["event"] == "entry_planned"]
    out["n_touches"] = len(touches)

    if not signals:
        return out

    out["fired"] = True
    trade = extract_trade(events)
    if trade and trade["closed"]:
        closed = trade["closed"]
        out["outcome"] = closed.get("reason")
        out["pnl"] = float(closed.get("realized_pnl", 0.0))
        out["entry_under"] = closed.get("entry_under")
        out["exit_under"] = closed.get("exit_under")
        out["symbol"] = closed.get("symbol")
        out["bars_held"] = closed.get("bars_held")

    return out


def print_session_detail(session_date: date) -> None:
    events = load_journal(session_date)
    print("=" * 70)
    print(f"Session report — {session_date}")
    print("=" * 70)

    if not events:
        print("\nNo journal file for this date. Either:")
        print("  - The runner was not started")
        print(f"  - The journal is at a different path (checked {JOURNAL_DIR})")
        return

    # Print timeline of interesting events
    keep_events = {
        "session_start", "prior_low_loaded", "account", "touch",
        "entry_planned", "entry_order_submitted", "entry_order_error",
        "target_hit", "stop_hit", "timeout_exit", "hard_close_exit",
        "position_closed", "exit_order_submitted", "exit_order_error",
        "signal_blocked", "signal_skipped", "session_end", "abort",
    }
    print("\nTimeline:")
    for e in events:
        if e["event"] not in keep_events:
            continue
        ts = e.get("ts", "")
        # Trim to HH:MM:SS
        if "T" in ts:
            ts = ts.split("T")[1][:8]
        data = {k: v for k, v in e.items() if k not in ("ts", "event")}
        print(f"  {ts}  {e['event']:<25}  {data}")

    # Summary
    summ = report_one_session(session_date)
    print("\nSummary:")
    print(f"  events logged:   {summ['n_events']}")
    print(f"  touches seen:    {summ.get('n_touches', 0)}")
    print(f"  signal fired:    {summ['fired']}")
    if summ["fired"]:
        print(f"  outcome:         {summ.get('outcome')}")
        print(f"  symbol:          {summ.get('symbol')}")
        print(f"  entry QQQ:       ${summ.get('entry_under', 0):.2f}")
        print(f"  exit QQQ:        ${summ.get('exit_under', 0):.2f}")
        print(f"  bars held:       {summ.get('bars_held')}")
        print(f"  realized P&L:    ${summ.get('pnl', 0):+.2f}")


def print_all_sessions() -> None:
    if not JOURNAL_DIR.exists():
        print(f"No journal dir at {JOURNAL_DIR}")
        return

    files = sorted(JOURNAL_DIR.glob("*.jsonl"))
    if not files:
        print("No paper sessions recorded yet.")
        return

    rows = []
    for f in files:
        sd = datetime.strptime(f.stem, "%Y-%m-%d").date()
        summ = report_one_session(sd)
        rows.append(summ)

    df = pd.DataFrame(rows)

    print("=" * 70)
    print(f"Paper trading history — {len(df)} sessions")
    print("=" * 70)

    fired = df[df["fired"]]
    print(f"\nSessions with signal:  {len(fired)} / {len(df)}")
    if not fired.empty:
        print(f"Cumulative P&L:        ${fired['pnl'].sum():+.2f}")
        print(f"Mean P&L / trade:      ${fired['pnl'].mean():+.2f}")
        wins = (fired["pnl"] > 0).sum()
        print(f"Wins / trades:         {wins} / {len(fired)}")
        if len(fired) > 0:
            print(f"Win rate:              {wins / len(fired):.0%}")

    print("\nPer session:")
    display_cols = ["session_date", "fired", "outcome", "pnl"]
    view = df[display_cols].copy()
    view["pnl"] = view["pnl"].map(lambda x: f"${x:+.0f}" if x != 0 else "—")
    view["outcome"] = view["outcome"].fillna("—")
    print(view.to_string(index=False))

    print("\nvs backtest expectation:")
    print("  backtest mean P&L:  +$17.64/trade")
    print("  backtest win rate:  64%")
    if not fired.empty:
        obs_mean = fired["pnl"].mean()
        obs_wr = (fired["pnl"] > 0).mean()
        print(f"  observed mean:      ${obs_mean:+.2f}/trade  (diff: {obs_mean - 17.64:+.2f})")
        print(f"  observed win rate:  {obs_wr:.0%}  (diff: {(obs_wr - 0.64) * 100:+.1f} pp)")


def main() -> int:
    args = _parse_args()

    if args.all:
        print_all_sessions()
        return 0

    target = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else date.today()
    )
    print_session_detail(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
