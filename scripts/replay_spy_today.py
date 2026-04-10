"""Replay today's SPY session through the S3 POC detector.

Pulls fresh bars from Massive (so we get whatever has traded so far),
feeds them into the detector with yesterday's POC as the level, prints
every signal with timestamp, direction, entry price, and ATR.

Also simulates what each signal would have done on the underlying:
target hit / stop hit / timeout / still-open.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.data.bars import download_range_async, load_bars
from mdq.data.calendar import filter_window
from mdq.live.feed import LiveBar
from mdq.live.spy_signal import HORIZON_BARS, SpyS3PocDetector

ET = pytz.timezone("America/New_York")
RES_C = RESULTS_DIR / "experiment_c"


def _load_prior_poc(target_date: date) -> tuple[date, float] | None:
    path = RES_C / "profiles__SPY.parquet"
    if not path.exists():
        return None
    profs = pd.read_parquet(path).sort_values("session_date")
    profs["session_date"] = pd.to_datetime(profs["session_date"]).dt.date
    prior = profs[profs["session_date"] < target_date]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return row["session_date"], float(row["poc"])


def simulate_forward(bars_win, signal_bar_idx, direction, target_move, stop_move, horizon=HORIZON_BARS):
    """Race target/stop on forward underlying bars."""
    entry = float(bars_win.iloc[signal_bar_idx]["c"])
    start = signal_bar_idx + 1
    end = min(start + horizon, len(bars_win))
    if start >= end:
        return {"outcome": "no_forward", "entry": entry}

    if direction == "short":
        target_px = entry - target_move
        stop_px = entry + stop_move
    else:
        target_px = entry + target_move
        stop_px = entry - stop_move

    for i in range(start, end):
        bar = bars_win.iloc[i]
        if direction == "short":
            hit_stop = bar["h"] >= stop_px
            hit_target = bar["l"] <= target_px
        else:
            hit_stop = bar["l"] <= stop_px
            hit_target = bar["h"] >= target_px
        if hit_stop:
            return {"outcome": "stop", "bars_held": i - signal_bar_idx,
                    "exit_px": stop_px, "exit_ts": bar["ts_et"]}
        if hit_target:
            return {"outcome": "target", "bars_held": i - signal_bar_idx,
                    "exit_px": target_px, "exit_ts": bar["ts_et"]}

    last = bars_win.iloc[end - 1]
    return {"outcome": "timeout", "bars_held": end - start - 1,
            "exit_px": float(last["c"]), "exit_ts": last["ts_et"]}


def main() -> int:
    today = date.today()
    print("=" * 75)
    print(f"SPY S3 POC detector replay for {today}")
    print("=" * 75)

    # 1. Pull fresh bars
    print("\n[1/4] Pulling today's SPY bars from Massive...")
    asyncio.run(
        download_range_async(
            ticker="SPY",
            start=today.isoformat(),
            end=today.isoformat(),
            overwrite=True,
        )
    )
    bars = load_bars("SPY", today.isoformat(), today.isoformat())
    print(f"  got {len(bars)} bars, range {bars['ts_et'].min()} -> {bars['ts_et'].max()}")

    # 2. Load prior POC
    print("\n[2/4] Loading yesterday's POC...")
    pl_res = _load_prior_poc(today)
    if pl_res is None:
        print("  FAIL: no profile")
        return 1
    prior_date, prior_poc = pl_res
    print(f"  prior POC from {prior_date}: ${prior_poc:.2f}")

    # 3. Feed bars through detector (use 09:00 warmup for ATR)
    print("\n[3/4] Feeding bars through detector...")
    bars_ext = filter_window(bars, "09:00", "11:30").reset_index(drop=True)
    print(f"  warmup + window bars: {len(bars_ext)}")

    detector = SpyS3PocDetector(level_name="poc", level=prior_poc)
    signals = []
    signal_indices = []

    for i, row in bars_ext.iterrows():
        lb = LiveBar(
            t=int(row["t"]),
            ts_et=row["ts_et"],
            o=float(row["o"]),
            h=float(row["h"]),
            l=float(row["l"]),
            c=float(row["c"]),
            v=float(row["v"]),
        )
        sig = detector.on_bar(lb)
        if sig is not None:
            signals.append(sig)
            signal_indices.append(i)

    # 4. Report
    print(f"\n[4/4] Results — {len(signals)} signal(s) fired\n")

    if not signals:
        print("  NO SIGNALS today on SPY S3 POC rule.")
        # Show how close we came — any touches of POC?
        bars_win = filter_window(bars, "09:30", "11:30")
        touches = bars_win[
            (bars_win["l"] <= prior_poc + 0.05)
            & (bars_win["h"] >= prior_poc - 0.05)
        ]
        if not touches.empty:
            print(f"\n  (but there WERE {len(touches)} POC touches today)")
            print("  The detector rejected them because they failed the wick or follow-through check.")
            print("  First 5 touches:")
            print(touches[["ts_et", "o", "h", "l", "c", "v"]].head().to_string(index=False))
        return 0

    # We have signals. Print each with simulated outcome.
    print(f"  {'time':>8}  {'dir':>6}  {'level':>8}  {'entry':>8}  {'atr':>6}  "
          f"{'target':>7}  {'stop':>7}  {'outcome':>8}  held")
    print("  " + "-" * 82)

    for sig, bar_idx in zip(signals, signal_indices):
        # Simulate forward with ATR-based target/stop
        sim = simulate_forward(
            bars_ext, bar_idx, sig.direction,
            sig.target_move, sig.stop_move, horizon=HORIZON_BARS,
        )
        ts_short = sig.bar.ts_et.strftime("%H:%M")
        outcome = sim.get("outcome", "—")
        held = sim.get("bars_held", "—")

        # Compute P&L in SPY dollars
        entry = sig.bar.c
        if outcome in ("target", "stop") and "exit_px" in sim:
            if sig.direction == "short":
                pnl = entry - sim["exit_px"]
            else:
                pnl = sim["exit_px"] - entry
        elif outcome == "timeout" and "exit_px" in sim:
            if sig.direction == "short":
                pnl = entry - sim["exit_px"]
            else:
                pnl = sim["exit_px"] - entry
        else:
            pnl = 0.0

        print(f"  {ts_short:>8}  {sig.direction:>6}  ${sig.level:>7.2f}  ${entry:>7.2f}  "
              f"{sig.atr:>6.3f}  ${sig.target_move:>6.3f}  ${sig.stop_move:>6.3f}  "
              f"{outcome:>8}  {held}  (pnl=${pnl:+.2f})")

    # Summary
    print()
    outcomes = [simulate_forward(bars_ext, idx, s.direction, s.target_move, s.stop_move)
                for s, idx in zip(signals, signal_indices)]
    n_target = sum(1 for o in outcomes if o.get("outcome") == "target")
    n_stop = sum(1 for o in outcomes if o.get("outcome") == "stop")
    n_timeout = sum(1 for o in outcomes if o.get("outcome") == "timeout")
    n_noforward = sum(1 for o in outcomes if o.get("outcome") == "no_forward")
    print(f"  outcomes: {n_target} targets, {n_stop} stops, {n_timeout} timeouts, {n_noforward} no-forward")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
