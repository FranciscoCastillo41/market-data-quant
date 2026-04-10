"""SPY Volume Study — Step 3C: Multi-Ticker Generalization.

Tests whether the POC fade edge observed on SPY generalizes to QQQ, NVDA, AMZN.

Rule (identical to the SPY POC study):
  - First touch of prior-day POC in 09:30-10:30 ET window
  - Fade direction: approaching from below → SHORT; from above → LONG
  - Entry at touch-bar close
  - ATR(20)-based 1st-passage race: target=1.5x, stop=1.5x, horizon=15 bars

Train / OOS split:
  - Train: 2023-01-03 to 2025-12-31
  - OOS:   2026-01-01 to 2026-04-09

SPY reference (from step 2):
  - Train: 99 events, 50.5% hit, +0.136 ATR expectancy
  - OOS:   8 events, 75.0% hit, +0.750 ATR expectancy
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR
from mdq.data.bars import download_range, load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.levels.volume_profile import compute_all_profiles
from mdq.stats.atr import compute_atr

# ── Constants ─────────────────────────────────────────────────────────────────
START = "2023-01-03"
END = "2026-04-09"
TRAIN_CUTOFF_YEAR = 2025
OOS_START_YEAR = 2026

WINDOW_START = "09:30"
WINDOW_END = "10:30"
POC_TOUCH_TOL = 0.05   # price distance to count as "touching" POC
ATR_MULT = 1.5         # target and stop both 1.5x ATR
HORIZON_BARS = 15      # max bars to hold after entry

TICKERS = ["QQQ", "NVDA", "AMZN"]

# SPY baseline from step 2 at 1.5/1.5 ATR, window=09:30-10:30, POC, V0
SPY_REFERENCE = {
    "train": {"n": 99, "hit_rate": 0.505, "exp_atr": 0.136},
    "oos":   {"n":  8, "hit_rate": 0.750, "exp_atr": 0.750},
}

OUT = RESULTS_DIR / "spy_volume_study"
OUT.mkdir(parents=True, exist_ok=True)


# ── Data helpers ───────────────────────────────────────────────────────────────

def ensure_downloaded(ticker: str) -> None:
    """Download bars if the full training range is not cached."""
    from mdq.config import RAW_DIR
    from pathlib import Path
    # Quick probe: does 2023-01 exist?
    probe = RAW_DIR / ticker / "2023-01.parquet"
    if not probe.exists():
        print(f"  Downloading {ticker} {START} → {END} …")
        download_range(ticker, START, END)
        print(f"  {ticker} download complete.")
    else:
        print(f"  {ticker}: cache found, skipping download.")


# ── Signal logic ───────────────────────────────────────────────────────────────

def find_first_poc_touch(
    window_bars: pd.DataFrame,
    session_bars: pd.DataFrame,
    poc: float,
) -> dict | None:
    """Return the first bar in window_bars that touches poc, or None.

    Fade direction: bar approached poc from below → SHORT (fade the rejection).
                    bar approached poc from above → LONG.
    """
    if window_bars.empty:
        return None

    h = window_bars["h"].to_numpy()
    l = window_bars["l"].to_numpy()
    c = window_bars["c"].to_numpy()
    t = window_bars["t"].to_numpy()
    ts = window_bars["ts_et"].to_numpy()

    full_t = session_bars["t"].to_numpy()

    for i in range(len(window_bars)):
        if l[i] <= poc + POC_TOUCH_TOL and h[i] >= poc - POC_TOUCH_TOL:
            # Determine approach direction from prior close (or first bar close)
            prev_close = c[i - 1] if i > 0 else c[i]
            from_below = prev_close < poc

            # Locate this bar in the full session for ATR look-back
            full_idx = int(np.searchsorted(full_t, t[i]))
            full_idx = min(full_idx, len(full_t) - 1)

            atr_arr = compute_atr(session_bars, window=20)
            atr_val = atr_arr[full_idx] if full_idx < len(atr_arr) else np.nan
            if np.isnan(atr_val) or atr_val <= 0:
                return None  # Cannot size without ATR — skip day

            direction = "short" if from_below else "long"

            return {
                "ts_et": ts[i],
                "t": int(t[i]),
                "entry_price": float(c[i]),
                "poc": poc,
                "from_below": from_below,
                "direction": direction,
                "bar_idx_in_session": full_idx,
                "atr": float(atr_val),
            }
    return None


def first_passage_outcome(
    session_bars: pd.DataFrame,
    entry_session_idx: int,
    entry_price: float,
    direction: str,
    atr: float,
) -> str:
    """Run target/stop race and return 'target', 'stop', or 'timeout'."""
    target_dist = atr * ATR_MULT
    stop_dist = atr * ATR_MULT

    if direction == "short":
        target_px = entry_price - target_dist
        stop_px = entry_price + stop_dist
    else:
        target_px = entry_price + target_dist
        stop_px = entry_price - stop_dist

    h = session_bars["h"].to_numpy()
    l = session_bars["l"].to_numpy()
    n = len(h)

    start = entry_session_idx + 1
    end = min(start + HORIZON_BARS, n)

    for i in range(start, end):
        if direction == "short":
            if h[i] >= stop_px:
                return "stop"
            if l[i] <= target_px:
                return "target"
        else:
            if l[i] <= stop_px:
                return "stop"
            if h[i] >= target_px:
                return "target"
    return "timeout"


# ── Per-ticker backtest ────────────────────────────────────────────────────────

def run_ticker_backtest(ticker: str) -> pd.DataFrame:
    """Return a DataFrame of one row per signal event for `ticker`."""
    print(f"\n{'─' * 60}")
    print(f"  Processing {ticker} …")

    bars = load_bars(ticker, START, END)
    profiles = compute_all_profiles(bars, bin_size=0.05)
    profiles["session_date"] = pd.to_datetime(profiles["session_date"]).dt.date

    rth = filter_rth(bars).copy()
    rth["session_date"] = pd.to_datetime(rth["session_date"]).dt.date

    session_dates = sorted(rth["session_date"].unique())
    print(f"    RTH bars: {len(rth):,}  |  Sessions: {len(session_dates)}")

    events: list[dict] = []

    for sd in session_dates:
        prior_profiles = profiles[profiles["session_date"] < sd]
        if prior_profiles.empty:
            continue

        poc = float(prior_profiles.iloc[-1]["poc"])
        session = rth[rth["session_date"] == sd].reset_index(drop=True)
        if session.empty:
            continue

        window = filter_window(session, WINDOW_START, WINDOW_END).reset_index(drop=True)
        touch = find_first_poc_touch(window, session, poc)
        if touch is None:
            continue

        outcome = first_passage_outcome(
            session,
            touch["bar_idx_in_session"],
            touch["entry_price"],
            touch["direction"],
            touch["atr"],
        )

        events.append({
            "ticker": ticker,
            "session_date": sd,
            "year": sd.year,
            "ts_et": touch["ts_et"],
            "poc": poc,
            "entry_price": touch["entry_price"],
            "direction": touch["direction"],
            "from_below": touch["from_below"],
            "atr": touch["atr"],
            "outcome": outcome,
        })

    df = pd.DataFrame(events)
    print(f"    Total events: {len(df)}")
    return df


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(events: pd.DataFrame, label: str) -> dict:
    """Compute hit rate, stop rate, timeout rate, and ATR expectancy."""
    n = len(events)
    if n == 0:
        return {"label": label, "n": 0, "hit_rate": np.nan,
                "stop_rate": np.nan, "timeout_rate": np.nan, "exp_atr": np.nan}

    n_target = (events["outcome"] == "target").sum()
    n_stop = (events["outcome"] == "stop").sum()
    n_timeout = (events["outcome"] == "timeout").sum()

    exp_atr = (n_target * ATR_MULT - n_stop * ATR_MULT) / n

    return {
        "label": label,
        "n": n,
        "hit_rate": n_target / n,
        "stop_rate": n_stop / n,
        "timeout_rate": n_timeout / n,
        "exp_atr": exp_atr,
    }


def signals_per_year(events: pd.DataFrame) -> float:
    """Average number of signals per year across the full sample."""
    if events.empty:
        return 0.0
    n_years = events["year"].nunique()
    return len(events) / n_years if n_years > 0 else 0.0


# ── Printing helpers ───────────────────────────────────────────────────────────

def _fmt(val, pct: bool = False, decimals: int = 3) -> str:
    if pd.isna(val):
        return "  —    "
    if pct:
        return f"{val * 100:6.1f}%"
    return f"{val:+.{decimals}f}" if isinstance(val, float) else str(val)


def print_ticker_summary(ticker: str, stats_train: dict, stats_oos: dict) -> None:
    print(f"\n  {ticker}")
    header = f"    {'Split':<8} {'N':>5}  {'Hit':>7}  {'Stop':>7}  {'Timeout':>9}  {'Exp(ATR)':>10}"
    print(header)
    print("    " + "-" * 56)
    for sp, st in [("TRAIN", stats_train), ("OOS", stats_oos)]:
        print(
            f"    {sp:<8} {st['n']:>5}  "
            f"{_fmt(st['hit_rate'], pct=True):>7}  "
            f"{_fmt(st['stop_rate'], pct=True):>7}  "
            f"{_fmt(st['timeout_rate'], pct=True):>9}  "
            f"{_fmt(st['exp_atr']):>10}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    t_start = time.perf_counter()

    print("=" * 70)
    print("SPY Volume Study — Step 3C: Multi-Ticker POC Fade Generalization")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Rule: first POC touch in {WINDOW_START}–{WINDOW_END}, fade, {ATR_MULT}/{ATR_MULT} ATR, {HORIZON_BARS}-bar horizon")
    print("=" * 70)

    # ── Step 1: Ensure data is available ──────────────────────────────────────
    print("\n[1] Checking / downloading bar data …")
    for ticker in TICKERS:
        ensure_downloaded(ticker)

    # ── Step 2: Run backtest per ticker ───────────────────────────────────────
    print("\n[2] Running per-ticker backtests …")
    all_frames: list[pd.DataFrame] = []
    ticker_results: dict[str, tuple[dict, dict]] = {}

    for ticker in TICKERS:
        events = run_ticker_backtest(ticker)
        if events.empty:
            ticker_results[ticker] = (
                compute_stats(events, f"{ticker} TRAIN"),
                compute_stats(events, f"{ticker} OOS"),
            )
            all_frames.append(events)
            continue

        train = events[events["year"] <= TRAIN_CUTOFF_YEAR]
        oos = events[events["year"] >= OOS_START_YEAR]

        ticker_results[ticker] = (
            compute_stats(train, f"{ticker} TRAIN"),
            compute_stats(oos, f"{ticker} OOS"),
        )
        all_frames.append(events)

    # ── Step 3: Print per-ticker summaries ────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("RESULTS — POC Fade  |  09:30–10:30  |  1.5/1.5 ATR  |  15-bar horizon")
    print("=" * 70)

    for ticker in TICKERS:
        st, so = ticker_results[ticker]
        print_ticker_summary(ticker, st, so)

    # SPY reference
    print("\n  SPY (reference from Step 2)")
    spy_ref_header = f"    {'Split':<8} {'N':>5}  {'Hit':>7}  {'Stop':>7}  {'Timeout':>9}  {'Exp(ATR)':>10}"
    print(spy_ref_header)
    print("    " + "-" * 56)
    for sp, ref_key in [("TRAIN", "train"), ("OOS", "oos")]:
        ref = SPY_REFERENCE[ref_key]
        stop_r = 1.0 - ref["hit_rate"] - (1.0 - ref["hit_rate"]) * 0.3  # approx
        print(
            f"    {sp:<8} {ref['n']:>5}  "
            f"{_fmt(ref['hit_rate'], pct=True):>7}  "
            f"    —    "
            f"         —   "
            f"  {_fmt(ref['exp_atr']):>10}"
        )

    # ── Step 4: Combined multi-ticker view ────────────────────────────────────
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    comb_train = combined[combined["year"] <= TRAIN_CUTOFF_YEAR] if not combined.empty else pd.DataFrame()
    comb_oos = combined[combined["year"] >= OOS_START_YEAR] if not combined.empty else pd.DataFrame()

    print("\n\n" + "=" * 70)
    print("COMBINED VIEW — All Tickers Pooled")
    print("=" * 70)
    print_ticker_summary(
        "QQQ+NVDA+AMZN",
        compute_stats(comb_train, "COMBINED TRAIN"),
        compute_stats(comb_oos, "COMBINED OOS"),
    )

    # ── Step 5: Signal frequency analysis ────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SIGNAL FREQUENCY — Signals per Year")
    print("=" * 70)
    total_per_year = 0.0
    for ticker in TICKERS:
        frames = [f for f in all_frames if not f.empty and (f["ticker"] == ticker).any()]
        if frames:
            t_events = pd.concat(frames)
            t_events = t_events[t_events["ticker"] == ticker]
            spy_count = signals_per_year(t_events)
            total_per_year += spy_count
            print(f"  {ticker:<6}: {spy_count:>5.1f} signals/year  (total {len(t_events)} over sample)")
        else:
            print(f"  {ticker:<6}: 0 signals/year")

    print(f"\n  SPY    :  ~33 signals/year  (reference, train period)")
    print(f"  ─────────────────────────────")
    print(f"  ALL 4  : ~{total_per_year + 33:.0f} signals/year combined (SPY + {'+'.join(TICKERS)})")

    # ── Step 6: Comparison table ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    cmp_header = f"  {'Ticker':<10} {'Split':<8} {'N':>5}  {'Hit%':>7}  {'Exp(ATR)':>10}  {'Edge?':>8}"
    print(cmp_header)
    print("  " + "-" * 56)

    def _edge_label(exp_atr, n) -> str:
        if n < 5:
            return "n/a (low n)"
        if exp_atr > 0.10:
            return "YES"
        if exp_atr > 0:
            return "marginal"
        return "NO"

    # SPY rows
    for sp, ref_key in [("train", "train"), ("oos", "oos")]:
        ref = SPY_REFERENCE[ref_key]
        print(
            f"  {'SPY':<10} {sp.upper():<8} {ref['n']:>5}  "
            f"{_fmt(ref['hit_rate'], pct=True):>7}  "
            f"  {_fmt(ref['exp_atr']):>10}  "
            f"{_edge_label(ref['exp_atr'], ref['n']):>8}"
        )

    for ticker in TICKERS:
        st, so = ticker_results[ticker]
        for sp, st_row in [("TRAIN", st), ("OOS", so)]:
            print(
                f"  {ticker:<10} {sp:<8} {st_row['n']:>5}  "
                f"{_fmt(st_row['hit_rate'], pct=True):>7}  "
                f"  {_fmt(st_row['exp_atr']):>10}  "
                f"{_edge_label(st_row['exp_atr'] if not pd.isna(st_row['exp_atr']) else -1, st_row['n']):>8}"
            )

    # ── Step 7: Save CSV ──────────────────────────────────────────────────────
    if not combined.empty:
        out_path = OUT / "step3c_multi_ticker.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n\nSaved: {out_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nStep 3C complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
