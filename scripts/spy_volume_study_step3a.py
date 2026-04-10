"""SPY Volume Study — Step 3a: Real Alpaca 0DTE option fills on Step 2 winning rule.

Winning rule from Step 2:
  - Symbol: SPY
  - Level: POC (prior-day point of control)
  - Window: 09:30-10:30 ET
  - Filter: V0 (no volume filter)
  - First touch only
  - ATR target/stop: 1.5 / 1.5
  - Horizon: 15 bars
  - Direction: fade the approach
      from_below → SHORT underlying → buy PUT
      from_above → LONG underlying → buy CALL

For each signal we fetch real Alpaca 1-min option bars and compute
actual P&L using bar closes at entry/exit timestamps.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from mdq.config import RESULTS_DIR
from mdq.data.bars import load_bars
from mdq.data.calendar import filter_rth, filter_window
from mdq.live.alpaca import AlpacaClient
from mdq.stats.atr import compute_atr

# ── Constants ────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
UTC = pytz.utc

STUDY_START = "2024-01-01"
STUDY_END = "2026-04-09"

PROFILES_PATH = RESULTS_DIR / "experiment_c" / "profiles__SPY.parquet"
OUT_DIR = RESULTS_DIR / "spy_volume_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "step3a_real_options.csv"

# Step 2 winning geometry
WINDOW_START = "09:30"
WINDOW_END = "10:30"
POC_TOUCH_TOL = 0.05     # bar must come within ±0.05 of POC to count as touch
ATR_WINDOW = 20
TARGET_MULT = 1.5
STOP_MULT = 1.5
HORIZON = 15             # max bars after entry

OPT_FETCH_MARGIN_MINS = 5   # extra minutes padded around signal window
OPT_BAR_TOLERANCE_MINS = 5  # max minutes to search for matching option bar


# ── OCC symbol builders ───────────────────────────────────────────────────────

def build_occ_call(exp_date: date, strike: float) -> str:
    """SPY{YYMMDD}C{strike*1000:08d}  e.g. SPY260409C00675000"""
    strike_int = int(round(strike * 1000))
    return f"SPY{exp_date.strftime('%y%m%d')}C{strike_int:08d}"


def build_occ_put(exp_date: date, strike: float) -> str:
    """SPY{YYMMDD}P{strike*1000:08d}  e.g. SPY260409P00675000"""
    strike_int = int(round(strike * 1000))
    return f"SPY{exp_date.strftime('%y%m%d')}P{strike_int:08d}"


# ── Signal detection ──────────────────────────────────────────────────────────

def find_poc_touch(
    window_bars: pd.DataFrame,
    session_bars: pd.DataFrame,
    poc: float,
) -> dict | None:
    """Return the first bar in window_bars that touches poc, or None."""
    if window_bars.empty or session_bars.empty:
        return None

    # Build full-session arrays for ATR and lookback
    sess_atr = compute_atr(session_bars, window=ATR_WINDOW)
    sess_t = session_bars["t"].to_numpy()

    h = window_bars["h"].to_numpy()
    l = window_bars["l"].to_numpy()
    c = window_bars["c"].to_numpy()
    o = window_bars["o"].to_numpy()
    t_ms = window_bars["t"].to_numpy()
    ts_et = window_bars["ts_et"].to_numpy()

    for i in range(len(window_bars)):
        if not (l[i] <= poc + POC_TOUCH_TOL and h[i] >= poc - POC_TOUCH_TOL):
            continue

        # Determine approach from prev bar close (or open if first bar)
        prev_close = c[i - 1] if i > 0 else o[i]
        from_below = prev_close < poc

        # Session index for ATR lookup
        sess_idx = int(np.searchsorted(sess_t, t_ms[i]))
        sess_idx = min(sess_idx, len(sess_atr) - 1)
        atr = float(sess_atr[sess_idx]) if not np.isnan(sess_atr[sess_idx]) else np.nan

        return {
            "touch_ts_et": ts_et[i],
            "touch_t_ms": int(t_ms[i]),
            "entry_price": float(c[i]),
            "from_below": bool(from_below),
            "sess_idx": sess_idx,
            "atr": atr,
        }
    return None


# ── First-passage race ────────────────────────────────────────────────────────

def run_first_passage(
    session_bars: pd.DataFrame,
    entry_sess_idx: int,
    entry_price: float,
    direction: str,   # "long" or "short"
    atr: float,
) -> dict:
    """Race ATR-based target vs stop over HORIZON bars. Returns outcome dict."""
    if np.isnan(atr) or atr <= 0:
        return {"outcome": "no_atr", "exit_price": np.nan, "exit_ts_et": None, "exit_t_ms": None}

    target_dist = atr * TARGET_MULT
    stop_dist = atr * STOP_MULT

    h = session_bars["h"].to_numpy()
    l = session_bars["l"].to_numpy()
    c = session_bars["c"].to_numpy()
    t_ms = session_bars["t"].to_numpy()
    ts_et = session_bars["ts_et"].to_numpy()
    n = len(session_bars)

    start = entry_sess_idx + 1
    end = min(start + HORIZON, n)

    if direction == "long":
        target_px = entry_price + target_dist
        stop_px = entry_price - stop_dist
    else:
        target_px = entry_price - target_dist
        stop_px = entry_price + stop_dist

    for i in range(start, end):
        if direction == "long":
            hit_stop = l[i] <= stop_px
            hit_target = h[i] >= target_px
        else:
            hit_stop = h[i] >= stop_px
            hit_target = l[i] <= target_px

        # Stop wins on simultaneous hit (conservative)
        if hit_stop:
            return {
                "outcome": "stop",
                "exit_price": float(c[i]),
                "exit_ts_et": ts_et[i],
                "exit_t_ms": int(t_ms[i]),
            }
        if hit_target:
            return {
                "outcome": "target",
                "exit_price": float(c[i]),
                "exit_ts_et": ts_et[i],
                "exit_t_ms": int(t_ms[i]),
            }

    # Timeout: use last bar in range
    last_i = end - 1
    return {
        "outcome": "timeout",
        "exit_price": float(c[last_i]),
        "exit_ts_et": ts_et[last_i],
        "exit_t_ms": int(t_ms[last_i]),
    }


# ── Option bar lookup ─────────────────────────────────────────────────────────

def _to_utc_naive(ts_et) -> datetime | None:
    """Convert an ET pandas Timestamp (or numpy datetime64) to UTC naive datetime."""
    if ts_et is None:
        return None
    if isinstance(ts_et, np.datetime64):
        ts_et = pd.Timestamp(ts_et)
    if isinstance(ts_et, pd.Timestamp):
        if ts_et.tzinfo is None:
            ts_et = ts_et.tz_localize(ET)
        else:
            ts_et = ts_et.tz_convert(ET)
        return ts_et.astimezone(UTC).replace(tzinfo=None)
    return None


def find_option_bar_close(
    opt_df: pd.DataFrame,
    target_ts_et,
    tolerance_mins: int = OPT_BAR_TOLERANCE_MINS,
) -> float | None:
    """Return the option bar close price closest to target_ts_et (within tolerance)."""
    if opt_df.empty:
        return None

    target_utc = _to_utc_naive(target_ts_et)
    if target_utc is None:
        return None

    tolerance = pd.Timedelta(minutes=tolerance_mins)

    # opt_df["t"] should already be UTC-naive datetimes from build_opt_df()
    diffs = (opt_df["t"] - target_utc).abs()
    min_diff = diffs.min()
    if min_diff > tolerance:
        return None
    return float(opt_df.loc[diffs.idxmin(), "c"])


def build_opt_df(raw_bars: list[dict]) -> pd.DataFrame:
    """Convert raw Alpaca option bar list to DataFrame with UTC-naive 't' column."""
    if not raw_bars:
        return pd.DataFrame()
    df = pd.DataFrame(raw_bars)
    # Alpaca returns timestamps in UTC; strip timezone info for uniform comparison
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(None)
    return df.sort_values("t").reset_index(drop=True)


def fetch_option_bars_with_fallback(
    client: AlpacaClient,
    session_date: date,
    strike: float,
    direction: str,   # "long" → call, "short" → put
) -> tuple[str, pd.DataFrame]:
    """Fetch option bars for ATM strike, falling back to ±1 strikes if empty."""
    exp_date = session_date  # 0DTE: expiry == session date
    fetch_start = datetime(
        session_date.year, session_date.month, session_date.day, 9, 25
    )
    fetch_end = datetime(
        session_date.year, session_date.month, session_date.day, 16, 5
    )

    builder = build_occ_call if direction == "long" else build_occ_put

    strikes_to_try = [strike, strike - 1, strike + 1]
    for s in strikes_to_try:
        symbol = builder(exp_date, s)
        try:
            raw = client.get_option_bars(symbol, fetch_start, fetch_end)
        except Exception:
            raw = []
        if raw:
            return symbol, build_opt_df(raw)

    # Return the primary symbol with empty DataFrame on complete failure
    return builder(exp_date, strike), pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 90)
    print("SPY Volume Study — Step 3a: Real Alpaca 0DTE Option Fills")
    print(f"  Rule: POC fade, 09:30-10:30 window, ATR {TARGET_MULT}/{STOP_MULT}, {HORIZON}-bar horizon")
    print(f"  Period: {STUDY_START} → {STUDY_END}")
    print("=" * 90)
    t0 = time.perf_counter()

    # ── Load SPY bars ─────────────────────────────────────────────────────────
    print("\nLoading SPY bars...")
    spy = load_bars("SPY", STUDY_START, STUDY_END)
    rth = filter_rth(spy).copy()
    rth["session_date"] = rth["session_date"].apply(
        lambda x: x.date() if hasattr(x, "date") else x
    )
    print(f"  RTH bars: {len(rth):,}   Sessions: {rth['session_date'].nunique()}")

    # ── Load volume profiles ──────────────────────────────────────────────────
    print("Loading SPY volume profiles...")
    if not Path(PROFILES_PATH).exists():
        raise FileNotFoundError(
            f"Profiles not found at {PROFILES_PATH}. Run update_profiles.py first."
        )
    profiles = pd.read_parquet(PROFILES_PATH)
    profiles["session_date"] = pd.to_datetime(profiles["session_date"]).dt.date
    print(f"  Profile rows: {len(profiles):,}  ({profiles['session_date'].min()} → {profiles['session_date'].max()})")

    # ── Generate signals ──────────────────────────────────────────────────────
    print("\nGenerating POC-fade signals...")
    signals: list[dict] = []

    all_sessions = sorted(rth["session_date"].unique())

    for sd in all_sessions:
        prior_profiles = profiles[profiles["session_date"] < sd]
        if prior_profiles.empty:
            continue
        prior = prior_profiles.iloc[-1]
        poc = float(prior["poc"])

        session = rth[rth["session_date"] == sd].reset_index(drop=True)
        if session.empty:
            continue

        window = filter_window(session, WINDOW_START, WINDOW_END).reset_index(drop=True)
        if window.empty:
            continue

        touch = find_poc_touch(window, session, poc)
        if touch is None:
            continue

        direction = "short" if touch["from_below"] else "long"
        atr = touch["atr"]

        race = run_first_passage(
            session,
            touch["sess_idx"],
            touch["entry_price"],
            direction,
            atr,
        )

        signals.append({
            "session_date": sd,
            "year": sd.year,
            "poc": poc,
            "entry_price": touch["entry_price"],
            "entry_ts_et": touch["touch_ts_et"],
            "entry_t_ms": touch["touch_t_ms"],
            "from_below": touch["from_below"],
            "direction": direction,
            "atr": atr,
            "outcome": race["outcome"],
            "exit_price": race["exit_price"],
            "exit_ts_et": race["exit_ts_et"],
            "exit_t_ms": race["exit_t_ms"],
        })

    print(f"  Total signals: {len(signals)}")

    # ── Fetch real option bars ────────────────────────────────────────────────
    print("\nFetching Alpaca 0DTE option fills...")
    client = AlpacaClient()
    results: list[dict] = []

    for i, sig in enumerate(signals, 1):
        sd: date = sig["session_date"]
        direction: str = sig["direction"]
        entry_price: float = sig["entry_price"]
        strike = round(entry_price)

        symbol, opt_df = fetch_option_bars_with_fallback(client, sd, strike, direction)

        entry_opt = find_option_bar_close(opt_df, sig["entry_ts_et"])
        exit_opt = find_option_bar_close(opt_df, sig["exit_ts_et"])

        if entry_opt is not None and exit_opt is not None:
            pnl_contract = (exit_opt - entry_opt) * 100
            pnl_pct = (exit_opt - entry_opt) / entry_opt if entry_opt > 0 else np.nan
            status = "ok"
        elif opt_df.empty:
            pnl_contract = np.nan
            pnl_pct = np.nan
            status = "no_option_bars"
            entry_opt = np.nan
            exit_opt = np.nan
        elif entry_opt is None:
            pnl_contract = np.nan
            pnl_pct = np.nan
            status = "no_entry_bar"
            exit_opt = exit_opt if exit_opt is not None else np.nan
        else:
            pnl_contract = np.nan
            pnl_pct = np.nan
            status = "no_exit_bar"
            entry_opt = entry_opt if entry_opt is not None else np.nan

        results.append({
            "session_date": sd.strftime("%Y-%m-%d"),
            "year": sig["year"],
            "direction": direction,
            "poc": sig["poc"],
            "entry_price": entry_price,
            "entry_ts_et": sig["entry_ts_et"],
            "exit_ts_et": sig["exit_ts_et"],
            "atr": sig["atr"],
            "outcome": sig["outcome"],
            "symbol": symbol,
            "status": status,
            "entry_opt": entry_opt,
            "exit_opt": exit_opt,
            "pnl_contract": pnl_contract,
            "pnl_pct": pnl_pct,
        })

        if i % 25 == 0 or i == len(signals):
            ok_n = sum(1 for r in results if r["status"] == "ok")
            elapsed = time.perf_counter() - t0
            print(f"  [{i:>4}/{len(signals)}]  ok={ok_n}  elapsed={elapsed:.0f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(df)

    total_elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")
    return 0


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    total_signals = len(df)
    ok = df[df["status"] == "ok"].copy()
    n_with_data = len(ok)

    print(f"\nTotal signals generated:     {total_signals}")
    print(f"Signals with option data:    {n_with_data}  ({n_with_data/total_signals:.1%} fill rate)")

    if not ok.empty:
        status_counts = df["status"].value_counts()
        print(f"\nStatus breakdown:")
        for s, c in status_counts.items():
            print(f"  {s:<20}: {c}")

    if ok.empty:
        print("\nNo trades with option data — check Alpaca coverage and date range.")
        return

    wins = ok[ok["pnl_contract"] > 0]
    losses = ok[ok["pnl_contract"] < 0]
    n = len(ok)
    n_wins = len(wins)
    n_losses = len(losses)

    print("\n── Option P&L per contract ──────────────────────────────────────────")
    print(f"  Mean P&L:      ${ok['pnl_contract'].mean():+.2f}")
    print(f"  Median P&L:    ${ok['pnl_contract'].median():+.2f}")
    print(f"  Std dev:       ${ok['pnl_contract'].std():.2f}")
    print(f"  Min:           ${ok['pnl_contract'].min():+.2f}")
    print(f"  Max:           ${ok['pnl_contract'].max():+.2f}")
    print(f"  Total:         ${ok['pnl_contract'].sum():+.2f}  ({n} trades)")

    print("\n── Win/Loss Stats ───────────────────────────────────────────────────")
    win_rate = n_wins / n if n > 0 else 0.0
    print(f"  Win rate:      {win_rate:.1%}  ({n_wins}/{n})")
    if n_wins > 0:
        print(f"  Avg win:       ${wins['pnl_contract'].mean():+.2f}")
    if n_losses > 0:
        print(f"  Avg loss:      ${losses['pnl_contract'].mean():+.2f}")
    if n_wins > 0 and n_losses > 0:
        gross_wins = wins["pnl_contract"].sum()
        gross_losses = abs(losses["pnl_contract"].sum())
        pf = gross_wins / gross_losses
        print(f"  Profit factor: {pf:.2f}")

    print("\n── By Year ──────────────────────────────────────────────────────────")
    ok_yr = ok.copy()
    ok_yr["year"] = ok_yr["year"].astype(int)
    by_year = ok_yr.groupby("year").agg(
        n=("pnl_contract", "size"),
        mean_pnl=("pnl_contract", "mean"),
        total_pnl=("pnl_contract", "sum"),
        win_rate=("pnl_contract", lambda x: (x > 0).mean()),
    )
    print(by_year.to_string(float_format=lambda x: f"{x:+.2f}"))

    print("\n── By Direction (long=CALL / short=PUT) ─────────────────────────────")
    by_dir = ok.groupby("direction").agg(
        n=("pnl_contract", "size"),
        mean_pnl=("pnl_contract", "mean"),
        total_pnl=("pnl_contract", "sum"),
        win_rate=("pnl_contract", lambda x: (x > 0).mean()),
    )
    print(by_dir.to_string(float_format=lambda x: f"{x:+.2f}"))

    print("\n── Most Recent 15 Trades ────────────────────────────────────────────")
    recent = ok.sort_values("session_date").tail(15)[
        ["session_date", "direction", "outcome", "entry_opt", "exit_opt", "pnl_contract"]
    ].copy()
    recent["entry_opt"] = recent["entry_opt"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
    recent["exit_opt"] = recent["exit_opt"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
    recent["pnl_contract"] = recent["pnl_contract"].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "—")
    print(recent.to_string(index=False))

    # Annualized estimate (roughly 252 trading days; signals per year)
    n_years = ok_yr["year"].nunique()
    trades_per_year = n / n_years if n_years > 0 else 0
    ann_pnl = ok["pnl_contract"].mean() * trades_per_year
    print(f"\n── Annualized Estimate ──────────────────────────────────────────────")
    print(f"  Avg trades/year:  {trades_per_year:.0f}")
    print(f"  Mean P&L/trade:   ${ok['pnl_contract'].mean():+.2f}")
    print(f"  Est. annual P&L:  ${ann_pnl:+.0f}/yr per contract")


if __name__ == "__main__":
    raise SystemExit(main())
