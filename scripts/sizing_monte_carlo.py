"""Monte Carlo: $1000 starting bankroll -> 1 year of Tier 1 trading.

Bootstraps real historical Alpaca option fills (with replacement) to generate
10,000 random 66-trade paths for each (DTE, sizing rule) combination. Reports
median, mean, tail percentiles, and P(double/5x/halve/ruin) so we can pick
the mathematically optimal strategy.

Key insight: "reinvest everything" is a sizing rule. Different sizing rules
produce dramatically different distributions of outcomes even with the same
underlying trade edge. The optimal rule maximizes log-wealth growth while
keeping ruin probability acceptably low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.config import RESULTS_DIR

N_TRADES_PER_YEAR = 66
N_PATHS = 10_000
START_BANKROLL = 1000.0
SEED = 42


def make_sizer(kind: str, alloc: float = 0.0):
    """Return a function (bankroll, cost) -> n_contracts."""
    if kind == "fixed_1":
        def sizer(bankroll, cost):
            return 1 if bankroll >= cost else 0
        return sizer
    if kind == "alloc":
        def sizer(bankroll, cost):
            n = int(alloc * bankroll / cost)
            if n == 0 and bankroll >= cost:
                return 1  # floor of 1 if affordable (don't skip trades early)
            return n
        return sizer
    if kind == "alloc_strict":
        def sizer(bankroll, cost):
            # No minimum floor — skip if allocation can't afford 1 contract
            return max(0, int(alloc * bankroll / cost))
        return sizer
    raise ValueError(kind)


def simulate_paths(
    trades_df: pd.DataFrame,
    sizer,
    n_paths: int,
    n_trades: int,
    start: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run n_paths bootstrap simulations of n_trades trades each.

    Returns (terminal_bankrolls, min_bankrolls, max_drawdowns_pct).
    """
    rng = np.random.default_rng(seed)
    costs = trades_df["cost"].to_numpy()
    pnls = trades_df["pnl_contract"].to_numpy()
    n_hist = len(trades_df)

    terminals = np.zeros(n_paths)
    mins = np.zeros(n_paths)
    max_dd = np.zeros(n_paths)

    for p in range(n_paths):
        bankroll = start
        min_b = bankroll
        peak = bankroll
        worst_dd = 0.0

        # Pre-sample all trade indices for this path (slight speedup)
        idxs = rng.integers(0, n_hist, size=n_trades)

        for i in range(n_trades):
            cost = costs[idxs[i]]
            pnl = pnls[idxs[i]]

            n_ctr = sizer(bankroll, cost)
            if n_ctr == 0:
                continue

            # Can't spend more than bankroll
            if n_ctr * cost > bankroll:
                n_ctr = int(bankroll / cost)
                if n_ctr == 0:
                    continue

            bankroll += n_ctr * pnl
            if bankroll < 0:
                bankroll = 0
                break

            if bankroll > peak:
                peak = bankroll
            if bankroll < min_b:
                min_b = bankroll

            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > worst_dd:
                worst_dd = dd

        terminals[p] = bankroll
        mins[p] = min_b
        max_dd[p] = worst_dd

    return terminals, mins, max_dd


def load_0dte_atm() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "tier1_real_options_backtest.csv")
    ok = df[df["status"] == "ok"].copy()
    ok["cost"] = ok["entry_opt"] * 100
    return ok[["cost", "pnl_contract"]].reset_index(drop=True)


def load_5dte_atm() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "dte_strike_sweep.csv")
    ok = df[
        (df["status"] == "ok")
        & (df["dte"] == 5)
        & (df["strike_label"] == "ATM")
    ].copy()
    ok["cost"] = ok["entry_opt"] * 100
    return ok[["cost", "pnl_contract"]].reset_index(drop=True)


def summarize(terminals: np.ndarray, max_dd: np.ndarray, start: float) -> dict:
    # Compute geometric mean return per trade (implied by final bankroll)
    # Guard against log(0)
    safe = np.maximum(terminals, 1.0)
    log_final = np.log(safe / start)
    log_mean = log_final.mean() / N_TRADES_PER_YEAR  # per-trade log return

    return {
        "median": float(np.median(terminals)),
        "mean": float(np.mean(terminals)),
        "p5": float(np.percentile(terminals, 5)),
        "p25": float(np.percentile(terminals, 25)),
        "p75": float(np.percentile(terminals, 75)),
        "p95": float(np.percentile(terminals, 95)),
        "p_double": float(np.mean(terminals >= 2 * start)),
        "p_5x": float(np.mean(terminals >= 5 * start)),
        "p_10x": float(np.mean(terminals >= 10 * start)),
        "p_halve": float(np.mean(terminals <= 0.5 * start)),
        "p_quarter": float(np.mean(terminals <= 0.25 * start)),
        "median_maxdd": float(np.median(max_dd)),
        "p95_maxdd": float(np.percentile(max_dd, 95)),
        "log_return_per_trade": float(log_mean),
    }


def main() -> int:
    print("=" * 95)
    print(f"Monte Carlo sizing study")
    print(f"  Starting bankroll: ${START_BANKROLL:,.0f}")
    print(f"  Trades per year:   {N_TRADES_PER_YEAR}")
    print(f"  Paths per config:  {N_PATHS:,}")
    print(f"  RNG seed:          {SEED}")
    print("=" * 95)

    configs = {
        "0DTE_ATM": load_0dte_atm(),
        "5DTE_ATM": load_5dte_atm(),
    }

    sizing_rules = [
        ("fixed_1ctr", make_sizer("fixed_1")),
        ("5%_min1",    make_sizer("alloc", 0.05)),
        ("10%_min1",   make_sizer("alloc", 0.10)),
        ("15%_min1",   make_sizer("alloc", 0.15)),
        ("20%_min1",   make_sizer("alloc", 0.20)),
        ("25%_min1",   make_sizer("alloc", 0.25)),
        ("33%_min1",   make_sizer("alloc", 0.33)),
        ("50%_min1",   make_sizer("alloc", 0.50)),
    ]

    results = []
    for cfg_name, trades_df in configs.items():
        print(f"\nRunning {cfg_name} (n_historical={len(trades_df)}, median cost=${trades_df['cost'].median():.0f})...")
        for sizing_name, sizer in sizing_rules:
            terminals, mins, max_dd = simulate_paths(
                trades_df, sizer, N_PATHS, N_TRADES_PER_YEAR, START_BANKROLL, SEED,
            )
            s = summarize(terminals, max_dd, START_BANKROLL)
            s["config"] = cfg_name
            s["sizing"] = sizing_name
            results.append(s)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "sizing_monte_carlo.csv", index=False)

    def fmt_money(x):
        return f"${x:,.0f}" if abs(x) < 1e6 else f"${x/1e6:,.1f}M"

    def fmt_pct(x):
        return f"{x:.1%}"

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.max_colwidth", 14)

    # ----------- Main results table -----------
    print("\n" + "=" * 95)
    print("RESULTS — terminal bankroll distributions after 66 trades (1 year)")
    print("=" * 95)

    disp = df.copy()
    for col in ("median", "mean", "p5", "p25", "p75", "p95"):
        disp[col] = disp[col].map(fmt_money)
    for col in ("p_double", "p_5x", "p_10x", "p_halve", "p_quarter",
                "median_maxdd", "p95_maxdd"):
        disp[col] = disp[col].map(fmt_pct)
    disp["log_return_per_trade"] = disp["log_return_per_trade"].map(lambda x: f"{x:+.4f}")

    cols1 = ["config", "sizing", "median", "mean", "p5", "p25", "p75", "p95"]
    print("\nTerminal bankroll percentiles:")
    print(disp[cols1].to_string(index=False))

    cols2 = ["config", "sizing", "p_double", "p_5x", "p_10x", "p_halve", "p_quarter"]
    print("\nMultiple-of-starting probabilities:")
    print(disp[cols2].to_string(index=False))

    cols3 = ["config", "sizing", "median_maxdd", "p95_maxdd", "log_return_per_trade"]
    print("\nDrawdowns and geometric growth:")
    print(disp[cols3].to_string(index=False))

    # ----------- Rank by geometric growth (Kelly-optimal) -----------
    print("\n" + "=" * 95)
    print("RANKED BY MEDIAN TERMINAL BANKROLL (Kelly-optimal on median)")
    print("=" * 95)
    ranked = df.sort_values("median", ascending=False)
    r_disp = ranked.copy()
    for col in ("median", "mean", "p5", "p95"):
        r_disp[col] = r_disp[col].map(fmt_money)
    for col in ("p_double", "p_halve", "p_quarter", "p95_maxdd"):
        r_disp[col] = r_disp[col].map(fmt_pct)
    r_disp["log_return_per_trade"] = r_disp["log_return_per_trade"].map(lambda x: f"{x:+.4f}")
    print(r_disp[["config", "sizing", "median", "mean", "p5", "p95",
                  "p_double", "p_halve", "p_quarter", "p95_maxdd",
                  "log_return_per_trade"]].to_string(index=False))

    # ----------- Ranked by expected log return (true Kelly metric) -----------
    print("\n" + "=" * 95)
    print("RANKED BY EXPECTED LOG RETURN PER TRADE (pure Kelly)")
    print("=" * 95)
    ranked_log = df.sort_values("log_return_per_trade", ascending=False)
    rl_disp = ranked_log.copy()
    for col in ("median", "mean", "p5", "p95"):
        rl_disp[col] = rl_disp[col].map(fmt_money)
    for col in ("p_double", "p_halve", "p_quarter"):
        rl_disp[col] = rl_disp[col].map(fmt_pct)
    rl_disp["log_return_per_trade"] = rl_disp["log_return_per_trade"].map(lambda x: f"{x:+.4f}")
    print(rl_disp[["config", "sizing", "log_return_per_trade",
                   "median", "p5", "p_double", "p_halve", "p_quarter"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
