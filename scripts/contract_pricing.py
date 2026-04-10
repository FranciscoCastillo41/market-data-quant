"""Show actual entry premium distribution for 0DTE vs 5DTE ATM from the sweep."""

from __future__ import annotations

import pandas as pd

from mdq.config import RESULTS_DIR


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "dte_strike_sweep.csv")
    ok = df[(df["status"] == "ok") & (df["strike_label"] == "ATM")].copy()

    print("Actual entry premium distribution (ATM puts):\n")
    for dte in sorted(ok["dte"].unique()):
        sub = ok[ok["dte"] == dte]
        print(f"DTE = {dte}")
        print(f"  n:       {len(sub)}")
        print(f"  mean:    ${sub['entry_opt'].mean():.2f}  (= ${sub['entry_opt'].mean()*100:.0f}/contract)")
        print(f"  median:  ${sub['entry_opt'].median():.2f}  (= ${sub['entry_opt'].median()*100:.0f}/contract)")
        print(f"  min:     ${sub['entry_opt'].min():.2f}")
        print(f"  25%ile:  ${sub['entry_opt'].quantile(0.25):.2f}")
        print(f"  75%ile:  ${sub['entry_opt'].quantile(0.75):.2f}")
        print(f"  max:     ${sub['entry_opt'].max():.2f}")
        print()


if __name__ == "__main__":
    main()
