import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from indicators import (
    compute_rsi,
    compute_bollinger_bands,
    compute_sma,
    compute_volume_profile,
)


def fetch_jpm_daily_bars(years: int = 1) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download("JPM", start=start, end=end, interval="1d", progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # momentum / mean-reversion oscillators
    df["rsi"] = compute_rsi(close)

    # bollinger bands
    bb = compute_bollinger_bands(close)
    df = pd.concat([df, bb], axis=1)

    # trend SMAs
    df["sma_9"]   = compute_sma(close, 9)
    df["sma_21"]  = compute_sma(close, 21)
    df["sma_200"] = compute_sma(close, 200)

    # %B: where price sits within the bands (0 = lower, 1 = upper)
    df["pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # distance from SMA 200 in % — key mean-reversion anchor
    df["dist_sma200_pct"] = (close - df["sma_200"]) / df["sma_200"] * 100

    return df


def get_volume_profiles(df: pd.DataFrame) -> dict:
    profiles = {}

    # full year
    profiles.update(compute_volume_profile(df, bins=50, label="1y"))

    # last 3 months
    cutoff_3m = df.index[-1] - pd.DateOffset(months=3)
    profiles.update(compute_volume_profile(df[df.index >= cutoff_3m], bins=30, label="3m"))

    # last 1 month
    cutoff_1m = df.index[-1] - pd.DateOffset(months=1)
    profiles.update(compute_volume_profile(df[df.index >= cutoff_1m], bins=20, label="1m"))

    return profiles


if __name__ == "__main__":
    bars = fetch_jpm_daily_bars()
    bars = apply_indicators(bars)

    print("=== JPM Daily Bars + Indicators (last 10 rows) ===")
    cols = ["Close", "rsi", "bb_upper", "bb_mid", "bb_lower",
            "pct_b", "sma_9", "sma_21", "sma_200", "dist_sma200_pct"]
    pd.set_option("display.float_format", "{:.2f}".format)
    print(bars[cols].tail(10).to_string())

    print("\n=== Volume Profile ===")
    vp = get_volume_profiles(bars)
    for key, val in vp.items():
        if isinstance(val, list):
            print(f"  {key}: {val[:5]} ...")  # trim list for readability
        else:
            print(f"  {key}: {val}")
