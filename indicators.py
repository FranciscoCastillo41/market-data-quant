import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename("rsi")


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame(
        {"bb_upper": mid + std_dev * std, "bb_mid": mid, "bb_lower": mid - std_dev * std}
    )


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean().rename(f"sma_{period}")


def compute_volume_profile(
    df: pd.DataFrame, bins: int = 50, label: str = ""
) -> dict:
    """
    Compute Volume Profile over a given OHLCV DataFrame.
    Returns POC, HVNs, and LVNs as price levels.
    """
    price_min = df["Low"].min()
    price_max = df["High"].max()
    price_bins = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2

    volume_at_price = np.zeros(bins)
    for _, row in df.iterrows():
        # distribute bar volume across price range of the bar
        bar_bins = (price_bins >= row["Low"]) & (price_bins <= row["High"])
        touched = np.where(bar_bins)[0]
        if len(touched) > 1:
            idxs = touched[:-1]
        elif len(touched) == 1:
            idxs = touched
        else:
            idxs = [np.digitize(row["Close"], price_bins) - 1]
        if len(idxs):
            share = row["Volume"] / max(len(idxs), 1)
            for i in idxs:
                if 0 <= i < bins:
                    volume_at_price[i] += share

    poc_idx = int(np.argmax(volume_at_price))
    poc = bin_centers[poc_idx]

    threshold_high = np.percentile(volume_at_price, 70)
    threshold_low = np.percentile(volume_at_price, 30)

    hvn = bin_centers[volume_at_price >= threshold_high].tolist()
    lvn = bin_centers[volume_at_price <= threshold_low].tolist()

    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}poc": round(poc, 2),
        f"{prefix}hvn": [round(p, 2) for p in hvn],
        f"{prefix}lvn": [round(p, 2) for p in lvn],
    }
