import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_spy_daily_bars(months: int = 3) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=months * 30)
    df = yf.download("SPY", start=start, end=end, interval="1d", progress=False)
    df.index = pd.to_datetime(df.index)
    return df


def daily_returns(bars: pd.DataFrame) -> pd.DataFrame:
    returns = bars[["Close"]].copy()
    returns["daily_return"] = returns["Close"].pct_change()
    returns["cumulative_return"] = (1 + returns["daily_return"]).cumprod() - 1
    return returns.dropna()


if __name__ == "__main__":
    bars = fetch_spy_daily_bars()
    returns = daily_returns(bars)
    print(returns)
