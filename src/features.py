import pandas as pd
import numpy as np

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs quantitative features for time-series and machine learning forecasting.
    Includes:
      - Daily percentage returns & log returns
      - Lagged closing prices (t-1 through t-5)
      - Trend moving averages (5, 10, 20-day MA)
      - Volatility regime features (10-day rolling return standard deviation)
      - Volume dynamics (volume percentage change and 10-day rolling average volume)
      - One-step ahead target: Close_{t+1}
    """
    df = data.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Ticker", "Date"], inplace=True)
    
    # 1. Returns
    df["daily_return"] = df.groupby("Ticker")["Close"].pct_change()
    df["pct_return"] = df["daily_return"]
    
    # 2. Lagged Close Prices
    df["close_lag"] = df.groupby("Ticker")["Close"].shift(1)
    for l in [1, 2, 3, 4, 5]:
        df[f"close_lag_{l}"] = df.groupby("Ticker")["Close"].shift(l)
        
    # 3. Rolling Moving Averages (Trend Features)
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = (
            df.groupby("Ticker")["Close"]
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
    # 4. Volatility Regimes (10-day rolling standard deviation of daily returns)
    df["volatility_10"] = (
        df.groupby("Ticker")["daily_return"]
        .rolling(10)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    # 5. Volume Features
    df["volume_change"] = df.groupby("Ticker")["Volume"].pct_change()
    df["avg_volume_10"] = (
        df.groupby("Ticker")["Volume"]
        .rolling(10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # 6. One-step-ahead target variable: Close(t+1)
    df["target"] = df.groupby("Ticker")["Close"].shift(-1)
    
    return df

FEATURE_COLUMNS = [
    "close_lag",
    "daily_return",
    "pct_return",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility_10"
]
