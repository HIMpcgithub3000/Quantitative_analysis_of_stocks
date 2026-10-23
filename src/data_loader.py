import os
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-01-01"

def download_ohlc(ticker, start_date=DEFAULT_START, end_date=DEFAULT_END, csv_dir="data/csv", json_dir="data/json"):
    """
    Downloads historical OHLCV data for a ticker and saves CSV and JSON representations.
    """
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    print(f"Downloading market data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        print(f"Warning: No data found for {ticker}")
        return None
        
    df = df.reset_index()
    # Flatten multi-index columns if returned by newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    df["Ticker"] = ticker
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    
    csv_path = os.path.join(csv_dir, f"{ticker}_ohlc.csv")
    json_path = os.path.join(json_dir, f"{ticker}_ohlc.json")
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", date_format="iso")
    print(f"Saved {ticker} -> {csv_path} and {json_path}")
    return df

def load_and_clean_data(csv_dir="data/csv", save_path="data/combined_ohlc.csv"):
    """
    Loads all raw ticker CSVs, cleans data, handles missing values,
    removes anomalies, and saves a combined dataset.
    """
    dfs = []
    for file in sorted(os.listdir(csv_dir)):
        if file.endswith(".csv"):
            filepath = os.path.join(csv_dir, file)
            df = pd.read_csv(filepath)
            dfs.append(df)
            
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
        
    data = pd.concat(dfs, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    
    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        
    data.sort_values(by=["Ticker", "Date"], inplace=True)
    
    # Modern forward-fill without deprecated fillna(method='ffill')
    data.ffill(inplace=True)
    data.dropna(subset=num_cols, inplace=True)
    
    # Filter anomalies (non-positive volume and non-positive prices)
    data = data[data["Volume"] > 0]
    for col in ["Open", "High", "Low", "Close"]:
        data = data[data[col] > 0]
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)
        print(f"Cleaned dataset saved: {save_path} ({len(data)} observations)")
        
    return data
