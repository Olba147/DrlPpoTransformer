import os
from datetime import date
import pandas as pd
import json
from polygon import RESTClient

# ---------------- CONFIGURATION ----------------
try:
    with open(r"secrets.json", "r") as f:
        secrets = json.load(f)
        polygon_API_KEY = secrets["POLYGON_API_KEY"]
        print(os.getcwd())
except Exception:
    print("failed to read secrets.json, please create it")
    exit

API_KEY = polygon_API_KEY
START_DATE = "2021-01-25"
END_DATE = "2026-01-25" 
TIMEFRAME = "minute" # 1 minute bars
MULTIPLIER = 1

# Current Dow 30 + 4 Major ETFs
TICKERS = [
    "MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "VZ", "V", "WMT",
    "DIA", "SPY", "QQQ" 
]

OUTPUT_DIR = r"../Data/polygon/data_raw_1m"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------



def download_ticker(client, ticker):
    print(f"Downloading {ticker}...")
    try:
        # Polygon's list_aggs handles pagination automatically
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=MULTIPLIER,
            timespan=TIMEFRAME,
            from_=START_DATE,
            to=END_DATE,
            limit=50000,
            adjusted=True  # IMPORTANT: Handles splits automatically
        ):
            aggs.append({
                "timestamp": a.timestamp,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
                "vwap": a.vwap,
                "transactions": a.transactions
            })
            
        if not aggs:
            print(f"No data found for {ticker}")
            return

        df = pd.DataFrame(aggs)
        
        # Convert ms timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        
        # Save to Parquet (Better than CSV for large datasets)
        save_path = os.path.join(OUTPUT_DIR, f"{ticker}.parquet")
        df.to_parquet(save_path)
        print(f"Saved {ticker}: {len(df)} rows")
        
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

def main():
    client = RESTClient(API_KEY)
    for ticker in TICKERS:
        download_ticker(client, ticker)

if __name__ == "__main__":
    main()