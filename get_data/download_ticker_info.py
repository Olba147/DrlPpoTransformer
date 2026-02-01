import argparse
import csv
import json
import os
from datetime import datetime, timedelta, date
import requests

API_BASE = "https://api.polygon.io"

TICKERS = [
    "MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "VZ", "V", "WMT",
    "DIA", "SPY", "QQQ",
]


def load_polygon_api_key() -> str:
    try:
        with open(r"secrets.json", "r") as f:
            secrets = json.load(f)
            return secrets["POLYGON_API_KEY"]
    except Exception:
        raise RuntimeError("failed to read secrets.json, please create it")


def fetch_ticker_overview(ticker: str, api_key: str, as_of_date: str | None) -> dict:
    params = {"apiKey": api_key}
    if as_of_date:
        params["date"] = as_of_date
    resp = requests.get(f"{API_BASE}/v3/reference/tickers/{ticker}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", {}) or {}


def fetch_avg_daily_volume(
    ticker: str,
    api_key: str,
    end_date: date,
    lookback_days: int = 30,
) -> float | None:
    start_date = end_date - timedelta(days=lookback_days)
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    url = f"{API_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return None
    volumes = []
    for r in results:
        if "v" in r:
            volumes.append(r["v"])
        elif "volume" in r:
            volumes.append(r["volume"])
    if not volumes:
        return None
    return sum(volumes) / len(volumes)


def _normalize_sector(overview: dict) -> str | None:
    sector = overview.get("sector") or overview.get("gics_sector") or overview.get("sector_name")
    if sector:
        return sector

    industry = (
        overview.get("industry_description")
        or overview.get("industry")
        or overview.get("industry_name")
        or overview.get("sic_description")
    )
    if not industry:
        return None

    industry_l = industry.lower()
    sector_keywords = {
        "Communication Services": ["telecom", "communication", "media", "entertainment", "interactive"],
        "Consumer Discretionary": ["retail", "automobile", "apparel", "leisure", "hotel", "restaurant", "travel"],
        "Consumer Staples": ["beverage", "food", "tobacco", "household", "personal products", "consumer staples"],
        "Energy": ["oil", "gas", "energy", "coal", "drilling", "refining", "pipeline"],
        "Financials": ["bank", "insurance", "financial", "capital markets", "broker", "asset management"],
        "Health Care": ["health", "pharma", "biotech", "medical", "hospital", "drug"],
        "Industrials": ["industrial", "aerospace", "defense", "machinery", "transportation", "construction"],
        "Information Technology": ["software", "hardware", "semiconductor", "it services", "technology"],
        "Materials": ["chemical", "materials", "mining", "metals", "paper", "forestry"],
        "Real Estate": ["reit", "real estate", "property", "mortgage"],
        "Utilities": ["utility", "electric", "gas utility", "water utility", "power generation"],
    }
    for sector_name, keywords in sector_keywords.items():
        if any(k in industry_l for k in keywords):
            return sector_name
    return None


def fetch_ticker_info(
    ticker: str,
    api_key: str,
    market_cap_date: str | None,
    avg_volume_days: int = 30,
) -> dict:
    overview = fetch_ticker_overview(ticker, api_key, market_cap_date)
    market = overview.get("market")
    name = overview.get("name")
    sector = _normalize_sector(overview) if market == "stocks" else None
    market_cap = overview.get("market_cap")

    if market_cap_date:
        end_date = datetime.strptime(market_cap_date, "%Y-%m-%d").date()
    else:
        end_date = datetime.utcnow().date()

    avg_daily_volume = fetch_avg_daily_volume(
        ticker=ticker,
        api_key=api_key,
        end_date=end_date,
        lookback_days=avg_volume_days,
    )

    return {
        "ticker": ticker,
        "company_name": name,
        "market": market,
        "sector": sector,
        "market_cap": market_cap,
        "avg_daily_volume": avg_daily_volume,
        "market_cap_date": market_cap_date,
    }


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-cap-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--avg-volume-days", type=int, default=30)
    parser.add_argument("--output", default=r"Data/polygon/ticker_info/ticker_info.csv")
    args = parser.parse_args()

    api_key = load_polygon_api_key()
    rows = []
    for ticker in TICKERS:
        try:
            info = fetch_ticker_info(
                ticker=ticker,
                api_key=api_key,
                market_cap_date=args.market_cap_date,
                avg_volume_days=args.avg_volume_days,
            )
            rows.append(info)
            print(f"Fetched {ticker}")
        except Exception as e:
            print(f"Failed {ticker}: {e}")

    write_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
