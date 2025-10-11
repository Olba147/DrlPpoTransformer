"""Fetch Binance BTCUSDT intraday klines from the earliest available record."""

import argparse
import csv
import datetime as dt
import sys
import time
from typing import Optional
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
COLUMN_NAMES = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def parse_time_argument(value: str) -> int:
    """Convert CLI time input into milliseconds since the Unix epoch."""
    value = value.strip()
    try:
        return int(float(value) * 1000)
    except ValueError:
        pass

    iso_candidate = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed = None

    if parsed is None:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = dt.datetime.strptime(value, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        raise argparse.ArgumentTypeError(f"Could not parse time value: {value!r}")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)

    return int(parsed.timestamp() * 1000)


def fetch_batch(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int],
    limit: int,
    pause_seconds: float,
    max_retries: int,
):
    """Fetch a single batch of klines from Binance with retry handling."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_ms,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise RuntimeError("Failed to reach Binance after multiple attempts") from exc
            time.sleep(pause_seconds * attempt)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code in {418, 429}:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_time = float(retry_after)
            except (TypeError, ValueError):
                wait_time = 1.0
            time.sleep(max(wait_time, pause_seconds))
            continue

        if attempt >= max_retries:
            raise RuntimeError(
                f"Binance API error {response.status_code}: {response.text.strip()}"
            )

        time.sleep(pause_seconds * attempt)


def download_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int],
    limit: int,
    pause_seconds: float,
    max_retries: int,
):
    """Yield successive klines until the API has no more data."""
    next_start = start_ms
    while True:
        batch = fetch_batch(
            symbol=symbol,
            interval=interval,
            start_ms=next_start,
            end_ms=end_ms,
            limit=limit,
            pause_seconds=pause_seconds,
            max_retries=max_retries,
        )

        if not batch:
            return

        row_appended = False
        stop = False
        next_batch_start: Optional[int] = None
        for entry in batch:
            close_time = int(entry[6])
            if end_ms is not None and close_time > end_ms:
                stop = True
                break

            yield entry
            row_appended = True
            next_batch_start = close_time + 1

        if not row_appended:
            return

        if stop:
            return

        if len(batch) < limit:
            return

        if next_batch_start is None:
            return

        next_start = next_batch_start
        time.sleep(pause_seconds)


def format_progress(timestamp_ms: int) -> str:
    """Return a readable UTC timestamp for progress logging."""
    utc_time = dt.datetime.fromtimestamp(timestamp_ms / 1000, tz=dt.timezone.utc)
    return utc_time.isoformat().replace("+00:00", "Z")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Binance intraday klines starting from the earliest available record "
            "and save them to a CSV file."
        )
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair to download (default: BTCUSDT)")
    parser.add_argument("--interval", default="1m", help="Kline interval per Binance API (default: 1m)")
    parser.add_argument(
        "--output",
        default="btcusdt_1m_klines.csv",
        help="Destination CSV file (default: btcusdt_1m_klines.csv)",
    )
    parser.add_argument(
        "--start",
        type=parse_time_argument,
        default=None,
        help="Optional start time (ISO 8601 or epoch seconds). Defaults to earliest.",
    )
    parser.add_argument(
        "--end",
        type=parse_time_argument,
        default=None,
        help="Optional end time (ISO 8601 or epoch seconds). Defaults to latest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows per API call (max 1000).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.2,
        help="Seconds to sleep between API calls to respect rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for transient API failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    symbol = args.symbol.upper()
    start_ms = args.start if args.start is not None else 0
    end_ms = args.end

    total_rows = 0
    latest_close: Optional[int] = None

    try:
        file_name =  f"../Data/binance/{symbol}_{args.interval}_{start_ms}_{end_ms}_klines.csv"
        with open(file_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMN_NAMES)

            for row in download_klines(
                symbol=symbol,
                interval=args.interval,
                start_ms=start_ms,
                end_ms=end_ms,
                limit=args.batch_size,
                pause_seconds=args.pause,
                max_retries=args.max_retries,
            ):
                writer.writerow(row[: len(COLUMN_NAMES)])
                total_rows += 1
                latest_close = int(row[6])

                if total_rows % 10000 == 0:
                    print(
                        f"Fetched {total_rows} rows through {format_progress(latest_close)}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("Download interrupted by user.", file=sys.stderr)
        return

    if total_rows == 0:
        print("No data returned. Check your symbol, interval, or network connection.", file=sys.stderr)
        return

    assert latest_close is not None
    print(
        f"Saved {total_rows} rows for {symbol} {args.interval} klines from {format_progress(start_ms if start_ms else 'earliest')} through {format_progress(latest_close)}\n"
        f"Output file: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
