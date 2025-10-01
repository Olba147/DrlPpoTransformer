"""Utilities for fetching Alpha Vantage intraday data with rate limiting."""

from collections import deque
import json
import time

import pandas as pd
from alpha_vantage.timeseries import TimeSeries


class AlphaVantageDataGetter:
    """Fetch Alpha Vantage intraday data without exceeding rate limits."""

    def __init__(self, api_key, max_requests_per_minute: int = 70):
        """Prepare the data getter with credentials and a capped request rate."""
        self.api_key = api_key
        self.max_requests_per_minute = max_requests_per_minute
        self._request_timestamps = deque()
        self._timeseries = TimeSeries(key=self.api_key, output_format="pandas")

    def _acquire_request_slot(self):
        """Block until a request slot is available beneath the rate limit."""
        window_seconds = 60
        while True:
            now = time.monotonic()
            # Drop timestamps that are outside the current window.
            while self._request_timestamps and now - self._request_timestamps[0] >= window_seconds:
                self._request_timestamps.popleft()

            if len(self._request_timestamps) < self.max_requests_per_minute:
                self._request_timestamps.append(now)
                return

            wait_time = window_seconds - (now - self._request_timestamps[0])
            if wait_time > 0:
                time.sleep(wait_time)

    def get_monthly_intraday(self, symbol, interval, month):
        """Fetch intraday data for a single month using Alpha Vantage."""
        self._acquire_request_slot()
        try:
            data, _ = self._timeseries.get_intraday(
                symbol=symbol,
                month=month,
                interval=interval,
                outputsize="full",
            )
            print(f"got: {symbol}, for month: {month}")
        except Exception as e:
            print(f"error getting: {symbol}, for month: {month} - ERROR: {e}")
            return None
        return data

    def get_intraday_history(self, symbol, start, end, interval):
        """Fetch and concatenate intraday data across multiple months."""
        monthly_datas = []
        for month in pd.date_range(start=start, end=end, freq="MS"):
            month_str = month.strftime("%Y-%m")
            monthly_data = self.get_monthly_intraday(symbol, interval, month_str)
            if monthly_data is not None:
                monthly_datas.append(monthly_data)

        if not monthly_datas:
            return None

        return pd.concat(monthly_datas)


if __name__ == "__main__":
    with open("secrets.json", "r") as f:
        secrets = json.load(f)
        api_key = secrets["ALPHA_VANTAGE_API_KEY"]
    getter = AlphaVantageDataGetter(api_key)
    data = getter.get_intraday_history("AAPL", "2025-01-01", "2025-09-01", "5min")

    if data is not None:
        data.to_csv("data.csv")
