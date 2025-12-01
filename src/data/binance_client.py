"""
Binance API client for fetching historical OHLCV data.
"""
import time
import requests
from datetime import datetime
from typing import List, Optional


class BinanceClient:
    """
    Client for interacting with Binance public API endpoints.
    No authentication required for market data.
    """

    BASE_URL = "https://api.binance.com"
    RATE_LIMIT_REQUESTS_PER_MINUTE = 1200

    def __init__(self):
        """Initialize Binance client with public endpoints."""
        self.session = requests.Session()
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()

    def _handle_rate_limit(self):
        """
        Implement rate limiting to stay within Binance's 1200 requests/minute limit.
        Uses a simple sliding window approach.
        """
        current_time = time.time()

        # Reset counter if more than 60 seconds have passed
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time

        # If we're approaching the limit, sleep until the window resets
        if self.request_count >= self.RATE_LIMIT_REQUESTS_PER_MINUTE - 10:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                print(f"Rate limit approaching, sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()

        # Small delay between requests to be respectful
        time_since_last = current_time - self.last_request_time
        if time_since_last < 0.05:  # 50ms minimum between requests
            time.sleep(0.05 - time_since_last)

        self.last_request_time = time.time()
        self.request_count += 1

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[List]:
        """
        Fetch OHLCV kline/candlestick bars from Binance.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            limit: Number of bars to fetch (max 1000, default 1000)

        Returns:
            List of klines, each kline is a list:
            [
                timestamp,      # Open time (ms)
                open,           # Open price
                high,           # High price
                low,            # Low price
                close,          # Close price
                volume,         # Volume
                close_time,     # Close time (ms)
                quote_volume,   # Quote asset volume
                trades,         # Number of trades
                taker_buy_base, # Taker buy base asset volume
                taker_buy_quote,# Taker buy quote asset volume
                ignore          # Unused field
            ]

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        self._handle_rate_limit()

        endpoint = f"{self.BASE_URL}/api/v3/klines"

        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)  # Binance max is 1000
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                error_data = response.json()
                raise ValueError(f"Invalid request parameters: {error_data.get('msg', str(e))}")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded. Please try again later.")
            else:
                raise Exception(f"HTTP error occurred: {e}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error occurred: {e}")

    def test_connection(self) -> bool:
        """
        Test connection to Binance API by pinging the server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            endpoint = f"{self.BASE_URL}/api/v3/ping"
            response = self.session.get(endpoint, timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_server_time(self) -> int:
        """
        Get Binance server time.

        Returns:
            Server time in milliseconds
        """
        endpoint = f"{self.BASE_URL}/api/v3/time"
        response = self.session.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.json()['serverTime']


if __name__ == "__main__":
    # Simple test
    import sys
    import io

    # Fix Unicode output on Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    client = BinanceClient()

    print("Testing connection...")
    if client.test_connection():
        print("[OK] Connection successful")
    else:
        print("[FAIL] Connection failed")
        exit(1)

    print("\nFetching server time...")
    server_time = client.get_server_time()
    server_dt = datetime.fromtimestamp(server_time / 1000)
    print(f"[OK] Server time: {server_dt}")

    print("\nFetching 5 BTCUSDT 1h klines...")
    klines = client.get_klines('BTCUSDT', '1h', limit=5)
    print(f"[OK] Fetched {len(klines)} klines")

    if klines:
        print("\nFirst kline:")
        first = klines[0]
        dt = datetime.fromtimestamp(first[0] / 1000)
        print(f"  Time: {dt}")
        print(f"  Open: {first[1]}")
        print(f"  High: {first[2]}")
        print(f"  Low: {first[3]}")
        print(f"  Close: {first[4]}")
        print(f"  Volume: {first[5]}")

        # Verify timestamps are sequential
        timestamps = [k[0] for k in klines]
        is_sequential = all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1))
        print(f"\n[OK] Timestamps sequential: {is_sequential}")
