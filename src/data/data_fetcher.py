"""
OHLCV data fetcher with pagination and retry logic.
"""
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Handle both relative and absolute imports
try:
    from .binance_client import BinanceClient
except ImportError:
    from binance_client import BinanceClient


class OHLCVDataFetcher:
    """
    Fetches historical OHLCV data with pagination support for large date ranges.
    """

    # Interval mapping to milliseconds
    INTERVAL_MS = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
        '1M': 30 * 24 * 60 * 60 * 1000,  # Approximate
    }

    def __init__(self):
        """Initialize data fetcher with Binance client."""
        self.client = BinanceClient()

    def _parse_date(self, date_str: str) -> int:
        """
        Parse date string to timestamp in milliseconds.

        Args:
            date_str: Date string in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'

        Returns:
            Timestamp in milliseconds
        """
        try:
            # Try parsing with time
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Try parsing date only
            dt = datetime.strptime(date_str, '%Y-%m-%d')

        return int(dt.timestamp() * 1000)

    def _paginate_requests(self, start_ts: int, end_ts: int, interval: str) -> list:
        """
        Calculate pagination parameters for API requests.
        Binance API returns max 1000 bars per request.

        Args:
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            interval: Kline interval (e.g., '1h')

        Returns:
            List of (start_ts, end_ts) tuples for each request
        """
        if interval not in self.INTERVAL_MS:
            raise ValueError(f"Invalid interval: {interval}")

        interval_ms = self.INTERVAL_MS[interval]
        max_bars_per_request = 1000

        # Calculate maximum time span per request
        max_time_span = interval_ms * max_bars_per_request

        requests = []
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + max_time_span, end_ts)
            requests.append((current_start, current_end))
            current_start = current_end

        return requests

    def _retry_request(self, request_func, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        Retry request with exponential backoff.

        Args:
            request_func: Function to call (should return data)
            max_retries: Maximum number of retries
            backoff_factor: Multiplier for backoff delay

        Returns:
            Result from request_func

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                return request_func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor ** attempt
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"All {max_retries} retry attempts failed")

        raise last_exception

    def fetch_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with automatic pagination.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            end_date: End date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            verbose: Print progress messages

        Returns:
            pandas DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValueError: If invalid parameters provided
            Exception: If data fetching fails
        """
        # Parse dates to timestamps
        start_ts = self._parse_date(start_date)
        end_ts = self._parse_date(end_date)

        if start_ts >= end_ts:
            raise ValueError("start_date must be before end_date")

        # Check if end date is in the future
        now_ts = int(datetime.now().timestamp() * 1000)
        if end_ts > now_ts:
            if verbose:
                print(f"Warning: end_date is in the future, adjusting to current time")
            end_ts = now_ts

        # Calculate pagination
        requests = self._paginate_requests(start_ts, end_ts, interval)

        if verbose:
            print(f"Fetching {symbol} {interval} data from {start_date} to {end_date}")
            print(f"Total requests required: {len(requests)}")

        # Fetch data in chunks
        all_klines = []

        for i, (chunk_start, chunk_end) in enumerate(requests):
            if verbose and len(requests) > 1:
                progress = (i + 1) / len(requests) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(requests)})")

            # Fetch with retry logic
            def fetch_chunk():
                return self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    limit=1000
                )

            klines = self._retry_request(fetch_chunk)

            if klines:
                all_klines.extend(klines)

        if not all_klines:
            raise Exception("No data returned from API")

        # Convert to DataFrame
        df = self._klines_to_dataframe(all_klines)

        # Remove duplicates (can occur at chunk boundaries)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        if verbose and len(df) < initial_count:
            print(f"Removed {initial_count - len(df)} duplicate bars")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        if verbose:
            print(f"Fetched {len(df)} bars successfully")

        return df

    def _calculate_expected_bars(self, start_ts: int, end_ts: int, interval: str) -> int:
        """
        Calculate expected number of bars for a given time range and interval.

        Args:
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            interval: Kline interval

        Returns:
            Expected number of bars
        """
        if interval not in self.INTERVAL_MS:
            raise ValueError(f"Invalid interval: {interval}")

        interval_ms = self.INTERVAL_MS[interval]
        total_time = end_ts - start_ts
        expected = int(total_time / interval_ms)

        return expected

    def _validate_continuity(self, df: pd.DataFrame, interval: str) -> dict:
        """
        Validate data continuity - check for gaps and duplicates.

        Args:
            df: DataFrame with timestamp column
            interval: Kline interval (e.g., '1h')

        Returns:
            Validation report dictionary with:
            - total_bars: Actual number of bars
            - expected_bars: Expected number of bars
            - missing_gaps: List of (start_ts, end_ts, gap_count) tuples
            - duplicates: List of duplicate timestamps
            - usable_segments: Estimate of usable bars (excluding gap-affected windows)
        """
        if interval not in self.INTERVAL_MS:
            raise ValueError(f"Invalid interval: {interval}")

        interval_ms = self.INTERVAL_MS[interval]

        # Calculate expected bars
        start_ts = df['timestamp'].iloc[0]
        end_ts = df['timestamp'].iloc[-1]
        expected_bars = self._calculate_expected_bars(start_ts, end_ts, interval) + 1

        # Check for duplicates
        duplicates = df[df['timestamp'].duplicated()]['timestamp'].tolist()

        # Check for gaps
        timestamps = df['timestamp'].values
        diffs = timestamps[1:] - timestamps[:-1]

        missing_gaps = []
        gap_indices = []

        for i, diff in enumerate(diffs):
            if diff > interval_ms:
                # Gap detected
                gap_start = timestamps[i]
                gap_end = timestamps[i + 1]
                gap_bars = int((diff - interval_ms) / interval_ms)

                missing_gaps.append({
                    'start_timestamp': int(gap_start),
                    'end_timestamp': int(gap_end),
                    'missing_bars': gap_bars
                })
                gap_indices.append(i)

        # Calculate usable segments
        # This is a rough estimate - windows overlapping with gaps would be excluded
        total_bars = len(df)
        usable_segments = total_bars - len(duplicates)

        report = {
            'total_bars': total_bars,
            'expected_bars': expected_bars,
            'missing_gaps': missing_gaps,
            'gap_count': len(missing_gaps),
            'total_missing_bars': sum(g['missing_bars'] for g in missing_gaps),
            'duplicates': duplicates,
            'duplicate_count': len(duplicates),
            'usable_segments': usable_segments,
            'data_quality': 'good' if len(missing_gaps) == 0 and len(duplicates) == 0 else 'has_issues'
        }

        return report

    def _klines_to_dataframe(self, klines: list) -> pd.DataFrame:
        """
        Convert raw klines data to pandas DataFrame.

        Args:
            klines: List of klines from Binance API

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Select and convert relevant columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Convert types (use int64 for timestamps to avoid overflow)
        df['timestamp'] = df['timestamp'].astype('int64')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        return df


if __name__ == "__main__":
    import sys
    import io

    # Fix Unicode output on Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    fetcher = OHLCVDataFetcher()

    # Test 1: Small fetch (< 1000 bars, single page)
    print("=" * 60)
    print("Test 1: Single-page fetch (10 days @ 1h = ~240 bars)")
    print("=" * 60)
    df1 = fetcher.fetch_historical_data(
        'BTCUSDT',
        '1h',
        '2024-01-01',
        '2024-01-10'
    )
    print(f"\nShape: {df1.shape}")
    print(f"Columns: {df1.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(df1.head(3))
    print("\nLast 3 rows:")
    print(df1.tail(3))
    print("\nData types:")
    print(df1.dtypes)

    # Test 2: Multi-page fetch (> 1000 bars)
    print("\n" + "=" * 60)
    print("Test 2: Multi-page fetch (60 days @ 1h = ~1440 bars)")
    print("=" * 60)
    df2 = fetcher.fetch_historical_data(
        'BTCUSDT',
        '1h',
        '2024-01-01',
        '2024-03-01'
    )
    print(f"\nShape: {df2.shape}")

    # Test 3: Validation report
    print("\n" + "=" * 60)
    print("Test 3: Data validation and continuity check")
    print("=" * 60)
    report = fetcher._validate_continuity(df2, '1h')

    print(f"\nValidation Report:")
    print(f"  Total bars: {report['total_bars']}")
    print(f"  Expected bars: {report['expected_bars']}")
    print(f"  Missing gaps: {report['gap_count']}")
    print(f"  Total missing bars: {report['total_missing_bars']}")
    print(f"  Duplicates: {report['duplicate_count']}")
    print(f"  Usable segments: {report['usable_segments']}")
    print(f"  Data quality: {report['data_quality']}")

    if report['gap_count'] > 0:
        print(f"\nFirst 5 gaps:")
        for gap in report['missing_gaps'][:5]:
            gap_start_dt = datetime.fromtimestamp(gap['start_timestamp'] / 1000)
            gap_end_dt = datetime.fromtimestamp(gap['end_timestamp'] / 1000)
            print(f"    {gap_start_dt} -> {gap_end_dt}: {gap['missing_bars']} missing bars")

    # Check for NaN values
    print(f"\nNaN check:")
    nan_counts = df2.isna().sum()
    print(nan_counts)
    if nan_counts.sum() == 0:
        print("  [OK] No NaN values found")

    print("\n[OK] All tests completed successfully!")
