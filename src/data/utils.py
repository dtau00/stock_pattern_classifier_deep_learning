"""
Utility functions for data management.
"""
from datetime import datetime
from typing import Union


# Interval mapping to seconds
INTERVAL_SECONDS = {
    '1m': 60,
    '3m': 3 * 60,
    '5m': 5 * 60,
    '15m': 15 * 60,
    '30m': 30 * 60,
    '1h': 60 * 60,
    '2h': 2 * 60 * 60,
    '4h': 4 * 60 * 60,
    '6h': 6 * 60 * 60,
    '8h': 8 * 60 * 60,
    '12h': 12 * 60 * 60,
    '1d': 24 * 60 * 60,
    '3d': 3 * 24 * 60 * 60,
    '1w': 7 * 24 * 60 * 60,
    '1M': 30 * 24 * 60 * 60,  # Approximate (30 days)
}

# Reference table for common timeframes (from spec)
REFERENCE_TABLE = {
    '1m': {'1day': 1440, '30days': 43200, '365days': 525600},
    '5m': {'1day': 288, '30days': 8640, '365days': 105120},
    '15m': {'1day': 96, '30days': 2880, '365days': 35040},
    '1h': {'1day': 24, '30days': 720, '365days': 8760},
    '4h': {'1day': 6, '30days': 180, '365days': 2190},
    '1d': {'1day': 1, '30days': 30, '365days': 365},
}


def estimate_bar_count(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str
) -> int:
    """
    Estimate number of bars for a given date range and timeframe.

    Args:
        start_date: Start date as string ('YYYY-MM-DD') or datetime object
        end_date: End date as string ('YYYY-MM-DD') or datetime object
        interval: Kline interval (e.g., '1m', '5m', '1h', '1d')

    Returns:
        Estimated number of bars

    Raises:
        ValueError: If invalid interval or date range

    Examples:
        >>> estimate_bar_count('2024-01-01', '2024-01-02', '1h')
        24
        >>> estimate_bar_count('2024-01-01', '2024-12-31', '1h')
        8760
        >>> estimate_bar_count('2024-01-01', '2024-01-02', '1m')
        1440
    """
    # Parse dates if strings
    if isinstance(start_date, str):
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    else:
        start_dt = start_date

    if isinstance(end_date, str):
        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    else:
        end_dt = end_date

    # Validate date range
    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date")

    # Validate interval
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Invalid interval: {interval}. Valid options: {list(INTERVAL_SECONDS.keys())}")

    # Calculate total seconds
    total_seconds = (end_dt - start_dt).total_seconds()

    # Get interval in seconds
    interval_seconds = INTERVAL_SECONDS[interval]

    # Calculate estimated bars
    estimated_bars = int(total_seconds / interval_seconds)

    return estimated_bars


def get_interval_info(interval: str) -> dict:
    """
    Get information about a specific interval.

    Args:
        interval: Kline interval

    Returns:
        Dictionary with interval information:
        - seconds: Interval duration in seconds
        - bars_per_day: Number of bars per day
        - bars_per_30days: Number of bars in 30 days
        - bars_per_365days: Number of bars in 365 days
    """
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"Invalid interval: {interval}")

    seconds = INTERVAL_SECONDS[interval]
    bars_per_day = 24 * 60 * 60 // seconds

    return {
        'interval': interval,
        'seconds': seconds,
        'bars_per_day': bars_per_day,
        'bars_per_30days': bars_per_day * 30,
        'bars_per_365days': bars_per_day * 365,
    }


if __name__ == "__main__":
    import sys
    import io

    # Fix Unicode output on Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("Bar Estimation Calculator Tests")
    print("=" * 60)

    # Test 1: 1 day @ 1h
    print("\nTest 1: 1 day @ 1h")
    count = estimate_bar_count('2024-01-01', '2024-01-02', '1h')
    print(f"  Result: {count} bars")
    print(f"  Expected: 24 bars")
    assert count == 24, f"Expected 24, got {count}"
    print("  [OK]")

    # Test 2: 365 days @ 1h
    print("\nTest 2: 365 days @ 1h")
    count = estimate_bar_count('2024-01-01', '2024-12-31', '1h')
    print(f"  Result: {count} bars")
    print(f"  Expected: 8760 bars")
    assert count == 8760, f"Expected 8760, got {count}"
    print("  [OK]")

    # Test 3: 1 day @ 1m
    print("\nTest 3: 1 day @ 1m")
    count = estimate_bar_count('2024-01-01', '2024-01-02', '1m')
    print(f"  Result: {count} bars")
    print(f"  Expected: 1440 bars")
    assert count == 1440, f"Expected 1440, got {count}"
    print("  [OK]")

    # Test 4: 2 years @ 1d (2024 is a leap year, so 731 days)
    print("\nTest 4: 2 years @ 1d")
    count = estimate_bar_count('2024-01-01', '2026-01-01', '1d')
    print(f"  Result: {count} bars")
    print(f"  Expected: 731 bars (2024 is a leap year)")
    assert count == 731, f"Expected 731, got {count}"
    print("  [OK]")

    # Test 5: Fractional days (36 hours @ 1h)
    print("\nTest 5: 36 hours @ 1h")
    count = estimate_bar_count('2024-01-01 00:00:00', '2024-01-02 12:00:00', '1h')
    print(f"  Result: {count} bars")
    print(f"  Expected: 36 bars")
    assert count == 36, f"Expected 36, got {count}"
    print("  [OK]")

    # Test 6: Verify against reference table
    print("\n" + "=" * 60)
    print("Reference Table Verification")
    print("=" * 60)

    for interval, reference in REFERENCE_TABLE.items():
        print(f"\nInterval: {interval}")

        # 1 day
        count_1d = estimate_bar_count('2024-01-01', '2024-01-02', interval)
        expected_1d = reference['1day']
        status_1d = "[OK]" if count_1d == expected_1d else "[FAIL]"
        print(f"  1 day: {count_1d} bars (expected {expected_1d}) {status_1d}")

        # 30 days
        count_30d = estimate_bar_count('2024-01-01', '2024-01-31', interval)
        expected_30d = reference['30days']
        status_30d = "[OK]" if count_30d == expected_30d else "[FAIL]"
        print(f"  30 days: {count_30d} bars (expected {expected_30d}) {status_30d}")

        # 365 days
        count_365d = estimate_bar_count('2024-01-01', '2024-12-31', interval)
        expected_365d = reference['365days']
        status_365d = "[OK]" if count_365d == expected_365d else "[FAIL]"
        print(f"  365 days: {count_365d} bars (expected {expected_365d}) {status_365d}")

    # Test interval info
    print("\n" + "=" * 60)
    print("Interval Information")
    print("=" * 60)

    for interval in ['1m', '5m', '1h', '4h', '1d']:
        info = get_interval_info(interval)
        print(f"\n{interval}:")
        print(f"  Seconds: {info['seconds']}")
        print(f"  Bars/day: {info['bars_per_day']}")
        print(f"  Bars/30days: {info['bars_per_30days']}")
        print(f"  Bars/365days: {info['bars_per_365days']}")

    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
    print("=" * 60)
