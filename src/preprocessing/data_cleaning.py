"""
Data cleaning module for detecting and handling missing bars in OHLCV data.

This module provides functions to:
- Detect missing bars in timestamp sequences
- Forward fill missing values
- Track which bars were filled
- Detect and exclude windows with large gaps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def detect_missing_bars(df: pd.DataFrame, interval: str) -> List[Tuple[int, int, int]]:
    """
    Detect missing bars in timestamp sequence.

    Args:
        df: DataFrame with 'timestamp' column (Unix milliseconds)
        interval: Time interval string (e.g., '1h', '5m', '1d')

    Returns:
        List of tuples: (missing_start_timestamp, missing_end_timestamp, count)
    """
    # Convert interval string to milliseconds
    interval_ms = _interval_to_milliseconds(interval)

    # Calculate timestamp differences
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    time_diffs = df_sorted['timestamp'].diff()

    # Find gaps (where diff > expected interval)
    gaps = []
    interval_td = pd.Timedelta(milliseconds=interval_ms)

    for idx in range(1, len(df_sorted)):
        diff = time_diffs.iloc[idx]
        if pd.notna(diff):
            # Convert Timedelta to milliseconds if needed
            diff_ms = diff.total_seconds() * 1000 if hasattr(diff, 'total_seconds') else diff

            if diff_ms > interval_ms:
                # Calculate number of missing bars
                num_missing = int((diff_ms / interval_ms) - 1)
                if num_missing > 0:
                    # Get timestamps (may be Timestamp or int)
                    ts_before = df_sorted.iloc[idx - 1]['timestamp']
                    ts_after = df_sorted.iloc[idx]['timestamp']

                    # Add/subtract using Timedelta for Timestamp objects, or raw ms for integers
                    if isinstance(ts_before, pd.Timestamp):
                        missing_start = ts_before + interval_td
                        missing_end = ts_after - interval_td
                        # Convert back to milliseconds for storage
                        missing_start = int(missing_start.timestamp() * 1000)
                        missing_end = int(missing_end.timestamp() * 1000)
                    else:
                        missing_start = ts_before + interval_ms
                        missing_end = ts_after - interval_ms

                    gaps.append((missing_start, missing_end, num_missing))

    return gaps


def forward_fill_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply forward fill to missing values and track filled rows.

    Args:
        df: DataFrame with potential NaN values

    Returns:
        Tuple of (filled DataFrame, fill report dict)
    """
    df_filled = df.copy()

    # Track which rows have NaN values before filling
    nan_mask = df_filled.isna().any(axis=1)
    nan_indices = df_filled[nan_mask].index.tolist()

    # Apply forward fill
    df_filled = df_filled.ffill()

    # Add flag for filled rows
    df_filled['is_filled'] = False
    df_filled.loc[nan_indices, 'is_filled'] = True

    # Create report
    report = {
        'filled_count': len(nan_indices),
        'filled_indices': nan_indices,
        'filled_columns': df.columns[df.isna().any()].tolist()
    }

    return df_filled, report


def _interval_to_milliseconds(interval: str) -> int:
    """
    Convert interval string to milliseconds.

    Args:
        interval: Interval string (e.g., '1m', '5m', '1h', '4h', '1d')

    Returns:
        Interval in milliseconds
    """
    # Parse interval string
    unit = interval[-1]
    value = int(interval[:-1])

    # Convert to milliseconds
    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    elif unit == 'w':
        return value * 7 * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")


def validate_continuity(df: pd.DataFrame, interval: str) -> Dict:
    """
    Validate data continuity and return comprehensive report.

    Args:
        df: DataFrame with 'timestamp' column
        interval: Time interval string

    Returns:
        Dictionary with validation metrics
    """
    # Calculate expected bar count
    if len(df) < 2:
        return {
            'total_bars': len(df),
            'expected_bars': len(df),
            'missing_gaps': [],
            'duplicates': [],
            'usable_bars': len(df)
        }

    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate expected bars
    start_ts = df_sorted['timestamp'].iloc[0]
    end_ts = df_sorted['timestamp'].iloc[-1]
    interval_ms = _interval_to_milliseconds(interval)
    expected_bars = int((end_ts - start_ts) / interval_ms) + 1

    # Detect gaps
    missing_gaps = detect_missing_bars(df, interval)

    # Detect duplicates
    duplicates = df_sorted[df_sorted['timestamp'].duplicated()]['timestamp'].tolist()

    # Calculate usable bars (excluding gaps)
    total_missing = sum(gap[2] for gap in missing_gaps)
    usable_bars = len(df) - len(duplicates)

    report = {
        'total_bars': len(df),
        'expected_bars': expected_bars,
        'missing_gaps': missing_gaps,
        'duplicates': duplicates,
        'usable_bars': usable_bars,
        'continuity_score': usable_bars / expected_bars if expected_bars > 0 else 0.0
    }

    return report


def detect_gaps(df: pd.DataFrame, interval: str, max_gap_length: int = 5) -> List[Tuple[int, int, int]]:
    """
    Detect gaps that exceed the maximum allowed gap length.

    Args:
        df: DataFrame with 'timestamp' column
        interval: Time interval string (e.g., '1h')
        max_gap_length: Maximum allowed gap length in bars (default: 5)

    Returns:
        List of tuples: (gap_start_idx, gap_end_idx, gap_length_bars)
    """
    # Get all gaps
    all_gaps = detect_missing_bars(df, interval)

    # Filter gaps that exceed threshold
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    large_gaps = []

    for gap_start_ts, gap_end_ts, gap_count in all_gaps:
        if gap_count >= max_gap_length:
            # Convert timestamps to the same type as the dataframe for comparison
            if isinstance(df_sorted['timestamp'].iloc[0], pd.Timestamp):
                gap_start_cmp = pd.Timestamp(gap_start_ts, unit='ms')
                gap_end_cmp = pd.Timestamp(gap_end_ts, unit='ms')
            else:
                gap_start_cmp = gap_start_ts
                gap_end_cmp = gap_end_ts

            # Find indices in sorted dataframe
            # Gap occurs between the bar before gap_start_ts and the bar after gap_end_ts
            start_idx = df_sorted[df_sorted['timestamp'] < gap_start_cmp].index[-1] if len(df_sorted[df_sorted['timestamp'] < gap_start_cmp]) > 0 else 0
            end_idx = df_sorted[df_sorted['timestamp'] > gap_end_cmp].index[0] if len(df_sorted[df_sorted['timestamp'] > gap_end_cmp]) > 0 else len(df_sorted) - 1

            large_gaps.append((start_idx, end_idx, gap_count))

    return large_gaps


def flag_exclusions(df: pd.DataFrame, gaps: List[Tuple[int, int, int]],
                    window_size: int = 127) -> np.ndarray:
    """
    Create boolean array flagging rows that should be excluded from window creation.

    A row should be excluded if it would be part of a window that overlaps with a gap.

    Args:
        df: DataFrame to analyze
        gaps: List of (start_idx, end_idx, gap_length) tuples from detect_gaps()
        window_size: Length of sliding windows (default: 127)

    Returns:
        Boolean numpy array of length len(df) where True = exclude this row
    """
    exclusion_flags = np.zeros(len(df), dtype=bool)

    # For each gap, mark all rows that would be part of an overlapping window
    for gap_start_idx, gap_end_idx, gap_length in gaps:
        # A window of length L starting at position i includes indices [i, i+L-1]
        # It overlaps with gap if any part of [i, i+L-1] intersects with the gap

        # Window can start as early as (gap_start_idx - window_size + 1)
        # Window can start as late as gap_end_idx
        window_start_min = max(0, gap_start_idx - window_size + 1)
        window_start_max = min(len(df) - 1, gap_end_idx)

        # Mark all rows that could be part of an excluded window
        affected_start = window_start_min
        affected_end = min(len(df) - 1, window_start_max + window_size - 1)

        exclusion_flags[affected_start:affected_end + 1] = True

    return exclusion_flags


def mark_excluded_windows(df: pd.DataFrame, gaps: List[Tuple[int, int, int]],
                          sequence_length: int = 127) -> pd.DataFrame:
    """
    Mark windows that overlap with gaps as excluded.

    A window overlaps with a gap if any part of the window includes the gap.
    For a window starting at position i with length L, it overlaps with a gap
    at position g if: i <= g < i + L

    Args:
        df: DataFrame to mark
        gaps: List of (start_idx, end_idx, gap_length) tuples
        sequence_length: Length of sliding windows (default: 127)

    Returns:
        DataFrame with 'is_excluded' column added
    """
    df_marked = df.copy()
    df_marked['is_excluded'] = flag_exclusions(df, gaps, sequence_length)
    return df_marked
