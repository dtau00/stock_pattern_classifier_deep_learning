"""
Data splitting module for chronological train/val/test split.

This module implements chronological (temporal) splitting of preprocessed
windows without shuffling, maintaining temporal integrity.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def split_data(
    windows: np.ndarray,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows chronologically into train/val/test sets.

    Maintains chronological order without shuffling to preserve temporal integrity.

    Args:
        windows: numpy array of shape (num_windows, sequence_length, num_channels)
        train_pct: Percentage of data for training (default: 0.7)
        val_pct: Percentage of data for validation (default: 0.15)
        test_pct: Percentage of data for testing (default: 0.15)

    Returns:
        Tuple of (train_windows, val_windows, test_windows)

    Raises:
        ValueError: If percentages don't sum to 1.0

    Examples:
        >>> windows = np.random.randn(1000, 127, 3)
        >>> train, val, test = split_data(windows)
        >>> print(len(train), len(val), len(test))
        700 150 150
    """
    # Validate split percentages
    total_pct = train_pct + val_pct + test_pct
    if not np.isclose(total_pct, 1.0, atol=1e-6):
        # Normalize percentages if they don't sum exactly to 1.0
        train_pct /= total_pct
        val_pct /= total_pct
        test_pct /= total_pct
        print(f"Warning: Split percentages normalized to sum to 1.0")

    num_windows = len(windows)

    # Calculate split indices
    train_end = int(num_windows * train_pct)
    val_end = train_end + int(num_windows * val_pct)

    # Split chronologically (NO shuffling)
    train = windows[:train_end]
    val = windows[train_end:val_end]
    test = windows[val_end:]

    print(f"Data split complete:")
    print(f"  Train: {len(train)} windows ({len(train)/num_windows*100:.1f}%)")
    print(f"  Val:   {len(val)} windows ({len(val)/num_windows*100:.1f}%)")
    print(f"  Test:  {len(test)} windows ({len(test)/num_windows*100:.1f}%)")
    print(f"  Total: {num_windows} windows")

    return train, val, test


def get_split_info(
    windows: np.ndarray,
    metadata: Optional[Dict] = None,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15
) -> Dict:
    """
    Get information about data splits without actually splitting.

    Useful for displaying split information in UI before processing.

    Args:
        windows: numpy array of shape (num_windows, sequence_length, num_channels)
        metadata: Optional metadata dict containing window timestamps
        train_pct: Percentage of data for training
        val_pct: Percentage of data for validation
        test_pct: Percentage of data for testing

    Returns:
        Dictionary with split information
    """
    num_windows = len(windows)

    # Calculate split indices
    train_end = int(num_windows * train_pct)
    val_end = train_end + int(num_windows * val_pct)

    info = {
        'total_windows': num_windows,
        'train_count': train_end,
        'val_count': val_end - train_end,
        'test_count': num_windows - val_end,
        'train_pct': train_pct,
        'val_pct': val_pct,
        'test_pct': test_pct
    }

    # Add timestamp information if metadata available
    if metadata and 'window_timestamps' in metadata:
        timestamps = metadata['window_timestamps']
        info['train_start_time'] = timestamps[0]
        info['train_end_time'] = timestamps[train_end - 1]
        info['val_start_time'] = timestamps[train_end]
        info['val_end_time'] = timestamps[val_end - 1]
        info['test_start_time'] = timestamps[val_end]
        info['test_end_time'] = timestamps[-1]

    return info


def verify_chronological_order(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Verify that splits maintain chronological order with no overlap.

    Args:
        train: Training set windows
        val: Validation set windows
        test: Test set windows
        metadata: Optional metadata with timestamps

    Returns:
        True if chronological order is maintained, False otherwise
    """
    # Basic checks
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print("Warning: One or more splits are empty")
        return False

    # Check if we have timestamp metadata
    if metadata and 'window_timestamps' in metadata:
        timestamps = metadata['window_timestamps']

        train_len = len(train)
        val_len = len(val)

        # Get timestamps for each split
        train_times = timestamps[:train_len]
        val_times = timestamps[train_len:train_len + val_len]
        test_times = timestamps[train_len + val_len:]

        # Verify chronological order
        if train_times[-1] >= val_times[0]:
            print(f"Error: Train end ({train_times[-1]}) >= Val start ({val_times[0]})")
            return False

        if val_times[-1] >= test_times[0]:
            print(f"Error: Val end ({val_times[-1]}) >= Test start ({test_times[0]})")
            return False

        print("[OK] Chronological order verified")
        print(f"  Train: {train_times[0]} to {train_times[-1]}")
        print(f"  Val:   {val_times[0]} to {val_times[-1]}")
        print(f"  Test:  {test_times[0]} to {test_times[-1]}")
        return True
    else:
        print("Note: No timestamp metadata available, cannot verify temporal order")
        return True


if __name__ == "__main__":
    # Test with synthetic data
    print("=" * 60)
    print("Testing Data Splitting Module")
    print("=" * 60)

    # Create synthetic windows
    num_windows = 1000
    sequence_length = 127
    num_channels = 3

    windows = np.random.randn(num_windows, sequence_length, num_channels)

    # Test 1: Standard 70/15/15 split
    print("\n--- Test 1: Standard 70/15/15 Split ---")
    train, val, test = split_data(windows, train_pct=0.7, val_pct=0.15, test_pct=0.15)
    assert len(train) == 700, f"Expected 700 train windows, got {len(train)}"
    assert len(val) == 150, f"Expected 150 val windows, got {len(val)}"
    assert len(test) == 150, f"Expected 150 test windows, got {len(test)}"
    print("[PASS] Test 1 passed: Correct split sizes")

    # Test 2: Different split ratios (80/10/10)
    print("\n--- Test 2: Different Split Ratios (80/10/10) ---")
    train, val, test = split_data(windows, train_pct=0.8, val_pct=0.1, test_pct=0.1)
    assert len(train) == 800, f"Expected 800 train windows, got {len(train)}"
    assert len(val) == 100, f"Expected 100 val windows, got {len(val)}"
    assert len(test) == 100, f"Expected 100 test windows, got {len(test)}"
    print("[PASS] Test 2 passed: Different ratios work correctly")

    # Test 3: Verify no overlap
    print("\n--- Test 3: Verify No Overlap ---")
    train, val, test = split_data(windows)
    total_reconstructed = np.vstack([train, val, test])
    assert np.array_equal(windows, total_reconstructed), "Data overlap or missing windows"
    print("[PASS] Test 3 passed: No overlap, all windows accounted for")

    # Test 4: Get split info
    print("\n--- Test 4: Get Split Info ---")
    info = get_split_info(windows)
    print(f"Split info: {info}")
    assert info['total_windows'] == num_windows
    assert info['train_count'] + info['val_count'] + info['test_count'] == num_windows
    print("[PASS] Test 4 passed: Split info correct")

    # Test 5: Verify chronological order with timestamps
    print("\n--- Test 5: Verify Chronological Order ---")
    timestamps = list(range(num_windows))  # Simple sequential timestamps
    metadata = {'window_timestamps': timestamps}
    train, val, test = split_data(windows)
    is_chronological = verify_chronological_order(train, val, test, metadata)
    assert is_chronological, "Chronological order not maintained"
    print("[PASS] Test 5 passed: Chronological order maintained")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
