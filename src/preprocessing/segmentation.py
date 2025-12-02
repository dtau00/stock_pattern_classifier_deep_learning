"""
Segmentation module for creating sliding windows from time series data.

This module provides functions to:
- Create sliding windows with configurable overlap
- Exclude windows overlapping with gaps
- Store window metadata (timestamps)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import h5py
import os


def create_sliding_windows(df: pd.DataFrame,
                           normalized_channels: List[str],
                           sequence_length: int = 127,
                           overlap: float = 0.5,
                           exclude_column: str = 'is_excluded') -> Tuple[np.ndarray, List[Dict]]:
    """
    Create sliding windows from normalized time series data.

    Args:
        df: DataFrame with normalized channels
        normalized_channels: List of normalized column names (e.g., ['returns_norm', 'volume_norm'])
        sequence_length: Length of each window in bars (default: 127)
        overlap: Overlap fraction between windows (default: 0.5 = 50%)
        exclude_column: Column name for exclusion flags (default: 'is_excluded')

    Returns:
        Tuple of:
        - windows: numpy array of shape (num_windows, sequence_length, num_channels)
        - metadata: list of dicts with window metadata (start_idx, end_idx, timestamps)
    """
    # Calculate step size
    step = int(sequence_length * (1 - overlap))
    if step < 1:
        step = 1

    # Check if exclusion column exists
    has_exclusion = exclude_column in df.columns

    windows = []
    metadata = []

    # Slide through the data
    for i in range(0, len(df) - sequence_length + 1, step):
        # Extract window
        window_df = df.iloc[i:i + sequence_length]

        # Check if window should be excluded
        if has_exclusion and window_df[exclude_column].any():
            # Skip this window (contains excluded data)
            continue

        # Extract normalized channels
        window_data = window_df[normalized_channels].values

        # Check for NaN values
        if np.isnan(window_data).any():
            # Skip windows with NaN
            continue

        # Verify shape
        if window_data.shape[0] != sequence_length:
            # Skip incomplete windows
            continue

        windows.append(window_data)

        # Store metadata
        window_metadata = {
            'start_idx': i,
            'end_idx': i + sequence_length - 1,
            'window_idx': len(windows) - 1
        }

        # Add timestamps if available
        if 'timestamp' in df.columns:
            start_ts = df.iloc[i]['timestamp']
            end_ts = df.iloc[i + sequence_length - 1]['timestamp']
            # Convert pandas Timestamp to Unix timestamp (milliseconds)
            window_metadata['start_timestamp'] = int(pd.Timestamp(start_ts).timestamp() * 1000)
            window_metadata['end_timestamp'] = int(pd.Timestamp(end_ts).timestamp() * 1000)

        metadata.append(window_metadata)

    # Convert to numpy array
    if len(windows) == 0:
        # No valid windows
        return np.array([]).reshape(0, sequence_length, len(normalized_channels)), []

    windows_array = np.stack(windows, axis=0)

    return windows_array, metadata


def save_preprocessed_package(windows: np.ndarray,
                              metadata: List[Dict],
                              normalization_stats: Dict,
                              filepath: str,
                              additional_metadata: Dict = None) -> str:
    """
    Save preprocessed windows to HDF5 or CSV file.

    Args:
        windows: Numpy array of shape (num_windows, sequence_length, num_channels)
        metadata: List of window metadata dicts
        normalization_stats: Normalization statistics used
        filepath: Path to save file (.h5 or .csv)
        additional_metadata: Optional additional metadata

    Returns:
        Path to saved file
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        # Save as HDF5
        with h5py.File(filepath, 'w') as f:
            # Save windows
            f.create_dataset('windows', data=windows, compression='gzip', compression_opts=4)

            # Save window metadata
            f.attrs['num_windows'] = windows.shape[0]
            f.attrs['sequence_length'] = windows.shape[1]
            f.attrs['num_channels'] = windows.shape[2]

            # Save metadata as JSON string
            import json
            f.attrs['window_metadata'] = json.dumps(metadata)
            f.attrs['normalization_stats'] = json.dumps(normalization_stats)

            if additional_metadata:
                f.attrs['additional_metadata'] = json.dumps(additional_metadata)

    elif filepath.endswith('.csv'):
        # Save as CSV (flattened)
        # Flatten windows: (num_windows, sequence_length * num_channels)
        num_windows, sequence_length, num_channels = windows.shape
        windows_flat = windows.reshape(num_windows, sequence_length * num_channels)

        # Create DataFrame
        column_names = [f'step_{i}_ch_{j}' for i in range(sequence_length) for j in range(num_channels)]
        df = pd.DataFrame(windows_flat, columns=column_names)

        # Add metadata columns
        if metadata:
            df['start_idx'] = [m['start_idx'] for m in metadata]
            df['end_idx'] = [m['end_idx'] for m in metadata]
            if 'start_timestamp' in metadata[0]:
                df['start_timestamp'] = [m['start_timestamp'] for m in metadata]
                df['end_timestamp'] = [m['end_timestamp'] for m in metadata]

        df.to_csv(filepath, index=False)

        # Save normalization stats separately
        stats_path = filepath.replace('.csv', '_stats.json')
        import json
        with open(stats_path, 'w') as f:
            save_data = {
                'normalization_stats': normalization_stats,
                'window_metadata': metadata[:10] if len(metadata) > 10 else metadata,  # Save first 10 only
                'shape': {
                    'num_windows': num_windows,
                    'sequence_length': sequence_length,
                    'num_channels': num_channels
                }
            }
            if additional_metadata:
                save_data['additional_metadata'] = additional_metadata
            json.dump(save_data, f, indent=2)

    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .h5, .hdf5, or .csv")

    return filepath


def load_preprocessed_package(filepath: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """
    Load preprocessed windows from HDF5 or CSV file.

    Args:
        filepath: Path to file (.h5 or .csv)

    Returns:
        Tuple of (windows array, metadata list, normalization_stats dict)
    """
    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        # Load from HDF5
        with h5py.File(filepath, 'r') as f:
            windows = f['windows'][:]

            # Load metadata
            import json
            metadata = json.loads(f.attrs['window_metadata'])
            normalization_stats = json.loads(f.attrs['normalization_stats'])

    elif filepath.endswith('.csv'):
        # Load from CSV
        df = pd.DataFrame(pd.read_csv(filepath))

        # Extract metadata columns
        metadata_cols = ['start_idx', 'end_idx', 'start_timestamp', 'end_timestamp']
        metadata = []
        for idx, row in df.iterrows():
            meta = {'window_idx': idx}
            for col in metadata_cols:
                if col in df.columns:
                    meta[col] = int(row[col])
            metadata.append(meta)

        # Remove metadata columns
        data_df = df.drop(columns=[col for col in metadata_cols if col in df.columns], errors='ignore')

        # Load normalization stats
        stats_path = filepath.replace('.csv', '_stats.json')
        import json
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)
            normalization_stats = stats_data['normalization_stats']
            shape = stats_data['shape']

        # Reshape to (num_windows, sequence_length, num_channels)
        windows = data_df.values.reshape(
            shape['num_windows'],
            shape['sequence_length'],
            shape['num_channels']
        )

    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .h5, .hdf5, or .csv")

    return windows, metadata, normalization_stats
