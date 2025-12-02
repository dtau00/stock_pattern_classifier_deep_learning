"""
Normalization module for data preprocessing.

This module provides functions to:
- Winsorize (clip outliers) using std or percentile methods
- Z-score normalization
- Normalization statistics persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union


def winsorize_channel(series: pd.Series,
                      method: str = 'std',
                      std_threshold: float = 3.0,
                      percentile_range: Tuple[float, float] = (1.0, 99.0)) -> pd.Series:
    """
    Winsorize (clip) extreme outliers in a data series.

    Args:
        series: Data series to winsorize
        method: 'std' for standard deviation or 'percentile' for percentile-based
        std_threshold: Number of standard deviations for clipping (default: 3.0)
        percentile_range: Tuple of (lower, upper) percentiles (default: (1.0, 99.0))

    Returns:
        Winsorized series
    """
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return series

    if method == 'std':
        # Calculate mean and std
        mean = series_clean.mean()
        std = series_clean.std()

        if std == 0:
            # All values are the same, no clipping needed
            return series

        # Calculate clipping bounds
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std

    elif method == 'percentile':
        # Calculate percentile bounds
        lower_bound = np.percentile(series_clean, percentile_range[0])
        upper_bound = np.percentile(series_clean, percentile_range[1])

    else:
        raise ValueError(f"Unsupported winsorization method: {method}")

    # Clip the series
    series_winsorized = series.clip(lower=lower_bound, upper=upper_bound)

    return series_winsorized


def winsorize_dataframe(df: pd.DataFrame,
                        channels: List[str],
                        method: str = 'std',
                        std_threshold: float = 3.0,
                        percentile_range: Tuple[float, float] = (1.0, 99.0)) -> pd.DataFrame:
    """
    Apply winsorization to multiple channels in a DataFrame.

    Args:
        df: DataFrame containing channels to winsorize
        channels: List of column names to winsorize
        method: 'std' or 'percentile'
        std_threshold: Threshold for std method
        percentile_range: Range for percentile method

    Returns:
        DataFrame with winsorized channels
    """
    df_winsorized = df.copy()

    for channel in channels:
        if channel in df.columns:
            df_winsorized[channel] = winsorize_channel(
                df[channel],
                method=method,
                std_threshold=std_threshold,
                percentile_range=percentile_range
            )

    return df_winsorized


def calculate_normalization_stats(train_df: pd.DataFrame,
                                   channels: List[str],
                                   winsorize: bool = True,
                                   winsorize_method: str = 'std',
                                   winsorize_std: float = 3.0,
                                   winsorize_percentile: Tuple[float, float] = (1.0, 99.0)) -> Dict:
    """
    Calculate normalization statistics (mean and std) from training data.

    IMPORTANT: Stats are calculated from TRAINING SET ONLY after winsorization.

    Args:
        train_df: Training DataFrame
        channels: List of column names to normalize
        winsorize: Whether to apply winsorization before calculating stats
        winsorize_method: 'std' or 'percentile'
        winsorize_std: Threshold for std winsorization
        winsorize_percentile: Range for percentile winsorization

    Returns:
        Dictionary mapping channel names to {'mean': float, 'std': float}
    """
    stats = {}

    # Apply winsorization if requested
    if winsorize:
        train_df_processed = winsorize_dataframe(
            train_df,
            channels,
            method=winsorize_method,
            std_threshold=winsorize_std,
            percentile_range=winsorize_percentile
        )
    else:
        train_df_processed = train_df.copy()

    # Calculate stats for each channel
    for channel in channels:
        if channel in train_df_processed.columns:
            channel_data = train_df_processed[channel].dropna()

            if len(channel_data) == 0:
                raise ValueError(f"Channel '{channel}' has no valid data")

            mean = float(channel_data.mean())
            std = float(channel_data.std())

            if std == 0:
                raise ValueError(f"Channel '{channel}' has zero standard deviation")

            stats[channel] = {
                'mean': mean,
                'std': std
            }

    return stats


def apply_normalization(df: pd.DataFrame,
                        stats: Dict,
                        suffix: str = '_norm') -> pd.DataFrame:
    """
    Apply z-score normalization using pre-calculated statistics.

    Formula: X_norm = (X - μ) / σ

    Args:
        df: DataFrame to normalize
        stats: Dictionary of normalization stats from calculate_normalization_stats()
        suffix: Suffix to add to normalized column names (default: '_norm')

    Returns:
        DataFrame with normalized columns added
    """
    df_normalized = df.copy()

    for channel, channel_stats in stats.items():
        if channel in df.columns:
            mean = channel_stats['mean']
            std = channel_stats['std']

            # Apply z-score normalization
            normalized_column = f"{channel}{suffix}"
            df_normalized[normalized_column] = (df[channel] - mean) / std

    return df_normalized


def get_winsorization_report(df_original: pd.DataFrame,
                             df_winsorized: pd.DataFrame,
                             channels: List[str]) -> Dict:
    """
    Generate report comparing original and winsorized data.

    Args:
        df_original: Original DataFrame
        df_winsorized: Winsorized DataFrame
        channels: List of channels that were winsorized

    Returns:
        Dictionary with clipping statistics for each channel
    """
    report = {}

    for channel in channels:
        if channel in df_original.columns and channel in df_winsorized.columns:
            original = df_original[channel].dropna()
            winsorized = df_winsorized[channel].dropna()

            # Count how many values were clipped
            clipped_mask = original != winsorized
            num_clipped = clipped_mask.sum()
            pct_clipped = (num_clipped / len(original) * 100) if len(original) > 0 else 0

            # Get clipping bounds
            min_clipped = winsorized.min()
            max_clipped = winsorized.max()

            report[channel] = {
                'total_values': len(original),
                'num_clipped': int(num_clipped),
                'pct_clipped': float(pct_clipped),
                'original_min': float(original.min()),
                'original_max': float(original.max()),
                'clipped_min': float(min_clipped),
                'clipped_max': float(max_clipped)
            }

    return report


def save_normalization_stats(stats: Dict,
                             filepath: str,
                             metadata: Dict = None) -> str:
    """
    Save normalization statistics to JSON file.

    Args:
        stats: Dictionary of normalization stats (from calculate_normalization_stats)
        filepath: Path to save JSON file
        metadata: Optional additional metadata to include

    Returns:
        Path to saved file
    """
    import json
    import os
    from datetime import datetime

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Build metadata
    output = {
        'channels': stats,
        'created_at': datetime.now().isoformat(),
    }

    # Add optional metadata
    if metadata:
        output.update(metadata)

    # Save as JSON
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    return filepath


def load_normalization_stats(filepath: str) -> Tuple[Dict, Dict]:
    """
    Load normalization statistics from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Tuple of (stats dict, metadata dict)
    """
    import json

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract stats
    stats = data.get('channels', {})

    # Extract metadata
    metadata = {k: v for k, v in data.items() if k != 'channels'}

    return stats, metadata
