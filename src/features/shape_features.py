"""
Shape-Based Feature Engineering Module

This module extracts price shape features for finding visually similar patterns.
Works alongside existing statistical features (returns, volume, volatility).

Two-stage clustering approach:
1. Shape-based pre-clustering: Group windows by visual price pattern similarity
2. Statistical refinement: Use contrastive learning within shape clusters

Shape Features:
- Normalized price curve (removes absolute level)
- Relative high/low positions
- Price momentum and curvature
- Turning points and trend direction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler


def normalize_price_window(window_prices: np.ndarray, method: str = 'first') -> np.ndarray:
    """
    Normalize price window to start at 0 for shape comparison.

    This removes absolute price level, making patterns comparable regardless
    of whether they occurred at $100 or $10,000.

    Args:
        window_prices: Array of shape (seq_len,) with close prices
        method: Normalization method
            - 'first': Subtract first price (percentage change from start)
            - 'minmax': Scale to [0, 1] range
            - 'zscore': Z-score normalization

    Returns:
        Normalized price curve of shape (seq_len,)

    Example:
        >>> prices = np.array([100, 102, 101, 105, 103])
        >>> normalized = normalize_price_window(prices, method='first')
        >>> normalized
        array([0.00, 0.02, 0.01, 0.05, 0.03])  # % change from start
    """
    if len(window_prices) == 0:
        return window_prices

    if method == 'first':
        # Percentage change from first price
        first_price = window_prices[0]
        if first_price == 0:
            return np.zeros_like(window_prices)
        normalized = (window_prices - first_price) / first_price

    elif method == 'minmax':
        # Scale to [0, 1]
        min_price = window_prices.min()
        max_price = window_prices.max()
        if max_price == min_price:
            return np.zeros_like(window_prices)
        normalized = (window_prices - min_price) / (max_price - min_price)

    elif method == 'zscore':
        # Z-score normalization
        mean = window_prices.mean()
        std = window_prices.std()
        if std == 0:
            return np.zeros_like(window_prices)
        normalized = (window_prices - mean) / std

    else:
        raise ValueError(f"Unknown method: {method}. Use 'first', 'minmax', or 'zscore'")

    return normalized


def extract_shape_features(df: pd.DataFrame,
                           window_size: int = 20,
                           normalize_method: str = 'first') -> pd.DataFrame:
    """
    Extract shape-based features for each bar.

    These features capture local price shape within a rolling window,
    making patterns comparable across different price levels and timeframes.

    Args:
        df: DataFrame with OHLCV columns
        window_size: Size of rolling window for shape analysis (default: 20)
        normalize_method: Method for price normalization

    Returns:
        DataFrame with added shape feature columns:
            - shape_norm_close: Normalized close price position
            - shape_norm_high: Normalized high price position
            - shape_norm_low: Normalized low price position
            - shape_momentum: Rolling price momentum
            - shape_curvature: Second derivative (acceleration)
            - shape_trend: Overall trend direction in window
    """
    df = df.copy()

    # 1. Normalized price positions (relative to window range)
    rolling_min = df['low'].rolling(window=window_size, min_periods=1).min()
    rolling_max = df['high'].rolling(window=window_size, min_periods=1).max()
    price_range = rolling_max - rolling_min
    price_range = price_range.replace(0, 1)  # Avoid division by zero

    df['shape_norm_close'] = (df['close'] - rolling_min) / price_range
    df['shape_norm_high'] = (df['high'] - rolling_min) / price_range
    df['shape_norm_low'] = (df['low'] - rolling_min) / price_range

    # 2. Price momentum (first derivative)
    df['shape_momentum'] = df['close'].pct_change(periods=1)

    # Smooth momentum with short EMA
    df['shape_momentum'] = df['shape_momentum'].ewm(span=3, adjust=False).mean()

    # 3. Price curvature (second derivative / acceleration)
    df['shape_curvature'] = df['shape_momentum'].diff()

    # 4. Trend direction (linear regression slope over window)
    df['shape_trend'] = df['close'].rolling(window=window_size, min_periods=5).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else 0,
        raw=True
    )

    # Normalize trend by price level
    df['shape_trend'] = df['shape_trend'] / df['close'].rolling(window=window_size).mean()

    return df


def compute_shape_vector(window_df: pd.DataFrame,
                         price_col: str = 'close',
                         ohlc_cols: bool = True) -> np.ndarray:
    """
    Compute shape feature vector for a single window.

    This creates a compact representation of the price pattern shape
    that can be used for clustering or similarity comparison.

    Args:
        window_df: DataFrame with price data for one window
        price_col: Column to use for shape analysis (default: 'close')
        ohlc_cols: Whether to include OHLC shape features

    Returns:
        Shape feature vector (flattened)

    Features included:
        - Normalized price curve (removes absolute level)
        - Start/end price change
        - Maximum drawdown and drawup
        - Number of turning points
        - Overall trend strength
        - Volatility (normalized by price level)
    """
    if len(window_df) == 0:
        return np.array([])

    prices = window_df[price_col].values

    # 1. Normalized price curve
    norm_prices = normalize_price_window(prices, method='first')

    # 2. Summary statistics
    start_end_change = norm_prices[-1] - norm_prices[0]
    max_drawdown = norm_prices.min()
    max_drawup = norm_prices.max()

    # 3. Turning points (local maxima/minima)
    # Simplified: count direction changes
    price_changes = np.diff(prices)
    direction_changes = np.diff(np.sign(price_changes))
    num_turning_points = np.sum(np.abs(direction_changes) > 0)

    # 4. Trend strength (linear fit R-squared)
    if len(prices) >= 2:
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, norm_prices, 1)
        trend_slope = coeffs[0]
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((norm_prices - y_pred) ** 2)
        ss_tot = np.sum((norm_prices - np.mean(norm_prices)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
    else:
        trend_slope = 0
        r_squared = 0

    # 5. Volatility (normalized)
    volatility = np.std(norm_prices)

    # 6. OHLC shape features (if available)
    ohlc_features = []
    if ohlc_cols and all(col in window_df.columns for col in ['open', 'high', 'low', 'close']):
        # Candlestick characteristics
        opens = window_df['open'].values
        highs = window_df['high'].values
        lows = window_df['low'].values
        closes = window_df['close'].values

        # Body size (open-close range)
        body_sizes = np.abs(closes - opens) / (closes + 1e-10)

        # Wick sizes
        upper_wicks = (highs - np.maximum(opens, closes)) / (closes + 1e-10)
        lower_wicks = (np.minimum(opens, closes) - lows) / (closes + 1e-10)

        ohlc_features = [
            np.mean(body_sizes),
            np.mean(upper_wicks),
            np.mean(lower_wicks),
            np.sum(closes > opens) / len(closes),  # Bullish candle ratio
        ]

    # Combine all features
    shape_vector = np.array([
        start_end_change,
        max_drawdown,
        max_drawup,
        num_turning_points / len(prices),  # Normalized by length
        trend_slope,
        r_squared,
        volatility,
    ] + ohlc_features)

    return shape_vector


def extract_windows_shape_features(windows_df: pd.DataFrame,
                                   window_size: int,
                                   metadata: List[dict],
                                   ohlc_cols: bool = True) -> np.ndarray:
    """
    Extract shape feature vectors for multiple windows.

    Args:
        windows_df: Original DataFrame with OHLCV data
        window_size: Size of each window
        metadata: List of window metadata with start/end indices
        ohlc_cols: Whether to include OHLC features

    Returns:
        Array of shape (num_windows, num_shape_features)
    """
    shape_features = []

    for meta in metadata:
        start_idx = meta['start_idx']
        end_idx = meta['end_idx']

        window_df = windows_df.iloc[start_idx:end_idx + 1]
        shape_vec = compute_shape_vector(window_df, ohlc_cols=ohlc_cols)
        shape_features.append(shape_vec)

    return np.array(shape_features)


def cluster_by_shape(shape_features: np.ndarray,
                     n_clusters: int = 10,
                     method: str = 'kmeans',
                     random_state: int = 42) -> Tuple[np.ndarray, object]:
    """
    Cluster windows based on shape features.

    This is the first stage of two-stage clustering:
    Groups windows by visual price pattern similarity.

    Args:
        shape_features: Array of shape (num_windows, num_features)
        n_clusters: Number of shape-based clusters
        method: Clustering method ('kmeans', 'hierarchical')
        random_state: Random seed

    Returns:
        Tuple of (cluster_labels, clustering_model)
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    # Standardize features
    scaler = StandardScaler()
    shape_features_scaled = scaler.fit_transform(shape_features)

    if method == 'kmeans':
        clusterer = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    labels = clusterer.fit_predict(shape_features_scaled)

    return labels, clusterer


def get_shape_cluster_stats(shape_labels: np.ndarray,
                            windows_df: pd.DataFrame,
                            metadata: List[dict]) -> pd.DataFrame:
    """
    Compute statistics for each shape cluster.

    Args:
        shape_labels: Cluster labels from cluster_by_shape()
        windows_df: Original DataFrame
        metadata: Window metadata

    Returns:
        DataFrame with cluster statistics
    """
    cluster_stats = []

    unique_clusters = np.unique(shape_labels)

    for cluster_id in unique_clusters:
        cluster_mask = shape_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Compute average price movement in cluster
        price_changes = []
        for idx in cluster_indices:
            meta = metadata[idx]
            window = windows_df.iloc[meta['start_idx']:meta['end_idx'] + 1]
            pct_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
            price_changes.append(pct_change)

        stats = {
            'cluster_id': cluster_id,
            'num_windows': len(cluster_indices),
            'avg_price_change': np.mean(price_changes),
            'std_price_change': np.std(price_changes),
            'median_price_change': np.median(price_changes),
        }
        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


if __name__ == '__main__':
    # Simple test
    print("Testing shape feature extraction...")

    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    })
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))

    # Extract shape features
    df_with_shape = extract_shape_features(df, window_size=20)

    print(f"✓ Added shape features: {[col for col in df_with_shape.columns if col.startswith('shape_')]}")

    # Test window shape vector
    window = df.iloc[:20]
    shape_vec = compute_shape_vector(window)

    print(f"✓ Shape vector dimension: {len(shape_vec)}")
    print(f"✓ Shape vector: {shape_vec}")

    print("\n✓ All tests passed!")
