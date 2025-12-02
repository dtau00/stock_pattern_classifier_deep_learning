"""
Feature Engineering Module

Implements the three core feature channels:
1. Returns Channel: Log returns for price movement
2. Volume/Liquidity Channel: OBV differenced and EMA smoothed
3. Volatility/Risk Channel: Normalized ATR (NATR)
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from close prices.

    Formula: log(P_t / P_{t-1})

    This ensures stationarity and is more appropriate for modeling
    multiplicative price movements.

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with added 'returns' column

    Notes:
        - First value will be NaN (no previous price to compare)
        - Log returns are approximately equal to percentage returns for small changes
        - More mathematically convenient than simple returns
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    # Create a copy to avoid modifying original
    df = df.copy()

    # Calculate log returns: log(P_t / P_{t-1})
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    return df


def calculate_obv_diff_ema(df: pd.DataFrame, ema_period: int = 20) -> pd.DataFrame:
    """
    Calculate first-order differenced OBV smoothed with EMA.

    Steps:
    1. Calculate OBV (On-Balance Volume)
    2. First-order differencing to ensure stationarity
    3. EMA smoothing to reduce noise

    Args:
        df: DataFrame with 'close' and 'volume' columns
        ema_period: Period for EMA smoothing (default: 20)

    Returns:
        DataFrame with added 'volume_liquidity' column

    Notes:
        - OBV: cumulative volume signed by price direction
        - Differencing makes it stationary
        - EMA smoothing reduces high-frequency noise
    """
    required_cols = ['close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")

    # Create a copy to avoid modifying original
    df = df.copy()

    # Calculate price direction: +1 for up, -1 for down, 0 for unchanged
    # Use sign of price difference
    price_diff = df['close'].diff()
    price_direction = np.sign(price_diff)

    # Calculate OBV: cumulative sum of signed volume
    # When price goes up, add volume; when down, subtract volume
    signed_volume = price_direction * df['volume']
    obv = signed_volume.cumsum()

    # First-order differencing to make stationary
    obv_diff = obv.diff()

    # EMA smoothing to reduce noise
    # adjust=False ensures we use exponential weighted average
    obv_ema = obv_diff.ewm(span=ema_period, adjust=False).mean()

    df['volume_liquidity'] = obv_ema

    return df


def calculate_natr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Normalized Average True Range (NATR).

    Steps:
    1. Calculate True Range (TR)
    2. Calculate Average True Range (ATR) as rolling mean of TR
    3. Normalize by close price: NATR = ATR / Close

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Period for ATR calculation (default: 14)

    Returns:
        DataFrame with added 'volatility_risk' column

    Notes:
        - True Range = max(H-L, |H-prev_C|, |L-prev_C|)
        - Normalization makes it regime-agnostic (percentage volatility)
        - Higher NATR = higher volatility/risk
    """
    required_cols = ['high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")

    # Create a copy to avoid modifying original
    df = df.copy()

    # Calculate True Range components
    high_low = df['high'] - df['low']
    high_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_prev_close = np.abs(df['low'] - df['close'].shift(1))

    # True Range is the maximum of the three components
    true_range = pd.DataFrame({
        'hl': high_low,
        'hpc': high_prev_close,
        'lpc': low_prev_close
    }).max(axis=1)

    # Calculate ATR as rolling mean of True Range
    atr = true_range.rolling(window=period, min_periods=1).mean()

    # Normalize by close price to get percentage volatility
    natr = atr / df['close']

    df['volatility_risk'] = natr

    return df


def compute_feature_correlation(
    df: pd.DataFrame,
    sample_size: int = 10000,
    window: Optional[str] = '30D',
    random_seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Validate feature independence using correlation analysis.

    Computes correlation matrix on normalized features to ensure they are
    sufficiently independent (low correlation).

    Args:
        df: DataFrame with normalized feature columns
        sample_size: Number of random samples to use (default: 10000)
        window: Rolling window for temporal correlation analysis (default: '30D')
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (correlation_matrix, warning_report)

    Warning Thresholds:
        - |corr| > 0.8: Warning (yellow flag)
        - |corr| > 0.9: Critical (red flag, block training)
    """
    feature_cols = ['returns', 'volume_liquidity', 'volatility_risk']

    # Check if normalized versions exist, otherwise use raw
    normalized_cols = [col + '_norm' for col in feature_cols]
    available_normalized = [col for col in normalized_cols if col in df.columns]

    if len(available_normalized) == len(feature_cols):
        # Use normalized features if available
        cols_to_use = available_normalized
    else:
        # Fall back to raw features
        cols_to_use = feature_cols

    # Check all required columns exist
    missing_cols = [col for col in cols_to_use if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Remove NaN values
    df_clean = df[cols_to_use].dropna()

    # Sample if dataset is larger than sample_size
    if len(df_clean) > sample_size:
        np.random.seed(random_seed)
        df_sampled = df_clean.sample(n=sample_size, random_state=random_seed)
    else:
        df_sampled = df_clean

    # Compute correlation matrix
    corr_matrix = df_sampled.corr()

    # Rename columns back to feature names (remove _norm suffix if present)
    if '_norm' in cols_to_use[0]:
        corr_matrix.columns = feature_cols
        corr_matrix.index = feature_cols

    # Analyze correlations and generate warnings
    warning_report = {
        'warnings': [],
        'critical': [],
        'max_correlation': 0.0,
        'safe_to_train': True
    }

    # Check off-diagonal correlations
    for i, feat1 in enumerate(feature_cols):
        for j, feat2 in enumerate(feature_cols):
            if i < j:  # Only check upper triangle (avoid duplicates)
                corr_value = abs(corr_matrix.loc[feat1, feat2])

                if corr_value > warning_report['max_correlation']:
                    warning_report['max_correlation'] = corr_value

                if corr_value > 0.9:
                    warning_report['critical'].append(
                        f"{feat1} ↔ {feat2}: {corr_value:.3f} (CRITICAL - blocks training)"
                    )
                    warning_report['safe_to_train'] = False
                elif corr_value > 0.8:
                    warning_report['warnings'].append(
                        f"{feat1} ↔ {feat2}: {corr_value:.3f} (WARNING - high correlation)"
                    )

    return corr_matrix, warning_report


def engineer_all_features(
    df: pd.DataFrame,
    obv_ema_period: int = 20,
    natr_period: int = 14
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations in sequence.

    This is a convenience function that applies:
    1. Log returns
    2. OBV differenced + EMA
    3. NATR

    Args:
        df: DataFrame with OHLCV data
        obv_ema_period: Period for OBV EMA smoothing (default: 20)
        natr_period: Period for NATR calculation (default: 14)

    Returns:
        DataFrame with all engineered features added
    """
    # Apply all transformations
    df = calculate_log_returns(df)
    df = calculate_obv_diff_ema(df, ema_period=obv_ema_period)
    df = calculate_natr(df, period=natr_period)

    return df
