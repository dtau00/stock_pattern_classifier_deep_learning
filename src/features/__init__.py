"""
Feature Engineering Module

Transforms raw OHLCV data into stationary feature channels for deep learning.

This module implements three core feature channels:
1. Returns Channel: Log returns for price movement
2. Volume/Liquidity Channel: OBV differenced and EMA smoothed
3. Volatility/Risk Channel: Normalized ATR (NATR)

Quick Start:
    >>> from src.features import engineer_all_features
    >>> df = engineer_all_features(df)
    >>> # Adds: 'returns', 'volume_liquidity', 'volatility_risk'

Individual Features:
    >>> from src.features import (
    ...     calculate_log_returns,
    ...     calculate_obv_diff_ema,
    ...     calculate_natr
    ... )

Validation:
    >>> from src.features import compute_feature_correlation
    >>> corr_matrix, report = compute_feature_correlation(df)
    >>> print(report['safe_to_train'])
"""

from .feature_engineering import (
    calculate_log_returns,
    calculate_obv_diff_ema,
    calculate_natr,
    compute_feature_correlation,
    engineer_all_features
)

__all__ = [
    'calculate_log_returns',
    'calculate_obv_diff_ema',
    'calculate_natr',
    'compute_feature_correlation',
    'engineer_all_features'
]

__version__ = '1.0.0'
