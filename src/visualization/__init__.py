"""
Visualization Module

TradingView-style interactive charts for visualizing OHLCV data and engineered features.

Quick Start:
    >>> from src.visualization import create_features_chart
    >>> fig = create_features_chart(df)
    >>> fig.show()

Available Charts:
    - create_tradingview_chart: Candlestick + volume (classic TradingView)
    - create_features_chart: All 3 features in one view
    - create_comparison_chart: Compare price vs individual feature
    - create_normalized_chart: Show normalized features (model input)
"""

from .tradingview_chart import (
    create_tradingview_chart,
    create_features_chart,
    create_comparison_chart,
    create_normalized_chart,
    save_chart
)

__all__ = [
    'create_tradingview_chart',
    'create_features_chart',
    'create_comparison_chart',
    'create_normalized_chart',
    'save_chart'
]

__version__ = '1.0.0'
