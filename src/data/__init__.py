"""
Data sourcing and retrieval module.

This module provides functionality for:
- Connecting to Binance API
- Fetching historical OHLCV data with pagination
- Validating data continuity and quality
- Estimating bar counts for date ranges
- Managing data package metadata
"""

from .binance_client import BinanceClient
from .data_fetcher import OHLCVDataFetcher
from .utils import estimate_bar_count, get_interval_info, INTERVAL_SECONDS
from .metadata_manager import MetadataManager

__all__ = [
    'BinanceClient',
    'OHLCVDataFetcher',
    'estimate_bar_count',
    'get_interval_info',
    'INTERVAL_SECONDS',
    'MetadataManager',
]
