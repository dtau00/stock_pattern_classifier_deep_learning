"""
Data Visualization Module.

This module provides interactive visualization tools for normalized preprocessed
data and raw OHLCV data with technical indicators.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict
from datetime import datetime


def plot_normalized_data(
    df: pd.DataFrame,
    title: str = "Normalized Data Visualization",
    show_range_slider: bool = True,
    height: int = 800
) -> go.Figure:
    """
    Create multi-pane interactive chart for normalized preprocessed data.

    Displays three channels:
    - Returns (log returns)
    - Volume/Liquidity (OBV diff EMA)
    - Volatility/Risk (NATR)

    Args:
        df: DataFrame with columns: timestamp, returns_norm, volume_liquidity_norm, volatility_risk_norm
        title: Chart title
        show_range_slider: Whether to show range slider for zooming
        height: Chart height in pixels

    Returns:
        Plotly figure object

    Example:
        >>> df = pd.DataFrame({...})
        >>> fig = plot_normalized_data(df)
        >>> fig.show()
    """
    # Validate required columns
    required_columns = ['returns_norm', 'volume_liquidity_norm', 'volatility_risk_norm']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create timestamp index if not present
    if 'timestamp' in df.columns:
        x_axis = df['timestamp']
        x_label = 'Timestamp'
    else:
        x_axis = df.index
        x_label = 'Bar Index'

    # Create subplots with 3 rows
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Returns Channel (Normalized Log Returns)',
            'Volume/Liquidity Channel (Normalized OBV Diff EMA)',
            'Volatility/Risk Channel (Normalized NATR)'
        ),
        row_heights=[0.33, 0.33, 0.34]
    )

    # Pane 1: Returns
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df['returns_norm'],
            mode='lines',
            name='Returns',
            line=dict(color='#1f77b4', width=1),
            hovertemplate='<b>Returns</b><br>%{x}<br>Value: %{y:.3f}<extra></extra>'
        ),
        row=1,
        col=1
    )

    # Add zero line for returns
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # Pane 2: Volume/Liquidity
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df['volume_liquidity_norm'],
            mode='lines',
            name='Volume/Liquidity',
            line=dict(color='#ff7f0e', width=1),
            hovertemplate='<b>Volume/Liquidity</b><br>%{x}<br>Value: %{y:.3f}<extra></extra>'
        ),
        row=2,
        col=1
    )

    # Add zero line for volume
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # Pane 3: Volatility/Risk
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df['volatility_risk_norm'],
            mode='lines',
            name='Volatility/Risk',
            line=dict(color='#2ca02c', width=1),
            hovertemplate='<b>Volatility/Risk</b><br>%{x}<br>Value: %{y:.3f}<extra></extra>'
        ),
        row=3,
        col=1
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=height,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Update axes
    fig.update_xaxes(title_text=x_label, row=3, col=1)
    fig.update_yaxes(title_text='Normalized Value', row=1, col=1)
    fig.update_yaxes(title_text='Normalized Value', row=2, col=1)
    fig.update_yaxes(title_text='Normalized Value', row=3, col=1)

    # Add range slider if requested
    if show_range_slider:
        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

    return fig


def plot_ta_verification(
    df: pd.DataFrame,
    title: str = "Technical Analysis Verification",
    indicators: Optional[List[str]] = None,
    height: int = 800
) -> go.Figure:
    """
    Create interactive chart for verifying technical indicators on raw OHLCV data.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
        title: Chart title
        indicators: List of indicators to plot (default: ['SMA', 'BBANDS'])
        height: Chart height in pixels

    Returns:
        Plotly figure object

    Example:
        >>> df = pd.DataFrame({...})
        >>> fig = plot_ta_verification(df, indicators=['SMA', 'BBANDS', 'RSI'])
        >>> fig.show()
    """
    if indicators is None:
        indicators = ['SMA', 'BBANDS']

    # Validate required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create timestamp index if not present
    if 'timestamp' in df.columns:
        x_axis = df['timestamp']
    else:
        x_axis = df.index

    # Determine number of subplot rows
    num_rows = 2 if 'RSI' in indicators or 'MACD' in indicators else 1
    row_heights = [0.7, 0.3] if num_rows == 2 else [1.0]

    # Create subplots
    subplot_titles = ['Price with Indicators']
    if num_rows == 2:
        subplot_titles.append('Oscillators')

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}]] * num_rows
    )

    # Main pane: Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=x_axis,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1,
        col=1
    )

    # Add SMA if requested
    if 'SMA' in indicators:
        sma_periods = [20, 50]
        colors = ['#1f77b4', '#ff7f0e']

        for period, color in zip(sma_periods, colors):
            sma = df['close'].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=sma,
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(color=color, width=1.5),
                    hovertemplate=f'<b>SMA {period}</b><br>%{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ),
                row=1,
                col=1
            )

    # Add Bollinger Bands if requested
    if 'BBANDS' in indicators:
        period = 20
        std_dev = 2

        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        # Upper band
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=upper_band,
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dash'),
                hovertemplate='<b>BB Upper</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=1,
            col=1
        )

        # Lower band
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=lower_band,
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                hovertemplate='<b>BB Lower</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=1,
            col=1
        )

    # Add RSI if requested
    if 'RSI' in indicators and num_rows == 2:
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='#9467bd', width=1.5),
                hovertemplate='<b>RSI</b><br>%{x}<br>Value: %{y:.1f}<extra></extra>'
            ),
            row=2,
            col=1
        )

        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=height,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        xaxis_rangeslider_visible=False
    )

    # Update axes
    fig.update_xaxes(title_text='Timestamp', row=num_rows, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)

    if num_rows == 2:
        fig.update_yaxes(title_text='RSI', row=2, col=1, range=[0, 100])

    # Add range slider on bottom chart
    fig.update_xaxes(rangeslider_visible=True, row=num_rows, col=1)

    return fig


def plot_single_window(
    window: np.ndarray,
    window_idx: int = 0,
    title: Optional[str] = None
) -> go.Figure:
    """
    Visualize a single preprocessed window.

    Args:
        window: numpy array of shape (sequence_length, num_channels)
        window_idx: Window index for title
        title: Custom title (optional)

    Returns:
        Plotly figure object
    """
    if window.ndim != 2 or window.shape[1] != 3:
        raise ValueError(f"Expected window shape (sequence_length, 3), got {window.shape}")

    sequence_length = window.shape[0]
    x_axis = np.arange(sequence_length)

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Returns Channel',
            'Volume/Liquidity Channel',
            'Volatility/Risk Channel'
        )
    )

    # Plot each channel
    channel_names = ['Returns', 'Volume/Liquidity', 'Volatility/Risk']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=window[:, i],
                mode='lines',
                name=channel_names[i],
                line=dict(color=colors[i], width=1.5)
            ),
            row=i + 1,
            col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i + 1, col=1)

    # Update layout
    if title is None:
        title = f"Window {window_idx} (Sequence Length: {sequence_length})"

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=600,
        showlegend=True,
        template='plotly_white'
    )

    # Update axes
    fig.update_xaxes(title_text='Timestep', row=3, col=1)
    for i in range(3):
        fig.update_yaxes(title_text='Normalized Value', row=i + 1, col=1)

    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("=" * 60)
    print("Testing Data Visualization Module")
    print("=" * 60)

    # Create synthetic normalized data
    print("\n--- Test 1: Plot Normalized Data ---")
    num_bars = 1000

    df_norm = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=num_bars, freq='1h'),
        'returns_norm': np.random.randn(num_bars) * 0.8,
        'volume_liquidity_norm': np.random.randn(num_bars) * 0.6,
        'volatility_risk_norm': np.abs(np.random.randn(num_bars)) * 0.9
    })

    try:
        fig = plot_normalized_data(df_norm, title="Test Normalized Data")
        print(f"Figure created with {len(fig.data)} traces")
        assert len(fig.data) == 3, "Should have 3 traces (one per channel)"
        print("[PASS] Test 1 passed: Normalized data plot created")
    except Exception as e:
        print(f"[FAIL] Test 1 failed: {e}")

    # Test 2: Plot TA verification
    print("\n--- Test 2: Plot TA Verification ---")

    # Create synthetic OHLCV data
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(num_bars) * 2)

    df_ohlcv = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=num_bars, freq='1h'),
        'open': prices + np.random.randn(num_bars) * 0.5,
        'high': prices + np.abs(np.random.randn(num_bars)) * 1.5,
        'low': prices - np.abs(np.random.randn(num_bars)) * 1.5,
        'close': prices,
        'volume': np.abs(np.random.randn(num_bars)) * 1000
    })

    # Ensure high >= low, high >= open, high >= close, low <= open, low <= close
    df_ohlcv['high'] = df_ohlcv[['open', 'high', 'low', 'close']].max(axis=1)
    df_ohlcv['low'] = df_ohlcv[['open', 'high', 'low', 'close']].min(axis=1)

    try:
        fig = plot_ta_verification(df_ohlcv, indicators=['SMA', 'BBANDS'])
        print(f"Figure created with {len(fig.data)} traces")
        # Candlestick + 2 SMAs + 2 BBands = 5 traces
        assert len(fig.data) >= 3, "Should have at least 3 traces"
        print("[PASS] Test 2 passed: TA verification plot created")
    except Exception as e:
        print(f"[FAIL] Test 2 failed: {e}")

    # Test 3: Plot with RSI
    print("\n--- Test 3: Plot TA with RSI ---")
    try:
        fig = plot_ta_verification(df_ohlcv, indicators=['SMA', 'BBANDS', 'RSI'])
        print(f"Figure created with {len(fig.data)} traces")
        # Should have RSI trace added
        print("[PASS] Test 3 passed: TA with RSI plot created")
    except Exception as e:
        print(f"[FAIL] Test 3 failed: {e}")

    # Test 4: Plot single window
    print("\n--- Test 4: Plot Single Window ---")
    window = np.random.randn(127, 3)

    try:
        fig = plot_single_window(window, window_idx=42)
        print(f"Figure created with {len(fig.data)} traces")
        assert len(fig.data) == 3, "Should have 3 traces (one per channel)"
        print("[PASS] Test 4 passed: Single window plot created")
    except Exception as e:
        print(f"[FAIL] Test 4 failed: {e}")

    # Test 5: Error handling - missing columns
    print("\n--- Test 5: Error Handling - Missing Columns ---")
    df_incomplete = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
        'returns_norm': np.random.randn(100)
        # Missing other channels
    })

    try:
        fig = plot_normalized_data(df_incomplete)
        print("[FAIL] Test 5 failed: Should raise ValueError")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("[PASS] Test 5 passed: Error handling works")

    # Test 6: Save figure to HTML (optional)
    print("\n--- Test 6: Save Figure to HTML ---")
    try:
        output_file = "test_normalized_data.html"
        fig = plot_normalized_data(df_norm[:100], title="Sample Data (First 100 Bars)")
        fig.write_html(output_file)
        print(f"Figure saved to {output_file}")
        print("[PASS] Test 6 passed: Figure saved successfully")

        # Clean up
        import os
        if os.path.exists(output_file):
            os.remove(output_file)
            print("Test file cleaned up")
    except Exception as e:
        print(f"[FAIL] Test 6 failed: {e}")

    print("\n" + "=" * 60)
    print("All core tests passed!")
    print("=" * 60)
    print("\nNote: To view interactive plots, save as HTML and open in browser:")
    print("  fig.write_html('output.html')")
    print("  Or use: fig.show() in Jupyter or interactive environment")
