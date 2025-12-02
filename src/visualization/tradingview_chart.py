"""
TradingView-Style Interactive Charts

Creates professional-looking interactive charts similar to TradingView
for visualizing:
1. Raw OHLCV data with candlesticks
2. Engineered features (returns, volume/liquidity, volatility/risk)
3. Feature comparison before/after engineering
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


def create_tradingview_chart(
    df: pd.DataFrame,
    title: str = "OHLCV Data",
    show_volume: bool = True,
    height: int = 800
) -> go.Figure:
    """
    Create a TradingView-style candlestick chart with volume.

    Args:
        df: DataFrame with OHLCV data and timestamp
        title: Chart title
        show_volume: Whether to show volume subplot
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    # Determine number of rows
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]

    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(title, "Volume") if show_volume else (title,)
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',  # TradingView green
            decreasing_line_color='#ef5350',  # TradingView red
        ),
        row=1, col=1
    )

    # Volume bars
    if show_volume:
        colors = ['#26a69a' if close >= open else '#ef5350'
                  for close, open in zip(df['close'], df['open'])]

        fig.add_trace(
            go.Bar(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )

    # Update layout - TradingView dark theme
    fig.update_layout(
        height=height,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        paper_bgcolor='#131722',  # TradingView background
        plot_bgcolor='#1e222d',   # TradingView plot area
        font=dict(color='#d1d4dc'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_xaxes(
        gridcolor='#2a2e39',
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor='#2a2e39',
        showgrid=True,
        zeroline=False
    )

    return fig


def create_features_chart(
    df: pd.DataFrame,
    title: str = "Engineered Features",
    height: int = 900
) -> go.Figure:
    """
    Create a TradingView-style chart showing all 3 engineered features.

    Args:
        df: DataFrame with engineered features
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    # Create subplots - 4 rows (price + 3 features)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=[
            f'{title} - Price',
            'Returns Channel (Log Returns)',
            'Volume/Liquidity Channel (OBV diff + EMA)',
            'Volatility/Risk Channel (NATR)'
        ]
    )

    x_axis = df['timestamp'] if 'timestamp' in df.columns else df.index

    # Row 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=x_axis,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    # Row 2: Returns
    if 'returns' in df.columns:
        colors_returns = ['#26a69a' if r >= 0 else '#ef5350'
                          for r in df['returns'].fillna(0)]

        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=df['returns'],
                name='Returns',
                marker_color=colors_returns,
                opacity=0.7
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      opacity=0.5, row=2, col=1)

    # Row 3: Volume/Liquidity
    if 'volume_liquidity' in df.columns:
        colors_vol = ['#2962ff' if v >= 0 else '#f23645'
                      for v in df['volume_liquidity'].fillna(0)]

        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=df['volume_liquidity'],
                name='Volume/Liquidity',
                marker_color=colors_vol,
                opacity=0.7
            ),
            row=3, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      opacity=0.5, row=3, col=1)

    # Row 4: Volatility/Risk
    if 'volatility_risk' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df['volatility_risk'],
                name='Volatility/Risk',
                line=dict(color='#ff9800', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 152, 0, 0.2)'
            ),
            row=4, col=1
        )

    # Update layout - TradingView dark theme
    fig.update_layout(
        height=height,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        paper_bgcolor='#131722',
        plot_bgcolor='#1e222d',
        font=dict(color='#d1d4dc'),
        showlegend=True,
    )

    # Update axes
    fig.update_xaxes(
        gridcolor='#2a2e39',
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor='#2a2e39',
        showgrid=True,
        zeroline=False
    )

    return fig


def create_comparison_chart(
    df: pd.DataFrame,
    feature: str = 'returns',
    title: Optional[str] = None,
    height: int = 600
) -> go.Figure:
    """
    Create a comparison chart showing raw price vs engineered feature.

    Args:
        df: DataFrame with price and features
        feature: Which feature to compare ('returns', 'volume_liquidity', 'volatility_risk')
        title: Chart title (auto-generated if None)
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    if title is None:
        title = f"Price vs {feature.replace('_', ' ').title()}"

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.5],
        subplot_titles=['Price (Raw)', f'{feature.replace("_", " ").title()} (Engineered)']
    )

    x_axis = df['timestamp'] if 'timestamp' in df.columns else df.index

    # Row 1: Price line
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df['close'],
            name='Close Price',
            line=dict(color='#2962ff', width=2)
        ),
        row=1, col=1
    )

    # Row 2: Feature
    if feature in df.columns:
        if feature == 'returns' or feature == 'volume_liquidity':
            # Bar chart for oscillators
            colors = ['#26a69a' if v >= 0 else '#ef5350'
                      for v in df[feature].fillna(0)]

            fig.add_trace(
                go.Bar(
                    x=x_axis,
                    y=df[feature],
                    name=feature.replace('_', ' ').title(),
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray",
                          opacity=0.5, row=2, col=1)

        elif feature == 'volatility_risk':
            # Area chart for volatility
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=df[feature],
                    name=feature.replace('_', ' ').title(),
                    line=dict(color='#ff9800', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 152, 0, 0.2)'
                ),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        height=height,
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        paper_bgcolor='#131722',
        plot_bgcolor='#1e222d',
        font=dict(color='#d1d4dc'),
        showlegend=True,
    )

    # Update axes
    fig.update_xaxes(gridcolor='#2a2e39', showgrid=True)
    fig.update_yaxes(gridcolor='#2a2e39', showgrid=True)

    return fig


def create_normalized_chart(
    df: pd.DataFrame,
    title: str = "Normalized Features (Model Input)",
    height: int = 900
) -> go.Figure:
    """
    Create a chart showing normalized features (what the model actually sees).

    Args:
        df: DataFrame with normalized features (*_norm columns)
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    # Check for normalized columns
    norm_cols = ['returns_norm', 'volume_liquidity_norm', 'volatility_risk_norm']
    available_norm = [col for col in norm_cols if col in df.columns]

    if not available_norm:
        # Fall back to raw features
        norm_cols = ['returns', 'volume_liquidity', 'volatility_risk']
        available_norm = [col for col in norm_cols if col in df.columns]

    if not available_norm:
        raise ValueError("No features found in DataFrame")

    # Create subplots - one per feature
    fig = make_subplots(
        rows=len(available_norm),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[col.replace('_norm', '').replace('_', ' ').title()
                        for col in available_norm]
    )

    x_axis = df['timestamp'] if 'timestamp' in df.columns else df.index

    colors = ['#2962ff', '#ff9800', '#4caf50']

    for i, (col, color) in enumerate(zip(available_norm, colors)):
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df[col],
                name=col.replace('_norm', '').replace('_', ' ').title(),
                line=dict(color=color, width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}'
            ),
            row=i+1, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      opacity=0.3, row=i+1, col=1)

    # Update layout
    fig.update_layout(
        height=height,
        title=title,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        paper_bgcolor='#131722',
        plot_bgcolor='#1e222d',
        font=dict(color='#d1d4dc'),
        showlegend=True,
    )

    # Update axes
    fig.update_xaxes(gridcolor='#2a2e39', showgrid=True)
    fig.update_yaxes(gridcolor='#2a2e39', showgrid=True)

    return fig


def save_chart(fig: go.Figure, filename: str, auto_open: bool = True):
    """
    Save chart to HTML file.

    Args:
        fig: Plotly Figure
        filename: Output filename (e.g., 'chart.html')
        auto_open: Whether to open in browser automatically
    """
    fig.write_html(filename, auto_open=auto_open)
    print(f"Chart saved to: {filename}")
