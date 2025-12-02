"""
Indicators Visualization

Displays OHLCV data from Binance packages or synthetic demo data
with configurable technical indicators (SMA, Bollinger Bands, RSI).
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Indicators Visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from visualization.data_viz import plot_ta_verification
from ui.components import load_package_with_date_range


# Main page content
st.title("📈 Indicators Visualization")
st.markdown("---")

# Use reusable component for loading package with date range
load_package_with_date_range(
    packages_dir="data/packages",
    session_key='ohlcv_data',
    key_prefix='load',
    file_extension='.csv',
    package_type="OHLCV"
)

# Display chart if data exists in session state
if 'ohlcv_data' in st.session_state:
    df_ohlcv = st.session_state['ohlcv_data']

    st.success(f"Displaying {len(df_ohlcv):,} bars of OHLCV data")

    # Indicator selection
    st.markdown("#### Select Indicators")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_sma = st.checkbox("SMA (20, 50)", value=True, key="real_sma")
    with col2:
        show_bbands = st.checkbox("Bollinger Bands", value=True, key="real_bb")
    with col3:
        show_rsi = st.checkbox("RSI", value=False, key="real_rsi")

    # Build indicators list
    indicators = []
    if show_sma:
        indicators.append('SMA')
    if show_bbands:
        indicators.append('BBANDS')
    if show_rsi:
        indicators.append('RSI')

    # Chart height
    chart_height = st.slider("Chart height (px)", 400, 1400, 800, step=50, key="real_height")

    # Display chart
    try:
        fig = plot_ta_verification(
            df_ohlcv,
            title="OHLCV Chart with Technical Indicators",
            indicators=indicators if indicators else ['SMA'],
            height=chart_height
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display statistics
        with st.expander("📊 OHLCV Statistics"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${df_ohlcv['close'].iloc[-1]:.2f}")
                st.metric("Price Change", f"{((df_ohlcv['close'].iloc[-1] / df_ohlcv['close'].iloc[0]) - 1) * 100:.2f}%")

            with col2:
                st.metric("High", f"${df_ohlcv['high'].max():.2f}")
                st.metric("Low", f"${df_ohlcv['low'].min():.2f}")

            with col3:
                st.metric("Avg Volume", f"{df_ohlcv['volume'].mean():.0f}")
                st.metric("Total Volume", f"{df_ohlcv['volume'].sum():.0f}")

            with col4:
                volatility = df_ohlcv['close'].pct_change().std() * 100
                st.metric("Volatility", f"{volatility:.2f}%")
                st.metric("Bars", f"{len(df_ohlcv):,}")

    except Exception as e:
        st.error(f"Error displaying chart: {e}")
