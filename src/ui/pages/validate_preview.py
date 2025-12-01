"""
Validate & Preview Page
Check data quality and preview OHLCV data
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import MetadataManager, OHLCVDataFetcher
import numpy as np


def convert_to_json_serializable(obj):
    """
    Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return convert_to_json_serializable(obj.tolist())
    else:
        return obj


def show():
    """Validate data quality and preview contents"""
    st.title("✅ Validate & Preview")
    st.markdown("---")

    st.info("**Purpose:** Check data quality metrics and preview OHLCV data")

    # Initialize metadata manager
    metadata_manager = MetadataManager()
    packages = metadata_manager.get_all_packages()

    if not packages:
        st.warning("📭 No data packages found")
        st.info("""
        **Get Started:**
        1. Go to **📊 OHLCV Manager** page
        2. Download a data package first
        """)
        return

    # Package selection
    st.subheader("📦 Select Package")

    # Create package selection list
    package_options = {}
    for pkg in packages:
        label = f"{pkg['symbol']} {pkg['interval']} ({pkg['start_date']} to {pkg['end_date']})"
        package_options[label] = pkg

    selected_label = st.selectbox(
        "Choose a package to validate",
        list(package_options.keys()),
        index=0
    )

    package = package_options[selected_label]

    # Load data
    try:
        df = pd.read_csv(package['file_path'])
    except Exception as e:
        st.error(f"❌ Failed to load package data: {e}")
        return

    # Display validation report
    st.markdown("---")
    st.subheader("📊 Data Quality Metrics")

    validation = package.get('validation_report', {})

    # Quality score
    total_bars = validation.get('total_bars', 0)
    expected_bars = validation.get('expected_bars', 0)
    gap_count = validation.get('gap_count', 0)
    duplicate_count = validation.get('duplicate_count', 0)

    # Calculate quality score (0-100)
    if expected_bars > 0:
        completeness_score = (total_bars / expected_bars) * 100
    else:
        completeness_score = 100

    gap_penalty = min(gap_count * 5, 30)  # Max 30 points penalty
    duplicate_penalty = min(duplicate_count * 10, 20)  # Max 20 points penalty

    quality_score = max(0, completeness_score - gap_penalty - duplicate_penalty)

    # Display quality score
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if quality_score >= 95:
            st.success(f"🟢 **Quality Score: {quality_score:.1f}%**")
            st.caption("Excellent data quality")
        elif quality_score >= 80:
            st.warning(f"🟡 **Quality Score: {quality_score:.1f}%**")
            st.caption("Good data quality with minor issues")
        else:
            st.error(f"🔴 **Quality Score: {quality_score:.1f}%**")
            st.caption("Data quality issues detected")

    with col2:
        quality_status = validation.get('data_quality', 'unknown')
        if quality_status == 'good':
            st.metric("Status", "✅ Good")
        else:
            st.metric("Status", "⚠️ Has Issues")

    with col3:
        st.metric("Total Bars", f"{total_bars:,}")

    # Detailed metrics
    st.markdown("---")
    st.subheader("📋 Detailed Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = total_bars - expected_bars if expected_bars > 0 else 0
        st.metric(
            "Expected Bars",
            f"{expected_bars:,}",
            delta=delta if delta != 0 else None,
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Missing Gaps",
            gap_count,
            help="Number of timestamp gaps detected"
        )

    with col3:
        st.metric(
            "Duplicates",
            duplicate_count,
            help="Number of duplicate timestamps"
        )

    with col4:
        usable = validation.get('usable_segments', total_bars)
        st.metric(
            "Usable Bars",
            f"{usable:,}",
            help="Bars available for analysis"
        )

    # Gap details
    if gap_count > 0:
        st.markdown("---")
        st.subheader(f"⚠️ Gap Analysis ({gap_count} gaps)")

        total_missing = validation.get('total_missing_bars', 0)
        st.warning(f"Total missing bars across all gaps: **{total_missing:,}**")

        # Show gap details
        with st.expander(f"View gap details ({min(gap_count, 20)} shown)"):
            gaps = validation.get('missing_gaps', [])

            # Create DataFrame for gaps
            gap_data = []
            for i, gap in enumerate(gaps[:20]):
                gap_start = datetime.fromtimestamp(gap['start_timestamp'] / 1000)
                gap_end = datetime.fromtimestamp(gap['end_timestamp'] / 1000)

                gap_data.append({
                    'Gap #': i + 1,
                    'Start': gap_start.strftime('%Y-%m-%d %H:%M'),
                    'End': gap_end.strftime('%Y-%m-%d %H:%M'),
                    'Missing Bars': gap['missing_bars']
                })

            gap_df = pd.DataFrame(gap_data)
            st.dataframe(gap_df, use_container_width=True, hide_index=True)

            if gap_count > 20:
                st.info(f"ℹ️ Showing first 20 of {gap_count} gaps")

    # Data preview
    st.markdown("---")
    st.subheader("👀 Data Preview")

    preview_tabs = st.tabs(["First 100", "Last 100", "Random Sample", "Statistics"])

    with preview_tabs[0]:
        st.markdown("**First 100 bars:**")
        st.dataframe(df.head(100), use_container_width=True)

    with preview_tabs[1]:
        st.markdown("**Last 100 bars:**")
        st.dataframe(df.tail(100), use_container_width=True)

    with preview_tabs[2]:
        st.markdown("**Random sample of 50 bars:**")
        sample = df.sample(min(50, len(df)))
        st.dataframe(sample, use_container_width=True)

    with preview_tabs[3]:
        st.markdown("**Statistical Summary:**")
        st.dataframe(df.describe(), use_container_width=True)

    # Price visualization
    st.markdown("---")
    st.subheader("📈 Price Visualization")

    # Limit data for visualization to avoid performance issues
    max_viz_rows = 10000
    df_viz = df.tail(max_viz_rows) if len(df) > max_viz_rows else df

    # Convert timestamp to datetime
    df_viz = df_viz.copy()
    df_viz['datetime'] = pd.to_datetime(df_viz['timestamp'], unit='ms')

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df_viz['datetime'],
        open=df_viz['open'],
        high=df_viz['high'],
        low=df_viz['low'],
        close=df_viz['close'],
        name='OHLC'
    )])

    fig.update_layout(
        title=f"{package['symbol']} {package['interval']} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    if len(df) > max_viz_rows:
        st.caption(f"ℹ️ Showing last {max_viz_rows:,} bars for performance")

    # Volume chart
    fig_volume = go.Figure()

    fig_volume.add_trace(go.Bar(
        x=df_viz['datetime'],
        y=df_viz['volume'],
        name='Volume',
        marker_color='lightblue'
    ))

    fig_volume.update_layout(
        title=f"{package['symbol']} {package['interval']} Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300
    )

    st.plotly_chart(fig_volume, use_container_width=True)
