"""
Manage Data Packages Page
Validate data quality and preview package contents
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import MetadataManager


def show():
    """Validate data quality and preview contents"""
    st.title("✅ Manage Data Packages")
    st.markdown("---")

    # Initialize metadata manager
    metadata_manager = MetadataManager()
    packages = metadata_manager.get_all_packages()

    if not packages:
        st.warning("📭 No data packages found")
        st.info("""
        **Get Started:**
        1. Go to **⬇️ Download OHLCV** page
        2. Download a data package first
        """)
        return

    # Create package selection list
    package_options = {}
    for pkg in packages:
        label = f"{pkg['symbol']} {pkg['interval']} ({pkg['start_date']} to {pkg['end_date']})"
        package_options[label] = pkg

    # Package selection with delete button on the same row
    col1, col2 = st.columns([5, 1])

    with col1:
        selected_label = st.selectbox(
            "Choose a package to validate",
            list(package_options.keys()),
            index=0
        )

    with col2:
        # Add spacing to align with selectbox
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️", type="secondary", help="Delete this package"):
            st.session_state['show_delete_confirmation'] = True

    package = package_options[selected_label]

    # Handle delete confirmation with dialog
    @st.dialog("Confirm Deletion")
    def confirm_delete():
        st.warning(f"⚠️ Are you sure you want to delete this package?")
        st.write(f"**{package['symbol']} {package['interval']}**")
        st.write(f"Date range: {package['start_date']} to {package['end_date']}")
        st.write(f"File size: {package.get('file_size_mb', 0):.2f} MB")
        st.write("")
        st.write("This action cannot be undone.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                success = metadata_manager.delete_package(package['package_id'])
                if success:
                    st.session_state['show_delete_confirmation'] = False
                    st.success("Package deleted successfully!")
                    st.rerun()
                else:
                    st.error("Failed to delete package")

        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state['show_delete_confirmation'] = False
                st.rerun()

    if st.session_state.get('show_delete_confirmation', False):
        confirm_delete()

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

    # Price visualization with TradingView Lightweight Charts
    st.markdown("---")
    st.subheader("📈 Price & Volume Visualization")

    # Limit data for visualization to avoid performance issues
    max_viz_rows = 10000
    df_viz = df.tail(max_viz_rows) if len(df) > max_viz_rows else df

    try:
        from streamlit_lightweight_charts import renderLightweightCharts

        # Prepare data for TradingView format (time in unix timestamp)
        tv_candlestick_data = []
        tv_volume_data = []

        for _, row in df_viz.iterrows():
            timestamp = int(row['timestamp'] / 1000)  # Convert ms to seconds

            # Candlestick data
            tv_candlestick_data.append({
                "time": timestamp,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close'])
            })

            # Volume data with color based on price movement
            volume_color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            tv_volume_data.append({
                "time": timestamp,
                "value": float(row['volume']),
                "color": volume_color
            })

        # Chart options
        chartOptions = {
            "layout": {
                "textColor": '#333',
                "background": {
                    "type": 'solid',
                    "color": 'white'
                }
            },
            "grid": {
                "vertLines": {
                    "color": 'rgba(197, 203, 206, 0.5)'
                },
                "horzLines": {
                    "color": 'rgba(197, 203, 206, 0.5)'
                }
            },
            "timeScale": {
                "timeVisible": True,
                "secondsVisible": False
            },
            "height": 600
        }

        # Series configuration - Candlestick and Volume
        seriesChart = [
            {
                "type": 'Candlestick',
                "data": tv_candlestick_data,
                "options": {
                    "upColor": '#26a69a',
                    "downColor": '#ef5350',
                    "borderVisible": False,
                    "wickUpColor": '#26a69a',
                    "wickDownColor": '#ef5350'
                }
            },
            {
                "type": 'Histogram',
                "data": tv_volume_data,
                "options": {
                    "priceFormat": {
                        "type": 'volume'
                    },
                    "priceScaleId": 'volume_scale'
                },
                "priceScale": {
                    "scaleMargins": {
                        "top": 0.8,
                        "bottom": 0
                    }
                }
            }
        ]

        # Render the chart
        renderLightweightCharts([
            {
                "chart": chartOptions,
                "series": seriesChart
            }
        ], 'price_volume_chart')

        if len(df) > max_viz_rows:
            st.caption(f"ℹ️ Showing last {max_viz_rows:,} bars for performance")

        st.caption("💡 **Chart Features:** Zoom with scroll wheel, pan by dragging, double-click to reset. Volume bars are color-coded: green for bullish candles, red for bearish.")

    except ImportError:
        st.warning("""
        ⚠️ **TradingView Lightweight Charts library not installed.**

        This visualization requires the streamlit-lightweight-charts package.

        **To install:**
        ```bash
        pip install streamlit-lightweight-charts
        ```

        **Fallback:** Using Plotly charts instead...
        """)

        # Fallback to Plotly if TradingView is not available
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

        if len(df) > max_viz_rows:
            st.caption(f"ℹ️ Showing last {max_viz_rows:,} bars for performance")

    except Exception as e:
        st.error(f"❌ Error rendering chart: {e}")
        st.caption("The chart may not display if there's insufficient data or formatting issues")


# Execute the main function
show()
