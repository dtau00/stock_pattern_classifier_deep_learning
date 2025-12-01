"""
OHLCV Manager Page
Download OHLCV data from Binance and manage data packages
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import OHLCVDataFetcher, estimate_bar_count, MetadataManager


# Popular trading pairs
POPULAR_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
    'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT',
    'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'LTCUSDT', 'TRXUSDT'
]

# Available intervals
INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']


def show():
    """OHLCV Manager - Download and manage data packages"""
    st.title("📊 OHLCV Manager")
    st.markdown("---")

    # Create tabs
    tab1, tab2 = st.tabs(["⬇️ Download OHLCV", "✅ Validate & Preview"])

    with tab1:
        show_download_tab()

    with tab2:
        show_validate_preview_tab()


def show_download_tab():
    """Download OHLCV data tab"""
    st.info("**Purpose:** Configure parameters and download historical OHLCV data from Binance")

    # Configuration Section
    st.subheader("📋 Download Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Symbol input with popular symbols
        symbol_mode = st.radio(
            "Symbol Selection",
            ["Popular Symbols", "Custom Symbol"],
            horizontal=True
        )

        if symbol_mode == "Popular Symbols":
            symbol = st.selectbox(
                "Select Symbol",
                POPULAR_SYMBOLS,
                index=0,
                help="Select from popular trading pairs"
            )
        else:
            symbol = st.text_input(
                "Enter Symbol",
                value="BTCUSDT",
                help="Enter trading pair symbol (e.g., BTCUSDT)"
            ).upper().strip()

        # Interval selection
        interval = st.selectbox(
            "Timeframe",
            INTERVALS,
            index=INTERVALS.index('1h'),
            help="Select candlestick interval"
        )

    with col2:
        # Date range selection
        default_start = datetime.now() - timedelta(days=30)
        default_end = datetime.now()

        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=datetime.now(),
            help="Select start date for data download"
        )

        end_date = st.date_input(
            "End Date",
            value=default_end,
            max_value=datetime.now(),
            help="Select end date for data download"
        )

    # Validation
    validation_errors = []

    if not symbol:
        validation_errors.append("❌ Symbol cannot be empty")

    if start_date >= end_date:
        validation_errors.append("❌ Start date must be before end date")

    if end_date > datetime.now().date():
        validation_errors.append("❌ End date cannot be in the future")

    # Display validation errors
    if validation_errors:
        for error in validation_errors:
            st.error(error)

    # Estimation Section
    st.markdown("---")
    st.subheader("📊 Estimation")

    if not validation_errors:
        try:
            # Estimate bar count
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            estimated_bars = estimate_bar_count(start_str, end_str, interval)

            # Estimate file size (rough approximation)
            # Each bar is approximately 48 bytes in CSV format
            bytes_per_bar = 48
            estimated_size_bytes = estimated_bars * bytes_per_bar
            estimated_size_kb = estimated_size_bytes / 1024
            estimated_size_mb = estimated_size_kb / 1024

            # Display estimation
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Estimated Bars",
                    f"{estimated_bars:,}",
                    help="Approximate number of candlestick bars"
                )

            with col2:
                if estimated_size_mb >= 1:
                    st.metric(
                        "Estimated Size",
                        f"{estimated_size_mb:.2f} MB",
                        help="Approximate file size (CSV format)"
                    )
                else:
                    st.metric(
                        "Estimated Size",
                        f"{estimated_size_kb:.2f} KB",
                        help="Approximate file size (CSV format)"
                    )

            with col3:
                days_range = (end_date - start_date).days
                st.metric(
                    "Date Range",
                    f"{days_range} days",
                    help="Number of days in selected range"
                )

            # Download section
            st.markdown("---")
            st.subheader("⬇️ Download")

            # Download button
            if st.button("🚀 Download Data Package", type="primary", use_container_width=True):
                download_package(symbol, interval, start_str, end_str, estimated_bars)

        except Exception as e:
            st.error(f"Error calculating estimate: {e}")
    else:
        st.warning("⚠️ Fix validation errors above to see estimation")

    # Information section
    st.markdown("---")
    with st.expander("ℹ️ Information & Tips"):
        st.markdown("""
        ### Download Guidelines

        **Recommended Ranges by Interval:**
        - **1m, 3m, 5m**: Max 7-30 days (high data volume)
        - **15m, 30m**: Max 30-90 days
        - **1h, 2h, 4h**: Max 6 months - 1 year
        - **1d**: 1-5 years (suitable for long-term analysis)

        **Data Quality:**
        - Binance provides high-quality historical data
        - Some gaps may exist due to exchange downtime
        - Validation report is generated after download

        **Storage:**
        - Downloaded data is saved in CSV format
        - Metadata is tracked for easy management
        - Files can be found in `data/packages/` directory

        **Tips:**
        - Start with smaller date ranges to test
        - Check estimation before downloading large datasets
        - Use appropriate interval for your analysis needs
        """)


def download_package(symbol: str, interval: str, start_date: str, end_date: str, estimated_bars: int):
    """
    Download data package and save to disk.

    Args:
        symbol: Trading pair symbol
        interval: Kline interval
        start_date: Start date string
        end_date: End date string
        estimated_bars: Estimated number of bars
    """
    try:
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Initialize fetcher
        status_text.text("🔄 Initializing data fetcher...")
        progress_bar.progress(10)
        fetcher = OHLCVDataFetcher()

        # Step 2: Fetch data
        status_text.text(f"📥 Downloading {symbol} {interval} data from Binance...")
        progress_bar.progress(20)

        df = fetcher.fetch_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            verbose=False
        )

        progress_bar.progress(60)

        # Step 3: Validate data
        status_text.text("✅ Validating data quality...")
        progress_bar.progress(70)

        validation_report = fetcher._validate_continuity(df, interval)

        # Step 4: Save to file
        status_text.text("💾 Saving data package...")
        progress_bar.progress(80)

        # Create filename
        package_id = f"{symbol}_{interval}_{start_date}_{end_date}".replace(':', '-').replace(' ', '_')
        file_path = Path(f"data/packages/{package_id}.csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save DataFrame
        df.to_csv(file_path, index=False)
        file_size = file_path.stat().st_size

        # Step 5: Update metadata
        status_text.text("📝 Updating metadata...")
        progress_bar.progress(90)

        metadata_manager = MetadataManager()
        metadata_manager.add_package(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            total_bars=len(df),
            file_path=str(file_path),
            file_size=file_size,
            validation_report=validation_report
        )

        # Complete
        progress_bar.progress(100)
        status_text.text("")

        # Display success message with summary
        st.success("✅ Download completed successfully!")

        # Display summary
        st.subheader("📦 Package Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Symbol", symbol)
            st.metric("Interval", interval)
            st.metric("Total Bars", f"{len(df):,}")

        with col2:
            st.metric("Start Date", start_date)
            st.metric("End Date", end_date)
            st.metric("File Size", f"{file_size / 1024 / 1024:.2f} MB")

        # Validation report
        st.subheader("✅ Validation Report")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Expected Bars", validation_report['expected_bars'])

        with col2:
            st.metric("Gaps", validation_report['gap_count'])

        with col3:
            st.metric("Duplicates", validation_report['duplicate_count'])

        with col4:
            quality_color = "🟢" if validation_report['data_quality'] == 'good' else "🟡"
            st.metric("Quality", f"{quality_color} {validation_report['data_quality']}")

        if validation_report['gap_count'] > 0:
            st.warning(f"⚠️ {validation_report['gap_count']} gap(s) detected. Total missing bars: {validation_report['total_missing_bars']}")

            with st.expander("View gap details"):
                for i, gap in enumerate(validation_report['missing_gaps'][:10]):
                    gap_start = datetime.fromtimestamp(gap['start_timestamp'] / 1000)
                    gap_end = datetime.fromtimestamp(gap['end_timestamp'] / 1000)
                    st.text(f"Gap {i+1}: {gap_start} → {gap_end} ({gap['missing_bars']} bars)")

                if len(validation_report['missing_gaps']) > 10:
                    st.text(f"... and {len(validation_report['missing_gaps']) - 10} more gaps")

        # Preview data
        st.subheader("👀 Data Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**First 5 rows:**")
            st.dataframe(df.head(5), use_container_width=True)

        with col2:
            st.markdown("**Last 5 rows:**")
            st.dataframe(df.tail(5), use_container_width=True)

        # Show next steps
        st.info("""
        **Next Steps:**
        - Go to **✅ Validate & Preview** tab to explore this package in detail
        - Go to **📈 Visualize Data** to see interactive charts (after preprocessing)
        """)

    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        st.exception(e)


def show_validate_preview_tab():
    """Validate data quality and preview contents"""
    st.info("**Purpose:** Check data quality metrics and preview OHLCV data")

    # Initialize metadata manager
    metadata_manager = MetadataManager()
    packages = metadata_manager.get_all_packages()

    if not packages:
        st.warning("📭 No data packages found")
        st.info("""
        **Get Started:**
        1. Go to **⬇️ Download OHLCV** tab
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

    # Delete button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Delete Package", type="secondary", use_container_width=True):
            st.session_state['show_delete_confirmation'] = True

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
