"""
Download OHLCV Page
Download OHLCV data from Binance
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

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
    """Download OHLCV data page"""
    st.title("⬇️ Download OHLCV")
    st.markdown("---")

    # Interval and date range - second row with 2 columns
    col1, col2 = st.columns(2)

    with col1:
        symbol_mode = st.radio(
            "Symbol Selection",
            ["Popular Symbols", "Custom Symbol"],
            horizontal=True
        )

        if symbol_mode == "Popular Symbols":
            symbol = st.selectbox(
                "Select Symbol",
                POPULAR_SYMBOLS,
                index=0
            )
        else:
            symbol = st.text_input(
                "Enter Symbol",
                value="BTCUSDT"
            ).upper().strip()

        interval = st.selectbox(
            "Timeframe",
            INTERVALS,
            index=INTERVALS.index('1h')
        )

    with col2:
        default_start = datetime.now() - timedelta(days=30)
        default_end = datetime.now()

        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=datetime.now()
        )

        end_date = st.date_input(
            "End Date",
            value=default_end,
            max_value=datetime.now()
        )

    # Validation
    validation_errors = []

    if not symbol:
        validation_errors.append("❌ Symbol cannot be empty")

    if start_date >= end_date:
        validation_errors.append("❌ Start date must be before end date")

    if end_date > datetime.now().date():
        validation_errors.append("❌ End date cannot be in the future")

    # Convert dates to strings and estimate bars (needed for button)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    estimated_bars = 0
    if not validation_errors:
        try:
            estimated_bars = estimate_bar_count(start_str, end_str, interval)
        except Exception as e:
            validation_errors.append(f"❌ Error estimating bars: {e}")

    # Download button in second column
    with col2:
        st.write("")
        st.write("")

    if st.button("Download Data Package", type="primary", use_container_width=True):
        download_package(symbol, interval, start_str, end_str, estimated_bars)

    # Display validation errors
    if validation_errors:
        for error in validation_errors:
            st.error(error)

    # Estimation Section
    st.markdown("---")

    if not validation_errors:
        try:

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
                    f"{estimated_bars:,}"
                )

            with col2:
                if estimated_size_mb >= 1:
                    st.metric(
                        "Estimated Size",
                        f"{estimated_size_mb:.2f} MB"
                    )
                else:
                    st.metric(
                        "Estimated Size",
                        f"{estimated_size_kb:.2f} KB"
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

        except Exception as e:
            st.error(f"Error calculating estimate: {e}")
    else:
        st.warning("⚠️ Fix validation errors above to see estimation")


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
        - Go to **✅ Manage Data Packages** to explore this package in detail
        - Go to **📈 Visualize Data** to see interactive charts (after preprocessing)
        """)

    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        st.exception(e)


# Execute the main function
show()
