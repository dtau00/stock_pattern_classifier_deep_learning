"""
Feature Visualization Page (formerly TA Verification)

Interactive TradingView-style charts showing:
- Raw OHLCV data
- Engineered features (returns, volume/liquidity, volatility)
- Feature correlation analysis
- Real-time visualization of selected data packages
"""
import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features import engineer_all_features, compute_feature_correlation
from src.visualization import (
    create_tradingview_chart,
    create_features_chart,
    create_comparison_chart
)


def show():
    """Feature Visualization - Interactive TradingView-style charts"""
    st.title("📊 Feature Visualization")
    st.markdown("---")

    st.info("**Section Purpose:** Visualize engineered features on OHLCV data with interactive TradingView-style charts")

    # Sidebar configuration
    st.sidebar.markdown("### Chart Configuration")

    # Data package selection
    st.markdown("## 📦 Load Data Package")

    df = None

    # Look for data packages
    packages_dir = project_root / "data" / "packages"

    if packages_dir.exists():
        package_files = list(packages_dir.glob("*.csv"))

        if package_files:
            package_names = [f.name for f in package_files]
            selected_package = st.selectbox(
                "Select data package:",
                package_names,
                help="Choose from downloaded OHLCV data packages"
            )

            # Automatically load the selected package
            try:
                package_path = packages_dir / selected_package

                with st.spinner(f"📥 Loading {selected_package}..."):
                    df = pd.read_csv(package_path)

                    # Convert timestamp if present
                    if 'timestamp' in df.columns:
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        except:
                            try:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                            except:
                                st.warning("⚠️ Could not parse timestamp column")

                    st.success(f"✓ Loaded {len(df):,} bars from {selected_package}")

                    # Show package info and get date range
                    if 'timestamp' in df.columns and df['timestamp'].notna().any():
                        min_date = df['timestamp'].min().date()
                        max_date = df['timestamp'].max().date()

                        # Date range selection
                        st.markdown("### 📆 Select Date Range")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            start_date = st.date_input(
                                "Start Date",
                                value=min_date,
                                min_value=min_date,
                                max_value=max_date,
                                help="Select the start date for analysis"
                            )

                        with col2:
                            end_date = st.date_input(
                                "End Date",
                                value=max_date,
                                min_value=min_date,
                                max_value=max_date,
                                help="Select the end date for analysis"
                            )

                        with col3:
                            # Show estimated bars in range
                            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
                            bars_in_range = mask.sum()
                            st.metric("Bars in selected range", f"{bars_in_range:,}")

                        # Validate date range
                        if start_date > end_date:
                            st.error("❌ Start date must be before end date")
                            df = None
                        else:
                            # Load button
                            if st.button("📊 Load & Visualize", type="primary"):
                                # Filter data by selected date range
                                original_len = len(df)
                                df = df[mask].copy()

                                if len(df) == 0:
                                    st.error("❌ No data in selected date range")
                                    df = None
                                else:
                                    # Show filtered data info
                                    filtered_info = f"✓ Loaded {len(df):,} bars"
                                    if len(df) < original_len:
                                        filtered_info += f" (filtered from {original_len:,})"
                                    filtered_info += f" | {start_date} to {end_date}"
                                    st.success(filtered_info)
                            else:
                                # Button not clicked yet, don't process data
                                df = None
                    else:
                        with st.expander("📋 Package Info"):
                            st.write(f"**File:** {selected_package}")
                            st.write(f"**Rows:** {len(df):,}")
                            st.write(f"**Columns:** {', '.join(df.columns)}")
                        st.warning("⚠️ No timestamp column found - using entire dataset")

                        # Load button for data without timestamp
                        if not st.button("📊 Load & Visualize", type="primary", key="load_no_timestamp"):
                            df = None

            except Exception as e:
                st.error(f"❌ Error loading package: {e}")
                df = None
        else:
            st.warning("⚠️ No data packages found in data/packages/")
            st.info("💡 Go to **OHLCV Manager** to download OHLCV data first")
    else:
        st.warning("⚠️ Data packages directory not found")
        st.info("💡 Go to **OHLCV Manager** to download OHLCV data first")

    # If we have data, process and display
    if df is not None:
        st.markdown("---")
        st.markdown("## 🔧 Feature Engineering")

        # Verify required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Required columns: open, high, low, close, volume")
        else:
            # Feature engineering parameters
            with st.expander("⚙️ Feature Engineering Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    obv_ema_period = st.slider(
                        "OBV EMA Period",
                        min_value=5,
                        max_value=50,
                        value=20,
                        help="Shorter = more responsive, Longer = more smoothed"
                    )
                with col2:
                    natr_period = st.slider(
                        "NATR Period",
                        min_value=7,
                        max_value=30,
                        value=14,
                        help="Period for volatility calculation"
                    )

            # Apply feature engineering
            with st.spinner("🔄 Applying feature engineering..."):
                try:
                    df_features = engineer_all_features(
                        df.copy(),
                        obv_ema_period=obv_ema_period,
                        natr_period=natr_period
                    )

                    st.success("✓ Feature engineering complete!")

                    # Feature statistics
                    with st.expander("📊 Feature Statistics"):
                        feature_stats = df_features[['returns', 'volume_liquidity', 'volatility_risk']].describe()
                        st.dataframe(feature_stats, use_container_width=True)

                    # Feature correlation
                    st.markdown("### 🔗 Feature Correlation Analysis")

                    corr_matrix, report = compute_feature_correlation(df_features)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Max Correlation",
                            f"{report['max_correlation']:.3f}",
                            delta="Safe" if report['safe_to_train'] else "Warning",
                            delta_color="normal" if report['safe_to_train'] else "inverse"
                        )
                    with col2:
                        st.metric(
                            "Safe to Train",
                            "✓ Yes" if report['safe_to_train'] else "✗ No",
                        )
                    with col3:
                        coverage = df_features['returns'].notna().sum() / len(df_features) * 100
                        st.metric("Data Coverage", f"{coverage:.1f}%")

                    # Show correlation matrix
                    st.dataframe(
                        corr_matrix.style.background_gradient(cmap='RdYlGn_r', vmin=-1, vmax=1),
                        use_container_width=True
                    )

                    if report['warnings']:
                        st.warning("⚠️ " + "\n".join(report['warnings']))
                    if report['critical']:
                        st.error("❌ " + "\n".join(report['critical']))

                    # Chart selection
                    st.markdown("---")
                    st.markdown("## 📈 Interactive Charts")

                    chart_type = st.selectbox(
                        "Select chart type:",
                        [
                            "All Features (Recommended)",
                            "Raw OHLCV Only",
                            "Price vs Returns",
                            "Price vs Volume/Liquidity",
                            "Price vs Volatility/Risk"
                        ]
                    )

                    # Chart height control
                    chart_height = st.sidebar.slider(
                        "Chart Height (px)",
                        min_value=600,
                        max_value=1400,
                        value=1000,
                        step=100
                    )

                    # Generate selected chart
                    with st.spinner("📊 Generating chart..."):
                        if chart_type == "All Features (Recommended)":
                            fig = create_features_chart(
                                df_features,
                                title="Feature Engineering - All Channels",
                                height=chart_height
                            )
                        elif chart_type == "Raw OHLCV Only":
                            fig = create_tradingview_chart(
                                df_features,
                                title="OHLCV Candlestick Chart",
                                show_volume=True,
                                height=chart_height
                            )
                        elif chart_type == "Price vs Returns":
                            fig = create_comparison_chart(
                                df_features,
                                feature='returns',
                                height=chart_height
                            )
                        elif chart_type == "Price vs Volume/Liquidity":
                            fig = create_comparison_chart(
                                df_features,
                                feature='volume_liquidity',
                                height=chart_height
                            )
                        elif chart_type == "Price vs Volatility/Risk":
                            fig = create_comparison_chart(
                                df_features,
                                feature='volatility_risk',
                                height=chart_height
                            )

                        # Display chart
                        st.plotly_chart(fig, use_container_width=True)

                    # Export options
                    st.markdown("---")
                    st.markdown("## 💾 Export Options")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Export features as CSV
                        csv = df_features.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Features CSV",
                            data=csv,
                            file_name="features_engineered.csv",
                            mime="text/csv",
                            help="Download DataFrame with engineered features"
                        )

                    with col2:
                        # Export chart as HTML
                        chart_html = fig.to_html()
                        st.download_button(
                            label="📊 Download Chart HTML",
                            data=chart_html,
                            file_name="chart_interactive.html",
                            mime="text/html",
                            help="Download interactive chart as standalone HTML"
                        )

                    with col3:
                        # Export correlation report
                        report_text = f"""Feature Correlation Report

Max Correlation: {report['max_correlation']:.3f}
Safe to Train: {report['safe_to_train']}

Correlation Matrix:
{corr_matrix.to_string()}

Warnings: {len(report['warnings'])}
Critical Issues: {len(report['critical'])}
"""
                        st.download_button(
                            label="📋 Download Report",
                            data=report_text,
                            file_name="correlation_report.txt",
                            mime="text/plain",
                            help="Download correlation analysis report"
                        )

                    # Data preview
                    with st.expander("🔍 Data Preview (First/Last 10 rows)"):
                        st.markdown("**First 10 rows:**")
                        st.dataframe(
                            df_features[['timestamp', 'close', 'returns', 'volume_liquidity', 'volatility_risk']].head(10),
                            use_container_width=True
                        )
                        st.markdown("**Last 10 rows:**")
                        st.dataframe(
                            df_features[['timestamp', 'close', 'returns', 'volume_liquidity', 'volatility_risk']].tail(10),
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"❌ Error during feature engineering: {e}")
                    st.exception(e)

    else:
        # Help section when no data loaded
        st.markdown("---")
        st.info("👆 Select a data source above to get started")

        with st.expander("📖 How to Use This Page"):
            st.markdown("""
            ### Quick Start Guide

            1. **Load Data Package:**
               - Select a data package from the dropdown
               - Data loads automatically when selected
               - View package info in the expandable section

            2. **Select Date Range:**
               - Choose start and end dates from the available range
               - Date range is based on the loaded package's data
               - Defaults to full package range
               - See bar count for selected range

            3. **Click "Load & Visualize":**
               - Processes the data with selected date range
               - Applies feature engineering
               - Generates interactive charts

            4. **Configure Parameters (Optional):**
               - Adjust OBV EMA period (default: 20)
               - Adjust NATR period (default: 14)

            5. **View Results:**
               - Feature statistics
               - Correlation analysis
               - Interactive TradingView-style charts

            6. **Export:**
               - Download engineered features as CSV
               - Save interactive charts as HTML
               - Export correlation reports

            ### What You'll See

            - **Returns Channel:** Log returns (price momentum)
            - **Volume/Liquidity Channel:** OBV flow (buying/selling pressure)
            - **Volatility/Risk Channel:** NATR (market volatility)

            ### Chart Types

            - **All Features:** See all 3 channels together (recommended)
            - **Raw OHLCV:** Traditional candlestick chart
            - **Comparisons:** Side-by-side price vs feature

            All charts are fully interactive:
            - Zoom: Click and drag
            - Pan: Shift + drag
            - Hover: See exact values
            - Reset: Double-click
            """)


if __name__ == "__main__":
    show()
