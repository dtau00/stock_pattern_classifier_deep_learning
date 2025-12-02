"""
Create Preprocessed Packages

Run the full preprocessing pipeline on downloaded OHLCV data:
1. Feature engineering (returns, OBV, NATR)
2. Data cleaning (gap detection)
3. Normalization (z-score per channel)
4. Segmentation (127-bar windows, 50% overlap)
5. Save as HDF5 package
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features import engineer_all_features, compute_feature_correlation
from src.preprocessing.data_cleaning import detect_gaps, mark_excluded_windows
from src.preprocessing.normalization import calculate_normalization_stats, apply_normalization
from src.preprocessing.segmentation import create_sliding_windows, save_preprocessed_package

# Import shared component
from src.ui.components import load_package_with_date_range


def show():
    """Create preprocessed packages from OHLCV data"""
    st.title("🔧 Create Preprocessed Packages")
    st.markdown("---")

    # Use reusable component for loading package with date range
    load_package_with_date_range(
        packages_dir=str(project_root / "data" / "packages"),
        session_key='preprocessing_data',
        key_prefix='preprocessing',
        file_extension='.csv',
        package_type="OHLCV"
    )

    # Get data from session state
    df = st.session_state.get('preprocessing_data', None)

    # If we don't have data yet, show instructions
    if df is None:
        return

    # Display loaded data info
    st.success(f"✅ Loaded {len(df):,} bars for preprocessing")

    # Extract metadata from the selected package filename if available
    selected_package = st.session_state.get('preprocessing_selected_package', 'unknown_package')
    # Parse symbol and interval from filename (e.g., "BTCUSDT_1h_2024-01-01_2024-12-31.csv")
    symbol = "unknown"
    interval = "unknown"
    if selected_package and selected_package != 'unknown_package':
        parts = selected_package.replace('.csv', '').split('_')
        if len(parts) >= 2:
            symbol = parts[0]
            interval = parts[1]

    # Preprocessing configuration
    st.markdown("---")
    st.subheader("⚙️ Preprocessing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        sequence_length = st.number_input(
            "Window Length (bars)",
            min_value=50,
            max_value=500,
            value=127,
            step=1,
            help="Number of bars per window. Default: 127 (prime number for FFT efficiency)"
        )

    with col2:
        overlap_percent = st.slider(
            "Window Overlap",
            min_value=0,
            max_value=90,
            value=50,
            step=5,
            format="%d%%",
            help="Overlap between consecutive windows. Default: 50%"
        )
        overlap = overlap_percent / 100.0
        st.caption(f"Step size: {int(sequence_length * (1 - overlap))} bars")

    # Feature engineering options
    with st.expander("🔧 Feature Engineering Options", expanded=False):
        st.markdown("**Volume/Liquidity Channel (OBV):**")
        obv_ema_period = st.number_input("OBV EMA Period", min_value=5, max_value=50, value=20, step=1)

        st.markdown("**Volatility/Risk Channel (NATR):**")
        col1, col2 = st.columns(2)
        with col1:
            atr_period = st.number_input("ATR Period", min_value=5, max_value=50, value=14, step=1)
        with col2:
            natr_smoothing = st.number_input("NATR EMA Smoothing", min_value=1, max_value=20, value=3, step=1)

    # Gap handling options
    with st.expander("🚨 Gap Handling", expanded=False):
        gap_buffer = st.number_input(
            "Gap Buffer (bars)",
            min_value=0,
            max_value=100,
            value=10,
            help="Additional bars to exclude around detected gaps"
        )

    # Output options
    st.markdown("---")
    st.subheader("💾 Output Configuration")

    output_name = st.text_input(
        "Package Name",
        value=f"{symbol}_{interval}_preprocessed",
        help="Name for the preprocessed package (will be saved as .h5)"
    )

    output_format = st.radio(
        "Output Format",
        ["HDF5 (.h5) - Recommended", "CSV (.csv)"],
        index=0,
        help="HDF5 is more efficient and preserves metadata"
    )

    # Run preprocessing
    st.markdown("---")

    if st.button("🚀 Run Preprocessing Pipeline", type="primary", use_container_width=True):
        try:
            # Data is already loaded in session state, use it directly
            st.info(f"Processing {len(df):,} bars from selected date range")

            # Step 1: Feature Engineering
            with st.spinner("Step 1/5: Engineering features..."):
                df = engineer_all_features(
                    df,
                    obv_ema_period=obv_ema_period,
                    natr_period=atr_period
                )

                # Validate feature independence
                corr_matrix, warning_report = compute_feature_correlation(df)

                if warning_report['safe_to_train']:
                    st.success("✅ Features engineered successfully")
                else:
                    st.error("❌ Critical correlation detected - features are not independent!")
                    for critical in warning_report['critical']:
                        st.error(f"  - {critical}")

                if warning_report['warnings']:
                    st.warning("⚠️ Feature correlation warnings:")
                    for warning in warning_report['warnings']:
                        st.warning(f"  - {warning}")

            # Step 2: Data Cleaning
            with st.spinner("Step 2/5: Detecting gaps and cleaning data..."):
                # Use interval from parsed filename, fallback to 1h if unknown
                interval_for_gaps = interval if interval != "unknown" else "1h"
                gaps = detect_gaps(df, interval=interval_for_gaps, max_gap_length=5)
                df = mark_excluded_windows(df, gaps, sequence_length=sequence_length)

                excluded_count = df['is_excluded'].sum() if 'is_excluded' in df.columns else 0
                st.success(f"✅ Detected {len(gaps)} gaps, excluded {excluded_count:,} bars")

            # Step 3: Normalization
            with st.spinner("Step 3/5: Normalizing channels..."):
                # Calculate normalization stats from the data (using all available data)
                channels_to_normalize = ['returns', 'volume_liquidity', 'volatility_risk']
                normalization_stats = calculate_normalization_stats(
                    df,
                    channels=channels_to_normalize,
                    winsorize=True
                )

                # Apply normalization
                df = apply_normalization(df, normalization_stats, suffix='_norm')

                normalized_channels = ['returns_norm', 'volume_liquidity_norm', 'volatility_risk_norm']
                st.success("✅ Normalized 3 channels (returns, volume, volatility)")

            # Step 4: Segmentation
            with st.spinner("Step 4/5: Creating sliding windows..."):
                windows, metadata = create_sliding_windows(
                    df,
                    normalized_channels=normalized_channels,
                    sequence_length=sequence_length,
                    overlap=overlap
                )

                if windows.shape[0] == 0:
                    st.error("❌ No valid windows created! Check your data quality and gap exclusions.")
                    return

                st.success(f"✅ Created {windows.shape[0]:,} windows of shape {windows.shape[1:2]}")

            # Step 5: Save
            with st.spinner("Step 5/5: Saving preprocessed package..."):
                # Determine output path
                preprocessed_dir = Path("data/preprocessed")
                preprocessed_dir.mkdir(parents=True, exist_ok=True)

                if output_format.startswith("HDF5"):
                    output_path = preprocessed_dir / f"{output_name}.h5"
                else:
                    output_path = preprocessed_dir / f"{output_name}.csv"

                # Additional metadata
                additional_metadata = {
                    'source_package': selected_package,
                    'symbol': symbol,
                    'interval': interval,
                    'source_bars': len(df),
                    'gaps_detected': len(gaps),
                    'excluded_bars': int(excluded_count),
                    'sequence_length': sequence_length,
                    'overlap': overlap,
                    'obv_ema_period': obv_ema_period,
                    'atr_period': atr_period,
                    'natr_smoothing': natr_smoothing
                }

                saved_path = save_preprocessed_package(
                    windows=windows,
                    metadata=metadata,
                    normalization_stats=normalization_stats,
                    filepath=str(output_path),
                    additional_metadata=additional_metadata
                )

                file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
                st.success(f"✅ Package saved: {saved_path} ({file_size_mb:.2f} MB)")

            # Summary
            st.markdown("---")
            st.subheader("📊 Preprocessing Summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Bars", f"{len(df):,}")
                st.metric("Gaps Detected", len(gaps))
            with col2:
                st.metric("Windows Created", f"{windows.shape[0]:,}")
                st.metric("Window Shape", f"{windows.shape[1]} x {windows.shape[2]}")
            with col3:
                st.metric("Output Size", f"{file_size_mb:.2f} MB")
                st.metric("Format", "HDF5" if output_path.suffix == '.h5' else "CSV")

            # Show normalization stats
            with st.expander("📈 Normalization Statistics"):
                st.json(normalization_stats)

            # Success message
            st.success("🎉 Preprocessing complete! You can now:")
            st.info("""
            - Explore your preprocessed data on the **Preprocessed Data Explorer** page
            - View distribution statistics and individual windows
            - Use this package for model training
            """)

        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
            import traceback
            with st.expander("Show full error"):
                st.code(traceback.format_exc())


# Execute the main function
show()
