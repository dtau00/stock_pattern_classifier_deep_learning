"""
Preprocessing Pipeline Validation Dashboard

Run validation tests for the preprocessing pipeline (data cleaning, normalization,
segmentation, splitting). These tests validate the data pipeline BEFORE model training.
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.validation.preprocessing_tests import PreprocessingValidator, run_preprocessing_validation

# Page config
st.set_page_config(page_title="Preprocessing Validation", page_icon="🔍", layout="wide")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Title
st.title("🔍 Preprocessing Pipeline Validation")
st.markdown("""
Validate your preprocessing pipeline components before training.

**What this tests:**
- Data cleaning and gap detection
- Feature normalization (z-score)
- Sliding window segmentation (127 bars, 50% overlap)
- Train/val/test splitting (70/15/15)
- HDF5 save/load integrity
- Full pipeline end-to-end
""")

st.markdown("---")

# Test runner section
st.header("🚀 Run Preprocessing Tests")

col1, col2 = st.columns([1, 3])

with col1:
    run_tests = st.button("▶️ Run Validation", type="primary", use_container_width=True)

with col2:
    st.info("Tests validate the preprocessing pipeline using your downloaded OHLCV data.")

# Data selection
st.subheader("Data Source")

# Check for available data files
data_dir = PROJECT_ROOT / "data" / "packages"

if data_dir.exists():
    csv_files = list(data_dir.glob("*.csv"))

    if csv_files:
        file_names = [f.stem for f in csv_files]
        selected_file = st.selectbox("Select OHLCV file:", file_names)

        # Show file info
        selected_path = data_dir / f"{selected_file}.csv"
        st.caption(f"📁 File: {selected_path}")
    else:
        st.error("❌ No OHLCV CSV files found in data/packages/. Please download data first using the OHLCV Manager.")
        selected_file = None
else:
    st.error("❌ Data directory not found. Please download data first using the OHLCV Manager.")
    selected_file = None

# Run validation
if run_tests:
    # Check if file is selected
    if selected_file is None:
        st.error("❌ Please download OHLCV data first using the OHLCV Manager (Page 10).")
    else:
        st.markdown("---")
        st.subheader("Running Validation Tests...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Load data
            status_text.text("Loading data...")
            progress_bar.progress(10)

            df_raw = pd.read_csv(selected_path)
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

            # Infer interval from filename (e.g., BTCUSDT_1h_...)
            parts = selected_file.split('_')
            interval = parts[1] if len(parts) > 1 else '1h'

            status_text.text(f"Running validation on {len(df_raw)} bars...")
            progress_bar.progress(30)

            # Create validator
            validator = PreprocessingValidator(verbose=False)

            # Import feature engineering
            from src.features.feature_engineering import engineer_features
            from src.preprocessing.data_cleaning import detect_gaps
            from src.preprocessing.normalization import normalize_features
            import numpy as np

            # Run tests
            tests = [
                ("Gap Detection", lambda: validator.test_gap_detection(df_raw, interval)),
                ("Normalization", lambda: validator.test_normalization(
                    engineer_features(df_raw),
                    ['returns', 'volume_liquidity', 'volatility_risk']
                )),
                ("Segmentation", lambda: validator.test_segmentation(
                    engineer_features(df_raw)[['returns', 'volume_liquidity', 'volatility_risk']].values
                )),
                ("Data Splitting", lambda: validator.test_data_splitting(
                    np.random.randn(1000, 127, 3)
                )),
                ("HDF5 I/O", lambda: validator.test_hdf5_io(
                    np.random.randn(100, 127, 3),
                    RESULTS_DIR
                )),
                ("Full Pipeline", lambda: validator.test_full_pipeline(df_raw, interval))
            ]

            for idx, (test_name, test_func) in enumerate(tests):
                status_text.text(f"Running {test_name}...")
                test_func()
                progress_bar.progress(30 + int((idx + 1) / len(tests) * 60))

            progress_bar.progress(100)
            status_text.text("Validation complete!")

            # Generate report
            report = validator.generate_report()

            # Save report
            report_file = RESULTS_DIR / "preprocessing_validation.json"
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data_file': selected_file,
                    'num_bars': len(df_raw),
                    'interval': interval,
                    'report': report
                }, f, indent=2)

            # Display results
            st.markdown("---")
            st.subheader("Validation Results")

            if report['all_passed']:
                st.success(f"✓ All {report['total_tests']} tests passed! Preprocessing pipeline is ready.")
            else:
                st.error(f"✗ {report['failed']} of {report['total_tests']} tests failed. Review errors below.")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tests", report['total_tests'])
            with col2:
                st.metric("Passed", report['passed'])
            with col3:
                st.metric("Failed", report['failed'])
            with col4:
                st.metric("Success Rate", report['success_rate'])

            # Detailed results
            st.markdown("---")
            st.subheader("Test Details")

            for test_name, result in report['results'].items():
                status_emoji = "✓" if result['status'] == 'PASS' else "✗"
                expanded = result['status'] != 'PASS'

                with st.expander(f"{status_emoji} {test_name.replace('_', ' ').title()} - {result['status']}", expanded=expanded):
                    if result['status'] == 'PASS':
                        st.success("Test passed successfully")

                        # Show details
                        details = {k: v for k, v in result.items() if k not in ['status', 'error']}
                        if details:
                            st.json(details)
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"Validation failed with error: {e}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            progress_bar.empty()


# Documentation
st.markdown("---")
st.header("📚 What Gets Tested")

with st.expander("1. Gap Detection"):
    st.markdown("""
**Purpose:** Verify gaps in OHLCV data are correctly identified.

**What it checks:**
- Gaps are detected based on expected interval
- Gap timestamps and missing bar counts are accurate
- No false positives

**Why it matters:** Windows overlapping gaps must be excluded to prevent training on incomplete data.
    """)

with st.expander("2. Normalization"):
    st.markdown("""
**Purpose:** Verify z-score normalization produces correct statistics.

**What it checks:**
- Each channel has mean ≈ 0 and std ≈ 1
- No NaN values introduced
- Statistics are tracked for inverse transform

**Why it matters:** Proper normalization is critical for model convergence and generalization.
    """)

with st.expander("3. Segmentation"):
    st.markdown("""
**Purpose:** Verify sliding windows are created correctly.

**What it checks:**
- Window shape is (num_windows, 127, 3)
- 50% overlap maintained (step_size = 63)
- No NaN values in windows
- Metadata is tracked

**Why it matters:** Incorrect window creation can lead to invalid training data.
    """)

with st.expander("4. Data Splitting"):
    st.markdown("""
**Purpose:** Verify train/val/test splits are correct.

**What it checks:**
- Split proportions are 70/15/15
- Chronological order is maintained
- No data leakage between splits

**Why it matters:** Proper splitting prevents look-ahead bias and ensures valid evaluation.
    """)

with st.expander("5. HDF5 I/O"):
    st.markdown("""
**Purpose:** Verify HDF5 save/load doesn't corrupt data.

**What it checks:**
- Windows save with compression
- Windows load without corruption
- Metadata is preserved

**Why it matters:** Data corruption would invalidate all downstream training.
    """)

with st.expander("6. Full Pipeline"):
    st.markdown("""
**Purpose:** Test complete preprocessing pipeline end-to-end.

**What it checks:**
- Raw OHLCV → Features → Normalized → Windows → Splits
- No errors in pipeline
- Output shapes are correct

**Why it matters:** Ensures all components work together correctly.
    """)


# Footer
st.markdown("---")
st.caption("Run preprocessing validation before training to ensure data quality.")
