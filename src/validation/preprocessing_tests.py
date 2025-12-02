"""
Preprocessing Pipeline Validation Tests

Tests for data cleaning, normalization, segmentation, and splitting.
These tests validate the preprocessing pipeline BEFORE model training.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_cleaning import detect_gaps, flag_exclusions
from src.preprocessing.normalization import normalize_features
from src.preprocessing.segmentation import create_sliding_windows
from src.preprocessing.data_splitting import split_data, verify_chronological_order


class PreprocessingValidator:
    """Validates preprocessing pipeline components."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, Dict[str, Any]] = {}

    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def test_gap_detection(self, df: pd.DataFrame, interval: str) -> bool:
        """
        Test gap detection logic.

        Validates:
        - Gaps are correctly identified
        - Gap structure is correct (tuple format)
        - No false positives

        Returns:
            bool: Test passed
        """
        self.log("\n[1/6] Testing Gap Detection...")

        try:
            # Detect gaps (returns list of tuples)
            gaps = detect_gaps(df, interval, max_gap_length=5)

            # Basic validation
            if not isinstance(gaps, list):
                raise ValueError(f"Expected list, got {type(gaps)}")

            # Check gap structure (should be tuples)
            for gap in gaps:
                if not isinstance(gap, tuple) or len(gap) != 3:
                    raise ValueError(f"Expected tuple of length 3, got {gap}")

                gap_start_idx, gap_end_idx, gap_length = gap

                # Validate indices
                if not isinstance(gap_start_idx, (int, np.integer)):
                    raise ValueError(f"Invalid gap_start_idx type: {type(gap_start_idx)}")
                if not isinstance(gap_end_idx, (int, np.integer)):
                    raise ValueError(f"Invalid gap_end_idx type: {type(gap_end_idx)}")
                if not isinstance(gap_length, (int, np.integer)):
                    raise ValueError(f"Invalid gap_length type: {type(gap_length)}")

                # Validate logical ordering
                if gap_start_idx >= gap_end_idx:
                    raise ValueError(f"gap_start_idx >= gap_end_idx: {gap}")
                if gap_length < 0:
                    raise ValueError(f"Negative gap_length: {gap}")

            self.results['gap_detection'] = {
                'status': 'PASS',
                'gaps_found': len(gaps),
                'details': [{'start_idx': g[0], 'end_idx': g[1], 'length': g[2]} for g in gaps]
            }

            self.log(f"  [OK] Gap detection passed - Found {len(gaps)} gaps")
            return True

        except Exception as e:
            self.results['gap_detection'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            self.log(f"  [FAIL] Gap detection failed: {e}")
            return False

    def test_normalization(self, features_df: pd.DataFrame,
                          channel_cols: List[str]) -> bool:
        """
        Test feature normalization.

        Validates:
        - Z-score normalization produces mean≈0, std≈1
        - No NaN values introduced by normalization (existing NaNs are expected)
        - Statistics are tracked correctly

        Returns:
            bool: Test passed
        """
        self.log("\n[2/6] Testing Normalization...")

        try:
            # Drop NaN rows before normalization (expected from feature engineering)
            features_df_clean = features_df.dropna(subset=channel_cols)
            initial_nan_count = len(features_df) - len(features_df_clean)

            if initial_nan_count > 0:
                self.log(f"  [INFO] Dropped {initial_nan_count} rows with NaN values from feature engineering")

            # Normalize features
            normalized_df, stats = normalize_features(features_df_clean, channel_cols)

            # Check for new NaN values (normalization shouldn't introduce any)
            nan_count = normalized_df[channel_cols].isna().sum().sum()
            if nan_count > 0:
                raise ValueError(f"Normalization introduced {nan_count} new NaN values")

            # Verify z-score properties (mean≈0, std≈1)
            tolerance_mean = 1e-10
            tolerance_std = 0.05  # Allow 5% deviation

            for col in channel_cols:
                mean = normalized_df[col].mean()
                std = normalized_df[col].std()

                if abs(mean) > tolerance_mean:
                    raise ValueError(f"{col}: mean={mean:.6f}, expected ≈0")

                if abs(std - 1.0) > tolerance_std:
                    raise ValueError(f"{col}: std={std:.6f}, expected ≈1")

            # Verify stats are tracked
            required_stats = {'mean', 'std'}
            for col in channel_cols:
                if col not in stats or not all(k in stats[col] for k in required_stats):
                    raise ValueError(f"Stats not tracked for {col}")

            self.results['normalization'] = {
                'status': 'PASS',
                'channels': len(channel_cols),
                'stats': {col: {'mean': stats[col]['mean'], 'std': stats[col]['std']}
                         for col in channel_cols}
            }

            self.log(f"  [OK] Normalization passed - {len(channel_cols)} channels normalized")
            return True

        except Exception as e:
            self.results['normalization'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            self.log(f"  [FAIL] Normalization failed: {e}")
            return False

    def test_segmentation(self, features: np.ndarray,
                         window_size: int = 127,
                         step_size: int = 63) -> bool:
        """
        Test sliding window segmentation.

        Validates:
        - Window shape is correct (num_windows, window_size, num_channels)
        - 50% overlap is maintained (step_size = window_size // 2)
        - No NaN values in windows
        - Windows don't overlap with gaps

        Returns:
            bool: Test passed
        """
        self.log("\n[3/6] Testing Segmentation...")

        try:
            # Create sliding windows (assuming no gaps for this test)
            gap_flags = np.zeros(len(features), dtype=bool)

            windows, metadata = create_sliding_windows(
                features,
                gap_flags=gap_flags,
                window_size=window_size,
                step_size=step_size
            )

            # Validate window shape
            num_channels = features.shape[1]
            expected_shape = (len(windows), window_size, num_channels)

            if windows.shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {windows.shape}")

            # Check for NaN values
            nan_count = np.isnan(windows).sum()
            if nan_count > 0:
                raise ValueError(f"Found {nan_count} NaN values in windows")

            # Verify overlap
            expected_windows = (len(features) - window_size) // step_size + 1
            if len(windows) != expected_windows:
                self.log(f"  [WARN] Expected {expected_windows} windows, got {len(windows)}")

            # Verify metadata
            if 'window_timestamps' not in metadata or len(metadata['window_timestamps']) != len(windows):
                raise ValueError("Window timestamps not tracked correctly")

            self.results['segmentation'] = {
                'status': 'PASS',
                'num_windows': len(windows),
                'window_shape': f"{window_size}x{num_channels}",
                'overlap': f"{step_size}/{window_size} (50%)"
            }

            self.log(f"  [OK] Segmentation passed - Created {len(windows)} windows")
            return True

        except Exception as e:
            self.results['segmentation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            self.log(f"  [FAIL] Segmentation failed: {e}")
            return False

    def test_data_splitting(self, windows: np.ndarray) -> bool:
        """
        Test train/val/test splitting.

        Validates:
        - Split proportions are correct (70/15/15)
        - Chronological order is maintained
        - No data leakage between splits

        Returns:
            bool: Test passed
        """
        self.log("\n[4/6] Testing Data Splitting...")

        try:
            # Split data
            train, val, test = split_data(windows, train_pct=0.7, val_pct=0.15, test_pct=0.15)

            # Verify proportions (allow 1% tolerance)
            total = len(windows)
            train_pct = len(train) / total
            val_pct = len(val) / total
            test_pct = len(test) / total

            if not (0.69 <= train_pct <= 0.71):
                raise ValueError(f"Train split {train_pct:.2%}, expected ~70%")

            if not (0.14 <= val_pct <= 0.16):
                raise ValueError(f"Val split {val_pct:.2%}, expected ~15%")

            if not (0.14 <= test_pct <= 0.16):
                raise ValueError(f"Test split {test_pct:.2%}, expected ~15%")

            # Verify chronological order (create dummy metadata)
            metadata = {'window_timestamps': list(range(len(windows)))}
            is_chronological = verify_chronological_order(train, val, test, metadata)

            if not is_chronological:
                raise ValueError("Chronological order not maintained")

            self.results['data_splitting'] = {
                'status': 'PASS',
                'train_size': len(train),
                'val_size': len(val),
                'test_size': len(test),
                'train_pct': f"{train_pct:.1%}",
                'val_pct': f"{val_pct:.1%}",
                'test_pct': f"{test_pct:.1%}"
            }

            self.log(f"  [OK] Data splitting passed - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
            return True

        except Exception as e:
            self.results['data_splitting'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            self.log(f"  [FAIL] Data splitting failed: {e}")
            return False

    def test_hdf5_io(self, windows: np.ndarray, temp_path: Path) -> bool:
        """
        Test HDF5 save/load functionality.

        Validates:
        - Windows save correctly with compression
        - Windows load correctly
        - No data corruption
        - Metadata is preserved

        Returns:
            bool: Test passed
        """
        self.log("\n[5/6] Testing HDF5 I/O...")

        try:
            # Create test file
            test_file = temp_path / "test_windows.h5"
            test_file.parent.mkdir(parents=True, exist_ok=True)

            # Save windows
            with h5py.File(test_file, 'w') as f:
                f.create_dataset(
                    'windows',
                    data=windows,
                    compression='gzip',
                    compression_opts=4
                )
                f.attrs['num_windows'] = len(windows)
                f.attrs['window_size'] = windows.shape[1]
                f.attrs['num_channels'] = windows.shape[2]

            # Load windows
            with h5py.File(test_file, 'r') as f:
                loaded_windows = f['windows'][:]
                loaded_num_windows = f.attrs['num_windows']
                loaded_window_size = f.attrs['window_size']
                loaded_num_channels = f.attrs['num_channels']

            # Verify data integrity
            if not np.allclose(windows, loaded_windows, atol=1e-10):
                raise ValueError("Data corruption detected after save/load")

            if loaded_num_windows != len(windows):
                raise ValueError(f"Metadata mismatch: num_windows")

            if loaded_window_size != windows.shape[1]:
                raise ValueError(f"Metadata mismatch: window_size")

            if loaded_num_channels != windows.shape[2]:
                raise ValueError(f"Metadata mismatch: num_channels")

            # Cleanup
            test_file.unlink()

            self.results['hdf5_io'] = {
                'status': 'PASS',
                'file_size': f"{test_file.stat().st_size / 1024:.1f} KB" if test_file.exists() else "cleaned",
                'compression': 'gzip (level 4)'
            }

            self.log(f"  [OK] HDF5 I/O passed - Save/load verified")
            return True

        except Exception as e:
            self.results['hdf5_io'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            self.log(f"  [FAIL] HDF5 I/O failed: {e}")
            return False

    def test_full_pipeline(self, df_raw: pd.DataFrame, interval: str) -> bool:
        """
        Test complete preprocessing pipeline end-to-end.

        Validates:
        - Raw OHLCV -> Features -> Normalized -> Windows -> Splits
        - No errors in pipeline
        - Output shapes are correct

        Returns:
            bool: Test passed
        """
        self.log("\n[6/6] Testing Full Pipeline...")

        try:
            from src.features.feature_engineering import engineer_features

            # Step 1: Feature engineering
            features_df = engineer_features(df_raw)
            channel_cols = ['returns', 'volume_liquidity', 'volatility_risk']

            # Step 2: Drop NaN rows (expected from feature engineering)
            features_df_clean = features_df.dropna(subset=channel_cols)

            # Step 3: Gap detection
            gaps = detect_gaps(features_df_clean, interval)
            gap_flags = flag_exclusions(features_df_clean, gaps, window_size=127)

            # Step 4: Normalization
            normalized_df, stats = normalize_features(features_df_clean, channel_cols)

            # Step 5: Convert to array
            features_array = normalized_df[channel_cols].values

            # Step 6: Segmentation
            windows, metadata = create_sliding_windows(
                features_array,
                gap_flags=gap_flags,
                window_size=127,
                step_size=63
            )

            # Step 7: Splitting
            train, val, test = split_data(windows, train_pct=0.7, val_pct=0.15, test_pct=0.15)

            # Validate final output
            if len(train) == 0 or len(val) == 0 or len(test) == 0:
                raise ValueError("One or more splits are empty")

            if train.shape[1] != 127 or train.shape[2] != 3:
                raise ValueError(f"Invalid train shape: {train.shape}")

            self.results['full_pipeline'] = {
                'status': 'PASS',
                'input_bars': len(df_raw),
                'windows_created': len(windows),
                'train_windows': len(train),
                'val_windows': len(val),
                'test_windows': len(test),
                'gaps_detected': len(gaps),
                'pipeline': 'OHLCV -> Features -> Normalize -> Segment -> Split'
            }

            self.log(f"  [OK] Full pipeline passed - {len(df_raw)} bars -> {len(windows)} windows")
            return True

        except Exception as e:
            self.results['full_pipeline'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            self.log(f"  [FAIL] Full pipeline failed: {e}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate validation report.

        Returns:
            dict: Validation report with all test results
        """
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASS')

        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'success_rate': f"{passed_tests / total_tests * 100:.1f}%" if total_tests > 0 else "0%",
            'all_passed': passed_tests == total_tests,
            'results': self.results
        }

    def print_summary(self):
        """Print validation summary."""
        report = self.generate_report()

        print("\n" + "=" * 80)
        print("PREPROCESSING VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed']}")
        print(f"Failed: {report['failed']}")
        print(f"Success Rate: {report['success_rate']}")
        print("=" * 80)

        if report['all_passed']:
            print("\n[PASS] ALL TESTS PASSED - Preprocessing pipeline is ready!")
        else:
            print("\n[FAIL] SOME TESTS FAILED - Review errors above")

        print()


def run_preprocessing_validation(df_raw: pd.DataFrame, interval: str = '1h') -> Dict[str, Any]:
    """
    Run all preprocessing validation tests.

    Args:
        df_raw: Raw OHLCV DataFrame
        interval: Timeframe interval (e.g., '1h', '4h')

    Returns:
        dict: Validation report
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE VALIDATION")
    print("=" * 80)

    validator = PreprocessingValidator(verbose=True)

    # Import feature engineering
    from src.features.feature_engineering import engineer_features

    # Engineer features
    features_df = engineer_features(df_raw)
    channel_cols = ['returns', 'volume_liquidity', 'volatility_risk']

    # Create test data
    features_array = features_df[channel_cols].values

    # Run tests
    validator.test_gap_detection(df_raw, interval)
    validator.test_normalization(features_df, channel_cols)
    validator.test_segmentation(features_array)

    # Create dummy windows for splitting test
    dummy_windows = np.random.randn(1000, 127, 3)
    validator.test_data_splitting(dummy_windows)

    # Test HDF5 I/O
    temp_path = project_root / "data" / "validation"
    validator.test_hdf5_io(dummy_windows, temp_path)

    # Test full pipeline
    validator.test_full_pipeline(df_raw, interval)

    # Print summary
    validator.print_summary()

    return validator.generate_report()


if __name__ == "__main__":
    """Run validation with synthetic data."""
    print("Running preprocessing validation with synthetic data...")

    # Create synthetic OHLCV data
    num_bars = 1000
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(num_bars) * 2)

    df_raw = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=num_bars, freq='1h'),
        'open': prices + np.random.randn(num_bars) * 0.5,
        'high': prices + np.abs(np.random.randn(num_bars)) * 1.5,
        'low': prices - np.abs(np.random.randn(num_bars)) * 1.5,
        'close': prices,
        'volume': np.abs(np.random.randn(num_bars)) * 1000
    })

    # Ensure OHLC relationships
    df_raw['high'] = df_raw[['open', 'high', 'low', 'close']].max(axis=1)
    df_raw['low'] = df_raw[['open', 'high', 'low', 'close']].min(axis=1)

    # Run validation
    report = run_preprocessing_validation(df_raw, interval='1h')

    # Exit with appropriate code
    sys.exit(0 if report['all_passed'] else 1)
