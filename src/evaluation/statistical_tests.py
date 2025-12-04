"""
Statistical Independence Tests Module

Implements statistical tests for model validation as specified in Design Document Section 5.

Tests implemented:
1. Volatility Independence Test (Kruskal-Wallis H-test)
   - Validates clustering is independent of volatility magnitude
   - Uses NATR (Normalized ATR) as volatility proxy
   - Non-parametric test (no normality assumptions)

2. Temporal Stability Test (Chi-Square test)
   - Validates clustering is independent of time periods
   - Tests for temporal banding in cluster distribution
   - Ensures model generalizes across time

Reference: Design Document Section 5, Statistical Tests Table
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import pandas as pd


class StatisticalTests:
    """
    Statistical independence tests for clustering validation.

    Tests ensure the model clusters based on pattern shape rather than
    magnitude-based features (volatility) or temporal artifacts.

    Example:
        >>> tester = StatisticalTests()
        >>> # Test volatility independence
        >>> result = tester.test_volatility_independence(cluster_ids, natr_values)
        >>> print(f"p-value: {result['p_value']:.3f}, Status: {result['status']}")
    """

    @staticmethod
    def test_volatility_independence(
        cluster_ids: np.ndarray,
        volatility_values: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test if clustering is independent of volatility using Kruskal-Wallis H-test.

        The Kruskal-Wallis H-test is a non-parametric test that determines whether
        samples from different groups (clusters) originate from the same distribution.

        Null Hypothesis (H0): Cluster assignments are independent of volatility
                             (all clusters have similar volatility distributions)
        Alternative Hypothesis (H1): Clustering depends on volatility
                                    (different clusters have different volatility)

        Args:
            cluster_ids: Cluster assignments of shape (N,)
            volatility_values: Volatility values (e.g., NATR) of shape (N,)
                              Should be the original volatility values, not normalized
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with:
                - 'test_name': 'Kruskal-Wallis H-test'
                - 'statistic': H-statistic value
                - 'p_value': P-value
                - 'alpha': Significance level
                - 'status': 'PASS', 'FAIL', or 'WARN'
                - 'message': Interpretation message
                - 'cluster_volatility_stats': Per-cluster volatility statistics

        Thresholds (Design Doc Section 5):
            - p > 0.05: PASS (independent of volatility)
            - p < 0.01: FAIL (clustering on magnitude, not shape)
            - 0.01 <= p <= 0.05: WARN (borderline, investigate further)

        Interpretation:
            - High p-value (> 0.05): Cannot reject H0, clustering is independent
                                    of volatility (GOOD)
            - Low p-value (< 0.01): Reject H0, clustering depends on volatility (BAD)
        """
        cluster_ids = np.asarray(cluster_ids)
        volatility_values = np.asarray(volatility_values)

        if len(cluster_ids) != len(volatility_values):
            raise ValueError(
                f"Array length mismatch: cluster_ids={len(cluster_ids)}, "
                f"volatility_values={len(volatility_values)}"
            )

        # Group volatility values by cluster
        unique_clusters = np.unique(cluster_ids)
        volatility_by_cluster = [
            volatility_values[cluster_ids == c]
            for c in unique_clusters
        ]

        # Remove empty clusters (shouldn't happen, but safety check)
        volatility_by_cluster = [v for v in volatility_by_cluster if len(v) > 0]

        # Require at least 2 clusters for test
        if len(volatility_by_cluster) < 2:
            return {
                'test_name': 'Kruskal-Wallis H-test',
                'statistic': np.nan,
                'p_value': np.nan,
                'alpha': alpha,
                'status': 'SKIP',
                'message': f'Insufficient clusters for test (need >= 2, got {len(volatility_by_cluster)})',
                'cluster_volatility_stats': {}
            }

        # Perform Kruskal-Wallis H-test
        h_statistic, p_value = stats.kruskal(*volatility_by_cluster)

        # Determine status
        if p_value > alpha:
            status = 'PASS'
            message = (
                f"[PASS] Clustering is independent of volatility (p={p_value:.4f} > {alpha}). "
                f"Model clusters based on shape, not magnitude."
            )
        elif p_value < 0.01:
            status = 'FAIL'
            message = (
                f"[FAIL] Clustering depends on volatility (p={p_value:.4f} < 0.01). "
                f"Model may be clustering on magnitude rather than shape. "
                f"CRITICAL: This indicates the model is not learning patterns correctly."
            )
        else:
            status = 'WARN'
            message = (
                f"[WARN] Borderline volatility independence (0.01 <= p={p_value:.4f} <= {alpha}). "
                f"Model shows weak dependence on volatility. Investigate cluster characteristics."
            )

        # Compute per-cluster volatility statistics
        cluster_stats = {}
        for c, vol_values in zip(unique_clusters, volatility_by_cluster):
            cluster_stats[int(c)] = {
                'mean': float(np.mean(vol_values)),
                'std': float(np.std(vol_values)),
                'median': float(np.median(vol_values)),
                'min': float(np.min(vol_values)),
                'max': float(np.max(vol_values)),
                'count': int(len(vol_values))
            }

        return {
            'test_name': 'Kruskal-Wallis H-test',
            'statistic': float(h_statistic),
            'p_value': float(p_value),
            'alpha': alpha,
            'status': status,
            'message': message,
            'cluster_volatility_stats': cluster_stats
        }

    @staticmethod
    def test_temporal_stability(
        cluster_ids: np.ndarray,
        timestamps: np.ndarray,
        n_periods: int = 5,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test if clustering is independent of time periods using Chi-Square test.

        Divides the data into chronological chunks and tests whether cluster
        distribution is uniform across time periods.

        Null Hypothesis (H0): Cluster distribution is independent of time
                             (no temporal banding)
        Alternative Hypothesis (H1): Cluster distribution varies across time
                                    (temporal banding present)

        Args:
            cluster_ids: Cluster assignments of shape (N,)
            timestamps: Timestamps of shape (N,) in chronological order
                       (e.g., Unix timestamps, datetime indices)
            n_periods: Number of time chunks to create (default: 5)
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with:
                - 'test_name': 'Chi-Square Test of Independence'
                - 'statistic': Chi-square statistic
                - 'p_value': P-value
                - 'degrees_of_freedom': Degrees of freedom
                - 'alpha': Significance level
                - 'status': 'PASS', 'FAIL', or 'WARN'
                - 'message': Interpretation message
                - 'contingency_table': Cluster distribution across time periods
                - 'expected_frequencies': Expected frequencies under independence

        Thresholds (Design Doc Section 5):
            - p > 0.05: PASS (independent of time)
            - p < 0.01: FAIL (time-dependent clustering)
            - 0.01 <= p <= 0.05: WARN (borderline, investigate)

        Note:
            Data should be chronologically ordered before calling this function.
            If data is not pre-sorted, sort by timestamps first.
        """
        cluster_ids = np.asarray(cluster_ids)
        timestamps = np.asarray(timestamps)

        if len(cluster_ids) != len(timestamps):
            raise ValueError(
                f"Array length mismatch: cluster_ids={len(cluster_ids)}, "
                f"timestamps={len(timestamps)}"
            )

        # Sort by timestamp (ensure chronological order)
        sort_idx = np.argsort(timestamps)
        cluster_ids_sorted = cluster_ids[sort_idx]
        timestamps_sorted = timestamps[sort_idx]

        # Divide into n_periods chronological chunks
        n_samples = len(cluster_ids_sorted)
        chunk_size = n_samples // n_periods
        remainder = n_samples % n_periods

        # Create period labels
        period_labels = np.zeros(n_samples, dtype=int)
        start_idx = 0

        for period_idx in range(n_periods):
            # Add one extra sample to first 'remainder' periods
            current_chunk_size = chunk_size + (1 if period_idx < remainder else 0)
            end_idx = start_idx + current_chunk_size
            period_labels[start_idx:end_idx] = period_idx
            start_idx = end_idx

        # Build contingency table: rows = clusters, columns = time periods
        unique_clusters = np.unique(cluster_ids_sorted)
        contingency_table = np.zeros((len(unique_clusters), n_periods), dtype=int)

        for i, cluster_id in enumerate(unique_clusters):
            for period_idx in range(n_periods):
                mask = (cluster_ids_sorted == cluster_id) & (period_labels == period_idx)
                contingency_table[i, period_idx] = mask.sum()

        # Perform Chi-Square test of independence
        try:
            chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
        except ValueError as e:
            # May fail if contingency table has issues (e.g., all zeros)
            return {
                'test_name': 'Chi-Square Test of Independence',
                'statistic': np.nan,
                'p_value': np.nan,
                'degrees_of_freedom': np.nan,
                'alpha': alpha,
                'status': 'ERROR',
                'message': f'Chi-Square test failed: {str(e)}',
                'contingency_table': contingency_table.tolist(),
                'expected_frequencies': None
            }

        # Determine status
        if p_value > alpha:
            status = 'PASS'
            message = (
                f"[PASS] Clustering is independent of time (p={p_value:.4f} > {alpha}). "
                f"No temporal banding detected. Model generalizes across time periods."
            )
        elif p_value < 0.01:
            status = 'FAIL'
            message = (
                f"[FAIL] Clustering depends on time (p={p_value:.4f} < 0.01). "
                f"Temporal banding detected. Model may be learning time-specific artifacts. "
                f"CRITICAL: This indicates poor generalization across time."
            )
        else:
            status = 'WARN'
            message = (
                f"[WARN] Borderline temporal independence (0.01 <= p={p_value:.4f} <= {alpha}). "
                f"Weak temporal dependence detected. Investigate cluster distribution across time."
            )

        return {
            'test_name': 'Chi-Square Test of Independence',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'alpha': alpha,
            'status': status,
            'message': message,
            'contingency_table': contingency_table.tolist(),
            'expected_frequencies': expected_freq.tolist()
        }

    @staticmethod
    def run_all_tests(
        cluster_ids: np.ndarray,
        volatility_values: np.ndarray,
        timestamps: np.ndarray,
        n_periods: int = 5,
        alpha: float = 0.05
    ) -> Dict:
        """
        Run all statistical independence tests.

        Args:
            cluster_ids: Cluster assignments of shape (N,)
            volatility_values: Volatility values (e.g., NATR) of shape (N,)
            timestamps: Timestamps of shape (N,)
            n_periods: Number of time chunks for temporal test (default: 5)
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with:
                - 'volatility_independence': Results from Kruskal-Wallis test
                - 'temporal_stability': Results from Chi-Square test
                - 'overall_status': 'PASS' if all tests pass, else 'FAIL'
                - 'summary': Summary message
        """
        tester = StatisticalTests()

        # Run volatility independence test
        vol_test = tester.test_volatility_independence(
            cluster_ids, volatility_values, alpha
        )

        # Run temporal stability test
        temp_test = tester.test_temporal_stability(
            cluster_ids, timestamps, n_periods, alpha
        )

        # Determine overall status
        statuses = [vol_test['status'], temp_test['status']]

        if 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARN' in statuses:
            overall_status = 'WARN'
        elif all(s == 'PASS' for s in statuses):
            overall_status = 'PASS'
        else:
            overall_status = 'UNKNOWN'

        # Summary message
        summary_lines = [
            "Statistical Independence Tests Summary:",
            f"  Volatility Independence: {vol_test['status']}",
            f"  Temporal Stability: {temp_test['status']}",
            f"  Overall: {overall_status}"
        ]

        summary = "\n".join(summary_lines)

        return {
            'volatility_independence': vol_test,
            'temporal_stability': temp_test,
            'overall_status': overall_status,
            'summary': summary
        }


def test_statistical_tests():
    """Unit tests for StatisticalTests."""
    print("Testing StatisticalTests...")

    tester = StatisticalTests()

    # Test 1: Volatility Independence - Independent case
    print("\n[Test 1] Volatility Independence - Independent clusters")

    # Create clusters that are independent of volatility
    np.random.seed(42)
    n_samples = 1000
    n_clusters = 4

    # Each cluster has random samples from same volatility distribution
    cluster_ids = np.random.randint(0, n_clusters, n_samples)
    volatility_values = np.random.exponential(scale=2.0, size=n_samples)  # Same distribution for all

    result = tester.test_volatility_independence(cluster_ids, volatility_values)

    print(f"  Test: {result['test_name']}")
    print(f"  H-statistic: {result['statistic']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Status: {result['status']}")
    print(f"  {result['message']}")

    assert result['status'] == 'PASS', "Expected PASS for independent volatility"
    print(f"  [PASS] Independent volatility correctly detected")

    # Test 2: Volatility Independence - Dependent case
    print("\n[Test 2] Volatility Independence - Dependent clusters")

    # Create clusters where each cluster has different volatility
    cluster_ids = []
    volatility_values = []

    for c in range(n_clusters):
        n_per_cluster = n_samples // n_clusters
        cluster_ids.extend([c] * n_per_cluster)
        # Each cluster has different mean volatility
        vol_mean = 0.5 + c * 0.5
        volatility_values.extend(np.random.exponential(scale=vol_mean, size=n_per_cluster))

    cluster_ids = np.array(cluster_ids)
    volatility_values = np.array(volatility_values)

    result = tester.test_volatility_independence(cluster_ids, volatility_values)

    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Status: {result['status']}")

    # Should fail (p-value should be very small)
    assert result['status'] == 'FAIL', "Expected FAIL for dependent volatility"
    print(f"  [PASS] Dependent volatility correctly detected")

    # Test 3: Temporal Stability - Independent case
    print("\n[Test 3] Temporal Stability - Time-independent clusters")

    # Create timestamps and clusters independent of time
    np.random.seed(42)
    timestamps = np.arange(n_samples)
    cluster_ids = np.random.randint(0, n_clusters, n_samples)

    result = tester.test_temporal_stability(cluster_ids, timestamps, n_periods=5)

    print(f"  Test: {result['test_name']}")
    print(f"  Chi-square: {result['statistic']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Degrees of freedom: {result['degrees_of_freedom']}")
    print(f"  Status: {result['status']}")

    assert result['status'] == 'PASS', "Expected PASS for time-independent clustering"
    print(f"  [PASS] Time-independent clustering correctly detected")

    # Test 4: Temporal Stability - Dependent case
    print("\n[Test 4] Temporal Stability - Time-dependent clusters")

    # Create strong temporal banding
    timestamps = np.arange(n_samples)
    cluster_ids = []

    chunk_size = n_samples // n_clusters
    for c in range(n_clusters):
        cluster_ids.extend([c] * chunk_size)

    cluster_ids.extend([0] * (n_samples - len(cluster_ids)))  # Fill remainder
    cluster_ids = np.array(cluster_ids)

    result = tester.test_temporal_stability(cluster_ids, timestamps, n_periods=5)

    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Status: {result['status']}")

    # Should fail (perfect temporal banding)
    assert result['status'] == 'FAIL', "Expected FAIL for time-dependent clustering"
    print(f"  [PASS] Time-dependent clustering correctly detected")

    # Test 5: Run all tests
    print("\n[Test 5] Run all tests together")

    # Create good clustering (independent of volatility and time)
    np.random.seed(42)
    timestamps = np.arange(n_samples)
    cluster_ids = np.random.randint(0, n_clusters, n_samples)
    volatility_values = np.random.exponential(scale=2.0, size=n_samples)

    all_results = tester.run_all_tests(cluster_ids, volatility_values, timestamps)

    print(f"\n{all_results['summary']}")
    assert all_results['overall_status'] == 'PASS', "Expected overall PASS"
    print(f"  [PASS] All tests passed")

    print("\n[SUCCESS] All StatisticalTests tests passed!")
    return True


if __name__ == '__main__':
    test_statistical_tests()
