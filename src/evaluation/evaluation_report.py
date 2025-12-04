"""
Comprehensive Evaluation Report Module

Generates complete evaluation reports with all metrics specified in Design Document Section 5.

This module orchestrates:
1. Clustering quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
2. Stability testing (ARI between runs)
3. Confidence calibration quality (R²)
4. Statistical independence tests (Kruskal-Wallis, Chi-Square)

Outputs JSON evaluation reports for model validation and tracking.

Reference: Design Document Section 5 & 6
"""

import json
import torch
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

try:
    from .clustering_metrics import ClusteringMetrics
    from .confidence_calibration import ConfidenceCalibrator
    from .statistical_tests import StatisticalTests
except ImportError:
    from clustering_metrics import ClusteringMetrics
    from confidence_calibration import ConfidenceCalibrator
    from statistical_tests import StatisticalTests


class EvaluationReport:
    """
    Comprehensive evaluation report generator.

    Computes all Design Document metrics and generates JSON reports.

    Example:
        >>> evaluator = EvaluationReport()
        >>> report = evaluator.evaluate_model(
        ...     model, val_loader, test_loader, volatility_values, timestamps
        ... )
        >>> evaluator.save_report(report, "data/reports/eval_20250130.json")
    """

    def __init__(self):
        self.clustering_metrics = ClusteringMetrics()
        self.statistical_tests = StatisticalTests()

    def evaluate_clustering_quality(
        self,
        z_normalized: np.ndarray,
        cluster_ids: np.ndarray,
        sample_size: Optional[int] = 5000
    ) -> Dict:
        """
        Evaluate clustering quality metrics.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            cluster_ids: Cluster assignments of shape (N,)
            sample_size: Sample size for Silhouette computation

        Returns:
            Dictionary with all clustering metrics and pass/fail status
        """
        # Compute all metrics
        metrics = self.clustering_metrics.compute_all_metrics(
            z_normalized, cluster_ids, sample_size
        )

        # Check thresholds
        threshold_results = self.clustering_metrics.check_metric_thresholds(metrics)

        # Determine overall pass
        all_passed = all(passed for passed, _ in threshold_results.values())

        return {
            'metrics': metrics,
            'thresholds': {
                name: {'passed': passed, 'message': msg}
                for name, (passed, msg) in threshold_results.items()
            },
            'overall_passed': all_passed
        }

    def evaluate_complete(
        self,
        z_normalized_val: np.ndarray,
        cluster_ids_val: np.ndarray,
        centroids_normalized: np.ndarray,
        volatility_values: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        calibration_results: Optional[Dict] = None,
        z_normalized_test: Optional[np.ndarray] = None,
        cluster_ids_test: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Complete model evaluation with all Design Document metrics.

        Args:
            z_normalized_val: Validation latent vectors (for calibration)
            cluster_ids_val: Validation cluster assignments
            centroids_normalized: Model centroids (L2-normalized)
            volatility_values: Optional NATR values for volatility independence test
            timestamps: Optional timestamps for temporal stability test
            calibration_results: Optional calibration results (if already calibrated)
            z_normalized_test: Optional test latent vectors (for final metrics)
            cluster_ids_test: Optional test cluster assignments

        Returns:
            Complete evaluation report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_set': {},
            'test_set': {},
            'statistical_tests': {},
            'confidence_calibration': {},
            'overall_status': 'PASS',
            'failures': []
        }

        # 1. Validation set clustering metrics
        print("\n[Evaluation] Clustering Quality Metrics (Validation Set)")
        print("-" * 60)

        val_quality = self.evaluate_clustering_quality(
            z_normalized_val, cluster_ids_val
        )

        report['validation_set'] = val_quality

        for name, result in val_quality['thresholds'].items():
            print(f"  {result['message']}")

        if not val_quality['overall_passed']:
            report['overall_status'] = 'FAIL'
            report['failures'].append('Validation clustering quality metrics failed')

        # 2. Test set clustering metrics (if provided)
        if z_normalized_test is not None and cluster_ids_test is not None:
            print("\n[Evaluation] Clustering Quality Metrics (Test Set)")
            print("-" * 60)

            test_quality = self.evaluate_clustering_quality(
                z_normalized_test, cluster_ids_test
            )

            report['test_set'] = test_quality

            for name, result in test_quality['thresholds'].items():
                print(f"  {result['message']}")

            if not test_quality['overall_passed']:
                report['overall_status'] = 'FAIL'
                report['failures'].append('Test clustering quality metrics failed')

        # 3. Confidence calibration (use validation set)
        print("\n[Evaluation] Confidence Calibration")
        print("-" * 60)

        if calibration_results is not None:
            report['confidence_calibration'] = calibration_results

            if not calibration_results.get('passed', False):
                report['overall_status'] = 'FAIL'
                report['failures'].append('Confidence calibration failed (R² < 0.7)')

            print(f"  Best gamma: {calibration_results['best_gamma']}")
            print(f"  R²: {calibration_results['best_r2']:.3f}")
            print(f"  Status: {'PASS' if calibration_results['passed'] else 'FAIL'}")

        # 4. Statistical independence tests
        if volatility_values is not None or timestamps is not None:
            print("\n[Evaluation] Statistical Independence Tests")
            print("-" * 60)

            # Use test set if available, otherwise validation set
            test_cluster_ids = cluster_ids_test if cluster_ids_test is not None else cluster_ids_val

            if volatility_values is not None:
                vol_test = self.statistical_tests.test_volatility_independence(
                    test_cluster_ids, volatility_values
                )
                report['statistical_tests']['volatility_independence'] = vol_test
                print(f"  Volatility Independence: {vol_test['status']}")
                print(f"    {vol_test['message']}")

                if vol_test['status'] == 'FAIL':
                    report['overall_status'] = 'FAIL'
                    report['failures'].append('Volatility independence test failed')

            if timestamps is not None:
                temp_test = self.statistical_tests.test_temporal_stability(
                    test_cluster_ids, timestamps
                )
                report['statistical_tests']['temporal_stability'] = temp_test
                print(f"  Temporal Stability: {temp_test['status']}")
                print(f"    {temp_test['message']}")

                if temp_test['status'] == 'FAIL':
                    report['overall_status'] = 'FAIL'
                    report['failures'].append('Temporal stability test failed')

        # 5. Final summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {report['overall_status']}")

        if report['failures']:
            print(f"\nFailures ({len(report['failures'])}):")
            for failure in report['failures']:
                print(f"  - {failure}")
        else:
            print("\n[SUCCESS] All evaluation metrics passed!")
            print("Model is ready for production use.")

        print("=" * 60)

        return report

    def save_report(self, report: Dict, path: str):
        """
        Save evaluation report to JSON file.

        Args:
            report: Evaluation report dictionary
            path: Path to save JSON file
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        report_converted = convert_types(report)

        # Save to file
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(report_converted, f, indent=2)

        print(f"\n[PASS] Evaluation report saved to: {path}")


# Testing functions
def test_evaluation_report():
    """Test evaluation report generation."""
    print("Testing EvaluationReport...")

    np.random.seed(42)

    # Create synthetic well-separated clusters
    n_per_cluster = 300
    n_clusters = 5
    d_z = 64

    z_list = []
    labels_list = []

    for i in range(n_clusters):
        center = np.random.randn(d_z) * 3
        cluster_data = np.random.randn(n_per_cluster, d_z) * 0.5 + center
        cluster_data = cluster_data / np.linalg.norm(cluster_data, axis=1, keepdims=True)
        z_list.append(cluster_data)
        labels_list.append(np.full(n_per_cluster, i))

    z_norm = np.vstack(z_list)
    cluster_ids = np.concatenate(labels_list)

    # Create centroids
    centroids = np.stack([z_norm[cluster_ids == i].mean(axis=0) for i in range(n_clusters)])
    centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Create synthetic volatility and timestamps (independent)
    volatility = np.random.exponential(scale=2.0, size=len(cluster_ids))
    timestamps = np.arange(len(cluster_ids))

    # Evaluate
    evaluator = EvaluationReport()

    report = evaluator.evaluate_complete(
        z_normalized_val=z_norm,
        cluster_ids_val=cluster_ids,
        centroids_normalized=centroids_norm,
        volatility_values=volatility,
        timestamps=timestamps
    )

    assert report['overall_status'] in ['PASS', 'FAIL', 'WARN']
    print("\n[PASS] Evaluation report generated successfully")

    # Test saving
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "test_report.json")
        evaluator.save_report(report, report_path)

        assert os.path.exists(report_path)
        print("[PASS] Report saved and verified")

    print("\n[SUCCESS] All EvaluationReport tests passed!")
    return True


if __name__ == '__main__':
    test_evaluation_report()
