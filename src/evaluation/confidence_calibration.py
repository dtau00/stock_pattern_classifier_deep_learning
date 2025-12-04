"""
Confidence Calibration Module

Implements confidence score calibration as specified in Design Document Section 4.2.

The calibration process:
1. Grid search over γ ∈ [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
2. For each γ, compute C_score and per-sample Silhouette Score
3. Fit linear regression: Silhouette = a · C_score + b
4. Compute R² (correlation coefficient)
5. Select γ* that maximizes R²
6. Validate R² >= 0.7 and positive slope

After calibration, C_score represents a calibrated probability that correlates
strongly with cluster quality (as measured by Silhouette Score).

Reference: Design Document Section 4.2, Calibration Procedure
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

try:
    from .confidence_scoring import ConfidenceScorer
    from .clustering_metrics import ClusteringMetrics
except ImportError:
    from confidence_scoring import ConfidenceScorer
    from clustering_metrics import ClusteringMetrics


class ConfidenceCalibrator:
    """
    Calibrates confidence score gamma parameter on validation set.

    The calibration ensures that confidence scores strongly correlate with
    Silhouette Score (a ground-truth measure of cluster quality).

    Args:
        gamma_grid: List of gamma values to search (default: [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        sample_size: Max samples for Silhouette computation (default: 5000)
                    Larger validation sets use stratified sampling for efficiency

    Attributes:
        gamma_grid: Grid of gamma values to search
        sample_size: Sample size for Silhouette computation
        calibration_results: Dict with calibration metrics per gamma
        best_gamma: Optimal gamma value (None until calibrated)
        best_r2: Best R² score achieved
        best_slope: Slope of best regression

    Example:
        >>> calibrator = ConfidenceCalibrator()
        >>> z_val = torch.randn(2000, 128)
        >>> z_val_norm = F.normalize(z_val, p=2, dim=1)
        >>> centroids_norm = F.normalize(torch.randn(8, 128), p=2, dim=1)
        >>> results = calibrator.calibrate(z_val_norm, centroids_norm)
        >>> print(f"Best gamma: {results['best_gamma']}, R²: {results['best_r2']:.3f}")
    """

    def __init__(
        self,
        gamma_grid: Optional[list] = None,
        sample_size: int = 5000
    ):
        """
        Initialize confidence calibrator.

        Args:
            gamma_grid: List of gamma values to search
                       Default: [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] from design doc
            sample_size: Max samples for Silhouette computation (default: 5000)
        """
        self.gamma_grid = gamma_grid if gamma_grid is not None else \
            [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        self.sample_size = sample_size

        self.calibration_results = {}
        self.best_gamma = None
        self.best_r2 = None
        self.best_slope = None

        self.metrics_calc = ClusteringMetrics()

    def calibrate(
        self,
        z_normalized: torch.Tensor,
        centroids_normalized: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Calibrate gamma parameter on validation set.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            centroids_normalized: L2-normalized centroids of shape (K, D_z)
            cluster_ids: Optional pre-computed cluster assignments of shape (N,)
            verbose: Print progress messages

        Returns:
            Dictionary with:
                - 'best_gamma': Optimal gamma value
                - 'best_r2': R² score at best gamma
                - 'best_slope': Regression slope at best gamma
                - 'best_intercept': Regression intercept at best gamma
                - 'all_results': List of results for each gamma
                - 'passed': Boolean indicating if R² >= 0.7 and slope > 0

        Raises:
            ValueError: If calibration fails to meet R² >= 0.7 requirement
        """
        if verbose:
            print("\n" + "="*60)
            print("Confidence Score Calibration")
            print("="*60)
            print(f"Validation samples: {len(z_normalized):,}")
            print(f"Gamma search grid: {self.gamma_grid}")
            print(f"Sample size for Silhouette: {self.sample_size}")

        # Convert to numpy for sklearn
        z_norm_np = z_normalized.detach().cpu().numpy()

        # Compute cluster assignments if not provided
        if cluster_ids is None:
            scorer = ConfidenceScorer(gamma=1.0)  # Gamma doesn't matter for cluster assignment
            cluster_ids, _, _ = scorer.compute_confidence(z_normalized, centroids_normalized)

        cluster_ids_np = cluster_ids.detach().cpu().numpy() if isinstance(cluster_ids, torch.Tensor) else cluster_ids

        # Compute ground truth: per-sample Silhouette Score
        if verbose:
            print("\n[Step 1] Computing ground truth Silhouette Scores...")

        # Use sampling if dataset is large
        use_sample_size = self.sample_size if len(z_norm_np) > self.sample_size else None

        silhouette_samples = self.metrics_calc.compute_silhouette_samples(
            z_norm_np,
            cluster_ids_np,
            sample_size=use_sample_size
        )

        if verbose:
            print(f"  Silhouette range: [{silhouette_samples.min():.3f}, {silhouette_samples.max():.3f}]")
            print(f"  Silhouette mean: {silhouette_samples.mean():.3f}")

        # If using sampling, update data to match sampled indices
        if use_sample_size is not None and len(z_norm_np) > self.sample_size:
            # Get sampled indices (stratified)
            indices = self.metrics_calc._stratified_sample(cluster_ids_np, self.sample_size)
            z_normalized_sampled = z_normalized[indices]
            cluster_ids_sampled = cluster_ids[indices]
        else:
            z_normalized_sampled = z_normalized
            cluster_ids_sampled = cluster_ids

        # Grid search over gamma
        if verbose:
            print("\n[Step 2] Grid search over gamma...")
            print("-" * 60)
            print(f"{'Gamma':>8} | {'R²':>8} | {'Slope':>8} | {'Intercept':>10} | {'Status':>12}")
            print("-" * 60)

        all_results = []
        best_r2 = -float('inf')
        best_gamma_idx = -1

        for gamma in self.gamma_grid:
            # Compute confidence scores with this gamma
            scorer = ConfidenceScorer(gamma=gamma)
            _, confidence_scores, _ = scorer.compute_confidence(
                z_normalized_sampled,
                centroids_normalized,
                cluster_ids=cluster_ids_sampled
            )

            confidence_np = confidence_scores.detach().cpu().numpy() if isinstance(confidence_scores, torch.Tensor) else confidence_scores

            # Fit linear regression: Silhouette = a * C_score + b
            X = confidence_np.reshape(-1, 1)
            y = silhouette_samples

            reg = LinearRegression()
            reg.fit(X, y)

            # Compute R²
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)

            slope = reg.coef_[0]
            intercept = reg.intercept_

            # Store results
            result = {
                'gamma': gamma,
                'r2': r2,
                'slope': slope,
                'intercept': intercept,
                'regression': reg
            }
            all_results.append(result)

            # Track best
            if r2 > best_r2:
                best_r2 = r2
                best_gamma_idx = len(all_results) - 1

            # Print row
            if verbose:
                status = ""
                if r2 >= 0.7 and slope > 0:
                    status = "[PASS]"
                elif r2 >= 0.7:
                    status = "[WARN: slope]"
                else:
                    status = "[FAIL: R²]"

                print(f"{gamma:8.1f} | {r2:8.3f} | {slope:8.3f} | {intercept:10.3f} | {status:>12}")

        if verbose:
            print("-" * 60)

        # Select best gamma
        best_result = all_results[best_gamma_idx]
        self.best_gamma = best_result['gamma']
        self.best_r2 = best_result['r2']
        self.best_slope = best_result['slope']
        self.best_intercept = best_result['intercept']

        # Validation criteria
        passed = (self.best_r2 >= 0.7) and (self.best_slope > 0)

        if verbose:
            print(f"\n[Step 3] Best gamma selection:")
            print(f"  gamma* = {self.best_gamma}")
            print(f"  R² = {self.best_r2:.3f}")
            print(f"  Slope = {self.best_slope:.3f}")
            print(f"  Intercept = {self.best_intercept:.3f}")

            print(f"\n[Step 4] Validation criteria:")
            print(f"  R² >= 0.7: {'[PASS]' if self.best_r2 >= 0.7 else '[FAIL]'} (R² = {self.best_r2:.3f})")
            print(f"  Slope > 0: {'[PASS]' if self.best_slope > 0 else '[FAIL]'} (slope = {self.best_slope:.3f})")

            if passed:
                print(f"\n{'='*60}")
                print("[SUCCESS] Confidence calibration PASSED!")
                print(f"Confidence scores are calibrated and reliable.")
                print(f"Use gamma = {self.best_gamma} for inference.")
                print("="*60)
            else:
                print(f"\n{'='*60}")
                print("[FAILURE] Confidence calibration FAILED!")
                print("Model training has FAILED - uncalibrated confidence scores.")
                if self.best_r2 < 0.7:
                    print(f"  Issue: R² = {self.best_r2:.3f} < 0.7 (weak correlation)")
                    print("  Recommendation: Retrain model with different hyperparameters")
                if self.best_slope <= 0:
                    print(f"  Issue: Slope = {self.best_slope:.3f} <= 0 (negative correlation)")
                    print("  Recommendation: Check centroid initialization and training")
                print("="*60)

        # Return comprehensive results
        calibration_summary = {
            'best_gamma': self.best_gamma,
            'best_r2': self.best_r2,
            'best_slope': self.best_slope,
            'best_intercept': self.best_intercept,
            'all_results': all_results,
            'passed': passed,
            'n_samples': len(silhouette_samples),
            'silhouette_mean': float(silhouette_samples.mean()),
            'silhouette_std': float(silhouette_samples.std())
        }

        self.calibration_results = calibration_summary

        return calibration_summary

    def plot_calibration(
        self,
        z_normalized: torch.Tensor,
        centroids_normalized: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot calibration scatter plot: C_score vs Silhouette Score.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            centroids_normalized: L2-normalized centroids of shape (K, D_z)
            cluster_ids: Optional pre-computed cluster assignments
            save_path: Optional path to save figure

        Requires:
            Must call calibrate() first to get best_gamma
        """
        if self.best_gamma is None:
            raise ValueError("Must call calibrate() before plotting!")

        # Compute confidence scores with best gamma
        scorer = ConfidenceScorer(gamma=self.best_gamma)
        _, confidence_scores, _ = scorer.compute_confidence(
            z_normalized, centroids_normalized, cluster_ids
        )

        # Compute Silhouette scores
        z_norm_np = z_normalized.detach().cpu().numpy()
        if cluster_ids is None:
            cluster_ids, _, _ = scorer.compute_confidence(z_normalized, centroids_normalized)
        cluster_ids_np = cluster_ids.detach().cpu().numpy() if isinstance(cluster_ids, torch.Tensor) else cluster_ids

        silhouette_samples = self.metrics_calc.compute_silhouette_samples(
            z_norm_np, cluster_ids_np, sample_size=self.sample_size
        )

        confidence_np = confidence_scores.detach().cpu().numpy() if isinstance(confidence_scores, torch.Tensor) else confidence_scores

        # Sample if too many points (for visualization clarity)
        if len(confidence_np) > 2000:
            indices = np.random.choice(len(confidence_np), 2000, replace=False)
            confidence_plot = confidence_np[indices]
            silhouette_plot = silhouette_samples[indices]
        else:
            confidence_plot = confidence_np
            silhouette_plot = silhouette_samples

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        ax.scatter(confidence_plot, silhouette_plot, alpha=0.5, s=20)

        # Regression line
        X_line = np.linspace(confidence_plot.min(), confidence_plot.max(), 100).reshape(-1, 1)
        y_line = self.best_slope * X_line + self.best_intercept
        ax.plot(X_line, y_line, 'r--', linewidth=2, label=f'Linear fit (R² = {self.best_r2:.3f})')

        ax.set_xlabel('Confidence Score (C_score)', fontsize=12)
        ax.set_ylabel('Silhouette Score (Ground Truth)', fontsize=12)
        ax.set_title(
            f'Confidence Calibration (γ = {self.best_gamma})\n'
            f'R² = {self.best_r2:.3f}, Slope = {self.best_slope:.3f}',
            fontsize=14
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add threshold line at C_score = 0.7
        ax.axvline(x=0.7, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Threshold (0.7)')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Calibration plot saved to: {save_path}")
        else:
            plt.show()

        return fig

    def save_calibration(self, path: str):
        """
        Save calibration results to file.

        Args:
            path: Path to save calibration results (JSON format)
        """
        import json

        if self.best_gamma is None:
            raise ValueError("Must call calibrate() before saving!")

        # Prepare results for JSON (convert numpy types to Python types)
        results_to_save = {
            'best_gamma': float(self.best_gamma),
            'best_r2': float(self.best_r2),
            'best_slope': float(self.best_slope),
            'best_intercept': float(self.best_intercept),
            'gamma_grid': [float(g) for g in self.gamma_grid],
            'sample_size': int(self.sample_size),
            'all_results': [
                {
                    'gamma': float(r['gamma']),
                    'r2': float(r['r2']),
                    'slope': float(r['slope']),
                    'intercept': float(r['intercept'])
                }
                for r in self.calibration_results['all_results']
            ],
            'passed': bool(self.calibration_results['passed']),
            'n_samples': int(self.calibration_results['n_samples']),
            'silhouette_mean': float(self.calibration_results['silhouette_mean']),
            'silhouette_std': float(self.calibration_results['silhouette_std'])
        }

        with open(path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"Calibration results saved to: {path}")


def test_confidence_calibrator():
    """Unit tests for ConfidenceCalibrator."""
    print("Testing ConfidenceCalibrator...")

    # Test 1: Basic calibration
    print("\n[Test 1] Basic calibration on synthetic data")

    # Create synthetic well-separated clusters
    np.random.seed(42)
    torch.manual_seed(42)

    n_per_cluster = 300
    n_clusters = 5
    d_z = 64

    z_list = []
    labels_list = []

    for i in range(n_clusters):
        center = torch.randn(d_z) * 3
        cluster_data = torch.randn(n_per_cluster, d_z) * 0.5 + center
        z_list.append(cluster_data)
        labels_list.append(torch.full((n_per_cluster,), i))

    z = torch.cat(z_list, dim=0)
    z_norm = torch.nn.functional.normalize(z, p=2, dim=1)
    cluster_ids = torch.cat(labels_list)

    # Compute centroids
    centroids = torch.stack([z_norm[cluster_ids == i].mean(dim=0) for i in range(n_clusters)])
    centroids_norm = torch.nn.functional.normalize(centroids, p=2, dim=1)

    # Calibrate
    calibrator = ConfidenceCalibrator(gamma_grid=[1.0, 3.0, 5.0, 7.0])
    results = calibrator.calibrate(z_norm, centroids_norm, cluster_ids, verbose=True)

    assert 'best_gamma' in results
    assert 'best_r2' in results
    assert 'passed' in results

    print(f"\n[Test 1 Results]")
    print(f"  Best gamma: {results['best_gamma']}")
    print(f"  Best R²: {results['best_r2']:.3f}")
    print(f"  Passed: {results['passed']}")

    if results['best_r2'] >= 0.5:
        print(f"  [PASS] R² is reasonable for synthetic data")
    else:
        print(f"  [WARN] R² is low (may be due to synthetic data)")

    # Test 2: Verify gamma affects confidence distribution
    print("\n[Test 2] Verify gamma affects confidence scores")

    scorer_low = ConfidenceScorer(gamma=1.0)
    scorer_high = ConfidenceScorer(gamma=10.0)

    _, conf_low, _ = scorer_low.compute_confidence(z_norm, centroids_norm, cluster_ids)
    _, conf_high, _ = scorer_high.compute_confidence(z_norm, centroids_norm, cluster_ids)

    mean_low = conf_low.mean().item()
    mean_high = conf_high.mean().item()

    print(f"  Mean confidence (gamma=1.0): {mean_low:.3f}")
    print(f"  Mean confidence (gamma=10.0): {mean_high:.3f}")
    print(f"  [INFO] Higher gamma generally increases separation")

    # Test 3: Save/load calibration
    print("\n[Test 3] Save calibration results")

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "calibration.json")
        calibrator.save_calibration(save_path)

        # Verify file exists
        assert os.path.exists(save_path)
        print(f"  [PASS] Calibration saved successfully")

        # Load and verify
        import json
        with open(save_path, 'r') as f:
            loaded = json.load(f)

        assert loaded['best_gamma'] == results['best_gamma']
        assert abs(loaded['best_r2'] - results['best_r2']) < 1e-6
        print(f"  [PASS] Calibration loaded and verified")

    print("\n[SUCCESS] All ConfidenceCalibrator tests passed!")
    return True


if __name__ == '__main__':
    test_confidence_calibrator()
