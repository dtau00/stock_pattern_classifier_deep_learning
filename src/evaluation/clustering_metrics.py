"""
Clustering Quality Metrics Module

Implements comprehensive clustering evaluation metrics as specified in Design Document Section 5.

Metrics implemented:
1. Adjusted Rand Index (ARI) - Stability between runs
2. Silhouette Score - Per-sample cluster quality
3. Davies-Bouldin Index - Cluster separation
4. Calinski-Harabasz Score - Cluster compactness vs separation

All metrics operate on L2-normalized latent vectors and cluster assignments.

Reference: Design Document Section 5, Table of Metrics
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.optimize import linear_sum_assignment


class ClusteringMetrics:
    """
    Comprehensive clustering quality metrics calculator.

    This class provides all clustering metrics required by the design document:
    - ARI: Adjusted Rand Index for stability testing
    - Silhouette Score: Per-sample and global cluster quality
    - Davies-Bouldin Index: Cluster separation quality
    - Calinski-Harabasz Score: Cluster compactness

    All methods accept both PyTorch tensors and NumPy arrays.

    Example:
        >>> metrics = ClusteringMetrics()
        >>> z_norm = torch.randn(1000, 128)
        >>> cluster_ids = torch.randint(0, 8, (1000,))
        >>> results = metrics.compute_all_metrics(z_norm, cluster_ids)
        >>> print(f"Silhouette: {results['silhouette_score']:.3f}")
    """

    @staticmethod
    def _to_numpy(x):
        """Convert PyTorch tensor or NumPy array to NumPy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def compute_silhouette_score(
        self,
        z_normalized: np.ndarray,
        cluster_ids: np.ndarray,
        sample_size: Optional[int] = None
    ) -> float:
        """
        Compute Silhouette Score (global average).

        Measures how well-separated clusters are. Range: [-1, 1], higher is better.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            cluster_ids: Cluster assignments of shape (N,)
            sample_size: If provided and N > sample_size, use stratified sampling
                        (for computational efficiency with large datasets)

        Returns:
            Silhouette score (scalar in [-1, 1])

        Thresholds (Design Doc Section 5):
            - >= 0.4: Good cluster separation
            - < 0.4: Poor separation, model may need improvement

        Note:
            Silhouette Score is O(N²) complexity. For large validation sets (> 10,000),
            use sample_size=5000 with stratified sampling as recommended in design doc.
        """
        z_normalized = self._to_numpy(z_normalized)
        cluster_ids = self._to_numpy(cluster_ids)

        # Check for sufficient clusters
        n_clusters = len(np.unique(cluster_ids))
        if n_clusters < 2:
            raise ValueError(
                f"Silhouette Score requires at least 2 clusters, got {n_clusters}. "
                f"Cannot compute cluster separation with only 1 cluster."
            )

        # Stratified sampling for large datasets
        if sample_size is not None and len(z_normalized) > sample_size:
            indices = self._stratified_sample(cluster_ids, sample_size)
            z_sampled = z_normalized[indices]
            labels_sampled = cluster_ids[indices]
        else:
            z_sampled = z_normalized
            labels_sampled = cluster_ids

        # Compute silhouette score
        score = silhouette_score(z_sampled, labels_sampled, metric='euclidean')

        return float(score)

    def compute_silhouette_samples(
        self,
        z_normalized: np.ndarray,
        cluster_ids: np.ndarray,
        sample_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute per-sample Silhouette Scores.

        Used for confidence score calibration (Design Doc Section 4.2).

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            cluster_ids: Cluster assignments of shape (N,)
            sample_size: If provided, use stratified sampling for efficiency

        Returns:
            Per-sample Silhouette scores of shape (N,) or (sample_size,)
            Range: [-1, 1] per sample

        Note:
            This is used as ground truth for calibrating confidence scores.
            High Silhouette score → sample is well-clustered.
        """
        z_normalized = self._to_numpy(z_normalized)
        cluster_ids = self._to_numpy(cluster_ids)

        # Check for sufficient clusters
        n_clusters = len(np.unique(cluster_ids))
        if n_clusters < 2:
            raise ValueError(f"Silhouette Score requires at least 2 clusters, got {n_clusters}")

        # Stratified sampling for large datasets
        if sample_size is not None and len(z_normalized) > sample_size:
            indices = self._stratified_sample(cluster_ids, sample_size)
            z_sampled = z_normalized[indices]
            labels_sampled = cluster_ids[indices]

            # Compute on sample
            scores_sampled = silhouette_samples(z_sampled, labels_sampled, metric='euclidean')
            return scores_sampled
        else:
            # Compute on full dataset
            scores = silhouette_samples(z_normalized, cluster_ids, metric='euclidean')
            return scores

    def compute_davies_bouldin_index(
        self,
        z_normalized: np.ndarray,
        cluster_ids: np.ndarray
    ) -> float:
        """
        Compute Davies-Bouldin Index.

        Measures average similarity between clusters. Lower is better.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            cluster_ids: Cluster assignments of shape (N,)

        Returns:
            Davies-Bouldin Index (scalar, lower is better)

        Thresholds (Design Doc Section 5):
            - <= 1.5: Good cluster compactness vs separation
            - > 1.5: Poor separation, clusters overlap too much

        Formula:
            DB = (1/K) * sum_i max_{j != i} (s_i + s_j) / d_{ij}
            where s_i = avg distance within cluster i
                  d_ij = distance between centroids i and j
        """
        z_normalized = self._to_numpy(z_normalized)
        cluster_ids = self._to_numpy(cluster_ids)

        # Check for sufficient clusters
        n_clusters = len(np.unique(cluster_ids))
        if n_clusters < 2:
            raise ValueError(f"Davies-Bouldin Index requires at least 2 clusters, got {n_clusters}")

        score = davies_bouldin_score(z_normalized, cluster_ids)
        return float(score)

    def compute_calinski_harabasz_score(
        self,
        z_normalized: np.ndarray,
        cluster_ids: np.ndarray
    ) -> float:
        """
        Compute Calinski-Harabasz Score (Variance Ratio Criterion).

        Ratio of between-cluster to within-cluster variance. Higher is better.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            cluster_ids: Cluster assignments of shape (N,)

        Returns:
            Calinski-Harabasz Score (scalar, higher is better)

        Thresholds (Design Doc Section 5):
            - >= 100: Good cluster definition
            - < 100: Poor separation, clusters not well-defined

        Formula:
            CH = [Tr(B_k) / (K-1)] / [Tr(W_k) / (N-K)]
            where B_k = between-cluster dispersion matrix
                  W_k = within-cluster dispersion matrix
                  K = number of clusters, N = number of samples
        """
        z_normalized = self._to_numpy(z_normalized)
        cluster_ids = self._to_numpy(cluster_ids)

        # Check for sufficient clusters
        n_clusters = len(np.unique(cluster_ids))
        if n_clusters < 2:
            raise ValueError(f"Calinski-Harabasz Score requires at least 2 clusters, got {n_clusters}")

        score = calinski_harabasz_score(z_normalized, cluster_ids)
        return float(score)

    def compute_adjusted_rand_index(
        self,
        cluster_ids_1: np.ndarray,
        cluster_ids_2: np.ndarray,
        use_hungarian: bool = True
    ) -> float:
        """
        Compute Adjusted Rand Index (ARI) for stability testing.

        Measures agreement between two clusterings from different training runs.
        Corrected for chance (random clustering gives ARI ≈ 0).

        Args:
            cluster_ids_1: Cluster assignments from run 1 of shape (N,)
            cluster_ids_2: Cluster assignments from run 2 of shape (N,)
            use_hungarian: If True, apply Hungarian algorithm to remap labels
                          before computing ARI (recommended for stability testing)

        Returns:
            ARI score in [-1, 1], where:
                - 1.0: Perfect agreement
                - 0.0: Random agreement
                - < 0: Worse than random

        Thresholds (Design Doc Section 5):
            - >= 0.85: Model is stable (CRITICAL for production use)
            - < 0.85: Model is unstable, needs investigation

        Note:
            Hungarian algorithm remapping is CRITICAL for stability testing
            because cluster IDs from different runs may be permuted.
        """
        cluster_ids_1 = self._to_numpy(cluster_ids_1)
        cluster_ids_2 = self._to_numpy(cluster_ids_2)

        if len(cluster_ids_1) != len(cluster_ids_2):
            raise ValueError(
                f"Cluster ID arrays must have same length: "
                f"{len(cluster_ids_1)} vs {len(cluster_ids_2)}"
            )

        # Apply Hungarian algorithm to find optimal label mapping
        if use_hungarian:
            cluster_ids_2_remapped = self._remap_labels_hungarian(
                cluster_ids_1, cluster_ids_2
            )
        else:
            cluster_ids_2_remapped = cluster_ids_2

        # Compute ARI
        ari = adjusted_rand_score(cluster_ids_1, cluster_ids_2_remapped)

        return float(ari)

    def _remap_labels_hungarian(
        self,
        cluster_ids_1: np.ndarray,
        cluster_ids_2: np.ndarray
    ) -> np.ndarray:
        """
        Remap cluster labels from run 2 to match run 1 using Hungarian algorithm.

        This is critical for ARI computation because cluster IDs are arbitrary
        and may be permuted between training runs.

        Args:
            cluster_ids_1: Reference cluster assignments of shape (N,)
            cluster_ids_2: Cluster assignments to remap of shape (N,)

        Returns:
            Remapped cluster_ids_2 that maximizes overlap with cluster_ids_1
        """
        # Get unique labels
        labels_1 = np.unique(cluster_ids_1)
        labels_2 = np.unique(cluster_ids_2)

        # Build confusion matrix
        # C[i, j] = number of samples with label i in run 1 and label j in run 2
        n_labels_1 = len(labels_1)
        n_labels_2 = len(labels_2)

        confusion = np.zeros((n_labels_1, n_labels_2), dtype=np.int64)
        for i, label_1 in enumerate(labels_1):
            for j, label_2 in enumerate(labels_2):
                mask_1 = (cluster_ids_1 == label_1)
                mask_2 = (cluster_ids_2 == label_2)
                confusion[i, j] = (mask_1 & mask_2).sum()

        # Hungarian algorithm: maximize overlap (minimize cost = -overlap)
        row_ind, col_ind = linear_sum_assignment(-confusion)

        # Build mapping: label_2 -> label_1
        label_mapping = {}
        for i, j in zip(row_ind, col_ind):
            label_mapping[labels_2[j]] = labels_1[i]

        # Handle unmapped labels (if K2 > K1)
        for label_2 in labels_2:
            if label_2 not in label_mapping:
                # Assign to a new label not in labels_1
                label_mapping[label_2] = max(labels_1) + 1

        # Remap cluster_ids_2
        cluster_ids_2_remapped = np.array([
            label_mapping[label] for label in cluster_ids_2
        ])

        return cluster_ids_2_remapped

    def _stratified_sample(
        self,
        cluster_ids: np.ndarray,
        sample_size: int
    ) -> np.ndarray:
        """
        Stratified random sampling proportional to cluster sizes.

        Ensures all clusters are represented in the sample.

        Args:
            cluster_ids: Cluster assignments of shape (N,)
            sample_size: Target sample size

        Returns:
            Indices of sampled points of shape (sample_size,)
        """
        unique_labels, counts = np.unique(cluster_ids, return_counts=True)
        n_total = len(cluster_ids)

        # Compute samples per cluster (proportional to size)
        samples_per_cluster = (counts / n_total * sample_size).astype(int)

        # Adjust to ensure total equals sample_size
        diff = sample_size - samples_per_cluster.sum()
        if diff > 0:
            # Add remaining samples to largest clusters
            largest_clusters = np.argsort(counts)[-diff:]
            samples_per_cluster[largest_clusters] += 1

        # Sample from each cluster
        indices = []
        for label, n_samples in zip(unique_labels, samples_per_cluster):
            cluster_indices = np.where(cluster_ids == label)[0]
            sampled = np.random.choice(cluster_indices, size=n_samples, replace=False)
            indices.extend(sampled)

        return np.array(indices)

    def compute_all_metrics(
        self,
        z_normalized,
        cluster_ids,
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute all clustering metrics at once.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            cluster_ids: Cluster assignments of shape (N,)
            sample_size: Optional sample size for Silhouette computation

        Returns:
            Dictionary with:
                - 'silhouette_score': Global Silhouette Score
                - 'davies_bouldin_index': Davies-Bouldin Index
                - 'calinski_harabasz_score': Calinski-Harabasz Score
                - 'n_samples': Number of samples
                - 'n_clusters': Number of unique clusters

        Note:
            ARI is not included (requires two clusterings).
            Use compute_adjusted_rand_index() separately for stability testing.
        """
        z_normalized = self._to_numpy(z_normalized)
        cluster_ids = self._to_numpy(cluster_ids)

        metrics = {
            'silhouette_score': self.compute_silhouette_score(
                z_normalized, cluster_ids, sample_size
            ),
            'davies_bouldin_index': self.compute_davies_bouldin_index(
                z_normalized, cluster_ids
            ),
            'calinski_harabasz_score': self.compute_calinski_harabasz_score(
                z_normalized, cluster_ids
            ),
            'n_samples': len(cluster_ids),
            'n_clusters': len(np.unique(cluster_ids))
        }

        return metrics

    def check_metric_thresholds(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Check if metrics meet design document thresholds.

        Args:
            metrics: Dictionary from compute_all_metrics()

        Returns:
            Dictionary with metric name -> (passed, message) tuples

        Thresholds (Design Doc Section 5):
            - Silhouette Score >= 0.4
            - Davies-Bouldin Index <= 1.5
            - Calinski-Harabasz Score >= 100
        """
        results = {}

        # Silhouette Score
        silhouette = metrics.get('silhouette_score', 0)
        if silhouette >= 0.4:
            results['silhouette_score'] = (
                True,
                f"[PASS] Silhouette = {silhouette:.3f} >= 0.4 (good separation)"
            )
        else:
            results['silhouette_score'] = (
                False,
                f"[FAIL] Silhouette = {silhouette:.3f} < 0.4 (poor separation)"
            )

        # Davies-Bouldin Index
        davies_bouldin = metrics.get('davies_bouldin_index', float('inf'))
        if davies_bouldin <= 1.5:
            results['davies_bouldin_index'] = (
                True,
                f"[PASS] Davies-Bouldin = {davies_bouldin:.3f} <= 1.5 (good compactness)"
            )
        else:
            results['davies_bouldin_index'] = (
                False,
                f"[FAIL] Davies-Bouldin = {davies_bouldin:.3f} > 1.5 (poor compactness)"
            )

        # Calinski-Harabasz Score
        calinski = metrics.get('calinski_harabasz_score', 0)
        if calinski >= 100:
            results['calinski_harabasz_score'] = (
                True,
                f"[PASS] Calinski-Harabasz = {calinski:.1f} >= 100 (well-defined clusters)"
            )
        else:
            results['calinski_harabasz_score'] = (
                False,
                f"[FAIL] Calinski-Harabasz = {calinski:.1f} < 100 (poorly defined clusters)"
            )

        return results


def test_clustering_metrics():
    """Unit tests for ClusteringMetrics."""
    print("Testing ClusteringMetrics...")

    metrics_calc = ClusteringMetrics()

    # Test 1: Synthetic well-separated clusters
    print("\n[Test 1] Well-separated synthetic clusters")
    np.random.seed(42)

    # Create 4 well-separated Gaussian clusters
    n_per_cluster = 100
    z_list = []
    labels_list = []

    for i in range(4):
        # Create cluster centered at different locations
        center = np.random.randn(128) * 5
        cluster_data = np.random.randn(n_per_cluster, 128) * 0.3 + center

        # Normalize
        cluster_data = cluster_data / np.linalg.norm(cluster_data, axis=1, keepdims=True)

        z_list.append(cluster_data)
        labels_list.append(np.full(n_per_cluster, i))

    z_normalized = np.vstack(z_list)
    cluster_ids = np.concatenate(labels_list)

    # Compute all metrics
    results = metrics_calc.compute_all_metrics(z_normalized, cluster_ids)

    print(f"  Silhouette Score: {results['silhouette_score']:.3f}")
    print(f"  Davies-Bouldin Index: {results['davies_bouldin_index']:.3f}")
    print(f"  Calinski-Harabasz Score: {results['calinski_harabasz_score']:.1f}")

    assert results['silhouette_score'] > 0.3, "Expected high Silhouette for well-separated clusters"
    print(f"  [PASS] Metrics computed successfully")

    # Test 2: Per-sample Silhouette
    print("\n[Test 2] Per-sample Silhouette scores")
    silhouette_samples = metrics_calc.compute_silhouette_samples(z_normalized, cluster_ids)

    assert len(silhouette_samples) == len(cluster_ids)
    print(f"  Computed {len(silhouette_samples)} per-sample scores")
    print(f"  Range: [{silhouette_samples.min():.3f}, {silhouette_samples.max():.3f}]")
    print(f"  Mean: {silhouette_samples.mean():.3f}")
    print(f"  [PASS] Per-sample scores computed")

    # Test 3: ARI with Hungarian remapping
    print("\n[Test 3] ARI with Hungarian algorithm")

    # Create two identical clusterings
    labels_1 = cluster_ids.copy()
    labels_2 = cluster_ids.copy()

    ari_identical = metrics_calc.compute_adjusted_rand_index(labels_1, labels_2)
    print(f"  ARI (identical): {ari_identical:.3f}")
    assert ari_identical == 1.0, "ARI should be 1.0 for identical clusterings"

    # Create permuted clustering (swap labels 0 <-> 1)
    labels_2_permuted = labels_2.copy()
    labels_2_permuted[labels_2 == 0] = 99  # Temp
    labels_2_permuted[labels_2 == 1] = 0
    labels_2_permuted[labels_2_permuted == 99] = 1

    ari_permuted_no_remap = metrics_calc.compute_adjusted_rand_index(
        labels_1, labels_2_permuted, use_hungarian=False
    )
    ari_permuted_with_remap = metrics_calc.compute_adjusted_rand_index(
        labels_1, labels_2_permuted, use_hungarian=True
    )

    print(f"  ARI (permuted, no remap): {ari_permuted_no_remap:.3f}")
    print(f"  ARI (permuted, with Hungarian): {ari_permuted_with_remap:.3f}")
    assert ari_permuted_with_remap == 1.0, "Hungarian should recover perfect match"
    print(f"  [PASS] Hungarian remapping works correctly")

    # Test 4: Stratified sampling
    print("\n[Test 4] Stratified sampling")
    sample_size = 200
    silhouette_full = metrics_calc.compute_silhouette_score(z_normalized, cluster_ids)
    silhouette_sampled = metrics_calc.compute_silhouette_score(
        z_normalized, cluster_ids, sample_size=sample_size
    )

    print(f"  Full dataset Silhouette: {silhouette_full:.3f}")
    print(f"  Sampled ({sample_size}) Silhouette: {silhouette_sampled:.3f}")
    print(f"  Difference: {abs(silhouette_full - silhouette_sampled):.3f}")
    print(f"  [INFO] Sampling provides approximation for efficiency")

    # Test 5: Threshold checking
    print("\n[Test 5] Threshold checking")
    threshold_results = metrics_calc.check_metric_thresholds(results)

    for metric_name, (passed, message) in threshold_results.items():
        print(f"  {message}")

    # Test 6: PyTorch tensor support
    print("\n[Test 6] PyTorch tensor support")
    z_torch = torch.from_numpy(z_normalized).float()
    labels_torch = torch.from_numpy(cluster_ids).long()

    results_torch = metrics_calc.compute_all_metrics(z_torch, labels_torch)
    print(f"  [PASS] PyTorch tensors converted and processed")
    print(f"  Silhouette (PyTorch input): {results_torch['silhouette_score']:.3f}")

    print("\n[SUCCESS] All ClusteringMetrics tests passed!")
    return True


if __name__ == '__main__':
    test_clustering_metrics()
