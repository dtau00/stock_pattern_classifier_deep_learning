"""
Confidence Scoring Module

Implements confidence score calculation as specified in Design Document Section 4.2.

The confidence score quantifies the model's certainty based on:
- D_assigned: Distance from latent vector to assigned centroid
- D_sep: Distance to second-closest centroid (separation distance)
- C_score: Calibrated probability via sigmoid transformation

Formula:
    C_score = Sigmoid(γ · (D_sep - D_assigned))

Where γ is calibrated on validation set to maximize R² with Silhouette Score.

CRITICAL REQUIREMENTS:
- All inputs must be L2-normalized (latent vectors and centroids)
- Distances are bounded in [0, 2] on unit hypersphere
- Output C_score ∈ [0, 1] represents calibrated probability of correct assignment
- No NaN or Inf values allowed in outputs
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class ConfidenceScorer:
    """
    Computes confidence scores for cluster assignments.

    The confidence score measures the certainty of a cluster assignment based on
    the separation between the assigned cluster and other clusters.

    Args:
        gamma: Calibration parameter for sigmoid (default: 5.0)
               Should be tuned on validation set using ConfidenceCalibrator

    Attributes:
        gamma: Sigmoid calibration parameter

    Example:
        >>> scorer = ConfidenceScorer(gamma=5.0)
        >>> z_norm = F.normalize(torch.randn(100, 128), p=2, dim=1)
        >>> centroids_norm = F.normalize(torch.randn(8, 128), p=2, dim=1)
        >>> cluster_ids, confidence_scores = scorer.compute_confidence(z_norm, centroids_norm)
        >>> confidence_scores.shape
        torch.Size([100])
    """

    def __init__(self, gamma: float = 5.0):
        """
        Initialize confidence scorer.

        Args:
            gamma: Calibration parameter for sigmoid (default: 5.0)
                   Typical range: [0.5, 10.0]
                   Higher γ = more aggressive confidence separation
        """
        self.gamma = gamma

    def compute_assigned_distance(
        self,
        z_normalized: torch.Tensor,
        centroids_normalized: torch.Tensor,
        cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance from each sample to its assigned centroid.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            centroids_normalized: L2-normalized centroids of shape (K, D_z)
            cluster_ids: Cluster assignments of shape (N,)

        Returns:
            D_assigned: Distances to assigned centroids of shape (N,)

        Formula:
            D_assigned[i] = ||z_normalized[i] - centroids_normalized[cluster_ids[i]]||_2
        """
        # Verify inputs are normalized
        self._verify_normalized(z_normalized, "z_normalized")
        self._verify_normalized(centroids_normalized, "centroids_normalized")

        # Get assigned centroids for each sample
        assigned_centroids = centroids_normalized[cluster_ids]  # (N, D_z)

        # Compute Euclidean distances
        distances = torch.norm(z_normalized - assigned_centroids, p=2, dim=1)  # (N,)

        return distances

    def compute_separation_distance(
        self,
        z_normalized: torch.Tensor,
        centroids_normalized: torch.Tensor,
        cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance from each sample to its second-closest centroid.

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            centroids_normalized: L2-normalized centroids of shape (K, D_z)
            cluster_ids: Cluster assignments of shape (N,)

        Returns:
            D_sep: Distances to second-closest centroids of shape (N,)

        Formula:
            D_sep[i] = min_{k != cluster_ids[i]} ||z_normalized[i] - centroids_normalized[k]||_2
        """
        # Verify inputs are normalized
        self._verify_normalized(z_normalized, "z_normalized")
        self._verify_normalized(centroids_normalized, "centroids_normalized")

        # Compute distances to ALL centroids
        # Shape: (N, K)
        all_distances = torch.cdist(z_normalized, centroids_normalized, p=2)

        # For each sample, mask out the assigned centroid distance
        # Set assigned centroid distance to infinity so it's not selected as min
        mask = torch.zeros_like(all_distances, dtype=torch.bool)
        mask[torch.arange(len(cluster_ids)), cluster_ids] = True

        # Replace assigned distances with infinity
        all_distances_masked = all_distances.clone()
        all_distances_masked[mask] = float('inf')

        # Find minimum distance among remaining centroids (second-closest)
        separation_distances = all_distances_masked.min(dim=1)[0]  # (N,)

        return separation_distances

    def compute_confidence_score(
        self,
        D_assigned: torch.Tensor,
        D_sep: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute calibrated confidence score using sigmoid transformation.

        Args:
            D_assigned: Distances to assigned centroids of shape (N,)
            D_sep: Distances to second-closest centroids of shape (N,)

        Returns:
            C_score: Confidence scores in [0, 1] of shape (N,)

        Formula:
            C_score = Sigmoid(γ · (D_sep - D_assigned))

        Interpretation:
            - D_sep > D_assigned → High confidence (well-separated from other clusters)
            - D_sep ≈ D_assigned → Low confidence (ambiguous assignment)
            - D_sep < D_assigned → Very low confidence (closer to another cluster)
        """
        # Compute margin: positive = high confidence, negative = low confidence
        margin = D_sep - D_assigned  # (N,)

        # Apply sigmoid with calibration parameter
        confidence_scores = torch.sigmoid(self.gamma * margin)  # (N,)

        return confidence_scores

    def compute_confidence(
        self,
        z_normalized: torch.Tensor,
        centroids_normalized: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cluster assignments and confidence scores (full pipeline).

        Args:
            z_normalized: L2-normalized latent vectors of shape (N, D_z)
            centroids_normalized: L2-normalized centroids of shape (K, D_z)
            cluster_ids: Optional pre-computed cluster assignments of shape (N,)
                        If None, computed from nearest centroid

        Returns:
            Tuple of (cluster_ids, confidence_scores, metrics):
                - cluster_ids: Cluster assignments of shape (N,)
                - confidence_scores: Confidence scores in [0, 1] of shape (N,)
                - metrics: Dictionary with:
                    - 'D_assigned': Distances to assigned centroids
                    - 'D_sep': Distances to second-closest centroids
                    - 'margin': D_sep - D_assigned

        Example:
            >>> scorer = ConfidenceScorer(gamma=5.0)
            >>> z_norm = F.normalize(torch.randn(100, 128), p=2, dim=1)
            >>> centroids_norm = F.normalize(torch.randn(8, 128), p=2, dim=1)
            >>> cluster_ids, scores, metrics = scorer.compute_confidence(z_norm, centroids_norm)
            >>> print(f"Confidence range: [{scores.min():.3f}, {scores.max():.3f}]")
        """
        # Verify inputs are normalized
        self._verify_normalized(z_normalized, "z_normalized")
        self._verify_normalized(centroids_normalized, "centroids_normalized")

        # Compute cluster assignments if not provided
        if cluster_ids is None:
            # Find nearest centroid
            distances_all = torch.cdist(z_normalized, centroids_normalized, p=2)
            cluster_ids = distances_all.argmin(dim=1)  # (N,)

        # Compute assigned distances
        D_assigned = self.compute_assigned_distance(
            z_normalized, centroids_normalized, cluster_ids
        )

        # Compute separation distances
        D_sep = self.compute_separation_distance(
            z_normalized, centroids_normalized, cluster_ids
        )

        # Compute confidence scores
        confidence_scores = self.compute_confidence_score(D_assigned, D_sep)

        # Verify outputs are valid
        self._verify_confidence_scores(confidence_scores)

        # Collect metrics
        metrics = {
            'D_assigned': D_assigned,
            'D_sep': D_sep,
            'margin': D_sep - D_assigned
        }

        return cluster_ids, confidence_scores, metrics

    def _verify_normalized(self, tensor: torch.Tensor, name: str, atol: float = 1e-4):
        """
        Verify that tensor is L2-normalized (norm = 1).

        Args:
            tensor: Tensor to verify of shape (N, D)
            name: Name for error message
            atol: Absolute tolerance for norm check

        Raises:
            ValueError: If tensor is not L2-normalized
        """
        norms = torch.norm(tensor, p=2, dim=1)
        max_deviation = (norms - 1.0).abs().max().item()

        if max_deviation > atol:
            raise ValueError(
                f"{name} is not L2-normalized! "
                f"Max deviation from 1.0: {max_deviation:.6f} (tolerance: {atol}). "
                f"Norm range: [{norms.min():.6f}, {norms.max():.6f}]. "
                f"Please normalize with F.normalize(tensor, p=2, dim=1) before passing."
            )

    def _verify_confidence_scores(self, scores: torch.Tensor):
        """
        Verify confidence scores are valid (no NaN, Inf, within [0, 1]).

        Args:
            scores: Confidence scores to verify

        Raises:
            ValueError: If scores contain invalid values
        """
        # Check for NaN
        if torch.isnan(scores).any():
            raise ValueError(
                f"Confidence scores contain NaN values! "
                f"This indicates numerical instability in sigmoid computation."
            )

        # Check for Inf
        if torch.isinf(scores).any():
            raise ValueError(
                f"Confidence scores contain Inf values! "
                f"This indicates overflow in sigmoid computation."
            )

        # Check bounds [0, 1]
        if (scores < 0).any() or (scores > 1).any():
            raise ValueError(
                f"Confidence scores outside [0, 1] range! "
                f"Range: [{scores.min():.6f}, {scores.max():.6f}]. "
                f"Sigmoid should always produce values in [0, 1]."
            )

    def get_high_confidence_mask(
        self,
        confidence_scores: torch.Tensor,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Get boolean mask for high-confidence samples.

        Args:
            confidence_scores: Confidence scores of shape (N,)
            threshold: Confidence threshold (default: 0.7)
                      As per design doc, C_score >= 0.7 = high confidence

        Returns:
            Boolean mask of shape (N,) where True = high confidence

        Example:
            >>> scorer = ConfidenceScorer()
            >>> confidence_scores = torch.tensor([0.9, 0.5, 0.8, 0.3, 0.95])
            >>> mask = scorer.get_high_confidence_mask(confidence_scores, threshold=0.7)
            >>> mask
            tensor([True, False, True, False, True])
        """
        return confidence_scores >= threshold

    def compute_confidence_statistics(
        self,
        confidence_scores: torch.Tensor,
        threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Compute summary statistics for confidence scores.

        Args:
            confidence_scores: Confidence scores of shape (N,)
            threshold: Confidence threshold for acceptance (default: 0.7)

        Returns:
            Dictionary with:
                - 'mean': Mean confidence score
                - 'std': Standard deviation
                - 'median': Median confidence score
                - 'min': Minimum confidence score
                - 'max': Maximum confidence score
                - 'acceptance_rate': Fraction of samples above threshold
                - 'high_confidence_count': Number of high-confidence samples
                - 'low_confidence_count': Number of low-confidence samples
        """
        high_confidence_mask = self.get_high_confidence_mask(confidence_scores, threshold)

        stats = {
            'mean': confidence_scores.mean().item(),
            'std': confidence_scores.std().item(),
            'median': confidence_scores.median().item(),
            'min': confidence_scores.min().item(),
            'max': confidence_scores.max().item(),
            'acceptance_rate': high_confidence_mask.float().mean().item(),
            'high_confidence_count': high_confidence_mask.sum().item(),
            'low_confidence_count': (~high_confidence_mask).sum().item()
        }

        return stats


def test_confidence_scorer():
    """
    Unit tests for ConfidenceScorer.

    Tests:
    1. Basic confidence computation
    2. D_assigned calculation
    3. D_sep calculation
    4. Confidence score bounds [0, 1]
    5. No NaN/Inf values
    6. High vs low confidence separation
    7. Normalization verification
    """
    print("Testing ConfidenceScorer...")

    # Test 1: Basic confidence computation
    print("\n[Test 1] Basic confidence computation")
    scorer = ConfidenceScorer(gamma=5.0)

    # Create synthetic data
    N, K, D_z = 100, 8, 128
    z = torch.randn(N, D_z)
    z_norm = F.normalize(z, p=2, dim=1)

    centroids = torch.randn(K, D_z)
    centroids_norm = F.normalize(centroids, p=2, dim=1)

    cluster_ids, confidence_scores, metrics = scorer.compute_confidence(z_norm, centroids_norm)

    assert cluster_ids.shape == (N,), f"Cluster IDs shape mismatch: {cluster_ids.shape}"
    assert confidence_scores.shape == (N,), f"Confidence scores shape mismatch: {confidence_scores.shape}"
    print(f"  [PASS] Output shapes correct")

    # Test 2: Confidence score bounds
    print("\n[Test 2] Confidence score bounds [0, 1]")
    assert (confidence_scores >= 0).all(), "Confidence scores below 0!"
    assert (confidence_scores <= 1).all(), "Confidence scores above 1!"
    print(f"  [PASS] All scores in [0, 1]: range [{confidence_scores.min():.3f}, {confidence_scores.max():.3f}]")

    # Test 3: No NaN/Inf
    print("\n[Test 3] No NaN/Inf values")
    assert not torch.isnan(confidence_scores).any(), "NaN values detected!"
    assert not torch.isinf(confidence_scores).any(), "Inf values detected!"
    print(f"  [PASS] No invalid values")

    # Test 4: D_assigned computation
    print("\n[Test 4] D_assigned computation")
    D_assigned = metrics['D_assigned']
    D_sep = metrics['D_sep']

    assert D_assigned.shape == (N,), f"D_assigned shape mismatch: {D_assigned.shape}"
    assert (D_assigned >= 0).all(), "Negative distances in D_assigned!"
    assert (D_assigned <= 2).all(), f"D_assigned > 2 (max on unit sphere): max={D_assigned.max():.3f}"
    print(f"  [PASS] D_assigned in [0, 2]: range [{D_assigned.min():.3f}, {D_assigned.max():.3f}]")

    # Test 5: D_sep computation
    print("\n[Test 5] D_sep computation")
    assert D_sep.shape == (N,), f"D_sep shape mismatch: {D_sep.shape}"
    assert (D_sep >= 0).all(), "Negative distances in D_sep!"
    assert (D_sep <= 2).all(), f"D_sep > 2 (max on unit sphere): max={D_sep.max():.3f}"
    print(f"  [PASS] D_sep in [0, 2]: range [{D_sep.min():.3f}, {D_sep.max():.3f}]")

    # Test 6: D_sep > D_assigned (should be true for well-assigned points)
    print("\n[Test 6] Separation property")
    well_separated = (D_sep > D_assigned).sum().item()
    separation_rate = well_separated / N
    print(f"  [INFO] {well_separated}/{N} samples have D_sep > D_assigned ({separation_rate*100:.1f}%)")
    print(f"  [INFO] Mean margin (D_sep - D_assigned): {metrics['margin'].mean():.3f}")

    # Test 7: Confidence statistics
    print("\n[Test 7] Confidence statistics")
    stats = scorer.compute_confidence_statistics(confidence_scores, threshold=0.7)
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Acceptance rate (>= 0.7): {stats['acceptance_rate']*100:.1f}%")
    print(f"  High confidence: {stats['high_confidence_count']}")
    print(f"  Low confidence: {stats['low_confidence_count']}")

    # Test 8: Normalization verification (should fail on non-normalized input)
    print("\n[Test 8] Normalization verification")
    try:
        # Try with non-normalized vectors
        z_bad = torch.randn(10, D_z)  # NOT normalized
        scorer.compute_confidence(z_bad, centroids_norm)
        print(f"  [FAIL] Should have raised ValueError for non-normalized input")
        assert False
    except ValueError as e:
        print(f"  [PASS] Correctly rejected non-normalized input")
        print(f"         Error: {str(e)[:80]}...")

    # Test 9: Different gamma values
    print("\n[Test 9] Effect of gamma parameter")
    for gamma in [0.5, 2.0, 5.0, 10.0]:
        scorer_gamma = ConfidenceScorer(gamma=gamma)
        _, scores, _ = scorer_gamma.compute_confidence(z_norm, centroids_norm)
        stats = scorer_gamma.compute_confidence_statistics(scores)
        print(f"  gamma={gamma:4.1f} | Mean: {stats['mean']:.3f} | Acceptance: {stats['acceptance_rate']*100:5.1f}%")

    print("\n[SUCCESS] All ConfidenceScorer tests passed!")
    return True


if __name__ == '__main__':
    test_confidence_scorer()
