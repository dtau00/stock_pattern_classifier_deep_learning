"""
Loss Functions for UCL-TSC Model

This module implements the loss functions for two-stage training:
- NT-Xent (Normalized Temperature-scaled Cross Entropy) for contrastive learning
- DEC (Deep Embedded Clustering) loss for clustering
- Lambda warm-up schedule for balancing losses in Stage 2

Reference: Implementation Guide Section 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.

    Used for contrastive learning in Stage 1 (pre-training) and Stage 2
    (joint fine-tuning). Maximizes agreement between augmented views of
    the same sample while minimizing agreement between different samples.

    Formula:
        L = -log( exp(sim(z_i, z'_i) / tau) / sum_k exp(sim(z_i, z_k) / tau) )

    Where:
        - sim(u, v) = u^T v / (||u|| ||v||) is cosine similarity
        - tau is temperature parameter (controls separation sharpness)
        - z_i, z'_i are two augmented views of the same sample (positive pair)
        - z_k are all other samples in the batch (negative pairs)

    Args:
        temperature (float): Temperature parameter tau (default: 0.5)
            - Lower values = harder contrastive task (sharper separation)
            - Typical range: [0.1, 1.0]

    Shape:
        - Input: Two tensors (z1, z2) each of shape (batch, d_h)
        - Output: Scalar loss value

    Example:
        >>> loss_fn = NTXentLoss(temperature=0.5)
        >>> z1 = F.normalize(torch.randn(32, 128), p=2, dim=1)
        >>> z2 = F.normalize(torch.randn(32, 128), p=2, dim=1)
        >>> loss = loss_fn(z1, z2)
        >>> loss.item()
        4.523

    Reference: Chen et al. (2020) - A Simple Framework for Contrastive Learning (SimCLR)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two views.

        Args:
            z1: L2-normalized projection vectors from view 1, shape (batch, d_h)
            z2: L2-normalized projection vectors from view 2, shape (batch, d_h)

        Returns:
            Scalar loss value

        Notes:
            - Inputs MUST be L2-normalized (norm=1.0 per vector)
            - Positive pairs: (z1[i], z2[i]) for each i
            - Negative pairs: all other samples in concatenated batch
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Concatenate z1 and z2 to create full batch of 2N samples
        # Shape: (2*batch, d_h)
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix for all pairs
        # Shape: (2*batch, 2*batch)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Create mask to remove self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        # Use -1e9 for FP32 or -65504 for FP16 compatibility
        mask_value = -1e9 if sim_matrix.dtype == torch.float32 else -65504.0
        sim_matrix.masked_fill_(mask, mask_value)

        # Create indices for positive pairs
        # For z1[i], positive is z2[i] (index i + batch_size)
        # For z2[i], positive is z1[i] (index i)
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),  # z1's positives
            torch.arange(0, batch_size, device=device)  # z2's positives
        ])

        # Extract positive similarities
        # Shape: (2*batch,)
        pos_sim = sim_matrix[torch.arange(2 * batch_size, device=device), pos_indices]

        # Compute loss: -log(exp(pos) / sum(exp(all)))
        # = -pos + log(sum(exp(all)))
        # Shape: (2*batch,)
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)

        # Average over all samples
        loss = loss.mean()

        return loss


class ClusteringLoss(nn.Module):
    """
    Deep Embedded Clustering (DEC) Loss.

    Used in Stage 2 (joint fine-tuning) to refine cluster assignments.
    Minimizes KL divergence between soft assignments Q and target distribution P.

    Formula:
        L_cluster = KL(P || Q) = sum_i sum_k P_ik * log(P_ik / Q_ik)

    Where:
        Q_ik = (1 + ||z_i - mu_k||^2 / alpha)^(-(alpha+1)/2)  [Student's t-distribution]
        P_ik = (Q_ik^2 / sum_i Q_ik) / sum_k' (Q_ik'^2 / sum_i Q_ik')  [Target distribution]

    Args:
        alpha (float): Degrees of freedom for Student's t-distribution (default: 1.0)
            - alpha=1 corresponds to Cauchy distribution
            - Recommended: keep at 1.0

    Shape:
        - Input:
            - z: Latent vectors of shape (batch, d_z)
            - centroids: Cluster centroids of shape (num_clusters, d_z)
        - Output: Scalar loss value

    Example:
        >>> loss_fn = ClusteringLoss(alpha=1.0)
        >>> z = F.normalize(torch.randn(32, 128), p=2, dim=1)
        >>> centroids = F.normalize(torch.randn(8, 128), p=2, dim=1)
        >>> loss = loss_fn(z, centroids)
        >>> loss.item()
        2.073

    Reference: Xie et al. (2016) - Unsupervised Deep Embedding for Clustering Analysis
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Compute DEC clustering loss.

        Args:
            z: L2-normalized latent vectors, shape (batch, d_z)
            centroids: L2-normalized cluster centroids, shape (num_clusters, d_z)

        Returns:
            Scalar loss value

        Notes:
            - Both z and centroids should be L2-normalized
            - Uses Student's t-distribution with alpha=1 (Cauchy)
            - Target distribution P sharpens soft assignments Q
        """
        # Compute soft assignments Q using Student's t-distribution
        # Shape: (batch, num_clusters)
        q = self._soft_assignment(z, centroids)

        # Compute target distribution P
        # Shape: (batch, num_clusters)
        p = self._target_distribution(q)

        # Compute KL divergence: KL(P || Q)
        # Add small epsilon to avoid log(0)
        loss = F.kl_div(
            (q + 1e-10).log(),
            p,
            reduction='batchmean'
        )

        return loss

    def _soft_assignment(self, z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments using Student's t-distribution.

        Args:
            z: Latent vectors, shape (batch, d_z)
            centroids: Cluster centroids, shape (num_clusters, d_z)

        Returns:
            Soft assignments Q, shape (batch, num_clusters)
        """
        # Compute squared Euclidean distances
        # Shape: (batch, num_clusters)
        distances_sq = torch.cdist(z, centroids, p=2).pow(2)

        # Student's t-distribution similarity
        # q_ik = (1 + d^2 / alpha)^(-(alpha+1)/2)
        q = (1.0 + distances_sq / self.alpha).pow(-(self.alpha + 1.0) / 2.0)

        # Normalize to get probabilities (sum to 1 over clusters)
        q = q / q.sum(dim=1, keepdim=True)

        return q

    def _target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution P from soft assignments Q.

        The target distribution sharpens the soft assignments to improve
        cluster purity and prevent degenerate solutions.

        Args:
            q: Soft assignments, shape (batch, num_clusters)

        Returns:
            Target distribution P, shape (batch, num_clusters)
        """
        # Square the soft assignments
        q_squared = q.pow(2)

        # Normalize by cluster frequency (sum over samples)
        # Shape: (num_clusters,)
        cluster_freq = q.sum(dim=0)

        # Compute P_ik = (Q_ik^2 / f_k) / sum_k' (Q_ik'^2 / f_k')
        # Shape: (batch, num_clusters)
        p = q_squared / cluster_freq

        # Normalize to get probabilities (sum to 1 over clusters)
        p = p / p.sum(dim=1, keepdim=True)

        return p


class LambdaSchedule:
    """
    Lambda warm-up schedule for Stage 2 joint fine-tuning.

    Controls the weight of the contrastive loss relative to clustering loss
    during Stage 2 training. Starts with low lambda (prioritize clustering)
    and gradually increases to balance both objectives.

    Schedule:
        lambda(t) = lambda_start + (lambda_end - lambda_start) * min(t / warmup_epochs, 1.0)

    Args:
        lambda_start (float): Starting lambda value (default: 0.1)
        lambda_end (float): Final lambda value (default: 1.0)
        warmup_epochs (int): Number of epochs for warm-up (default: 10)

    Example:
        >>> schedule = LambdaSchedule(lambda_start=0.1, lambda_end=1.0, warmup_epochs=10)
        >>> schedule.get_lambda(epoch=0)
        0.1
        >>> schedule.get_lambda(epoch=5)
        0.55
        >>> schedule.get_lambda(epoch=10)
        1.0
        >>> schedule.get_lambda(epoch=20)
        1.0
    """

    def __init__(
        self,
        lambda_start: float = 0.1,
        lambda_end: float = 1.0,
        warmup_epochs: int = 10
    ):
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.warmup_epochs = warmup_epochs

    def get_lambda(self, epoch: int) -> float:
        """
        Get lambda value for current epoch.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Lambda value for current epoch
        """
        if epoch >= self.warmup_epochs:
            return self.lambda_end

        # Linear interpolation during warm-up
        progress = epoch / self.warmup_epochs
        lambda_value = self.lambda_start + (self.lambda_end - self.lambda_start) * progress

        return lambda_value

    def __call__(self, epoch: int) -> float:
        """Allow using schedule as a callable."""
        return self.get_lambda(epoch)


def test_losses():
    """
    Unit tests for loss functions.

    Tests:
    1. NT-Xent loss computation
    2. NT-Xent loss properties (positive pairs closer than negatives)
    3. Clustering loss computation
    4. Lambda schedule
    """
    print("Testing Loss Functions...")

    batch_size = 32
    d_h = 128
    num_clusters = 8

    # Test 1: NT-Xent Loss
    print("\n[Test 1] NT-Xent Loss")
    loss_fn = NTXentLoss(temperature=0.5)

    # Create two views (normalized)
    z1 = F.normalize(torch.randn(batch_size, d_h), p=2, dim=1)
    z2 = F.normalize(torch.randn(batch_size, d_h), p=2, dim=1)

    loss = loss_fn(z1, z2)
    print(f"  [PASS] NT-Xent loss computed: {loss.item():.3f}")

    # Loss should be positive
    assert loss.item() > 0, "NT-Xent loss should be positive"
    print(f"  [PASS] Loss is positive: {loss.item():.3f}")

    # Test 2: Positive pairs should have lower loss
    print("\n[Test 2] Positive pair similarity")

    # Create identical views (perfect positive pairs)
    z1_identical = F.normalize(torch.randn(batch_size, d_h), p=2, dim=1)
    z2_identical = z1_identical.clone()
    loss_identical = loss_fn(z1_identical, z2_identical)

    # Create random views (poor positive pairs)
    z1_random = F.normalize(torch.randn(batch_size, d_h), p=2, dim=1)
    z2_random = F.normalize(torch.randn(batch_size, d_h), p=2, dim=1)
    loss_random = loss_fn(z1_random, z2_random)

    print(f"  Loss (identical views): {loss_identical.item():.3f}")
    print(f"  Loss (random views): {loss_random.item():.3f}")
    assert loss_identical < loss_random, "Identical views should have lower loss"
    print(f"  [PASS] Identical views have lower loss")

    # Test 3: Clustering Loss
    print("\n[Test 3] Clustering Loss")
    cluster_loss_fn = ClusteringLoss(alpha=1.0)

    z = F.normalize(torch.randn(batch_size, d_h), p=2, dim=1)
    centroids = F.normalize(torch.randn(num_clusters, d_h), p=2, dim=1)

    cluster_loss = cluster_loss_fn(z, centroids)
    print(f"  [PASS] Clustering loss computed: {cluster_loss.item():.3f}")

    # Loss should be positive
    assert cluster_loss.item() >= 0, "Clustering loss should be non-negative"
    print(f"  [PASS] Loss is non-negative: {cluster_loss.item():.3f}")

    # Test soft assignments sum to 1
    q = cluster_loss_fn._soft_assignment(z, centroids)
    q_sum = q.sum(dim=1)
    assert torch.allclose(q_sum, torch.ones(batch_size), atol=1e-5), \
        "Soft assignments should sum to 1"
    print(f"  [PASS] Soft assignments sum to 1.0 (mean={q_sum.mean():.6f})")

    # Test 4: Lambda Schedule
    print("\n[Test 4] Lambda Schedule")
    schedule = LambdaSchedule(lambda_start=0.1, lambda_end=1.0, warmup_epochs=10)

    # Test epoch 0
    lambda_0 = schedule.get_lambda(0)
    assert abs(lambda_0 - 0.1) < 1e-5, "Lambda at epoch 0 should be lambda_start"
    print(f"  [PASS] Epoch 0: lambda={lambda_0:.3f} (should be 0.1)")

    # Test epoch 5 (midpoint)
    lambda_5 = schedule.get_lambda(5)
    expected_5 = 0.1 + (1.0 - 0.1) * 0.5
    assert abs(lambda_5 - expected_5) < 1e-5, "Lambda at epoch 5 should be midpoint"
    print(f"  [PASS] Epoch 5: lambda={lambda_5:.3f} (should be ~0.55)")

    # Test epoch 10 (end of warmup)
    lambda_10 = schedule.get_lambda(10)
    assert abs(lambda_10 - 1.0) < 1e-5, "Lambda at epoch 10 should be lambda_end"
    print(f"  [PASS] Epoch 10: lambda={lambda_10:.3f} (should be 1.0)")

    # Test epoch 20 (after warmup)
    lambda_20 = schedule.get_lambda(20)
    assert abs(lambda_20 - 1.0) < 1e-5, "Lambda after warmup should stay at lambda_end"
    print(f"  [PASS] Epoch 20: lambda={lambda_20:.3f} (should be 1.0)")

    # Test monotonic increase
    lambdas = [schedule.get_lambda(i) for i in range(15)]
    is_monotonic = all(lambdas[i] <= lambdas[i+1] for i in range(len(lambdas)-1))
    assert is_monotonic, "Lambda schedule should be monotonically increasing"
    print(f"  [PASS] Lambda schedule is monotonically increasing")

    # Test 5: Gradient flow
    print("\n[Test 5] Gradient flow")

    # NT-Xent (use leaf tensors)
    z1_raw = torch.randn(batch_size, d_h, requires_grad=True)
    z2_raw = torch.randn(batch_size, d_h, requires_grad=True)
    z1 = F.normalize(z1_raw, p=2, dim=1)
    z2 = F.normalize(z2_raw, p=2, dim=1)
    loss = loss_fn(z1, z2)
    loss.backward()
    assert z1_raw.grad is not None, "NT-Xent gradients should flow"
    print(f"  [PASS] NT-Xent gradients flow to inputs")
    print(f"  [INFO] Gradient norm: {z1_raw.grad.norm():.3f}")

    # Clustering (use leaf tensors)
    z_raw = torch.randn(batch_size, d_h, requires_grad=True)
    centroids_raw = torch.randn(num_clusters, d_h, requires_grad=True)
    z = F.normalize(z_raw, p=2, dim=1)
    centroids = F.normalize(centroids_raw, p=2, dim=1)
    cluster_loss = cluster_loss_fn(z, centroids)
    cluster_loss.backward()
    assert z_raw.grad is not None, "Clustering gradients should flow to z"
    assert centroids_raw.grad is not None, "Clustering gradients should flow to centroids"
    print(f"  [PASS] Clustering gradients flow to z and centroids")
    print(f"  [INFO] z gradient norm: {z_raw.grad.norm():.3f}, centroid gradient norm: {centroids_raw.grad.norm():.3f}")

    print("\n[SUCCESS] All loss function tests passed!")
    return True


if __name__ == '__main__':
    test_losses()
