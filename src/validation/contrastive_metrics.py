"""
Contrastive Learning Quality Metrics
Computes Alignment and Uniformity for collapse detection during Stage 1 training.

References:
    Wang & Isola (2020): "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ContrastiveMetrics:
    """
    Computes Alignment and Uniformity metrics for contrastive learning quality assessment.

    Alignment: Measures how close positive pairs are (lower is better)
    Uniformity: Measures how uniformly distributed representations are (lower is better)
    """

    @staticmethod
    def compute_alignment(
        z1: torch.Tensor,
        z2: torch.Tensor,
        normalize: bool = True
    ) -> float:
        """
        Compute Alignment metric for positive pairs.

        Measures how close positive pairs (augmented views) are:
            L_align = E[ ||z_i - z'_i||^2 ]

        Args:
            z1: First augmented view latent vectors (N, D_z)
            z2: Second augmented view latent vectors (N, D_z)
            normalize: Whether to L2-normalize before computation

        Returns:
            alignment: Scalar alignment loss (lower is better, typically 0-2)
        """
        if normalize:
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)

        # Compute pairwise squared distances for positive pairs
        # ||z1 - z2||^2 = ||z1||^2 + ||z2||^2 - 2<z1, z2>
        alignment = (z1 - z2).pow(2).sum(dim=1).mean()

        return alignment.item()

    @staticmethod
    def compute_uniformity(
        z: torch.Tensor,
        normalize: bool = True,
        t: float = 2.0
    ) -> float:
        """
        Compute Uniformity metric for representation distribution.

        Measures how uniformly representations are distributed on hypersphere:
            L_uniform = log E[ exp(-t * ||z_i - z_j||^2) ]

        Args:
            z: Latent vectors (N, D_z)
            normalize: Whether to L2-normalize before computation
            t: Temperature parameter (default: 2.0)

        Returns:
            uniformity: Scalar uniformity loss (lower is better)
        """
        if normalize:
            z = F.normalize(z, p=2, dim=1)

        # Compute pairwise squared distances
        n = z.shape[0]

        # Efficient computation using matrix operations
        # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2<z_i, z_j>
        sq_dists = (
            z.pow(2).sum(dim=1, keepdim=True) +
            z.pow(2).sum(dim=1, keepdim=True).t() -
            2 * torch.mm(z, z.t())
        )

        # Remove diagonal (self-distances)
        mask = torch.eye(n, device=z.device).bool()
        sq_dists = sq_dists[~mask]

        # Compute uniformity
        uniformity = torch.log(torch.exp(-t * sq_dists).mean())

        return uniformity.item()

    @staticmethod
    def compute_both(
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute both Alignment and Uniformity metrics.

        Args:
            z1: First augmented view (N, D_z)
            z2: Second augmented view (N, D_z)

        Returns:
            alignment: Alignment loss
            uniformity: Uniformity loss
        """
        # Alignment uses positive pairs
        alignment = ContrastiveMetrics.compute_alignment(z1, z2)

        # Uniformity uses all samples (concatenate both views)
        z_all = torch.cat([z1, z2], dim=0)
        uniformity = ContrastiveMetrics.compute_uniformity(z_all)

        return alignment, uniformity

    @staticmethod
    def check_collapse(
        alignment: float,
        uniformity: float,
        prev_alignment: float,
        prev_uniformity: float,
        alignment_threshold: float = 1.2,
        uniformity_threshold: float = 1.1
    ) -> Tuple[bool, str]:
        """
        Check for representation collapse based on metric trends.

        Args:
            alignment: Current alignment
            uniformity: Current uniformity
            prev_alignment: Previous alignment
            prev_uniformity: Previous uniformity
            alignment_threshold: Alert if alignment increases by this factor
            uniformity_threshold: Alert if uniformity increases by this factor

        Returns:
            collapsed: True if collapse detected
            message: Warning message with recommendations
        """
        collapsed = False
        message = ""

        # Check alignment increase (bad - positive pairs drifting apart)
        if alignment > prev_alignment * alignment_threshold:
            collapsed = True
            message += f"WARNING: Alignment increased {alignment/prev_alignment:.2f}x! "
            message += "Positive pairs are drifting apart. "

        # Check uniformity increase (bad - clustering together)
        if uniformity > prev_uniformity * uniformity_threshold:
            collapsed = True
            message += f"WARNING: Uniformity increased {uniformity/prev_uniformity:.2f}x! "
            message += "Representations collapsing to small region. "

        if collapsed:
            message += "\nRecommendations:\n"
            message += "  - Reduce learning rate\n"
            message += "  - Increase temperature tau\n"
            message += "  - Add L2 regularization to encoder\n"
            message += "  - Check for gradient explosion\n"

        return collapsed, message


class ContrastiveMetricsTracker:
    """Helper class for tracking metrics during training."""

    def __init__(self):
        self.history = {
            'epochs': [],
            'alignment': [],
            'uniformity': []
        }

    def update(
        self,
        epoch: int,
        z1: torch.Tensor,
        z2: torch.Tensor
    ):
        """
        Compute and store metrics for current epoch.

        Args:
            epoch: Current epoch
            z1: First view latent vectors
            z2: Second view latent vectors
        """
        alignment, uniformity = ContrastiveMetrics.compute_both(z1, z2)

        self.history['epochs'].append(epoch)
        self.history['alignment'].append(alignment)
        self.history['uniformity'].append(uniformity)

        print(f"Epoch {epoch}: Alignment = {alignment:.4f}, Uniformity = {uniformity:.4f}")

        # Check for collapse
        if epoch > 0:
            prev_alignment = self.history['alignment'][-2]
            prev_uniformity = self.history['uniformity'][-2]

            collapsed, message = ContrastiveMetrics.check_collapse(
                alignment, uniformity,
                prev_alignment, prev_uniformity
            )

            if collapsed:
                print(message)

        return alignment, uniformity

    def get_history(self):
        """Return metrics history."""
        return self.history


# Example usage
if __name__ == '__main__':
    # Example: Compute metrics on random data
    print("Testing Contrastive Metrics...")

    # Generate random latent vectors
    z1 = torch.randn(100, 128)
    z2 = torch.randn(100, 128)

    # Normalize
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Compute metrics
    alignment, uniformity = ContrastiveMetrics.compute_both(z1, z2)

    print(f"Alignment: {alignment:.4f}")
    print(f"Uniformity: {uniformity:.4f}")
    print("\nMetrics computed successfully!")
