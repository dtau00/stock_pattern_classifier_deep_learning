"""
Projection Head for Contrastive Learning

This module implements a non-linear projection head used ONLY for NT-Xent
contrastive loss during Stage 1 training. The projection head maps the latent
representation z to a higher-dimensional space h for contrastive learning.

CRITICAL: The projection head is discarded after Stage 1. Clustering (Stage 2)
uses the raw latent vector z, NOT the projected vector h.

Key Features:
- 2-layer MLP with ReLU activation
- Optional bottleneck architecture for additional non-linearity
- L2 normalization of output for cosine similarity
- Used ONLY for NT-Xent loss, never for clustering

Reference: Implementation Guide Section 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Projects latent vector z to projection vector h for NT-Xent loss.
    This head is used ONLY during contrastive pre-training (Stage 1) and
    is discarded before clustering (Stage 2).

    Architecture:
        Standard mode:
            Linear(d_z → d_h) → ReLU → Linear(d_h → d_h)

        Bottleneck mode (optional):
            Linear(d_z → bottleneck_dim) → ReLU → Linear(bottleneck_dim → d_h)

    Args:
        d_z (int): Latent dimension (input from encoder) (default: 128)
        d_h (int): Projection dimension (output for NT-Xent) (default: 128)
        use_bottleneck (bool): If True, use bottleneck architecture (default: False)
        bottleneck_dim (int): Dimension of bottleneck layer (default: 64)

    Shape:
        - Input: (batch, d_z)
        - Output: (batch, d_h)

    Example:
        >>> proj = ProjectionHead(d_z=128, d_h=128)
        >>> z = torch.randn(32, 128)
        >>> h = proj(z)
        >>> h.shape
        torch.Size([32, 128])

    Notes:
        - Output h should be L2-normalized before NT-Xent loss
        - Projection head is saved during Stage 1 checkpointing
        - Projection head is discarded for Stage 2 and inference
        - Clustering always uses z (encoder output), never h
    """

    def __init__(
        self,
        d_z: int = 128,
        d_h: int = 128,
        use_bottleneck: bool = False,
        bottleneck_dim: int = 64
    ):
        super().__init__()

        self.d_z = d_z
        self.d_h = d_h
        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            # Bottleneck architecture for additional non-linearity
            self.mlp = nn.Sequential(
                nn.Linear(d_z, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, d_h)
            )
        else:
            # Standard 2-layer MLP
            self.mlp = nn.Sequential(
                nn.Linear(d_z, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_h)
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.

        Args:
            z: Latent vector from encoder, shape (batch, d_z)

        Returns:
            Projection vector h, shape (batch, d_h)

        Notes:
            - Output h is NOT L2-normalized (normalization done in loss function)
            - This allows flexibility in loss computation
        """
        h = self.mlp(z)
        return h

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.

        This is a convenience method that combines projection and normalization
        for use in contrastive loss computation.

        Args:
            z: Latent vector from encoder, shape (batch, d_z)

        Returns:
            L2-normalized projection vector h, shape (batch, d_h)

        Notes:
            - Each vector is projected onto unit hypersphere
            - Enables cosine similarity computation via dot product
        """
        h = self.mlp(z)
        h_normalized = F.normalize(h, p=2, dim=1)
        return h_normalized


def test_projection_head():
    """
    Unit tests for ProjectionHead.

    Tests:
    1. Standard mode (no bottleneck)
    2. Bottleneck mode
    3. L2 normalization
    4. Gradient flow
    5. Parameter counts
    """
    print("Testing ProjectionHead...")

    batch_size, d_z, d_h = 32, 128, 128

    # Test 1: Standard mode
    print("\n[Test 1] Standard mode (no bottleneck)")
    proj_std = ProjectionHead(d_z=d_z, d_h=d_h, use_bottleneck=False)

    z = torch.randn(batch_size, d_z)
    h = proj_std(z)

    assert h.shape == (batch_size, d_h), f"Shape mismatch: {h.shape}"
    print(f"  [PASS] Input shape: {z.shape}")
    print(f"  [PASS] Output shape: {h.shape}")

    # Test 2: Bottleneck mode
    print("\n[Test 2] Bottleneck mode")
    proj_bn = ProjectionHead(
        d_z=d_z,
        d_h=d_h,
        use_bottleneck=True,
        bottleneck_dim=64
    )

    h_bn = proj_bn(z)

    assert h_bn.shape == (batch_size, d_h), f"Shape mismatch: {h_bn.shape}"
    print(f"  [PASS] Input shape: {z.shape}")
    print(f"  [PASS] Output shape: {h_bn.shape}")

    # Test 3: L2 normalization
    print("\n[Test 3] L2 normalization")

    # Test unnormalized output
    h_raw = proj_std(z)
    norms_raw = torch.norm(h_raw, p=2, dim=1)
    print(f"  [INFO] Raw output norms (should vary):")
    print(f"         Mean: {norms_raw.mean():.3f}, Std: {norms_raw.std():.3f}")
    print(f"         Min: {norms_raw.min():.3f}, Max: {norms_raw.max():.3f}")

    # Test normalized output
    h_norm = proj_std.forward_normalized(z)
    norms_norm = torch.norm(h_norm, p=2, dim=1)

    # All norms should be ~1.0
    assert torch.allclose(norms_norm, torch.ones(batch_size), atol=1e-5), \
        f"L2 normalization failed: norms not close to 1.0"

    print(f"  [PASS] Normalized output norms (should be 1.0):")
    print(f"         Mean: {norms_norm.mean():.6f}, Std: {norms_norm.std():.6f}")
    print(f"         Min: {norms_norm.min():.6f}, Max: {norms_norm.max():.6f}")

    # Test 4: Gradient flow
    print("\n[Test 4] Gradient flow")
    proj_std = ProjectionHead(d_z=d_z, d_h=d_h, use_bottleneck=False)
    z = torch.randn(batch_size, d_z, requires_grad=True)

    h = proj_std(z)
    loss = h.sum()
    loss.backward()

    assert z.grad is not None, "No gradient for input z"
    print(f"  [PASS] Gradients flow to input z")
    print(f"  [INFO] Gradient norm: {z.grad.norm():.3f}")

    # Test 5: Parameter count
    print("\n[Test 5] Parameter count")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_std = count_parameters(proj_std)
    params_bn = count_parameters(proj_bn)

    print(f"  Standard mode: {params_std:,} parameters")
    print(f"  Bottleneck mode: {params_bn:,} parameters")

    # Test 6: Different dimensions
    print("\n[Test 6] Different dimensions")
    test_cases = [
        (128, 128, False),  # Standard: same dim
        (128, 256, False),  # Standard: project to higher dim
        (64, 128, False),   # Standard: project from lower dim
        (128, 128, True),   # Bottleneck: same dim
        (128, 256, True),   # Bottleneck: project to higher dim
    ]

    for d_z_test, d_h_test, use_bn in test_cases:
        proj = ProjectionHead(
            d_z=d_z_test,
            d_h=d_h_test,
            use_bottleneck=use_bn
        )
        z_test = torch.randn(batch_size, d_z_test)
        h_test = proj(z_test)

        assert h_test.shape == (batch_size, d_h_test), \
            f"Shape mismatch for d_z={d_z_test}, d_h={d_h_test}"

        mode_str = "bottleneck" if use_bn else "standard"
        print(f"  [PASS] {mode_str}: {d_z_test} -> {d_h_test} "
              f"({count_parameters(proj):,} params)")

    # Test 7: Cosine similarity after normalization
    print("\n[Test 7] Cosine similarity after normalization")
    proj = ProjectionHead(d_z=128, d_h=128)

    # Create two similar latent vectors
    z1 = torch.randn(1, 128)
    z2 = z1 + 0.1 * torch.randn(1, 128)  # Similar to z1

    # Get normalized projections
    h1 = proj.forward_normalized(z1)
    h2 = proj.forward_normalized(z2)

    # Compute cosine similarity (dot product of normalized vectors)
    cos_sim = (h1 * h2).sum(dim=1).item()

    print(f"  Cosine similarity between similar inputs: {cos_sim:.3f}")
    assert -1.0 <= cos_sim <= 1.0, "Cosine similarity out of range"
    print(f"  [PASS] Cosine similarity in valid range [-1, 1]")

    # Create dissimilar vector
    z3 = torch.randn(1, 128)  # Unrelated to z1
    h3 = proj.forward_normalized(z3)

    cos_sim_dissim = (h1 * h3).sum(dim=1).item()
    print(f"  Cosine similarity between dissimilar inputs: {cos_sim_dissim:.3f}")

    print("\n[SUCCESS] All projection head tests passed!")
    return True


if __name__ == '__main__':
    test_projection_head()
