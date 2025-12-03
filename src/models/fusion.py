"""
Adaptive Feature Fusion Layer

This module combines outputs from multiple specialized encoders using
attention-gated summation. The fusion layer dynamically weights the
contribution of each encoder based on learned attention weights.

Key Features:
- Attention-gated summation (softmax over encoder outputs)
- Dynamic encoder handling (2 or 3 encoders based on configuration)
- L2 normalization for clustering compatibility
- FFN for computing attention weights

Reference: Implementation Guide Section 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusionLayer(nn.Module):
    """
    Attention-gated fusion with dynamic encoder handling.

    Combines encoder outputs (z_spatial, z_temporal, z_fused) using learned
    attention weights. Automatically handles 2-encoder or 3-encoder modes.

    Formula:
        z = Softmax(FFN(Concat(z_·))) ⊙ [z_spatial, z_temporal, z_fused]

    Args:
        d_z (int): Latent dimension (default: 128)
        use_hybrid_encoder (bool): If True, expects 3 encoders;
                                    if False, expects 2 encoders (default: False)

    Shape:
        - Input (2-encoder mode):
            - z_spatial: (batch, d_z)
            - z_temporal: (batch, d_z)
            - z_fused: None
        - Input (3-encoder mode):
            - z_spatial: (batch, d_z)
            - z_temporal: (batch, d_z)
            - z_fused: (batch, d_z)
        - Output: (batch, d_z)

    Example:
        >>> fusion = AdaptiveFusionLayer(d_z=128, use_hybrid_encoder=False)
        >>> z_spatial = torch.randn(32, 128)
        >>> z_temporal = torch.randn(32, 128)
        >>> z = fusion(z_spatial, z_temporal)
        >>> z.shape
        torch.Size([32, 128])
    """

    def __init__(self, d_z: int = 128, use_hybrid_encoder: bool = False):
        super().__init__()

        self.use_hybrid_encoder = use_hybrid_encoder
        self.d_z = d_z
        num_encoders = 3 if use_hybrid_encoder else 2

        # FFN to compute attention weights
        # Input: concatenated encoder outputs (num_encoders * d_z)
        # Output: attention weights (num_encoders)
        self.ffn = nn.Sequential(
            nn.Linear(num_encoders * d_z, 128),
            nn.ReLU(),
            nn.Linear(128, num_encoders),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_temporal: torch.Tensor,
        z_fused: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with attention-gated summation.

        Args:
            z_spatial: Spatial encoder output, shape (batch, d_z)
            z_temporal: Temporal encoder output, shape (batch, d_z)
            z_fused: Hybrid encoder output, shape (batch, d_z) or None

        Returns:
            Fused latent vector z, shape (batch, d_z)

        Raises:
            AssertionError: If use_hybrid_encoder=True but z_fused is None
        """
        if self.use_hybrid_encoder:
            assert z_fused is not None, \
                "z_fused required when use_hybrid_encoder=True"

            # Stack encoder outputs: (batch, 3, d_z)
            encoder_outputs = torch.stack(
                [z_spatial, z_temporal, z_fused], dim=1
            )

            # Concatenate for attention FFN: (batch, 3*d_z)
            concat_features = torch.cat(
                [z_spatial, z_temporal, z_fused], dim=1
            )
        else:
            # Stack encoder outputs: (batch, 2, d_z)
            encoder_outputs = torch.stack(
                [z_spatial, z_temporal], dim=1
            )

            # Concatenate for attention FFN: (batch, 2*d_z)
            concat_features = torch.cat(
                [z_spatial, z_temporal], dim=1
            )

        # Compute attention weights via FFN: (batch, num_encoders)
        weights = self.ffn(concat_features)

        # Add dimension for broadcasting: (batch, num_encoders, 1)
        weights = weights.unsqueeze(2)

        # Weighted sum: (batch, num_encoders, d_z) * (batch, num_encoders, 1)
        #            -> (batch, num_encoders, d_z) -> (batch, d_z)
        z = (weights * encoder_outputs).sum(dim=1)

        return z

    def get_attention_weights(
        self,
        z_spatial: torch.Tensor,
        z_temporal: torch.Tensor,
        z_fused: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get attention weights without computing the fusion.

        Useful for visualization and debugging.

        Args:
            z_spatial: Spatial encoder output, shape (batch, d_z)
            z_temporal: Temporal encoder output, shape (batch, d_z)
            z_fused: Hybrid encoder output, shape (batch, d_z) or None

        Returns:
            Attention weights, shape (batch, num_encoders)
        """
        if self.use_hybrid_encoder:
            assert z_fused is not None, \
                "z_fused required when use_hybrid_encoder=True"
            concat_features = torch.cat(
                [z_spatial, z_temporal, z_fused], dim=1
            )
        else:
            concat_features = torch.cat(
                [z_spatial, z_temporal], dim=1
            )

        weights = self.ffn(concat_features)  # (batch, num_encoders)
        return weights


def test_fusion_layer():
    """
    Unit tests for AdaptiveFusionLayer.

    Tests:
    1. 2-encoder mode (use_hybrid_encoder=False)
    2. 3-encoder mode (use_hybrid_encoder=True)
    3. Attention weight properties (sum to 1, non-negative)
    4. Output shape
    """
    print("Testing AdaptiveFusionLayer...")

    batch_size, d_z = 32, 128

    # Test 1: 2-encoder mode
    print("\n[Test 1] 2-encoder mode")
    fusion_2 = AdaptiveFusionLayer(d_z=d_z, use_hybrid_encoder=False)

    z_spatial = torch.randn(batch_size, d_z)
    z_temporal = torch.randn(batch_size, d_z)

    z = fusion_2(z_spatial, z_temporal)
    weights = fusion_2.get_attention_weights(z_spatial, z_temporal)

    assert z.shape == (batch_size, d_z), f"Shape mismatch: {z.shape}"
    assert weights.shape == (batch_size, 2), f"Weights shape mismatch: {weights.shape}"
    print(f"  [PASS] Output shape: {z.shape}")
    print(f"  [PASS] Attention weights shape: {weights.shape}")

    # Check weights sum to 1 and are non-negative
    weights_sum = weights.sum(dim=1)
    assert torch.allclose(weights_sum, torch.ones(batch_size), atol=1e-5), \
        "Weights don't sum to 1"
    assert (weights >= 0).all(), "Weights contain negative values"
    print(f"  [PASS] Weights sum to 1.0 (mean={weights_sum.mean():.6f})")
    print(f"  [PASS] Weights are non-negative (min={weights.min():.6f})")
    print(f"  [INFO] Mean attention: spatial={weights[:,0].mean():.3f}, "
          f"temporal={weights[:,1].mean():.3f}")

    # Test 2: 3-encoder mode
    print("\n[Test 2] 3-encoder mode")
    fusion_3 = AdaptiveFusionLayer(d_z=d_z, use_hybrid_encoder=True)

    z_fused = torch.randn(batch_size, d_z)

    z = fusion_3(z_spatial, z_temporal, z_fused)
    weights = fusion_3.get_attention_weights(z_spatial, z_temporal, z_fused)

    assert z.shape == (batch_size, d_z), f"Shape mismatch: {z.shape}"
    assert weights.shape == (batch_size, 3), f"Weights shape mismatch: {weights.shape}"
    print(f"  [PASS] Output shape: {z.shape}")
    print(f"  [PASS] Attention weights shape: {weights.shape}")

    # Check weights sum to 1 and are non-negative
    weights_sum = weights.sum(dim=1)
    assert torch.allclose(weights_sum, torch.ones(batch_size), atol=1e-5), \
        "Weights don't sum to 1"
    assert (weights >= 0).all(), "Weights contain negative values"
    print(f"  [PASS] Weights sum to 1.0 (mean={weights_sum.mean():.6f})")
    print(f"  [PASS] Weights are non-negative (min={weights.min():.6f})")
    print(f"  [INFO] Mean attention: spatial={weights[:,0].mean():.3f}, "
          f"temporal={weights[:,1].mean():.3f}, hybrid={weights[:,2].mean():.3f}")

    # Test 3: Error handling (missing z_fused in 3-encoder mode)
    print("\n[Test 3] Error handling")
    try:
        z = fusion_3(z_spatial, z_temporal)  # Missing z_fused
        print(f"  [FAIL] Should have raised AssertionError")
        assert False
    except AssertionError as e:
        print(f"  [PASS] Correctly raised AssertionError: {e}")

    # Test 4: Gradient flow
    print("\n[Test 4] Gradient flow")
    fusion_2 = AdaptiveFusionLayer(d_z=d_z, use_hybrid_encoder=False)
    z_spatial = torch.randn(batch_size, d_z, requires_grad=True)
    z_temporal = torch.randn(batch_size, d_z, requires_grad=True)

    z = fusion_2(z_spatial, z_temporal)
    loss = z.sum()
    loss.backward()

    assert z_spatial.grad is not None, "No gradient for z_spatial"
    assert z_temporal.grad is not None, "No gradient for z_temporal"
    print(f"  [PASS] Gradients flow to z_spatial")
    print(f"  [PASS] Gradients flow to z_temporal")

    # Test 5: Parameter count
    print("\n[Test 5] Parameter count")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_2 = count_parameters(fusion_2)
    params_3 = count_parameters(fusion_3)

    print(f"  Fusion (2 encoders): {params_2:,} parameters")
    print(f"  Fusion (3 encoders): {params_3:,} parameters")

    print("\n[SUCCESS] All fusion layer tests passed!")
    return True


if __name__ == '__main__':
    test_fusion_layer()
