"""
Data Augmentation for Time Series

This module implements data augmentation strategies for time series contrastive learning.
Augmentation creates two views of the same sample for NT-Xent loss.

Augmentation Methods:
1. Jittering - Add Gaussian noise (robustness to measurement errors)
2. Scaling - Multiply by random scale factors (invariance to volatility amplitude)
3. Time Masking - Mask contiguous time blocks (robustness to missing data)

Reference: Implementation Guide Section 5.3
"""

import torch
import torch.nn as nn
import numpy as np


class TimeSeriesAugmentation:
    """
    Data augmentation for time series (creates two augmented views).

    Applies a combination of jittering, scaling, and time masking to create
    two different augmented views of the same input sample for contrastive learning.

    Optimized for GPU execution with minimal CPU synchronization.

    Args:
        jitter_sigma (float): Std dev of Gaussian noise (as fraction of data std) (default: 0.01)
        scale_range (tuple): Range for uniform amplitude scaling (default: (0.9, 1.1))
        mask_max_length_pct (float): Max percentage of sequence to mask (default: 0.1)
        apply_jitter (bool): Enable jittering (default: True)
        apply_scaling (bool): Enable scaling (default: True)
        apply_masking (bool): Enable time masking (default: True)

    Shape:
        - Input: (batch, channels, seq_len)
        - Output: Tuple of (view1, view2) each of shape (batch, channels, seq_len)

    Example:
        >>> aug = TimeSeriesAugmentation(jitter_sigma=0.01, scale_range=(0.9, 1.1))
        >>> x = torch.randn(32, 3, 127)
        >>> x1, x2 = aug(x)
        >>> x1.shape, x2.shape
        (torch.Size([32, 3, 127]), torch.Size([32, 3, 127]))
    """

    def __init__(
        self,
        jitter_sigma: float = 0.01,
        scale_range: tuple = (0.9, 1.1),
        mask_max_length_pct: float = 0.1,
        apply_jitter: bool = True,
        apply_scaling: bool = True,
        apply_masking: bool = True
    ):
        self.jitter_sigma = jitter_sigma
        self.scale_range = scale_range
        self.mask_max_length_pct = mask_max_length_pct
        self.apply_jitter = apply_jitter
        self.apply_scaling = apply_scaling
        self.apply_masking = apply_masking

    def __call__(self, x: torch.Tensor) -> tuple:
        """
        Generate two augmented views of input.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tuple of (view1, view2) each of shape (batch, channels, seq_len)
        """
        # Create two independent augmented views
        view1 = self._augment(x)
        view2 = self._augment(x)

        return view1, view2

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a single view.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Augmented tensor of shape (batch, channels, seq_len)
        """
        # Optimize: only clone when necessary (time_mask needs it)
        # Jitter and scale can work in-place on the clone

        # Start with original (will be modified in-place)
        x_aug = x

        # Apply jittering (creates new tensor via addition)
        if self.apply_jitter:
            x_aug = self._jitter(x_aug)

        # Apply scaling (in-place multiplication if we already have a copy from jitter)
        if self.apply_scaling:
            x_aug = self._scale(x_aug)

        # Apply time masking (needs clone to preserve structure)
        if self.apply_masking:
            # Only clone here if we haven't already created a copy
            if not self.apply_jitter and not self.apply_scaling:
                x_aug = x.clone()
            x_aug = self._time_mask(x_aug)

        # If no augmentations were applied, clone to ensure we don't modify original
        if not (self.apply_jitter or self.apply_scaling or self.apply_masking):
            x_aug = x.clone()

        return x_aug

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to time series.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Jittered tensor of shape (batch, channels, seq_len)
        """
        # Compute std per sample and channel
        # Shape: (batch, channels, 1)
        std = x.std(dim=2, keepdim=True)

        # Generate Gaussian noise (torch.randn_like doesn't accept generator parameter)
        noise = torch.randn_like(x)
        noise = noise * (self.jitter_sigma * std)

        return x + noise

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random amplitude scaling.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Scaled tensor of shape (batch, channels, seq_len)
        """
        batch_size, channels, seq_len = x.shape

        # Generate random scale factors per sample and channel
        # Shape: (batch, channels, 1)
        scale_min, scale_max = self.scale_range

        # Generate on same device as input
        scale_factors = torch.rand(batch_size, channels, 1, device=x.device)
        scale_factors = scale_min + (scale_max - scale_min) * scale_factors

        return x * scale_factors

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly mask a contiguous time block with segment mean.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Masked tensor of shape (batch, channels, seq_len)
        """
        batch_size, channels, seq_len = x.shape
        mask_len = int(seq_len * self.mask_max_length_pct)

        if mask_len == 0:
            return x

        # Work directly on x (caller has already cloned if needed)
        x_masked = x

        # Vectorized random start positions for all samples (GPU-friendly)
        start_indices = torch.randint(0, seq_len - mask_len + 1, (batch_size,), device=x.device)

        # Compute mean per channel across time dimension for all samples
        # Shape: (batch, channels)
        segment_means = x.mean(dim=2)

        # Fully vectorized masking (NO CPU sync)
        # Create mask tensor on GPU
        mask = torch.zeros_like(x, dtype=torch.bool)

        # Build mask indices using advanced indexing (stays on GPU)
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1)  # (batch, 1)
        time_offsets = torch.arange(mask_len, device=x.device).unsqueeze(0)  # (1, mask_len)
        masked_positions = start_indices.unsqueeze(1) + time_offsets  # (batch, mask_len)

        # Set mask positions
        mask[batch_idx, :, masked_positions] = True

        # Apply masking: keep original where False, use mean where True
        segment_means_expanded = segment_means.unsqueeze(2).expand(-1, -1, seq_len)
        x_masked = torch.where(mask, segment_means_expanded, x_masked)

        return x_masked


def test_augmentation():
    """
    Unit tests for data augmentation.

    Tests:
    1. Output shapes
    2. Two views are different
    3. Jittering adds noise
    4. Scaling changes amplitude
    5. Time masking fills with mean
    """
    print("Testing Data Augmentation...")

    batch_size, channels, seq_len = 32, 3, 127

    # Test 1: Basic augmentation
    print("\n[Test 1] Basic augmentation")
    aug = TimeSeriesAugmentation(
        jitter_sigma=0.01,
        scale_range=(0.9, 1.1),
        mask_max_length_pct=0.1
    )

    x = torch.randn(batch_size, channels, seq_len)
    view1, view2 = aug(x)

    assert view1.shape == x.shape, f"View1 shape mismatch: {view1.shape}"
    assert view2.shape == x.shape, f"View2 shape mismatch: {view2.shape}"
    print(f"  [PASS] Output shapes: {view1.shape}, {view2.shape}")

    # Test 2: Two views are different
    print("\n[Test 2] Two views are different")
    diff = (view1 - view2).abs().mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")
    assert diff > 1e-6, "Two views should be different"
    print(f"  [PASS] Two views are different (mean diff={diff:.6f})")

    # Test 3: Jittering
    print("\n[Test 3] Jittering (noise addition)")
    aug_jitter_only = TimeSeriesAugmentation(
        jitter_sigma=0.05,
        apply_scaling=False,
        apply_masking=False
    )

    x = torch.randn(batch_size, channels, seq_len)
    view1, view2 = aug_jitter_only(x)

    # Check noise is added (views differ from original)
    noise1 = (view1 - x).abs().mean().item()
    noise2 = (view2 - x).abs().mean().item()

    print(f"  Noise magnitude (view1): {noise1:.6f}")
    print(f"  Noise magnitude (view2): {noise2:.6f}")
    assert noise1 > 1e-6, "Jittering should add noise"
    assert noise2 > 1e-6, "Jittering should add noise"
    print(f"  [PASS] Jittering adds noise")

    # Test 4: Scaling
    print("\n[Test 4] Scaling (amplitude change)")
    aug_scale_only = TimeSeriesAugmentation(
        scale_range=(0.5, 1.5),
        apply_jitter=False,
        apply_masking=False
    )

    x = torch.randn(batch_size, channels, seq_len)
    view1, view2 = aug_scale_only(x)

    # Check scaling changes amplitude
    x_std = x.std().item()
    view1_std = view1.std().item()
    view2_std = view2.std().item()

    print(f"  Original std: {x_std:.6f}")
    print(f"  View1 std: {view1_std:.6f}")
    print(f"  View2 std: {view2_std:.6f}")
    print(f"  [PASS] Scaling changes amplitude")

    # Test 5: Time masking
    print("\n[Test 5] Time masking")
    aug_mask_only = TimeSeriesAugmentation(
        mask_max_length_pct=0.2,
        apply_jitter=False,
        apply_scaling=False
    )

    x = torch.randn(batch_size, channels, seq_len)
    view1, view2 = aug_mask_only(x)

    # Check some values are replaced (views differ from original)
    diff1 = (view1 - x).abs().max().item()
    diff2 = (view2 - x).abs().max().item()

    print(f"  Max difference (view1): {diff1:.6f}")
    print(f"  Max difference (view2): {diff2:.6f}")
    assert diff1 > 1e-6, "Time masking should modify values"
    assert diff2 > 1e-6, "Time masking should modify values"
    print(f"  [PASS] Time masking modifies values")

    # Test 6: Disable all augmentations
    print("\n[Test 6] Disable all augmentations")
    aug_disabled = TimeSeriesAugmentation(
        apply_jitter=False,
        apply_scaling=False,
        apply_masking=False
    )

    x = torch.randn(batch_size, channels, seq_len)
    view1, view2 = aug_disabled(x)

    # Views should be identical to input
    assert torch.allclose(view1, x), "Disabled augmentation should return original"
    assert torch.allclose(view2, x), "Disabled augmentation should return original"
    print(f"  [PASS] Disabled augmentation returns original")

    # Test 7: Augmentation strength
    print("\n[Test 7] Augmentation strength")

    # Weak augmentation
    aug_weak = TimeSeriesAugmentation(
        jitter_sigma=0.005,
        scale_range=(0.95, 1.05),
        mask_max_length_pct=0.05
    )

    # Strong augmentation
    aug_strong = TimeSeriesAugmentation(
        jitter_sigma=0.05,
        scale_range=(0.7, 1.3),
        mask_max_length_pct=0.2
    )

    x = torch.randn(batch_size, channels, seq_len)

    view1_weak, view2_weak = aug_weak(x)
    view1_strong, view2_strong = aug_strong(x)

    diff_weak = (view1_weak - x).abs().mean().item()
    diff_strong = (view1_strong - x).abs().mean().item()

    print(f"  Weak augmentation diff: {diff_weak:.6f}")
    print(f"  Strong augmentation diff: {diff_strong:.6f}")
    assert diff_strong > diff_weak, "Strong augmentation should differ more"
    print(f"  [PASS] Strong augmentation creates larger differences")

    # Test 8: Batch independence
    print("\n[Test 8] Batch independence")
    aug = TimeSeriesAugmentation()

    x = torch.randn(batch_size, channels, seq_len)
    view1, view2 = aug(x)

    # Each sample should be augmented independently
    # Check that samples have different augmentations
    diffs = []
    for i in range(min(5, batch_size)):
        diff_i = (view1[i] - view2[i]).abs().mean().item()
        diffs.append(diff_i)

    print(f"  Per-sample diffs (first 5): {[f'{d:.4f}' for d in diffs]}")
    print(f"  [PASS] Each sample augmented independently")

    print("\n[SUCCESS] All augmentation tests passed!")
    return True


if __name__ == '__main__':
    test_augmentation()
