"""
Causal 1D Convolution Module

This module implements 1D convolution with causal padding to prevent future
information leakage in time series modeling. This is critical for financial
time series where predictions must be based only on past and current data.

Key Features:
- Manual left-padding only (no right padding)
- Prevents future information leakage
- Compatible with dilated convolutions
- Maintains temporal causality for trading applications

Reference: Implementation Guide Section 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution with manual left-padding.

    Standard PyTorch Conv1d with default padding looks at both past and future
    time steps, violating causality. This module applies padding ONLY on the
    left side to ensure no future information leakage.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (filters)
        kernel_size (int): Size of the convolving kernel
        dilation (int): Spacing between kernel elements (default: 1)
        stride (int): Stride of the convolution (default: 1)
        groups (int): Number of blocked connections (default: 1)
        bias (bool): If True, adds a learnable bias (default: True)

    Shape:
        - Input: (batch_size, in_channels, sequence_length)
        - Output: (batch_size, out_channels, sequence_length)

    Example:
        >>> conv = CausalConv1d(in_channels=3, out_channels=64, kernel_size=5, dilation=1)
        >>> x = torch.randn(32, 3, 127)  # batch=32, channels=3, seq_len=127
        >>> out = conv(x)
        >>> out.shape
        torch.Size([32, 64, 127])

    Notes:
        - Left padding = (kernel_size - 1) * dilation
        - Output sequence length = input sequence length (same)
        - No future information is used at any timestep
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        # Calculate causal padding (left side only)
        self.padding = (kernel_size - 1) * dilation

        # Create standard Conv1d with NO padding (we handle it manually)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # CRITICAL: Must be 0 (we apply manual left padding)
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Store parameters for debugging
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding.

        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        # Apply left padding only using F.pad
        # F.pad format: (left, right) for last dimension
        x_padded = F.pad(x, (self.padding, 0))

        # Apply convolution
        return self.conv(x_padded)

    def get_receptive_field(self) -> int:
        """
        Calculate the receptive field of this layer.

        Returns:
            Receptive field in number of time steps
        """
        return 1 + (self.kernel_size - 1) * self.dilation

    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (
            f'CausalConv1d('
            f'in_channels={self.conv.in_channels}, '
            f'out_channels={self.conv.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'dilation={self.dilation}, '
            f'padding={self.padding}, '
            f'receptive_field={self.get_receptive_field()})'
        )


def test_causal_conv():
    """
    Unit test for CausalConv1d to verify:
    1. Output shape matches input shape
    2. No future information leakage (causality)
    3. Receptive field calculation is correct
    """
    print("Testing CausalConv1d...")

    # Test 1: Shape preservation
    print("\n[Test 1] Shape preservation")
    batch, channels, seq_len = 8, 3, 127
    conv = CausalConv1d(in_channels=3, out_channels=64, kernel_size=5, dilation=1)
    x = torch.randn(batch, channels, seq_len)
    out = conv(x)

    assert out.shape == (batch, 64, seq_len), f"Shape mismatch: {out.shape} != {(batch, 64, seq_len)}"
    print(f"  [PASS] Input: {x.shape} -> Output: {out.shape}")

    # Test 2: Causality (no future leakage)
    print("\n[Test 2] Causality verification")
    conv.eval()  # Set to eval mode for deterministic behavior

    with torch.no_grad():
        # Create two inputs that differ only at timestep t=T (last position)
        x1 = torch.randn(1, 3, 127)
        x2 = x1.clone()
        x2[:, :, -1] = torch.randn(1, 3)  # Modify only last timestep

        # Get outputs
        out1 = conv(x1)
        out2 = conv(x2)

        # Outputs at t=T-1 should be IDENTICAL (no future leakage)
        diff = (out1[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
        print(f"  Max diff at t=0..T-1: {diff:.2e}")

        if diff < 1e-6:
            print(f"  [PASS] No future information leakage detected")
        else:
            print(f"  [FAIL] Future information leakage detected!")
            raise AssertionError(f"Causality violated: diff={diff}")

    # Test 3: Receptive field
    print("\n[Test 3] Receptive field calculation")
    test_cases = [
        (5, 1, 5),      # kernel=5, dilation=1 -> RF=5
        (3, 1, 3),      # kernel=3, dilation=1 -> RF=3
        (3, 2, 5),      # kernel=3, dilation=2 -> RF=5
        (3, 4, 9),      # kernel=3, dilation=4 -> RF=9
        (3, 8, 17),     # kernel=3, dilation=8 -> RF=17
    ]

    for kernel, dilation, expected_rf in test_cases:
        conv = CausalConv1d(3, 64, kernel_size=kernel, dilation=dilation)
        rf = conv.get_receptive_field()
        assert rf == expected_rf, f"RF mismatch: {rf} != {expected_rf}"
        print(f"  [PASS] kernel={kernel}, dilation={dilation} -> RF={rf}")

    # Test 4: Dilated convolutions
    print("\n[Test 4] Dilated convolutions")
    dilations = [1, 2, 4, 8, 16, 32]
    total_rf = 1

    for dilation in dilations:
        conv = CausalConv1d(64, 64, kernel_size=3, dilation=dilation)
        x = torch.randn(1, 64, 127)
        out = conv(x)

        rf = conv.get_receptive_field()
        total_rf += (rf - 1)

        assert out.shape == (1, 64, 127), f"Dilated conv shape mismatch"
        print(f"  [PASS] dilation={dilation:2d} -> RF={rf:3d} (cumulative RF={total_rf})")

    print(f"\n  Final cumulative RF={total_rf} (should be 127 for TCN encoder)")

    print("\n[SUCCESS] All CausalConv1d tests passed!")
    return True


if __name__ == '__main__':
    test_causal_conv()
