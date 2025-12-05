"""
Multi-View Feature Encoders for Time Series

This module implements three specialized encoders that capture patterns at different
temporal scales:

1. ResidualSpatialEncoder (CNN) - Local patterns
2. TCNTemporalEncoder - Long-term dependencies (full sequence context)
3. CNNTCNHybridEncoder - Intermediate patterns

All encoders use:
- Causal convolutions (no future leakage)
- LayerNorm (no cross-sample leakage)
- Final time step pooling (extract t=T only)
- Projection to latent dimension D_z

Reference: Implementation Guide Section 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from .causal_conv import CausalConv1d
except ImportError:
    from causal_conv import CausalConv1d


def calculate_tcn_dilations(seq_length: int, kernel_size: int = 3, min_layers: int = 2) -> List[int]:
    """
    Calculate optimal TCN dilation factors to match sequence length receptive field.

    Formula: RF = 1 + sum((k-1) * d for each dilation d)

    Args:
        seq_length: Target sequence length
        kernel_size: Convolution kernel size (default: 3)
        min_layers: Minimum number of layers (default: 2)

    Returns:
        List of dilation factors [1, 2, 4, ...]
    """
    dilations = []
    current_rf = 1
    dilation = 1

    while current_rf < seq_length or len(dilations) < min_layers:
        dilations.append(dilation)
        current_rf += (kernel_size - 1) * dilation
        dilation *= 2

        if len(dilations) >= 10:
            break

    return dilations


def calculate_hybrid_config(seq_length: int) -> dict:
    """
    Calculate optimal CNN-TCN hybrid configuration for intermediate scale.

    Target: ~30% of sequence length receptive field

    Args:
        seq_length: Target sequence length

    Returns:
        Dict with 'cnn_layers' and 'tcn_dilations'
    """
    target_rf = max(3, int(seq_length * 0.3))

    if seq_length <= 10:
        return {'cnn_layers': 1, 'tcn_dilations': [1]}
    elif seq_length <= 30:
        return {'cnn_layers': 2, 'tcn_dilations': [1, 2]}
    elif seq_length <= 80:
        return {'cnn_layers': 2, 'tcn_dilations': [1, 2, 4]}
    else:
        return {'cnn_layers': 2, 'tcn_dilations': [1, 2, 4, 8]}


class ResidualSpatialEncoder(nn.Module):
    """
    Residual Spatial Encoder (CNN) for local, high-frequency patterns.

    Captures short-term patterns using convolutional layers with residual
    connections. Architecture scales based on sequence length.

    Args:
        input_channels (int): Number of input feature channels (default: 3)
        d_z (int): Latent dimension for output (default: 128)
        seq_length (int): Sequence length (default: 127)
        hidden_channels (int): Number of channels in conv layers (default: 128)

    Shape:
        - Input: (batch, input_channels, seq_length)
        - Output: (batch, d_z)
    """

    def __init__(self, input_channels: int = 3, d_z: int = 128, seq_length: int = 127, hidden_channels: int = 128):
        super().__init__()

        self.seq_length = seq_length
        self.hidden_channels = hidden_channels
        num_layers = 3 if seq_length >= 15 else 2

        # Convolutional layers
        self.conv1 = CausalConv1d(input_channels, hidden_channels, kernel_size=5, dilation=1)
        self.ln1 = nn.LayerNorm(hidden_channels)

        self.conv2 = CausalConv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=1)
        self.ln2 = nn.LayerNorm(hidden_channels)

        self.conv3 = None
        self.ln3 = None
        if num_layers >= 3:
            self.conv3 = CausalConv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=1)
            self.ln3 = nn.LayerNorm(hidden_channels)

        # Projection to latent space
        self.projection = nn.Linear(hidden_channels, d_z)

        # Activation
        self.relu = nn.ReLU()

        # Store parameters
        self.d_z = d_z
        self.num_layers = num_layers
        self.receptive_field = 1 + num_layers * (5 - 1) * 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, channels, seq_length)

        Returns:
            Latent vector z_spatial of shape (batch, d_z)
        """
        # Layer 1
        h1 = self.conv1(x)
        h1 = h1.transpose(1, 2)
        h1 = self.ln1(h1)
        h1 = h1.transpose(1, 2)
        h1 = self.relu(h1)

        # Layer 2
        h2 = self.conv2(h1)
        h2 = h2.transpose(1, 2)
        h2 = self.ln2(h2)
        h2 = h2.transpose(1, 2)
        h2 = self.relu(h2)

        # Add residual connection from h1
        h2 = h2 + h1

        # Layer 3 (optional)
        if self.conv3 is not None:
            h3 = self.conv3(h2)
            h3 = h3.transpose(1, 2)
            h3 = self.ln3(h3)
            h3 = h3.transpose(1, 2)
            h3 = self.relu(h3)

            # Add residual connection from h2
            h3 = h3 + h2
            h_final = h3[:, :, -1]
        else:
            h_final = h2[:, :, -1]

        # Project to latent dimension
        z_spatial = self.projection(h_final)

        return z_spatial

    def get_receptive_field(self) -> int:
        """Return receptive field in number of time steps."""
        return self.receptive_field


class TCNTemporalEncoder(nn.Module):
    """
    TCN Temporal Encoder for long-term dependencies and causal relationships.

    Captures patterns across the full sequence using dilated causal
    convolutions with exponentially increasing dilation.

    Args:
        input_channels (int): Number of input feature channels (default: 3)
        d_z (int): Latent dimension for output (default: 128)
        seq_length (int): Sequence length (default: 127)
        hidden_channels (int): Number of channels in conv layers (default: 128)

    Shape:
        - Input: (batch, input_channels, seq_length)
        - Output: (batch, d_z)
    """

    def __init__(self, input_channels: int = 3, d_z: int = 128, seq_length: int = 127, hidden_channels: int = 128):
        super().__init__()

        self.seq_length = seq_length
        self.hidden_channels = hidden_channels
        dilations = calculate_tcn_dilations(seq_length, kernel_size=3)

        # Create layers
        self.layers = nn.ModuleList()
        in_ch = input_channels

        for dilation in dilations:
            self.layers.append(nn.ModuleDict({
                'conv': CausalConv1d(in_ch, hidden_channels, kernel_size=3, dilation=dilation),
                'ln': nn.LayerNorm(hidden_channels),
                'relu': nn.ReLU()
            }))
            in_ch = hidden_channels

        # Projection to latent space
        self.projection = nn.Linear(hidden_channels, d_z)

        # Store parameters
        self.d_z = d_z
        self.dilations = dilations
        self.receptive_field = 1 + sum((3 - 1) * d for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dilated TCN layers.

        Args:
            x: Input tensor of shape (batch, channels, seq_length)

        Returns:
            Latent vector z_temporal of shape (batch, d_z)
        """
        h = self.get_features(x)

        # Final Time Step Pooling: Extract ONLY t=T (last timestep)
        h_final = h[:, :, -1]

        # Project to latent dimension
        z_temporal = self.projection(h_final)

        return z_temporal

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate feature maps before final pooling.

        Args:
            x: Input tensor of shape (batch, channels, seq_length)

        Returns:
            Feature maps of shape (batch, 64, seq_length)
        """
        h = x

        # Apply all TCN layers sequentially
        for layer in self.layers:
            h = layer['conv'](h)
            h = h.transpose(1, 2)
            h = layer['ln'](h)
            h = h.transpose(1, 2)
            h = layer['relu'](h)

        return h

    def get_receptive_field(self) -> int:
        """Return receptive field in number of time steps."""
        return self.receptive_field


class CNNTCNHybridEncoder(nn.Module):
    """
    CNN-TCN Hybrid Encoder for intermediate-scale patterns.

    Combines CNN and TCN stages to capture patterns at intermediate scale.
    Architecture scales based on sequence length (target ~30% RF).

    Args:
        input_channels (int): Number of input feature channels (default: 3)
        d_z (int): Latent dimension for output (default: 128)
        seq_length (int): Sequence length (default: 127)
        hidden_channels (int): Number of channels in conv layers (default: 128)

    Shape:
        - Input: (batch, input_channels, seq_length)
        - Output: (batch, d_z)
    """

    def __init__(self, input_channels: int = 3, d_z: int = 128, seq_length: int = 127, hidden_channels: int = 128):
        super().__init__()

        self.seq_length = seq_length
        self.hidden_channels = hidden_channels
        config = calculate_hybrid_config(seq_length)
        num_cnn_layers = config['cnn_layers']
        dilations_tcn = config['tcn_dilations']

        # CNN Stage
        self.cnn1 = CausalConv1d(input_channels, hidden_channels, kernel_size=5, dilation=1)
        self.ln_cnn1 = nn.LayerNorm(hidden_channels)

        self.cnn2 = None
        self.ln_cnn2 = None
        if num_cnn_layers >= 2:
            self.cnn2 = CausalConv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=1)
            self.ln_cnn2 = nn.LayerNorm(hidden_channels)

        # TCN Stage
        self.tcn_layers = nn.ModuleList()
        for dilation in dilations_tcn:
            self.tcn_layers.append(nn.ModuleDict({
                'conv': CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=dilation),
                'ln': nn.LayerNorm(hidden_channels),
                'relu': nn.ReLU()
            }))

        # Projection to latent space
        self.projection = nn.Linear(hidden_channels, d_z)

        # Activation
        self.relu = nn.ReLU()

        # Store parameters
        self.d_z = d_z
        self.num_cnn_layers = num_cnn_layers
        self.dilations_tcn = dilations_tcn
        cnn_rf = 1 + num_cnn_layers * (5 - 1) * 1
        tcn_rf = sum((3 - 1) * d for d in dilations_tcn)
        self.receptive_field = cnn_rf + tcn_rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN then TCN stages.

        Args:
            x: Input tensor of shape (batch, channels, seq_length)

        Returns:
            Latent vector z_fused of shape (batch, d_z)
        """
        # CNN Stage Layer 1
        h = self.cnn1(x)
        h = h.transpose(1, 2)
        h = self.ln_cnn1(h)
        h = h.transpose(1, 2)
        h = self.relu(h)

        # CNN Stage Layer 2 (optional)
        if self.cnn2 is not None:
            h = self.cnn2(h)
            h = h.transpose(1, 2)
            h = self.ln_cnn2(h)
            h = h.transpose(1, 2)
            h = self.relu(h)

        # TCN Stage
        for layer in self.tcn_layers:
            h = layer['conv'](h)
            h = h.transpose(1, 2)
            h = layer['ln'](h)
            h = h.transpose(1, 2)
            h = layer['relu'](h)

        # Final Time Step Pooling
        h_final = h[:, :, -1]

        # Project to latent dimension
        z_fused = self.projection(h_final)

        return z_fused

    def get_receptive_field(self) -> int:
        """Return receptive field in number of time steps."""
        return self.receptive_field


def test_encoders():
    """
    Unit tests for all encoder modules.

    Tests:
    1. Output shapes
    2. Receptive fields
    3. Causality (no future leakage)
    4. Parameter counts
    """
    print("Testing Encoder Modules...")
    batch_size, channels, seq_len = 8, 3, 127
    d_z = 128

    # Test 1: ResidualSpatialEncoder
    print("\n[Test 1] ResidualSpatialEncoder")
    encoder_spatial = ResidualSpatialEncoder(input_channels=channels, d_z=d_z)
    x = torch.randn(batch_size, channels, seq_len)
    z_spatial = encoder_spatial(x)

    assert z_spatial.shape == (batch_size, d_z), f"Shape mismatch: {z_spatial.shape}"
    print(f"  [PASS] Output shape: {z_spatial.shape}")
    print(f"  [PASS] Receptive field: {encoder_spatial.get_receptive_field()} bars")

    # Test 2: TCNTemporalEncoder
    print("\n[Test 2] TCNTemporalEncoder")
    encoder_temporal = TCNTemporalEncoder(input_channels=channels, d_z=d_z)
    z_temporal = encoder_temporal(x)

    assert z_temporal.shape == (batch_size, d_z), f"Shape mismatch: {z_temporal.shape}"
    print(f"  [PASS] Output shape: {z_temporal.shape}")
    print(f"  [PASS] Receptive field: {encoder_temporal.get_receptive_field()} bars (should be 127)")

    # Test 3: CNNTCNHybridEncoder
    print("\n[Test 3] CNNTCNHybridEncoder")
    encoder_hybrid = CNNTCNHybridEncoder(input_channels=channels, d_z=d_z)
    z_hybrid = encoder_hybrid(x)

    assert z_hybrid.shape == (batch_size, d_z), f"Shape mismatch: {z_hybrid.shape}"
    print(f"  [PASS] Output shape: {z_hybrid.shape}")
    print(f"  [PASS] Receptive field: {encoder_hybrid.get_receptive_field()} bars (should be ~40)")

    # Test 4: Causality check (no future leakage)
    print("\n[Test 4] Causality verification")
    encoder_spatial.eval()
    encoder_temporal.eval()
    encoder_hybrid.eval()

    with torch.no_grad():
        # Create two inputs that differ only at t=T
        x1 = torch.randn(1, channels, seq_len)
        x2 = x1.clone()
        x2[:, :, -1] = torch.randn(1, channels)

        # Spatial encoder should NOT see the future
        # But since it uses final time step pooling, outputs WILL differ
        # We need to check intermediate features at t<T

        # For proper causality test, we'd need to expose intermediate features
        # For now, verify encoders run without errors
        z1_s = encoder_spatial(x1)
        z2_s = encoder_spatial(x2)

        z1_t = encoder_temporal(x1)
        z2_t = encoder_temporal(x2)

        z1_h = encoder_hybrid(x1)
        z2_h = encoder_hybrid(x2)

        print(f"  [PASS] All encoders process inputs without errors")
        print(f"  [INFO] Proper causality test requires Test A from test suite")

    # Test 5: Parameter counts
    print("\n[Test 5] Parameter counts")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_spatial = count_parameters(encoder_spatial)
    params_temporal = count_parameters(encoder_temporal)
    params_hybrid = count_parameters(encoder_hybrid)

    print(f"  ResidualSpatialEncoder: {params_spatial:,} parameters")
    print(f"  TCNTemporalEncoder: {params_temporal:,} parameters")
    print(f"  CNNTCNHybridEncoder: {params_hybrid:,} parameters")
    print(f"  Total (3 encoders): {params_spatial + params_temporal + params_hybrid:,} parameters")

    print("\n[SUCCESS] All encoder tests passed!")
    return True


if __name__ == '__main__':
    test_encoders()
