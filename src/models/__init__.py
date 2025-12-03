"""
Model architecture components for stock pattern classification.

This module contains the PyTorch model components for the UCL-TSC
(Unsupervised Contrastive Learning for Time Series Clustering) architecture.
"""

from .causal_conv import CausalConv1d
from .encoders import ResidualSpatialEncoder, TCNTemporalEncoder, CNNTCNHybridEncoder
from .fusion import AdaptiveFusionLayer
from .projection import ProjectionHead
from .ucl_tsc_model import MultiViewEncoder, UCLTSCModel

__all__ = [
    'CausalConv1d',
    'ResidualSpatialEncoder',
    'TCNTemporalEncoder',
    'CNNTCNHybridEncoder',
    'AdaptiveFusionLayer',
    'ProjectionHead',
    'MultiViewEncoder',
    'UCLTSCModel',
]
