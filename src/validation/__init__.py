"""
Validation tools for UCL-TSC deep learning system.

This package provides critical validation tooling to ensure:
1. Model stability across training runs
2. Latent space quality (no collapse)
3. Correct architecture implementation
"""

__all__ = [
    'StabilityTester',
    'LatentSpaceVisualizer',
    'TrainingVisualizer',
    'ContrastiveMetrics',
    'AugmentationVerifier',
    'ClusterVisualizer',
    'CentroidMonitor',
]
