"""
Training pipeline components for UCL-TSC model.

This module contains loss functions, trainers, and utilities for
two-stage training (contrastive pre-training + joint fine-tuning).
"""

from .losses import NTXentLoss, ClusteringLoss, LambdaSchedule
from .augmentation import TimeSeriesAugmentation
from .trainer import TwoStageTrainer

__all__ = [
    'NTXentLoss',
    'ClusteringLoss',
    'LambdaSchedule',
    'TimeSeriesAugmentation',
    'TwoStageTrainer',
]
