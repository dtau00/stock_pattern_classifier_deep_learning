"""
Evaluation and Metrics Module

This module provides comprehensive evaluation tools for the UCL-TSC model:
- Confidence scoring and calibration
- Clustering quality metrics
- Statistical independence tests
- Stability testing
- Experiment tracking

Reference: Design Document Section 4-7
"""

from .confidence_scoring import ConfidenceScorer
from .clustering_metrics import ClusteringMetrics
from .confidence_calibration import ConfidenceCalibrator
from .statistical_tests import StatisticalTests
from .evaluation_report import EvaluationReport
from .stage1_metrics import (
    evaluate_stage1,
    embedding_variance,
    effective_rank,
    alignment_uniformity,
    knn_accuracy
)

__all__ = [
    'ConfidenceScorer',
    'ClusteringMetrics',
    'ConfidenceCalibrator',
    'StatisticalTests',
    'EvaluationReport',
    'evaluate_stage1',
    'embedding_variance',
    'effective_rank',
    'alignment_uniformity',
    'knn_accuracy',
]
