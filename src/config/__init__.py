"""
Configuration system for UCL-TSC model.

This module provides dataclasses for organizing model, training, and data
hyperparameters in a structured way.
"""

from .config import ModelConfig, TrainingConfig, DataConfig, Config

__all__ = ['ModelConfig', 'TrainingConfig', 'DataConfig', 'Config']
