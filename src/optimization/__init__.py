"""
Hyperparameter Optimization Module

Provides automated hyperparameter tuning using Optuna for both Stage 1
(contrastive learning) and Stage 2 (DEC clustering) training.
"""

from .config_manager import (
    save_hpo_config,
    load_hpo_config,
    list_saved_configs,
    get_default_param_space
)
from .trial_handler import TrialHandler
from .optimizer import HyperparameterOptimizer

__all__ = [
    'save_hpo_config',
    'load_hpo_config',
    'list_saved_configs',
    'get_default_param_space',
    'TrialHandler',
    'HyperparameterOptimizer',
]
