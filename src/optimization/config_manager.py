"""
HPO Configuration Manager

Handles saving, loading, and managing hyperparameter optimization configurations.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


def save_hpo_config(
    name: str,
    param_space: Dict[str, Any],
    strategy: str,
    metric: str,
    n_trials: int,
    stage: str = "combined"
) -> str:
    """
    Save HPO configuration to JSON.

    Args:
        name: Configuration name
        param_space: Dictionary of parameters and their search ranges
        strategy: 'grid' or 'bayesian'
        metric: Target metric ('silhouette', 'davies_bouldin', 'val_loss')
        n_trials: Number of trials for Bayesian optimization
        stage: Training stage ('stage1', 'stage2', or 'combined')

    Returns:
        Path to saved configuration file
    """
    config = {
        'param_space': param_space,
        'strategy': strategy,
        'metric': metric,
        'n_trials': n_trials,
        'stage': stage,
        'created_at': datetime.now().isoformat()
    }

    path = f"data/hpo_configs/{name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

    return path


def load_hpo_config(name: str) -> Dict[str, Any]:
    """
    Load HPO configuration from JSON.

    Args:
        name: Configuration name (without .json extension)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If configuration file doesn't exist
    """
    path = f"data/hpo_configs/{name}.json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration '{name}' not found at {path}")

    with open(path, 'r') as f:
        return json.load(f)


def list_saved_configs() -> List[str]:
    """
    Return list of saved configuration names.

    Returns:
        List of configuration names (without .json extension)
    """
    path = "data/hpo_configs/"

    if not os.path.exists(path):
        return []

    configs = [
        f.replace('.json', '')
        for f in os.listdir(path)
        if f.endswith('.json')
    ]

    return sorted(configs)


def delete_hpo_config(name: str) -> bool:
    """
    Delete an HPO configuration.

    Args:
        name: Configuration name (without .json extension)

    Returns:
        True if deleted successfully, False otherwise
    """
    path = f"data/hpo_configs/{name}.json"

    if os.path.exists(path):
        os.remove(path)
        return True

    return False


def get_default_param_space(stage: str = "combined") -> Dict[str, List[Any]]:
    """
    Get default parameter search space based on config.py.

    Args:
        stage: Training stage ('stage1', 'stage2', or 'combined')

    Returns:
        Dictionary of parameters and their suggested ranges
    """
    # Common parameters for both stages
    common_params = {
        'batch_size': [64, 128, 256, 512],
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
        'gradient_accumulation_steps': [1, 2, 4],
        'adam_beta1': [0.9, 0.95],
        'adam_beta2': [0.999, 0.9999],
        'adam_eps': [1e-8, 1e-7],
        'weight_decay': [0.0, 1e-5, 1e-4],
        'lr_warmup_epochs': [5, 10, 20],
    }

    # Stage 1 specific parameters
    stage1_params = {
        'temperature': [0.05, 0.07, 0.1, 0.2],  # NT-Xent temperature
        'd_z': [64, 128, 256],  # Projection/latent dimension
        'jitter_sigma': [0.005, 0.01, 0.02],  # Augmentation noise
        'scale_range_min': [0.85, 0.9, 0.95],
        'scale_range_max': [1.05, 1.1, 1.15],
    }

    # Stage 2 specific parameters
    stage2_params = {
        'num_clusters': [5, 8, 10, 12, 15],
        'lambda_start': [0.05, 0.1, 0.2],  # DEC loss weight
        'lambda_end': [0.5, 1.0, 2.0],
        'lambda_warmup_epochs': [5, 10, 15],
        'centroid_normalize_every_n_batches': [5, 10, 20],
    }

    if stage == 'stage1':
        return {**common_params, **stage1_params}
    elif stage == 'stage2':
        return {**common_params, **stage2_params}
    else:  # combined
        return {**common_params, **stage1_params, **stage2_params}


def get_param_space_summary(param_space: Dict[str, List[Any]]) -> str:
    """
    Generate a human-readable summary of parameter space.

    Args:
        param_space: Parameter space dictionary

    Returns:
        Formatted summary string
    """
    lines = ["Parameter Space Summary:"]
    lines.append("=" * 50)

    total_combinations = 1
    for param_name, values in sorted(param_space.items()):
        num_values = len(values)
        total_combinations *= num_values

        if isinstance(values[0], float):
            value_str = f"[{min(values):.2e} - {max(values):.2e}] ({num_values} values)"
        else:
            value_str = f"{values} ({num_values} values)"

        lines.append(f"  {param_name}: {value_str}")

    lines.append("=" * 50)
    lines.append(f"Total combinations (grid search): {total_combinations:,}")

    return "\n".join(lines)


def validate_param_space(param_space: Dict[str, List[Any]]) -> tuple[bool, Optional[str]]:
    """
    Validate parameter space configuration.

    Args:
        param_space: Parameter space dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not param_space:
        return False, "Parameter space is empty"

    for param_name, values in param_space.items():
        if not isinstance(values, list):
            return False, f"Parameter '{param_name}' must be a list of values"

        if len(values) == 0:
            return False, f"Parameter '{param_name}' has no values"

        # Check for consistent types
        first_type = type(values[0])
        if not all(isinstance(v, first_type) for v in values):
            return False, f"Parameter '{param_name}' has mixed types"

    return True, None


# Test function
def test_config_manager():
    """Test configuration manager functionality."""
    print("Testing Config Manager...")

    # Test 1: Get default param space
    print("\n[Test 1] Default parameter space")
    param_space = get_default_param_space('combined')
    print(f"  [PASS] Loaded {len(param_space)} parameters")

    # Test 2: Validate param space
    print("\n[Test 2] Validate parameter space")
    is_valid, error = validate_param_space(param_space)
    assert is_valid, f"Validation failed: {error}"
    print(f"  [PASS] Parameter space is valid")

    # Test 3: Save configuration
    print("\n[Test 3] Save configuration")
    test_name = "test_config_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_hpo_config(
        name=test_name,
        param_space=param_space,
        strategy='bayesian',
        metric='silhouette',
        n_trials=50
    )
    assert os.path.exists(path)
    print(f"  [PASS] Saved to {path}")

    # Test 4: List configurations
    print("\n[Test 4] List configurations")
    configs = list_saved_configs()
    assert test_name in configs
    print(f"  [PASS] Found {len(configs)} configurations")

    # Test 5: Load configuration
    print("\n[Test 5] Load configuration")
    loaded = load_hpo_config(test_name)
    assert loaded['strategy'] == 'bayesian'
    assert loaded['metric'] == 'silhouette'
    assert loaded['n_trials'] == 50
    print(f"  [PASS] Loaded configuration successfully")

    # Test 6: Delete configuration
    print("\n[Test 6] Delete configuration")
    deleted = delete_hpo_config(test_name)
    assert deleted
    assert test_name not in list_saved_configs()
    print(f"  [PASS] Deleted configuration")

    # Test 7: Parameter space summary
    print("\n[Test 7] Parameter space summary")
    summary = get_param_space_summary(param_space)
    print(summary)
    print(f"  [PASS] Generated summary")

    print("\n[SUCCESS] All config manager tests passed!")


if __name__ == '__main__':
    test_config_manager()
