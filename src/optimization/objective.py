"""
Objective Functions for Hyperparameter Optimization

Defines Optuna objective functions for Stage 1, Stage 2, and combined training.
"""

import os
import torch
import numpy as np
import h5py
from typing import Dict, Any, Optional
import optuna
from pathlib import Path

try:
    from ..models.ucl_tsc_model import UCLTSCModel
    from ..config.config import Config, ModelConfig, TrainingConfig, AugmentationConfig
    from ..training.trainer import TwoStageTrainer
    from .trial_handler import TrialHandler
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.ucl_tsc_model import UCLTSCModel
    from config.config import Config, ModelConfig, TrainingConfig, AugmentationConfig
    from training.trainer import TwoStageTrainer
    from optimization.trial_handler import TrialHandler


def load_data_from_hdf5(
    data_path: str,
    subset_fraction: float = 1.0
) -> tuple:
    """
    Load preprocessed data from HDF5 file.

    Args:
        data_path: Path to HDF5 file
        subset_fraction: Fraction of data to use (0.0-1.0)

    Returns:
        Tuple of (train_data, val_data, test_data) as numpy arrays
    """
    with h5py.File(data_path, 'r') as f:
        # Load data
        train_data = f['train'][:]
        val_data = f['val'][:]
        test_data = f['test'][:]

    # Subsample if requested
    if 0.0 < subset_fraction < 1.0:
        n_train = int(len(train_data) * subset_fraction)
        n_val = int(len(val_data) * subset_fraction)
        n_test = int(len(test_data) * subset_fraction)

        train_data = train_data[:n_train]
        val_data = val_data[:n_val]
        test_data = test_data[:n_test]

    return train_data, val_data, test_data


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    preload_to_gpu: bool = False
):
    """
    Create PyTorch DataLoaders from numpy arrays.

    Args:
        train_data: Training data (N, C, T)
        val_data: Validation data (N, C, T)
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        pin_memory: Enable pin_memory
        preload_to_gpu: Preload entire dataset to GPU

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Convert to tensors
    train_tensor = torch.from_numpy(train_data).float()
    val_tensor = torch.from_numpy(val_data).float()

    # Preload to GPU if requested and possible
    if preload_to_gpu and torch.cuda.is_available():
        try:
            train_tensor = train_tensor.cuda()
            val_tensor = val_tensor.cuda()
            pin_memory = False  # Not needed if data on GPU
            num_workers = 0  # Must be 0 for CUDA tensors
        except RuntimeError:
            # If OOM, keep on CPU
            pass

    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def compute_clustering_metrics(model, data_loader, device):
    """
    Compute clustering quality metrics.

    Args:
        model: Trained model
        data_loader: DataLoader with data
        device: Device to use

    Returns:
        Dictionary with silhouette_score and davies_bouldin_index
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    model.eval()
    embeddings = []
    cluster_assignments = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)

            # Get embeddings and cluster assignments
            z = model.get_embeddings(x)
            q = model.cluster_head(z)
            clusters = torch.argmax(q, dim=1)

            embeddings.append(z.cpu().numpy())
            cluster_assignments.append(clusters.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    cluster_assignments = np.concatenate(cluster_assignments, axis=0)

    # Check if we have more than one cluster
    n_unique_clusters = len(np.unique(cluster_assignments))
    if n_unique_clusters < 2:
        return {
            'silhouette_score': -1.0,
            'davies_bouldin_index': float('inf')
        }

    # Compute metrics
    silhouette = silhouette_score(embeddings, cluster_assignments)
    davies_bouldin = davies_bouldin_score(embeddings, cluster_assignments)

    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin
    }


def stage1_objective(
    trial: optuna.Trial,
    data_path: str,
    param_space: Dict[str, Any],
    metric: str,
    trial_handler: TrialHandler,
    subset_fraction: float = 0.3,
    max_epochs: int = 30
) -> float:
    """
    Objective function for Stage 1 (contrastive learning) optimization.

    Args:
        trial: Optuna trial object
        data_path: Path to HDF5 data file
        param_space: Parameter search space
        metric: Target metric ('val_loss')
        trial_handler: Trial handler for OOM/early stopping
        subset_fraction: Fraction of data to use (for speed)
        max_epochs: Maximum training epochs

    Returns:
        Metric value to optimize
    """
    try:
        # Suggest parameters
        params = {}
        for param_name, values in param_space.items():
            if param_name in ['batch_size', 'gradient_accumulation_steps', 'd_z',
                             'lr_warmup_epochs', 'num_workers', 'encoder_hidden_channels',
                             'projection_hidden_dim', 'fusion_hidden_dim']:
                params[param_name] = trial.suggest_categorical(param_name, values)
            elif param_name in ['learning_rate', 'temperature', 'jitter_sigma',
                               'adam_beta1', 'adam_beta2', 'adam_eps', 'weight_decay']:
                params[param_name] = trial.suggest_categorical(param_name, values)
            elif param_name == 'scale_range_min':
                params['scale_range_min'] = trial.suggest_categorical(param_name, values)
            elif param_name == 'scale_range_max':
                params['scale_range_max'] = trial.suggest_categorical(param_name, values)
            elif param_name == 'use_projection_bottleneck':
                params[param_name] = trial.suggest_categorical(param_name, values)

        trial_handler.log_trial_start(trial.number, params)

        # Create configuration
        model_config = ModelConfig(
            d_z=params.get('d_z', 128),
            tau=params.get('temperature', 0.1),
            encoder_hidden_channels=params.get('encoder_hidden_channels', 128),
            projection_hidden_dim=params.get('projection_hidden_dim', 512),
            fusion_hidden_dim=params.get('fusion_hidden_dim', 256),
            use_projection_bottleneck=params.get('use_projection_bottleneck', False)
        )

        training_config = TrainingConfig(
            batch_size=params.get('batch_size', 256),
            learning_rate=params.get('learning_rate', 1e-3),
            max_epochs_stage1=max_epochs,
            max_epochs_stage2=0,  # Skip Stage 2
            gradient_accumulation_steps_stage1=params.get('gradient_accumulation_steps', 1),
            lr_warmup_epochs=params.get('lr_warmup_epochs', 5),
            num_workers=params.get('num_workers', 0),
            use_mixed_precision=True,
            preload_to_gpu=False,  # Keep on CPU to avoid OOM
            early_stopping_patience=10
        )

        augmentation_config = AugmentationConfig(
            jitter_sigma=params.get('jitter_sigma', 0.01),
            scale_range=(
                params.get('scale_range_min', 0.9),
                params.get('scale_range_max', 1.1)
            )
        )

        config = Config(
            model=model_config,
            training=training_config,
            augmentation=augmentation_config
        )

        # Load data
        train_data, val_data, _ = load_data_from_hdf5(data_path, subset_fraction)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_data, val_data,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            preload_to_gpu=config.training.preload_to_gpu
        )

        # Create model
        model = UCLTSCModel(
            input_channels=3,
            d_z=model_config.d_z,
            num_clusters=8,  # Fixed for Stage 1
            use_hybrid_encoder=False,
            seq_length=127,
            encoder_hidden_channels=model_config.encoder_hidden_channels,
            projection_hidden_dim=model_config.projection_hidden_dim,
            fusion_hidden_dim=model_config.fusion_hidden_dim,
            use_projection_bottleneck=model_config.use_projection_bottleneck
        )

        # Create trainer
        trainer = TwoStageTrainer(model, config)

        # Train Stage 1 only
        history = trainer._train_stage1(train_loader, val_loader, callback=None)

        # Get final validation loss
        val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')

        # Check for early stopping
        if trial_handler.should_prune(trial.number, val_loss, metric):
            trial_handler.log_trial_pruned(trial.number, f"{metric}={val_loss:.4f}")
            raise optuna.TrialPruned()

        trial_handler.log_trial_complete(trial.number, val_loss, metric)

        # Clean up
        del model, trainer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return val_loss

    except torch.cuda.OutOfMemoryError:
        trial_handler.handle_oom(trial.number)
        return float('inf')  # Worst score for minimization

    except Exception as e:
        trial_handler.log_trial_failed(trial.number, str(e))
        raise


def stage2_objective(
    trial: optuna.Trial,
    data_path: str,
    param_space: Dict[str, Any],
    metric: str,
    trial_handler: TrialHandler,
    stage1_checkpoint: str,
    subset_fraction: float = 0.3,
    max_epochs: int = 30
) -> float:
    """
    Objective function for Stage 2 (DEC clustering) optimization.

    Args:
        trial: Optuna trial object
        data_path: Path to HDF5 data file
        param_space: Parameter search space
        metric: Target metric ('silhouette' or 'davies_bouldin')
        trial_handler: Trial handler for OOM/early stopping
        stage1_checkpoint: Path to Stage 1 checkpoint
        subset_fraction: Fraction of data to use
        max_epochs: Maximum training epochs

    Returns:
        Metric value to optimize
    """
    try:
        # Suggest parameters
        params = {}
        for param_name, values in param_space.items():
            if param_name in ['batch_size', 'gradient_accumulation_steps', 'num_clusters',
                             'lambda_warmup_epochs', 'centroid_normalize_every_n_batches']:
                params[param_name] = trial.suggest_categorical(param_name, values)
            elif param_name in ['learning_rate', 'lambda_start', 'lambda_end',
                               'adam_beta1', 'adam_beta2', 'adam_eps', 'weight_decay']:
                params[param_name] = trial.suggest_categorical(param_name, values)

        trial_handler.log_trial_start(trial.number, params)

        # Load Stage 1 checkpoint to get architecture parameters
        checkpoint = torch.load(stage1_checkpoint)
        checkpoint_config = checkpoint.get('config', {})
        d_z = checkpoint_config.get('d_z', 128)

        # Get architecture parameters from checkpoint with backward compatibility
        if hasattr(checkpoint_config, 'model'):
            encoder_hidden_channels = getattr(checkpoint_config.model, 'encoder_hidden_channels', 128)
            projection_hidden_dim = getattr(checkpoint_config.model, 'projection_hidden_dim', 512)
            fusion_hidden_dim = getattr(checkpoint_config.model, 'fusion_hidden_dim', 256)
            use_projection_bottleneck = getattr(checkpoint_config.model, 'use_projection_bottleneck', False)
        else:
            encoder_hidden_channels = 128
            projection_hidden_dim = 512
            fusion_hidden_dim = 256
            use_projection_bottleneck = False

        # Create configuration
        model_config = ModelConfig(
            d_z=d_z,
            num_clusters=params.get('num_clusters', 8),
            encoder_hidden_channels=encoder_hidden_channels,
            projection_hidden_dim=projection_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            use_projection_bottleneck=use_projection_bottleneck
        )

        training_config = TrainingConfig(
            batch_size=params.get('batch_size', 256),
            learning_rate=params.get('learning_rate', 1e-4),
            max_epochs_stage1=0,  # Already done
            max_epochs_stage2=max_epochs,
            gradient_accumulation_steps_stage2=params.get('gradient_accumulation_steps', 1),
            lambda_start=params.get('lambda_start', 0.1),
            lambda_end=params.get('lambda_end', 1.0),
            lambda_warmup_epochs=params.get('lambda_warmup_epochs', 10),
            centroid_normalize_every_n_batches=params.get('centroid_normalize_every_n_batches', 10),
            num_workers=0,
            preload_to_gpu=False
        )

        config = Config(
            model=model_config,
            training=training_config
        )

        # Load data
        train_data, val_data, _ = load_data_from_hdf5(data_path, subset_fraction)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_data, val_data,
            batch_size=config.training.batch_size
        )

        # Create model and load Stage 1 weights
        model = UCLTSCModel(
            input_channels=3,
            d_z=d_z,
            num_clusters=model_config.num_clusters,
            seq_length=127,
            encoder_hidden_channels=model_config.encoder_hidden_channels,
            projection_hidden_dim=model_config.projection_hidden_dim,
            fusion_hidden_dim=model_config.fusion_hidden_dim,
            use_projection_bottleneck=model_config.use_projection_bottleneck
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Create trainer
        trainer = TwoStageTrainer(model, config)

        # Initialize centroids
        trainer._initialize_centroids(train_loader)

        # Train Stage 2
        history = trainer._train_stage2(train_loader, val_loader, callback=None)

        # Compute clustering metrics
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        metrics = compute_clustering_metrics(model, val_loader, device)

        # Get metric value
        if metric == 'silhouette':
            metric_value = metrics['silhouette_score']
            # For maximization, return negative
            result = -metric_value
        else:  # davies_bouldin
            metric_value = metrics['davies_bouldin_index']
            result = metric_value

        # Check for early stopping
        if trial_handler.should_prune(trial.number, metric_value, metric):
            trial_handler.log_trial_pruned(trial.number, f"{metric}={metric_value:.4f}")
            raise optuna.TrialPruned()

        trial_handler.log_trial_complete(trial.number, metric_value, metric)

        # Clean up
        del model, trainer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except torch.cuda.OutOfMemoryError:
        trial_handler.handle_oom(trial.number)
        return float('inf') if metric == 'davies_bouldin' else float('-inf')

    except Exception as e:
        trial_handler.log_trial_failed(trial.number, str(e))
        raise


def combined_objective(
    trial: optuna.Trial,
    data_path: str,
    param_space: Dict[str, Any],
    metric: str,
    trial_handler: TrialHandler,
    subset_fraction: float = 0.3,
    stage1_epochs: int = 30,
    stage2_epochs: int = 30
) -> float:
    """
    Objective function for combined Stage 1 + Stage 2 optimization.

    Args:
        trial: Optuna trial object
        data_path: Path to HDF5 data file
        param_space: Parameter search space
        metric: Target metric
        trial_handler: Trial handler
        subset_fraction: Fraction of data to use
        stage1_epochs: Stage 1 epochs
        stage2_epochs: Stage 2 epochs

    Returns:
        Metric value to optimize
    """
    try:
        # Suggest parameters
        params = {}
        for param_name, values in param_space.items():
            params[param_name] = trial.suggest_categorical(param_name, values)

        trial_handler.log_trial_start(trial.number, params)

        # Create full configuration
        model_config = ModelConfig(
            d_z=params.get('d_z', 128),
            tau=params.get('temperature', 0.1),
            num_clusters=params.get('num_clusters', 8),
            encoder_hidden_channels=params.get('encoder_hidden_channels', 128),
            projection_hidden_dim=params.get('projection_hidden_dim', 512),
            fusion_hidden_dim=params.get('fusion_hidden_dim', 256),
            use_projection_bottleneck=params.get('use_projection_bottleneck', False)
        )

        training_config = TrainingConfig(
            batch_size=params.get('batch_size', 256),
            learning_rate=params.get('learning_rate', 1e-3),
            max_epochs_stage1=stage1_epochs,
            max_epochs_stage2=stage2_epochs,
            gradient_accumulation_steps_stage1=params.get('gradient_accumulation_steps', 1),
            gradient_accumulation_steps_stage2=params.get('gradient_accumulation_steps', 1),
            lr_warmup_epochs=params.get('lr_warmup_epochs', 5),
            lambda_start=params.get('lambda_start', 0.1),
            lambda_end=params.get('lambda_end', 1.0),
            lambda_warmup_epochs=params.get('lambda_warmup_epochs', 10),
            centroid_normalize_every_n_batches=params.get('centroid_normalize_every_n_batches', 10),
            num_workers=0,
            preload_to_gpu=False
        )

        augmentation_config = AugmentationConfig(
            jitter_sigma=params.get('jitter_sigma', 0.01),
            scale_range=(
                params.get('scale_range_min', 0.9),
                params.get('scale_range_max', 1.1)
            )
        )

        config = Config(
            model=model_config,
            training=training_config,
            augmentation=augmentation_config
        )

        # Load data
        train_data, val_data, _ = load_data_from_hdf5(data_path, subset_fraction)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_data, val_data,
            batch_size=config.training.batch_size
        )

        # Create model
        model = UCLTSCModel(
            input_channels=3,
            d_z=model_config.d_z,
            num_clusters=model_config.num_clusters,
            seq_length=127,
            encoder_hidden_channels=model_config.encoder_hidden_channels,
            projection_hidden_dim=model_config.projection_hidden_dim,
            fusion_hidden_dim=model_config.fusion_hidden_dim,
            use_projection_bottleneck=model_config.use_projection_bottleneck
        )

        # Create trainer and run full training
        trainer = TwoStageTrainer(model, config)
        history = trainer.train(train_loader, val_loader)

        # Get metric based on type
        if metric == 'val_loss':
            # Use Stage 2 validation loss
            metric_value = history['stage2']['val_loss'][-1] if history['stage2']['val_loss'] else float('inf')
            result = metric_value
        else:
            # Compute clustering metrics
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            metrics = compute_clustering_metrics(model, val_loader, device)

            if metric == 'silhouette':
                metric_value = metrics['silhouette_score']
                result = -metric_value  # Maximize
            else:  # davies_bouldin
                metric_value = metrics['davies_bouldin_index']
                result = metric_value  # Minimize

        # Check for early stopping
        if trial_handler.should_prune(trial.number, metric_value, metric):
            trial_handler.log_trial_pruned(trial.number, f"{metric}={metric_value:.4f}")
            raise optuna.TrialPruned()

        trial_handler.log_trial_complete(trial.number, metric_value, metric)

        # Clean up
        del model, trainer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except torch.cuda.OutOfMemoryError:
        trial_handler.handle_oom(trial.number)
        if metric == 'silhouette':
            return float('-inf')
        else:
            return float('inf')

    except Exception as e:
        trial_handler.log_trial_failed(trial.number, str(e))
        raise
