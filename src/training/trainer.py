"""
Two-Stage Trainer for UCL-TSC Model

This module implements the complete training pipeline:
- Stage 1: Contrastive pre-training with NT-Xent loss
- Stage 2: Joint fine-tuning with clustering + contrastive loss

Reference: Implementation Guide Section 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from .losses import NTXentLoss, ClusteringLoss, LambdaSchedule
    from .augmentation import TimeSeriesAugmentation
except ImportError:
    from losses import NTXentLoss, ClusteringLoss, LambdaSchedule
    from augmentation import TimeSeriesAugmentation


class TwoStageTrainer:
    """
    Two-stage trainer for UCL-TSC model.

    Stage 1: Contrastive pre-training
        - NT-Xent loss only
        - Learning rate warm-up (epochs 1-5)
        - Feature collapse detection
        - Best checkpoint selection

    Stage 2: Joint fine-tuning
        - Combined clustering + contrastive loss
        - Lambda warm-up schedule
        - Centroid normalization
        - Reduced learning rate

    Args:
        model: UCLTSCModel instance
        config: Configuration object
        device: Device for training (default: 'cuda' if available)

    Example:
        >>> from src.models import UCLTSCModel
        >>> from src.config import Config
        >>>
        >>> model = UCLTSCModel(d_z=128, num_clusters=8)
        >>> config = Config()
        >>> trainer = TwoStageTrainer(model, config)
        >>>
        >>> # Train on data
        >>> history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model,
        config,
        device: str = None
    ):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Convert to channels-last if requested (better cache locality for CNNs)
        if config.training.use_channels_last and self.device == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
            print(f"[Optimization] Using channels-last memory format")

        # Compile model with torch.compile() if available and requested
        if config.training.use_torch_compile and hasattr(torch, 'compile'):
            print(f"[Optimization] Compiling model with mode='{config.training.compile_mode}'")
            try:
                self.model = torch.compile(self.model, mode=config.training.compile_mode)
                print(f"[Optimization] Model compiled successfully")
            except Exception as e:
                print(f"[Warning] Model compilation failed: {e}. Continuing without compilation.")

        # Loss functions
        self.contrastive_loss_fn = NTXentLoss(temperature=config.model.tau)
        self.clustering_loss_fn = ClusteringLoss(alpha=1.0)

        # Lambda schedule for Stage 2
        self.lambda_schedule = LambdaSchedule(
            lambda_start=config.training.lambda_start,
            lambda_end=config.training.lambda_end,
            warmup_epochs=config.training.lambda_warmup_epochs
        )

        # Augmentation (keep on CPU initially, move tensors to device during training)
        self.augmentation = TimeSeriesAugmentation(
            jitter_sigma=config.augmentation.jitter_sigma,
            scale_range=config.augmentation.scale_range,
            mask_max_length_pct=config.augmentation.mask_max_length_pct,
            apply_jitter=config.augmentation.apply_jitter,
            apply_scaling=config.augmentation.apply_scaling,
            apply_masking=config.augmentation.apply_masking
        )

        # Disable cudnn benchmarking to save memory
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = False

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'stage1': {'train_loss': [], 'val_loss': []},
            'stage2': {'train_loss': [], 'val_loss': [], 'cluster_loss': [], 'contrastive_loss': []}
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        callback = None
    ) -> Dict:
        """
        Run full two-stage training.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            callback: Optional callback object with on_epoch_end(stage, epoch, total_epochs, metrics)

        Returns:
            Training history dictionary
        """
        self.callback = callback
        print("\n" + "="*60)
        print("Starting Two-Stage Training")
        print("="*60)

        # Display configuration
        print("\n[Configuration]")
        print("-" * 60)
        print(f"Device: {self.device}")

        # GPU memory info
        if self.device == 'cuda':
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_mem = torch.cuda.memory_allocated(0) / 1e9
            free_mem = total_mem - allocated_mem
            print(f"GPU Memory: {allocated_mem:.2f}GB used / {total_mem:.2f}GB total ({free_mem:.2f}GB free)")

        print(f"Batch Size: {self.config.training.batch_size}")
        print(f"Mixed Precision: {self.config.training.use_mixed_precision}")
        print(f"Pin Memory: {self.config.training.pin_memory}")
        print(f"Preload to GPU: {self.config.training.preload_to_gpu}")
        print(f"Num Workers: {self.config.training.num_workers}")

        # Check if data is actually on GPU
        sample_batch = next(iter(train_loader))
        data_device = sample_batch[0].device
        print(f"\n[Data Status]")
        print(f"Location: {data_device.type}" + (f" (GPU {data_device.index})" if data_device.type == 'cuda' else ""))
        data_on_gpu = data_device.type == 'cuda'
        print(f"Preloaded: {'YES - Data already on GPU' if data_on_gpu else 'NO - Will transfer each batch'}")

        if self.config.training.preload_to_gpu and not data_on_gpu:
            print("\n[WARNING] preload_to_gpu=True but data is on CPU!")
            print("         Data may not have fit in GPU memory.")

        print("\n[Augmentation]")
        print("-" * 60)
        print(f"Jitter Sigma: {self.config.augmentation.jitter_sigma}")
        print(f"Scale Range: {self.config.augmentation.scale_range}")
        print(f"Mask %: {self.config.augmentation.mask_max_length_pct}")
        print(f"Location: GPU (on-the-fly during training)")

        # Stage 1: Contrastive Pre-training
        print("\n[Stage 1] Contrastive Pre-training")
        print("-" * 60)
        self._train_stage1(train_loader, val_loader, callback)

        # Initialize centroids using best Stage 1 checkpoint
        print("\n[Initialization] K-Means++ Centroids")
        print("-" * 60)
        self._initialize_centroids(train_loader)

        # Stage 2: Joint Fine-tuning
        print("\n[Stage 2] Joint Fine-tuning")
        print("-" * 60)
        self._train_stage2(train_loader, val_loader, callback)

        # Final evaluation
        if test_loader is not None:
            print("\n[Evaluation] Test Set")
            print("-" * 60)
            test_metrics = self._evaluate(test_loader, stage=2)
            print(f"Test Loss: {test_metrics['total_loss']:.4f}")
            print(f"  Clustering: {test_metrics['cluster_loss']:.4f}")
            print(f"  Contrastive: {test_metrics['contrastive_loss']:.4f}")

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

        return self.history

    def _train_stage1(self, train_loader: DataLoader, val_loader: DataLoader, callback=None):
        """Stage 1: Contrastive pre-training."""

        # Optimizer with LR warm-up (use fused kernels if available and requested)
        use_fused = self.config.training.use_fused_optimizer and self.device == 'cuda'
        optimizer_kwargs = {'lr': self.config.training.learning_rate}

        # Try to use fused optimizer (requires PyTorch 2.0+)
        if use_fused:
            try:
                optimizer = torch.optim.Adam(self.model.parameters(), fused=True, **optimizer_kwargs)
                print(f"[Optimization] Using fused Adam optimizer")
            except TypeError:
                # Fused not available, fall back to standard
                optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)
                print(f"[Info] Fused optimizer not available (requires PyTorch 2.0+), using standard Adam")
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)

        # Mixed precision scaler (only on CUDA)
        scaler = torch.cuda.amp.GradScaler() if (self.config.training.use_mixed_precision and self.device == 'cuda') else None

        best_val_loss = float('inf')
        patience_counter = 0

        import time
        for epoch in range(self.config.training.max_epochs_stage1):
            epoch_start_time = time.time()

            # Learning rate warm-up
            if epoch < self.config.training.lr_warmup_epochs:
                lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.lr_warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = self.config.training.learning_rate

            # Training
            train_loss = self._train_epoch_stage1(train_loader, optimizer, scaler)

            # Validation
            val_loss = self._validate_stage1(val_loader)

            # Calculate elapsed time
            epoch_time = time.time() - epoch_start_time

            # Logging
            self.history['stage1']['train_loss'].append(train_loss)
            self.history['stage1']['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1:3d}/{self.config.training.max_epochs_stage1} | "
                  f"LR: {lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            # Callback
            if callback is not None:
                callback.on_epoch_end(
                    stage=1,
                    epoch=epoch+1,
                    total_epochs=self.config.training.max_epochs_stage1,
                    metrics={'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr}
                )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint('best_stage1.pt')
                print(f"  -> Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                print(f"  -> Early stopping triggered (patience: {patience_counter})")
                break

        # Load best checkpoint
        self._load_checkpoint('best_stage1.pt')
        print(f"\nStage 1 complete. Best val loss: {best_val_loss:.4f}")

    def _train_stage2(self, train_loader: DataLoader, val_loader: DataLoader, callback=None):
        """Stage 2: Joint fine-tuning."""

        # Optimizer with reduced learning rate (use fused kernels if available and requested)
        lr_stage2 = self.config.training.learning_rate * self.config.training.stage2_lr_factor
        use_fused = self.config.training.use_fused_optimizer and self.device == 'cuda'
        optimizer_kwargs = {'lr': lr_stage2}

        # Try to use fused optimizer (requires PyTorch 2.0+)
        if use_fused:
            try:
                optimizer = torch.optim.Adam(self.model.parameters(), fused=True, **optimizer_kwargs)
            except TypeError:
                # Fused not available, fall back to standard
                optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)

        # Mixed precision scaler (only on CUDA)
        scaler = torch.cuda.amp.GradScaler() if (self.config.training.use_mixed_precision and self.device == 'cuda') else None

        best_cluster_loss = float('inf')
        patience_counter = 0

        import time
        for epoch in range(self.config.training.max_epochs_stage2):
            epoch_start_time = time.time()

            # Get lambda for current epoch
            lambda_t = self.lambda_schedule(epoch)

            # Training
            metrics = self._train_epoch_stage2(train_loader, optimizer, scaler, lambda_t)

            # Validation
            val_metrics = self._validate_stage2(val_loader, lambda_t)

            # Calculate elapsed time
            epoch_time = time.time() - epoch_start_time

            # Logging
            self.history['stage2']['train_loss'].append(metrics['total_loss'])
            self.history['stage2']['val_loss'].append(val_metrics['total_loss'])
            self.history['stage2']['cluster_loss'].append(val_metrics['cluster_loss'])
            self.history['stage2']['contrastive_loss'].append(val_metrics['contrastive_loss'])

            print(f"Epoch {epoch+1:3d}/{self.config.training.max_epochs_stage2} | "
                  f"Lambda: {lambda_t:.3f} | "
                  f"Train: {metrics['total_loss']:.4f} | "
                  f"Val: {val_metrics['total_loss']:.4f} "
                  f"(C: {val_metrics['cluster_loss']:.4f}, "
                  f"N: {val_metrics['contrastive_loss']:.4f}) | "
                  f"Time: {epoch_time:.2f}s")

            # Callback
            if callback is not None:
                callback.on_epoch_end(
                    stage=2,
                    epoch=epoch+1,
                    total_epochs=self.config.training.max_epochs_stage2,
                    metrics={
                        'train_loss': metrics['total_loss'],
                        'val_loss': val_metrics['total_loss'],
                        'cluster_loss': val_metrics['cluster_loss'],
                        'contrastive_loss': val_metrics['contrastive_loss'],
                        'lambda': lambda_t
                    }
                )

            # Early stopping on clustering loss
            if val_metrics['cluster_loss'] < best_cluster_loss:
                best_cluster_loss = val_metrics['cluster_loss']
                patience_counter = 0
                self._save_checkpoint('best_stage2.pt')
                print(f"  -> Best model saved (cluster_loss: {val_metrics['cluster_loss']:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                print(f"  -> Early stopping triggered (patience: {patience_counter})")
                break

        # Load best checkpoint
        self._load_checkpoint('best_stage2.pt')
        print(f"\nStage 2 complete. Best cluster loss: {best_cluster_loss:.4f}")

    def _train_epoch_stage1(self, train_loader, optimizer, scaler) -> float:
        """Train one epoch of Stage 1."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (x,) in enumerate(train_loader):
            # Move to device if not already there
            if x.device.type != self.device:
                x = x.to(self.device, non_blocking=True)

            # Convert to channels-last if enabled
            if self.config.training.use_channels_last and self.device == 'cuda':
                x = x.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Create two augmented views ON GPU (in autocast context)
                    x1, x2 = self.augmentation(x)

                    # Forward pass through encoder and projection head
                    z1, h1 = self.model.encoder.forward_projection(x1)
                    z2, h2 = self.model.encoder.forward_projection(x2)

                    # Normalize projections
                    h1_norm = F.normalize(h1, p=2, dim=1)
                    h2_norm = F.normalize(h2, p=2, dim=1)

                    # NT-Xent loss
                    loss = self.contrastive_loss_fn(h1_norm, h2_norm)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Create two augmented views ON GPU
                x1, x2 = self.augmentation(x)

                # Forward pass
                z1, h1 = self.model.encoder.forward_projection(x1)
                z2, h2 = self.model.encoder.forward_projection(x2)

                # Normalize projections
                h1_norm = F.normalize(h1, p=2, dim=1)
                h2_norm = F.normalize(h2, p=2, dim=1)

                # NT-Xent loss
                loss = self.contrastive_loss_fn(h1_norm, h2_norm)

                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _train_epoch_stage2(self, train_loader, optimizer, scaler, lambda_t: float) -> Dict:
        """Train one epoch of Stage 2."""
        self.model.train()
        total_loss = 0.0
        total_cluster_loss = 0.0
        total_contrastive_loss = 0.0

        for batch_idx, (x,) in enumerate(train_loader):
            # Move to device if not already there
            if x.device.type != self.device:
                x = x.to(self.device, non_blocking=True)

            # Convert to channels-last if enabled
            if self.config.training.use_channels_last and self.device == 'cuda':
                x = x.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Get latent vectors (normalized)
                    z = self.model.encoder(x)
                    z_norm = F.normalize(z, p=2, dim=1)

                    # Clustering loss
                    loss_cluster = self.clustering_loss_fn(z_norm, self.model.centroids)

                    # Create two augmented views ON GPU (in autocast context)
                    x1, x2 = self.augmentation(x)

                    # Contrastive loss
                    z1, h1 = self.model.encoder.forward_projection(x1)
                    z2, h2 = self.model.encoder.forward_projection(x2)
                    h1_norm = F.normalize(h1, p=2, dim=1)
                    h2_norm = F.normalize(h2, p=2, dim=1)
                    loss_contrastive = self.contrastive_loss_fn(h1_norm, h2_norm)

                    # Combined loss
                    loss = loss_cluster + lambda_t * loss_contrastive

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Get latent vectors (normalized)
                z = self.model.encoder(x)
                z_norm = F.normalize(z, p=2, dim=1)

                # Clustering loss
                loss_cluster = self.clustering_loss_fn(z_norm, self.model.centroids)

                # Create two augmented views ON GPU
                x1, x2 = self.augmentation(x)

                # Contrastive loss
                z1, h1 = self.model.encoder.forward_projection(x1)
                z2, h2 = self.model.encoder.forward_projection(x2)
                h1_norm = F.normalize(h1, p=2, dim=1)
                h2_norm = F.normalize(h2, p=2, dim=1)
                loss_contrastive = self.contrastive_loss_fn(h1_norm, h2_norm)

                # Combined loss
                loss = loss_cluster + lambda_t * loss_contrastive

                loss.backward()
                optimizer.step()

            # Normalize centroids after optimizer step
            self.model.normalize_centroids()

            total_loss += loss.item()
            total_cluster_loss += loss_cluster.item()
            total_contrastive_loss += loss_contrastive.item()

        return {
            'total_loss': total_loss / len(train_loader),
            'cluster_loss': total_cluster_loss / len(train_loader),
            'contrastive_loss': total_contrastive_loss / len(train_loader)
        }

    def _validate_stage1(self, val_loader) -> float:
        """Validate Stage 1."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for (x,) in val_loader:
                # Move to device if not already there
                if x.device.type != self.device:
                    x = x.to(self.device, non_blocking=True)

                # Create two augmented views ON GPU
                x1, x2 = self.augmentation(x)

                # Forward pass
                z1, h1 = self.model.encoder.forward_projection(x1)
                z2, h2 = self.model.encoder.forward_projection(x2)

                # Normalize projections
                h1_norm = F.normalize(h1, p=2, dim=1)
                h2_norm = F.normalize(h2, p=2, dim=1)

                # NT-Xent loss
                loss = self.contrastive_loss_fn(h1_norm, h2_norm)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _validate_stage2(self, val_loader, lambda_t: float) -> Dict:
        """Validate Stage 2."""
        self.model.eval()
        total_loss = 0.0
        total_cluster_loss = 0.0
        total_contrastive_loss = 0.0

        with torch.no_grad():
            for (x,) in val_loader:
                # Move to device if not already there
                if x.device.type != self.device:
                    x = x.to(self.device, non_blocking=True)

                # Get latent vectors
                z = self.model.encoder(x)
                z_norm = F.normalize(z, p=2, dim=1)

                # Clustering loss
                loss_cluster = self.clustering_loss_fn(z_norm, self.model.centroids)

                # Create augmented views ON GPU for contrastive loss
                x1, x2 = self.augmentation(x)
                z1, h1 = self.model.encoder.forward_projection(x1)
                z2, h2 = self.model.encoder.forward_projection(x2)
                h1_norm = F.normalize(h1, p=2, dim=1)
                h2_norm = F.normalize(h2, p=2, dim=1)
                loss_contrastive = self.contrastive_loss_fn(h1_norm, h2_norm)

                # Combined loss
                loss = loss_cluster + lambda_t * loss_contrastive

                total_loss += loss.item()
                total_cluster_loss += loss_cluster.item()
                total_contrastive_loss += loss_contrastive.item()

        return {
            'total_loss': total_loss / len(val_loader),
            'cluster_loss': total_cluster_loss / len(val_loader),
            'contrastive_loss': total_contrastive_loss / len(val_loader)
        }

    def _evaluate(self, test_loader, stage: int) -> Dict:
        """Evaluate on test set."""
        if stage == 1:
            return {'loss': self._validate_stage1(test_loader)}
        else:
            return self._validate_stage2(test_loader, lambda_t=1.0)

    def _initialize_centroids(self, train_loader):
        """Initialize centroids using K-Means++."""
        self.model.eval()

        # Collect latent vectors from training set
        latents = []
        with torch.no_grad():
            for (x,) in train_loader:
                # Move to device if not already there
                if x.device.type != self.device:
                    x = x.to(self.device, non_blocking=True)
                z = self.model.encoder(x)
                z_norm = F.normalize(z, p=2, dim=1)
                latents.append(z_norm.cpu())

        latents = torch.cat(latents, dim=0)

        # Initialize centroids
        self.model.initialize_centroids(latents)

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, checkpoint_dir / filename)

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_dir = Path('checkpoints')
        checkpoint = torch.load(checkpoint_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def test_trainer():
    """Test trainer with synthetic data."""
    print("Testing TwoStageTrainer...")

    # Import required modules
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.ucl_tsc_model import UCLTSCModel
    from config.config import get_small_config

    # Create small config for fast testing
    config = get_small_config()
    config.training.max_epochs_stage1 = 3
    config.training.max_epochs_stage2 = 3
    config.training.batch_size = 32

    # Create synthetic data
    print("\n[Test 1] Creating synthetic data")
    n_samples = 100
    x = torch.randn(n_samples, 3, 127)

    train_dataset = TensorDataset(x[:70])
    val_dataset = TensorDataset(x[70:85])
    test_dataset = TensorDataset(x[85:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Create model
    print("\n[Test 2] Creating model")
    seq_len = train_loader.dataset.tensors[0].shape[2]
    model = UCLTSCModel(
        input_channels=3,
        d_z=config.model.d_z,
        num_clusters=config.model.num_clusters,
        use_hybrid_encoder=False,
        seq_length=seq_len
    )
    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    print("\n[Test 3] Creating trainer")
    trainer = TwoStageTrainer(model, config, device='cpu')
    print(f"  Trainer created on device: {trainer.device}")

    # Train
    print("\n[Test 4] Training (quick test)")
    history = trainer.train(train_loader, val_loader, test_loader)

    print("\n[Test 5] Checking history")
    assert len(history['stage1']['train_loss']) > 0
    assert len(history['stage2']['train_loss']) > 0
    print(f"  Stage 1 epochs: {len(history['stage1']['train_loss'])}")
    print(f"  Stage 2 epochs: {len(history['stage2']['train_loss'])}")

    print("\n[SUCCESS] Trainer test passed!")
    return True


if __name__ == '__main__':
    test_trainer()
