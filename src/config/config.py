"""
Configuration Dataclasses for UCL-TSC Model

This module defines configuration dataclasses for model architecture,
training hyperparameters, and data parameters.

Reference: Implementation Guide Section 1
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """
    Model architecture hyperparameters.

    Attributes:
        input_channels: Number of input feature channels (default: 3)
        d_z: Latent dimension (default: 128)
        num_clusters: Number of clusters for clustering head (default: 8)
        use_hybrid_encoder: If True, use 3 encoders; else 2 (default: False)
        tau: Temperature for NT-Xent loss (default: 0.5)
    """
    input_channels: int = 3
    d_z: int = 128
    num_clusters: int = 8
    use_hybrid_encoder: bool = False
    tau: float = 0.5

    def __post_init__(self):
        """Validate configuration parameters."""
        assert 1 <= self.input_channels <= 10, \
            f"input_channels must be in [1, 10], got {self.input_channels}"
        assert 32 <= self.d_z <= 256, \
            f"d_z must be in [32, 256], got {self.d_z}"
        assert 5 <= self.num_clusters <= 1000, \
            f"num_clusters must be in [5, 15], got {self.num_clusters}"
        assert 0.1 <= self.tau <= 1.0, \
            f"tau must be in [0.1, 1.0], got {self.tau}"


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.

    Attributes:
        batch_size: Samples per training batch (default: 512)
        learning_rate: Learning rate for optimizer (default: 0.001)
        max_epochs_stage1: Maximum epochs for Stage 1 (default: 50)
        max_epochs_stage2: Maximum epochs for Stage 2 (default: 50)
        lambda_start: Starting lambda for Stage 2 (default: 0.1)
        lambda_end: Final lambda for Stage 2 (default: 1.0)
        lambda_warmup_epochs: Epochs for lambda warm-up (default: 10)
        early_stopping_patience: Patience for early stopping (default: 10)
        use_mixed_precision: Enable FP16 training (default: True)
        gradient_accumulation_steps: Mini-batches before optimizer step (default: 1)
        num_workers: Dataloader workers (default: 4)
        lr_warmup_epochs: LR warm-up epochs for Stage 1 (default: 5)
        stage2_lr_factor: LR multiplier for Stage 2 (default: 0.1)
        pin_memory: Enable pin_memory for DataLoader (default: False)
        preload_to_gpu: Load entire dataset to GPU at start (default: False)
    """
    batch_size: int = 256
    learning_rate: float = 0.001
    max_epochs_stage1: int = 50
    max_epochs_stage2: int = 50
    lambda_start: float = 0.1
    lambda_end: float = 1.0
    lambda_warmup_epochs: int = 10
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    lr_warmup_epochs: int = 5
    stage2_lr_factor: float = 0.1
    pin_memory: bool = False
    preload_to_gpu: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        assert 32 <= self.batch_size <= 16384, \
            f"batch_size must be in [32, 16384], got {self.batch_size}"
        assert 0.0001 <= self.learning_rate <= 0.01, \
            f"learning_rate must be in [0.0001, 0.01], got {self.learning_rate}"
        assert 20 <= self.max_epochs_stage1 <= 100, \
            f"max_epochs_stage1 must be in [20, 100], got {self.max_epochs_stage1}"
        assert 20 <= self.max_epochs_stage2 <= 100, \
            f"max_epochs_stage2 must be in [20, 100], got {self.max_epochs_stage2}"
        assert 0.0 <= self.lambda_start <= 1.0, \
            f"lambda_start must be in [0.0, 1.0], got {self.lambda_start}"
        assert 0.0 <= self.lambda_end <= 5.0, \
            f"lambda_end must be in [0.0, 5.0], got {self.lambda_end}"
        assert 5 <= self.lambda_warmup_epochs <= 20, \
            f"lambda_warmup_epochs must be in [5, 20], got {self.lambda_warmup_epochs}"
        assert 5 <= self.early_stopping_patience <= 20, \
            f"early_stopping_patience must be in [5, 20], got {self.early_stopping_patience}"
        assert 1 <= self.gradient_accumulation_steps <= 8, \
            f"gradient_accumulation_steps must be in [1, 8], got {self.gradient_accumulation_steps}"
        assert 0 <= self.num_workers <= 8, \
            f"num_workers must be in [0, 8], got {self.num_workers}"


@dataclass
class AugmentationConfig:
    """
    Data augmentation hyperparameters.

    Attributes:
        jitter_sigma: Std dev of Gaussian noise (default: 0.01)
        scale_range: Range for amplitude scaling (default: (0.9, 1.1))
        mask_max_length_pct: Max percentage to mask (default: 0.1)
        apply_jitter: Enable jittering (default: True)
        apply_scaling: Enable scaling (default: True)
        apply_masking: Enable time masking (default: True)
    """
    jitter_sigma: float = 0.01
    scale_range: Tuple[float, float] = (0.9, 1.1)
    mask_max_length_pct: float = 0.1
    apply_jitter: bool = True
    apply_scaling: bool = True
    apply_masking: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0.005 <= self.jitter_sigma <= 0.05, \
            f"jitter_sigma must be in [0.005, 0.05], got {self.jitter_sigma}"
        assert len(self.scale_range) == 2, \
            f"scale_range must be tuple of 2 values, got {self.scale_range}"
        assert 0.8 <= self.scale_range[0] <= 1.0, \
            f"scale_range[0] must be in [0.8, 1.0], got {self.scale_range[0]}"
        assert 1.0 <= self.scale_range[1] <= 1.2, \
            f"scale_range[1] must be in [1.0, 1.2], got {self.scale_range[1]}"
        assert 0.05 <= self.mask_max_length_pct <= 0.2, \
            f"mask_max_length_pct must be in [0.05, 0.2], got {self.mask_max_length_pct}"


@dataclass
class DataConfig:
    """
    Data pipeline parameters.

    Attributes:
        sequence_length: Length of time series windows (default: 127)
        sequence_overlap: Overlap between windows (default: 0.5)
        train_split: Fraction for training (default: 0.7)
        val_split: Fraction for validation (default: 0.15)
        test_split: Fraction for testing (default: 0.15)
    """
    sequence_length: int = 127
    sequence_overlap: float = 0.5
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.sequence_length > 0, \
            f"sequence_length must be positive, got {self.sequence_length}"
        assert 0.0 <= self.sequence_overlap < 1.0, \
            f"sequence_overlap must be in [0.0, 1.0), got {self.sequence_overlap}"
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            f"Splits must sum to 1.0, got {self.train_split + self.val_split + self.test_split}"


@dataclass
class Config:
    """
    Unified configuration for UCL-TSC model.

    Attributes:
        model: Model architecture configuration
        training: Training hyperparameters
        augmentation: Data augmentation parameters
        data: Data pipeline parameters
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["UCL-TSC Configuration:"]
        lines.append("\n[Model]")
        for key, value in self.model.__dict__.items():
            lines.append(f"  {key}: {value}")

        lines.append("\n[Training]")
        for key, value in self.training.__dict__.items():
            lines.append(f"  {key}: {value}")

        lines.append("\n[Augmentation]")
        for key, value in self.augmentation.__dict__.items():
            lines.append(f"  {key}: {value}")

        lines.append("\n[Data]")
        for key, value in self.data.__dict__.items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Config with default values
    """
    return Config()


def get_small_config() -> Config:
    """
    Get configuration for small/fast experiments.

    Suitable for:
    - Quick testing
    - Limited GPU memory
    - Fast iteration

    Returns:
        Config with reduced parameters
    """
    return Config(
        model=ModelConfig(
            d_z=64,
            num_clusters=5,
            use_hybrid_encoder=False
        ),
        training=TrainingConfig(
            batch_size=64,
            max_epochs_stage1=20,
            max_epochs_stage2=20,
            learning_rate=0.001
        )
    )


def get_large_config() -> Config:
    """
    Get configuration for large-scale training.

    Suitable for:
    - Production training
    - High GPU memory
    - Best performance

    Returns:
        Config with larger parameters
    """
    return Config(
        model=ModelConfig(
            d_z=256,
            num_clusters=12,
            use_hybrid_encoder=True
        ),
        training=TrainingConfig(
            batch_size=512,
            max_epochs_stage1=100,
            max_epochs_stage2=100,
            learning_rate=0.001
        )
    )


def test_config():
    """Test configuration system."""
    print("Testing Configuration System...")

    # Test 1: Default config
    print("\n[Test 1] Default configuration")
    config = get_default_config()
    print(config)
    print(f"  [PASS] Default config created")

    # Test 2: Small config
    print("\n[Test 2] Small configuration")
    config_small = get_small_config()
    assert config_small.model.d_z == 64
    assert config_small.training.batch_size == 128
    print(f"  [PASS] Small config: d_z={config_small.model.d_z}, batch={config_small.training.batch_size}")

    # Test 3: Large config
    print("\n[Test 3] Large configuration")
    config_large = get_large_config()
    assert config_large.model.d_z == 256
    assert config_large.model.use_hybrid_encoder == True
    print(f"  [PASS] Large config: d_z={config_large.model.d_z}, hybrid={config_large.model.use_hybrid_encoder}")

    # Test 4: Custom config
    print("\n[Test 4] Custom configuration")
    config_custom = Config(
        model=ModelConfig(d_z=128, num_clusters=10),
        training=TrainingConfig(batch_size=256, max_epochs_stage1=30)
    )
    assert config_custom.model.num_clusters == 10
    assert config_custom.training.batch_size == 256
    print(f"  [PASS] Custom config: clusters={config_custom.model.num_clusters}, batch={config_custom.training.batch_size}")

    # Test 5: Validation (should fail for invalid values)
    print("\n[Test 5] Parameter validation")
    try:
        invalid_config = ModelConfig(d_z=10)  # Too small
        print(f"  [FAIL] Should have raised assertion error")
        assert False
    except AssertionError as e:
        print(f"  [PASS] Correctly rejected invalid d_z: {e}")

    try:
        invalid_config = TrainingConfig(batch_size=50)  # Too small
        print(f"  [FAIL] Should have raised assertion error")
        assert False
    except AssertionError as e:
        print(f"  [PASS] Correctly rejected invalid batch_size: {e}")

    # Test 6: Split validation
    print("\n[Test 6] Data split validation")
    try:
        invalid_data = DataConfig(train_split=0.6, val_split=0.2, test_split=0.1)  # Sum != 1.0
        print(f"  [FAIL] Should have raised assertion error")
        assert False
    except AssertionError as e:
        print(f"  [PASS] Correctly rejected invalid splits: {e}")

    print("\n[SUCCESS] All configuration tests passed!")
    return True


if __name__ == '__main__':
    test_config()
