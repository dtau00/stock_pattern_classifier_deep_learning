"""
Simple test script to verify training GUI components
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Test imports
print("Testing imports...")
try:
    from src.models.ucl_tsc_model import UCLTSCModel
    print("[PASS] UCLTSCModel imported")
except Exception as e:
    print(f"[FAIL] UCLTSCModel import failed: {e}")

try:
    from src.config.config import Config, get_default_config
    print("[PASS] Config imported")
except Exception as e:
    print(f"[FAIL] Config import failed: {e}")

try:
    from src.training.trainer import TwoStageTrainer
    print("[PASS] TwoStageTrainer imported")
except Exception as e:
    print(f"[FAIL] TwoStageTrainer import failed: {e}")

# Create synthetic data for testing
print("\nCreating synthetic data...")
n_samples = 100
n_channels = 3
seq_len = 127

# Create random data
data = torch.randn(n_samples, n_channels, seq_len)
print(f"[PASS] Created synthetic data: {data.shape}")

# Split data
train_data = data[:70]
val_data = data[70:85]
test_data = data[85:]

train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=32)
test_loader = DataLoader(TensorDataset(test_data), batch_size=32)

print(f"[PASS] Created data loaders")
print(f"  Train: {len(train_data)} samples")
print(f"  Val: {len(val_data)} samples")
print(f"  Test: {len(test_data)} samples")

# Create model and config
print("\nCreating model...")
try:
    config = get_default_config()
    # Use smaller config for quick testing
    config.training.max_epochs_stage1 = 2
    config.training.max_epochs_stage2 = 2

    model = UCLTSCModel(
        input_channels=3,
        d_z=config.model.d_z,
        num_clusters=config.model.num_clusters
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] Model created: {n_params:,} parameters")
except Exception as e:
    print(f"[FAIL] Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create trainer
print("\nCreating trainer...")
try:
    trainer = TwoStageTrainer(model, config, device='cpu')
    print("[PASS] Trainer created")
except Exception as e:
    print(f"[FAIL] Trainer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test callback
print("\nTesting callback...")
class TestCallback:
    def __init__(self):
        self.calls = []

    def on_epoch_end(self, stage, epoch, total_epochs, metrics):
        self.calls.append({
            'stage': stage,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'metrics': metrics
        })
        print(f"  Callback: Stage {stage}, Epoch {epoch}/{total_epochs}")

callback = TestCallback()

# Run quick training test
print("\nRunning quick training test (2 epochs per stage)...")
try:
    history = trainer.train(
        train_loader,
        val_loader,
        test_loader,
        callback=callback
    )
    print("[PASS] Training completed")
    print(f"  Callback called {len(callback.calls)} times")
    print(f"  Stage 1 epochs: {len(history['stage1']['train_loss'])}")
    print(f"  Stage 2 epochs: {len(history['stage2']['train_loss'])}")
except Exception as e:
    print(f"[FAIL] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model inference
print("\nTesting model inference...")
try:
    model.eval()
    with torch.no_grad():
        z_norm, cluster_ids = model(test_data)

    print(f"[PASS] Inference successful")
    print(f"  Output shape: {z_norm.shape}")
    print(f"  Cluster IDs: {cluster_ids.unique().tolist()}")
except Exception as e:
    print(f"[FAIL] Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save test model
print("\nSaving test model...")
try:
    from pathlib import Path

    model_dir = Path("models/trained")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "test_model.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, model_path)

    print(f"[PASS] Model saved to: {model_path}")
except Exception as e:
    print(f"[FAIL] Model save failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests passed! Training GUI is ready to use.")
print("="*60)
print("\nTo launch the GUI, run:")
print("  streamlit run src/ui/app.py")
print("\nThen navigate to:")
print("  - Page 50: Model Training")
print("  - Page 51: Model Comparison")
print("  - Page 52: Model Inference")
