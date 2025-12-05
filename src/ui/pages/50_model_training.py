"""
Model Training Page

This page provides a UI for training the UCL-TSC model with two-stage training:
- Stage 1: Contrastive pre-training
- Stage 2: Joint fine-tuning with clustering

Features:
- Configuration builder
- Training progress visualization
- Real-time loss plots
- Model management (save/load)
- Training history
"""

import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
import h5py
from datetime import datetime

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ucl_tsc_model import UCLTSCModel
from src.config.config import Config, ModelConfig, TrainingConfig, AugmentationConfig, DataConfig, save_preset_config, load_preset_config
from src.training.trainer import TwoStageTrainer
from src.evaluation import ConfidenceCalibrator
import torch.nn.functional as F

# Register safe globals for torch.load (PyTorch 2.6+)
torch.serialization.add_safe_globals([Config, ModelConfig, TrainingConfig, AugmentationConfig, DataConfig])


def save_training_summary(data_package, config, history, end_after_stage1, device, train_split, val_split, test_split, dataset_info=None):
    """
    Save training summary to JSON file for later comparison.

    Args:
        data_package: Name of the data package used
        config: Training configuration
        history: Training history with losses and metrics
        end_after_stage1: Whether training ended after stage 1
        device: Device used for training
        train_split: Training split percentage
        val_split: Validation split percentage
        test_split: Test split percentage
        dataset_info: Optional dict with dataset properties (total_windows, window_length, num_channels)
    """
    summary_dir = Path("data/training_logs")
    summary_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract Stage 1 metrics
    stage1_metrics = history.get('stage1', {}).get('metrics', {})

    # Build summary dictionary
    summary = {
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'data_package': data_package,
        'end_after_stage1': end_after_stage1,
        'device': device,

        # Dataset properties (for AI analysis)
        'dataset_total_windows': dataset_info.get('total_windows') if dataset_info else None,
        'dataset_window_length': dataset_info.get('window_length') if dataset_info else None,
        'dataset_num_channels': dataset_info.get('num_channels') if dataset_info else None,

        # Data splits
        'train_split': train_split,
        'val_split': val_split,
        'test_split': test_split,

        # Model config
        'model': {
            'd_z': config.model.d_z,
            'num_clusters': config.model.num_clusters,
            'tau': config.model.tau,
            'use_hybrid_encoder': config.model.use_hybrid_encoder,
            'seq_length': config.model.seq_length,
            'encoder_hidden_channels': config.model.encoder_hidden_channels,
            'projection_hidden_dim': config.model.projection_hidden_dim,
            'fusion_hidden_dim': config.model.fusion_hidden_dim,
            'use_projection_bottleneck': config.model.use_projection_bottleneck,
        },

        # Training config
        'training': {
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'max_epochs_stage1': config.training.max_epochs_stage1,
            'max_epochs_stage2': config.training.max_epochs_stage2,
            'lr_warmup_epochs': config.training.lr_warmup_epochs,
            'stage2_lr_factor': config.training.stage2_lr_factor,
            'early_stopping_patience': config.training.early_stopping_patience,
            'use_mixed_precision': config.training.use_mixed_precision,
            'gradient_accumulation_steps_stage1': config.training.gradient_accumulation_steps_stage1,
            'gradient_accumulation_steps_stage2': config.training.gradient_accumulation_steps_stage2,
            'num_workers': config.training.num_workers,
            'pin_memory': config.training.pin_memory,
            'preload_to_gpu': config.training.preload_to_gpu,
            'persistent_workers': config.training.persistent_workers,
            'prefetch_factor': config.training.prefetch_factor,
            'use_fused_optimizer': config.training.use_fused_optimizer,
            'use_torch_compile': config.training.use_torch_compile,
            'compile_mode': config.training.compile_mode,
            'use_channels_last': config.training.use_channels_last,
            'centroid_normalize_every_n_batches': config.training.centroid_normalize_every_n_batches,
            'lambda_start': config.training.lambda_start,
            'lambda_end': config.training.lambda_end,
            'lambda_warmup_epochs': config.training.lambda_warmup_epochs,
        },

        # Augmentation config
        'augmentation': {
            'jitter_sigma': config.augmentation.jitter_sigma,
            'scale_range': list(config.augmentation.scale_range),
            'mask_max_length_pct': config.augmentation.mask_max_length_pct,
            'apply_jitter': config.augmentation.apply_jitter,
            'apply_scaling': config.augmentation.apply_scaling,
            'apply_masking': config.augmentation.apply_masking,
        },

        # Results - Stage 1
        'results': {
            'stage1_epochs_completed': len(history['stage1']['train_loss']),
            'stage1_best_val_loss': float(min(history['stage1']['val_loss'])) if history['stage1']['val_loss'] else None,
            'stage1_final_train_loss': float(history['stage1']['train_loss'][-1]) if history['stage1']['train_loss'] else None,
            'stage1_final_val_loss': float(history['stage1']['val_loss'][-1]) if history['stage1']['val_loss'] else None,
        }
    }

    # Add Stage 1 quality metrics if available
    if stage1_metrics:
        summary['results']['variance'] = float(stage1_metrics.get('variance', 0))
        summary['results']['variance_pass'] = bool(stage1_metrics.get('variance_pass', False))
        summary['results']['effective_rank'] = float(stage1_metrics.get('effective_rank', 0))
        summary['results']['effective_rank_pass'] = bool(stage1_metrics.get('effective_rank_pass', False))
        summary['results']['alignment'] = float(stage1_metrics.get('alignment', 0))
        summary['results']['alignment_pass'] = bool(stage1_metrics.get('alignment_pass', False))
        summary['results']['uniformity'] = float(stage1_metrics.get('uniformity', 0))
        summary['results']['uniformity_pass'] = bool(stage1_metrics.get('uniformity_pass', False))
        summary['results']['knn_accuracy'] = float(stage1_metrics.get('knn_accuracy', 0))
        summary['results']['knn_accuracy_pass'] = bool(stage1_metrics.get('knn_accuracy_pass', False))
        summary['results']['overall_score'] = float(stage1_metrics.get('overall_score', 0))
        summary['results']['n_passed'] = int(stage1_metrics.get('n_passed', 0))
        summary['results']['n_total'] = int(stage1_metrics.get('n_total', 0))

    # Add Stage 2 results if not stage1_only
    if not end_after_stage1 and 'stage2' in history and history['stage2']['train_loss']:
        summary['results']['stage2_epochs_completed'] = len(history['stage2']['train_loss'])
        summary['results']['stage2_best_cluster_loss'] = float(min(history['stage2']['cluster_loss'])) if history['stage2']['cluster_loss'] else None
        summary['results']['stage2_final_train_loss'] = float(history['stage2']['train_loss'][-1])
        summary['results']['stage2_final_val_loss'] = float(history['stage2']['val_loss'][-1])
        summary['results']['stage2_final_cluster_loss'] = float(history['stage2']['cluster_loss'][-1])
        summary['results']['stage2_final_contrastive_loss'] = float(history['stage2']['contrastive_loss'][-1])

    # Save to JSON
    summary_file = summary_dir / f"training_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to: {summary_file}")


def calibrate_confidence_scores(model, val_loader, device):
    """
    Calibrate confidence scores on validation set.

    Args:
        model: Trained UCLTSCModel
        val_loader: Validation data loader
        device: Device (cuda/cpu)

    Returns:
        Calibration results dictionary
    """
    model.eval()

    # Collect validation latent vectors and cluster assignments
    z_val_list = []
    cluster_ids_list = []

    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)

            # Get latent vectors
            z = model.encoder(x)
            z_norm = F.normalize(z, p=2, dim=1)

            # Get cluster assignments
            cluster_ids = model.get_cluster_assignment(z_norm)

            z_val_list.append(z_norm.cpu())
            cluster_ids_list.append(cluster_ids.cpu())

    # Concatenate all batches
    z_val = torch.cat(z_val_list, dim=0)
    cluster_ids_val = torch.cat(cluster_ids_list, dim=0)

    # Get centroids
    centroids_norm = F.normalize(model.centroids, p=2, dim=1).cpu()

    # Calibrate
    calibrator = ConfidenceCalibrator(
        gamma_grid=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        sample_size=5000
    )

    calibration_results = calibrator.calibrate(
        z_val,
        centroids_norm,
        cluster_ids_val,
        verbose=False  # Don't print to console in UI
    )

    return calibration_results


def main():
    st.title("🚀 Model Training")
    st.markdown("Train UCL-TSC model with two-stage training pipeline")

    # Sidebar for navigation
    with st.sidebar:
        st.header("Training Steps")
        step = st.radio(
            "Select Step:",
            ["1. Load Data", "2. Configure Model", "3. Train", "4. Evaluate", "5. Manage Models"],
            index=0
        )

    # Step 1: Load Data
    if step == "1. Load Data":
        show_data_loading()

    # Step 2: Configure Model
    elif step == "2. Configure Model":
        show_model_configuration()

    # Step 3: Train
    elif step == "3. Train":
        show_training()

    # Step 4: Evaluate
    elif step == "4. Evaluate":
        show_evaluation()

    # Step 5: Manage Models
    elif step == "5. Manage Models":
        show_model_management()


def show_data_loading():
    """Step 1: Load preprocessed data."""
    st.header("Step 1: Load Preprocessed Data")
    st.markdown("Load preprocessed windows from HDF5 files created in the preprocessing step.")

    # File selection
    data_dir = Path("data/preprocessed")
    if not data_dir.exists():
        st.warning("No preprocessed data directory found. Please run preprocessing first.")
        st.info("📁 Expected directory: `data/preprocessed/`")
        return

    # Get available HDF5 files
    hdf5_files = list(data_dir.glob("*.h5"))

    if not hdf5_files:
        st.warning("No HDF5 files found in data/preprocessed/")
        st.info("💡 Run the preprocessing pipeline first (Page 20)")
        return

    # File selector
    selected_file = st.selectbox(
        "Select preprocessed data file:",
        options=hdf5_files,
        format_func=lambda x: x.name
    )

    if selected_file:
        try:
            # Load data info
            with h5py.File(selected_file, 'r') as f:
                n_samples = f['windows'].shape[0]
                seq_len = f['windows'].shape[1]
                n_channels = f['windows'].shape[2]

                # Get metadata (use window_metadata if available, fallback to empty dict)
                metadata = {}
                if 'window_metadata' in f.attrs:
                    metadata = json.loads(f.attrs['window_metadata'])
                elif 'metadata' in f.attrs:
                    metadata = json.loads(f.attrs['metadata'])

            st.success(f"✅ Found preprocessed data: **{selected_file.name}**")

            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{n_samples:,}")
            with col2:
                st.metric("Sequence Length", seq_len)
            with col3:
                st.metric("Channels", n_channels)

            # Show metadata
            with st.expander("📊 Data Metadata"):
                st.json(metadata)

            # Load button
            if st.button("Load Data into Session", type="primary"):
                with st.spinner("Loading data..."):
                    with h5py.File(selected_file, 'r') as f:
                        windows = f['windows'][:]
                        # Load metadata
                        window_metadata = json.loads(f.attrs['window_metadata'])

                    # Convert to torch tensor
                    windows_tensor = torch.from_numpy(windows).float()

                    # Transpose from (N, T, C) to (N, C, T)
                    windows_tensor = windows_tensor.permute(0, 2, 1)

                    # Load OHLCV data for visualization
                    try:
                        with h5py.File(selected_file, 'r') as f:
                            additional_metadata = json.loads(f.attrs['additional_metadata'])

                        source_package = additional_metadata.get('source_package')
                        if source_package:
                            ohlcv_path = Path("data/packages") / source_package
                            if ohlcv_path.exists():
                                ohlcv_df = pd.read_csv(ohlcv_path)
                                if 'timestamp' in ohlcv_df.columns:
                                    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
                                elif 'open_time' in ohlcv_df.columns:
                                    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['open_time'], unit='ms')
                                st.session_state['ohlcv_data'] = ohlcv_df
                                st.session_state['window_metadata'] = window_metadata
                    except Exception as e:
                        pass  # OHLCV data is optional

                    # Store in session state
                    st.session_state['data'] = windows_tensor
                    st.session_state['data_file'] = selected_file.name
                    st.session_state['n_samples'] = n_samples

                    st.success(f"✅ Loaded {n_samples:,} windows into session")
                    st.info("👉 Proceed to **Step 2: Configure Model**")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Show loaded data status
    if 'data' in st.session_state:
        st.divider()
        st.success(f"✅ Data loaded: {st.session_state['data_file']}")
        st.metric("Samples in Session", f"{st.session_state['n_samples']:,}")


def show_model_configuration():
    """Step 2: Configure model and training parameters."""
    st.header("Step 2: Configure Model")
    st.markdown("Set model architecture and training hyperparameters")

    # Check if data is loaded
    if 'data' not in st.session_state:
        st.warning("⚠️ Please load data first (Step 1)")
        return

    # Preset selector
    st.subheader("⚙️ Configuration Preset")

    # Initialize preset in session state if not exists
    if 'selected_preset' not in st.session_state:
        st.session_state['selected_preset'] = "Default"

    preset = st.selectbox(
        "Choose a preset:",
        ["Custom", "Default", "Small (Fast)", "Large (Best Quality)"],
        index=["Custom", "Default", "Small (Fast)", "Large (Best Quality)"].index(st.session_state['selected_preset']),
        key="preset_selector"
    )

    # Update session state if preset changed
    if preset != st.session_state['selected_preset']:
        st.session_state['selected_preset'] = preset
        st.rerun()

    # Load config from disk (will use built-in defaults if file doesn't exist)
    config = load_preset_config(preset)

    # Model Configuration
    st.subheader("🏗️ Model Architecture")
    col1, col2 = st.columns(2)

    with col1:
        d_z = st.number_input(
            "Latent Dimension (d_z)",
            min_value=32, max_value=256, value=config.model.d_z,
            help="Dimensionality of latent representations"
        )
        num_clusters = st.number_input(
            "Number of Clusters",
            min_value=5, max_value=1000, value=config.model.num_clusters,
            help="Number of pattern clusters to identify"
        )

    with col2:
        tau = st.slider(
            "Temperature (τ)",
            min_value=0.1, max_value=1.0, value=config.model.tau, step=0.05,
            help="Temperature for NT-Xent loss (lower = harder task)"
        )
        use_hybrid = st.checkbox(
            "Use Hybrid Encoder",
            value=config.model.use_hybrid_encoder,
            help="Enable 3rd encoder for intermediate patterns (scales with window size)"
        )

    # Advanced Architecture Configuration
    with st.expander("🔧 Advanced Architecture Settings (Affects Model Capacity)", expanded=False):
        st.markdown("**These parameters control the internal dimensions of the model architecture.**")
        st.info("💡 **Tip:** Increase these values to improve embedding quality and prevent dimensional collapse. Higher values = more capacity but slower training.")

        col1, col2 = st.columns(2)

        with col1:
            encoder_hidden_channels = st.number_input(
                "Encoder Hidden Channels",
                min_value=32, max_value=512, value=config.model.encoder_hidden_channels,
                help="Number of channels in CNN/TCN layers. Default: 128. Higher = more feature capacity."
            )
            projection_hidden_dim = st.number_input(
                "Projection Head Hidden Dim",
                min_value=64, max_value=2048, value=config.model.projection_hidden_dim,
                help="Hidden dimension in projection MLP. Default: 512 (4x d_z). Higher = richer contrastive representations."
            )

        with col2:
            fusion_hidden_dim = st.number_input(
                "Fusion Attention Hidden Dim",
                min_value=64, max_value=1024, value=config.model.fusion_hidden_dim,
                help="Hidden dimension in fusion FFN. Default: 256 (2x d_z). Higher = better encoder fusion."
            )
            use_projection_bottleneck = st.checkbox(
                "Use Projection Bottleneck",
                value=config.model.use_projection_bottleneck,
                help="Use bottleneck architecture in projection head (more non-linearity)"
            )

    # Training Configuration
    st.subheader("📈 Training Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Stage 1: Contrastive Pre-training**")
        max_epochs_stage1 = st.number_input(
            "Max Epochs",
            min_value=20, max_value=100, value=config.training.max_epochs_stage1,
            key="stage1_epochs"
        )
        lr = st.number_input(
            "Learning Rate",
            min_value=0.0001, max_value=0.01, value=config.training.learning_rate,
            format="%.4f"
        )

    with col2:
        st.markdown("**Stage 2: Joint Fine-tuning**")
        max_epochs_stage2 = st.number_input(
            "Max Epochs",
            min_value=20, max_value=100, value=config.training.max_epochs_stage2,
            key="stage2_epochs"
        )
        batch_size_options = [2 ,4 ,8 ,16 ,32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        # Find index of current batch size, default to 512 if not in list
        try:
            batch_size_index = batch_size_options.index(config.training.batch_size)
        except ValueError:
            batch_size_index = 4  # Default to 512

        batch_size = st.selectbox(
            "Batch Size",
            options=batch_size_options,
            index=batch_size_index
        )

    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Optimization**")
            early_stopping = st.number_input(
                "Early Stopping Patience",
                min_value=1, max_value=100, value=config.training.early_stopping_patience
            )
            num_workers = st.number_input(
                "DataLoader Workers",
                min_value=0, max_value=32, value=config.training.num_workers,
                help="Number of subprocesses for data loading (0=main process only)"
            )
            persistent_workers = st.checkbox(
                "Persistent Workers",
                value=config.training.persistent_workers,
                help="Keep workers alive between epochs (reduces overhead when num_workers > 0)"
            )
            prefetch_factor = st.number_input(
                "Prefetch Factor",
                min_value=2, max_value=100, value=config.training.prefetch_factor,
                help="Number of batches to prefetch per worker (only when num_workers > 0)"
            )
            use_mixed_precision = st.checkbox(
                "Mixed Precision (FP16)",
                value=config.training.use_mixed_precision,
                help="Faster training on CUDA, 50% memory reduction"
            )
            pin_memory = st.checkbox(
                "Pin Memory",
                value=config.training.pin_memory,
                help="Enable pinned memory for faster CPU->GPU transfer"
            )
            preload_to_gpu = st.checkbox(
                "Preload Dataset to GPU",
                value=config.training.preload_to_gpu,
                help="Load entire dataset to GPU at start (fastest but requires sufficient VRAM)"
            )
            st.markdown("**Advanced Optimizations (PyTorch 2.0+)**")
            use_fused_optimizer = st.checkbox(
                "Fused Optimizer",
                value=config.training.use_fused_optimizer,
                help="Use fused CUDA kernels for Adam optimizer (5-10% speedup)"
            )
            use_torch_compile = st.checkbox(
                "Compile Model (torch.compile)",
                value=config.training.use_torch_compile,
                help="JIT compile model with PyTorch 2.0+ (20-40% speedup, requires first-epoch warmup)"
            )
            compile_mode = st.selectbox(
                "Compile Mode",
                options=['default', 'reduce-overhead', 'max-autotune'],
                index=['default', 'reduce-overhead', 'max-autotune'].index(config.training.compile_mode),
                help="default=balanced, reduce-overhead=faster startup, max-autotune=best performance"
            )
            use_channels_last = st.checkbox(
                "Channels-Last Memory",
                value=config.training.use_channels_last,
                help="Better cache locality for CNNs (5-15% speedup on modern GPUs)"
            )

            st.markdown("**Gradient Accumulation (Per Stage)**")
            gradient_accumulation_steps_stage1 = st.number_input(
                "Stage 1 Gradient Accumulation",
                min_value=1, max_value=100, value=config.training.gradient_accumulation_steps_stage1,
                help="Accumulate gradients in Stage 1 (benefits from larger effective batch for NT-Xent, default: 2)"
            )
            gradient_accumulation_steps_stage2 = st.number_input(
                "Stage 2 Gradient Accumulation",
                min_value=1, max_value=100, value=config.training.gradient_accumulation_steps_stage2,
                help="Accumulate gradients in Stage 2 (default: 1 for frequent centroid updates)"
            )

            st.markdown("**Stage 2 Optimizations**")
            centroid_normalize_every_n_batches = st.number_input(
                "Centroid Normalize Every N Batches",
                min_value=1, max_value=1000, value=config.training.centroid_normalize_every_n_batches,
                help="Normalize centroids every N batches (reduces overhead, default: 10)"
            )

        with col2:
            st.markdown("**Data Augmentation**")
            jitter_sigma = st.slider(
                "Jitter Sigma",
                min_value=0.005, max_value=0.2, value=config.augmentation.jitter_sigma,
                step=0.005,
                format="%.3f"
            )
            mask_pct = st.slider(
                "Time Mask %",
                min_value=0.05, max_value=0.4, value=config.augmentation.mask_max_length_pct,
                format="%.2f"
            )

            col_scale1, col_scale2 = st.columns(2)
            with col_scale1:
                scale_min = st.number_input(
                    "Scale Min",
                    min_value=0.8, max_value=1.0, value=config.augmentation.scale_range[0],
                    step=0.01,
                    format="%.2f"
                )
            with col_scale2:
                scale_max = st.number_input(
                    "Scale Max",
                    min_value=1.0, max_value=1.2, value=config.augmentation.scale_range[1],
                    step=0.01,
                    format="%.2f"
                )

    # Create config object
    custom_config = Config(
        model=ModelConfig(
            d_z=d_z,
            num_clusters=num_clusters,
            tau=tau,
            use_hybrid_encoder=use_hybrid,
            encoder_hidden_channels=encoder_hidden_channels,
            projection_hidden_dim=projection_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            use_projection_bottleneck=use_projection_bottleneck
        ),
        training=TrainingConfig(
            max_epochs_stage1=max_epochs_stage1,
            max_epochs_stage2=max_epochs_stage2,
            learning_rate=lr,
            batch_size=batch_size,
            early_stopping_patience=early_stopping,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            use_mixed_precision=use_mixed_precision,
            pin_memory=pin_memory,
            preload_to_gpu=preload_to_gpu,
            use_fused_optimizer=use_fused_optimizer,
            use_torch_compile=use_torch_compile,
            compile_mode=compile_mode,
            use_channels_last=use_channels_last,
            gradient_accumulation_steps_stage1=gradient_accumulation_steps_stage1,
            gradient_accumulation_steps_stage2=gradient_accumulation_steps_stage2,
            centroid_normalize_every_n_batches=centroid_normalize_every_n_batches
        ),
        augmentation=AugmentationConfig(
            jitter_sigma=jitter_sigma,
            mask_max_length_pct=mask_pct,
            scale_range=(scale_min, scale_max)
        )
    )

    # Save config button
    col1, col2 = st.columns([1, 3])
    with col1:
        save_button = st.button("💾 Save", type="primary")
    with col2:
        queue_button = st.button("➕ Add to Queue", type="secondary")

    if save_button:
        # Save to the selected preset
        save_preset_config(preset, custom_config)
        st.session_state['config'] = custom_config
        st.success(f"✅ Configuration saved to preset: **{preset}**")
        st.info("👉 Proceed to **Step 3: Train**")

    if queue_button:
        # Add to training queue
        if 'config_queue' not in st.session_state:
            st.session_state['config_queue'] = []

        # Prompt for configuration name
        config_name = st.text_input(
            "Enter configuration name:",
            value=f"Config_{len(st.session_state['config_queue'])+1}",
            key="queue_config_name"
        )
        if config_name:
            config_entry = {
                'name': config_name,
                'config': custom_config
            }
            st.session_state['config_queue'].append(config_entry)
            st.session_state['config'] = custom_config  # Also set as current config
            st.success(f"✅ Added '{config_name}' to queue ({len(st.session_state['config_queue'])} total)")
            st.info("👉 Configure another model or proceed to **Step 3: Train**")

    # Show current config
    if 'config' in st.session_state:
        st.divider()
        with st.expander("📋 Current Configuration"):
            st.text(str(st.session_state['config']))

    # Show configuration queue
    if 'config_queue' in st.session_state and len(st.session_state['config_queue']) > 0:
        st.divider()
        st.subheader("📋 Configuration Queue")
        st.markdown(f"**{len(st.session_state['config_queue'])} configuration(s) queued for training**")

        for idx, config_entry in enumerate(st.session_state['config_queue']):
            with st.expander(f"Config {idx+1}: {config_entry['name']}", expanded=False):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    cfg = config_entry['config']
                    st.json({
                        'name': config_entry['name'],
                        'model': {
                            'd_z': cfg.model.d_z,
                            'num_clusters': cfg.model.num_clusters,
                            'tau': cfg.model.tau,
                            'use_hybrid_encoder': cfg.model.use_hybrid_encoder,
                            'encoder_hidden_channels': cfg.model.encoder_hidden_channels,
                            'projection_hidden_dim': cfg.model.projection_hidden_dim,
                            'fusion_hidden_dim': cfg.model.fusion_hidden_dim,
                            'use_projection_bottleneck': cfg.model.use_projection_bottleneck,
                        },
                        'training': {
                            'batch_size': cfg.training.batch_size,
                            'learning_rate': cfg.training.learning_rate,
                            'max_epochs_stage1': cfg.training.max_epochs_stage1,
                            'max_epochs_stage2': cfg.training.max_epochs_stage2,
                            'lr_warmup_epochs': cfg.training.lr_warmup_epochs,
                            'stage2_lr_factor': cfg.training.stage2_lr_factor,
                            'early_stopping_patience': cfg.training.early_stopping_patience,
                            'gradient_accumulation_steps_stage1': cfg.training.gradient_accumulation_steps_stage1,
                            'gradient_accumulation_steps_stage2': cfg.training.gradient_accumulation_steps_stage2,
                            'use_mixed_precision': cfg.training.use_mixed_precision,
                            'use_fused_optimizer': cfg.training.use_fused_optimizer,
                            'use_torch_compile': cfg.training.use_torch_compile,
                            'compile_mode': cfg.training.compile_mode,
                            'use_channels_last': cfg.training.use_channels_last,
                            'centroid_normalize_every_n_batches': cfg.training.centroid_normalize_every_n_batches,
                            'lambda_start': cfg.training.lambda_start,
                            'lambda_end': cfg.training.lambda_end,
                            'lambda_warmup_epochs': cfg.training.lambda_warmup_epochs,
                        },
                        'augmentation': {
                            'jitter_sigma': cfg.augmentation.jitter_sigma,
                            'scale_range': list(cfg.augmentation.scale_range),
                            'mask_max_length_pct': cfg.augmentation.mask_max_length_pct,
                            'apply_jitter': cfg.augmentation.apply_jitter,
                            'apply_scaling': cfg.augmentation.apply_scaling,
                            'apply_masking': cfg.augmentation.apply_masking,
                        }
                    })
                with col_b:
                    if st.button("🗑️ Remove", key=f"remove_config_{idx}"):
                        st.session_state['config_queue'].pop(idx)
                        st.rerun()

        # Clear queue button
        if st.button("🗑️ Clear Queue"):
            st.session_state['config_queue'] = []
            st.rerun()

def show_training():
    """Step 3: Train queued configurations sequentially."""
    st.header("Step 3: Train Model")
    st.markdown("Train all queued configurations sequentially")

    # Check prerequisites
    if 'data' not in st.session_state:
        st.warning("⚠️ Please load data first (Step 1)")
        return

    # Check if there are configs in the queue
    if 'config_queue' not in st.session_state or len(st.session_state['config_queue']) == 0:
        st.warning("⚠️ No configurations in queue. Please add configurations in Step 2 first.")
        st.info("💡 Go to Step 2: Configure Model and click '➕ Add to Queue' to queue configurations")
        return

    # Show queued configurations
    st.markdown(f"**{len(st.session_state['config_queue'])} configuration(s) queued for training**")

    for idx, config_entry in enumerate(st.session_state['config_queue']):
        with st.expander(f"Config {idx+1}: {config_entry['name']}", expanded=False):
            cfg = config_entry['config']
            st.json({
                'name': config_entry['name'],
                'model': {
                    'd_z': cfg.model.d_z,
                    'num_clusters': cfg.model.num_clusters,
                    'tau': cfg.model.tau,
                    'use_hybrid_encoder': cfg.model.use_hybrid_encoder,
                    'encoder_hidden_channels': cfg.model.encoder_hidden_channels,
                    'projection_hidden_dim': cfg.model.projection_hidden_dim,
                    'fusion_hidden_dim': cfg.model.fusion_hidden_dim,
                    'use_projection_bottleneck': cfg.model.use_projection_bottleneck,
                },
                'training': {
                    'batch_size': cfg.training.batch_size,
                    'learning_rate': cfg.training.learning_rate,
                    'max_epochs_stage1': cfg.training.max_epochs_stage1,
                    'max_epochs_stage2': cfg.training.max_epochs_stage2,
                    'lr_warmup_epochs': cfg.training.lr_warmup_epochs,
                    'stage2_lr_factor': cfg.training.stage2_lr_factor,
                    'early_stopping_patience': cfg.training.early_stopping_patience,
                    'gradient_accumulation_steps_stage1': cfg.training.gradient_accumulation_steps_stage1,
                    'gradient_accumulation_steps_stage2': cfg.training.gradient_accumulation_steps_stage2,
                    'use_mixed_precision': cfg.training.use_mixed_precision,
                    'use_fused_optimizer': cfg.training.use_fused_optimizer,
                    'use_torch_compile': cfg.training.use_torch_compile,
                    'compile_mode': cfg.training.compile_mode,
                    'use_channels_last': cfg.training.use_channels_last,
                    'centroid_normalize_every_n_batches': cfg.training.centroid_normalize_every_n_batches,
                    'lambda_start': cfg.training.lambda_start,
                    'lambda_end': cfg.training.lambda_end,
                    'lambda_warmup_epochs': cfg.training.lambda_warmup_epochs,
                },
                'augmentation': {
                    'jitter_sigma': cfg.augmentation.jitter_sigma,
                    'scale_range': list(cfg.augmentation.scale_range),
                    'mask_max_length_pct': cfg.augmentation.mask_max_length_pct,
                    'apply_jitter': cfg.augmentation.apply_jitter,
                    'apply_scaling': cfg.augmentation.apply_scaling,
                    'apply_masking': cfg.augmentation.apply_masking,
                }
            })

    st.divider()

    # Training settings (shared for all configs)
    st.subheader("🎯 Training Setup")
    st.markdown("These settings will apply to all queued configurations")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_split = st.slider("Train %", 50, 80, 70, key="multi_train_split")
    with col2:
        val_split = st.slider("Val %", 10, 30, 15, key="multi_val_split")
    with col3:
        test_split = 100 - train_split - val_split
        st.metric("Test %", test_split)

    # Device selection
    device = st.selectbox(
        "Device",
        options=['cuda' if torch.cuda.is_available() else 'cpu', 'cpu'],
        index=0,
        key="multi_device"
    )

    if device == 'cuda' and not torch.cuda.is_available():
        st.error("CUDA not available. Falling back to CPU.")
        device = 'cpu'

    # Stage 1 only mode toggle
    st.divider()
    end_after_stage1 = st.checkbox(
        "End training after Stage 1 (for all configs)",
        value=False,
        help="Complete only Stage 1 (contrastive pre-training) and skip Stage 2 (clustering)."
    )

    # Start training button
    st.divider()
    if st.button("🚀 Start Sequential Training", type="primary", key="start_multi_training"):
        # Add training settings to each config entry
        config_queue_with_settings = []
        for config_entry in st.session_state['config_queue']:
            entry_with_settings = config_entry.copy()
            entry_with_settings['train_split'] = train_split
            entry_with_settings['val_split'] = val_split
            entry_with_settings['test_split'] = test_split
            entry_with_settings['end_after_stage1'] = end_after_stage1
            config_queue_with_settings.append(entry_with_settings)

        run_multi_config_training(config_queue_with_settings, device)

    # Show training summaries comparison table
    st.divider()
    st.subheader("📊 Training History Comparison")
    show_training_summaries()


def run_multi_config_training(config_queue, device):
    """Execute training for all configurations in the queue sequentially."""
    st.subheader("🔄 Sequential Training Progress")

    total_configs = len(config_queue)
    overall_progress = st.progress(0)
    overall_status = st.empty()

    # Create containers for current config training
    current_config_container = st.container()

    for config_idx, config_entry in enumerate(config_queue):
        overall_status.markdown(f"### Training Configuration {config_idx+1}/{total_configs}: {config_entry['name']}")

        with current_config_container:
            st.divider()
            st.markdown(f"#### Configuration: {config_entry['name']}")

            # Display config info
            st.info(f"""
            **Configuration:**
            - d_z: {config_entry['config'].model.d_z}
            - Clusters: {config_entry['config'].model.num_clusters}
            - Batch Size: {config_entry['config'].training.batch_size}
            - Learning Rate: {config_entry['config'].training.learning_rate}
            - Train/Val/Test: {config_entry['train_split']}/{config_entry['val_split']}/{config_entry['test_split']}%
            - Stage 1 Only: {config_entry['end_after_stage1']}
            """)

            try:
                # Prepare data splits
                data = st.session_state['data']
                n = len(data)
                train_size = int(n * config_entry['train_split'] / 100)
                val_size = int(n * config_entry['val_split'] / 100)

                train_data = data[:train_size]
                val_data = data[train_size:train_size+val_size]
                test_data = data[train_size+val_size:]

                # Handle GPU preloading if enabled
                config = config_entry['config']
                if config.training.preload_to_gpu and device == 'cuda':
                    with st.spinner("Preloading dataset to GPU..."):
                        try:
                            if torch.cuda.is_available():
                                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                                dataset_size = train_data.element_size() * train_data.nelement()
                                dataset_size += val_data.element_size() * val_data.nelement()
                                dataset_size += test_data.element_size() * test_data.nelement()

                                if dataset_size > free_memory * 0.8:
                                    st.warning(f"Dataset ({dataset_size/1e9:.2f}GB) may not fit in GPU memory. Using standard loading.")
                                    config.training.preload_to_gpu = False
                                else:
                                    train_data = train_data.to(device)
                                    val_data = val_data.to(device)
                                    test_data = test_data.to(device)
                                    st.success(f"✅ Dataset preloaded to GPU ({dataset_size/1e9:.2f}GB)")
                        except Exception as e:
                            st.warning(f"Failed to preload to GPU: {e}. Using standard loading.")
                            config.training.preload_to_gpu = False

                # Adjust DataLoader settings
                use_pin_memory = config.training.pin_memory and not config.training.preload_to_gpu and device == 'cuda'
                use_num_workers = 0 if config.training.preload_to_gpu else config.training.num_workers
                use_persistent_workers = config.training.persistent_workers and use_num_workers > 0
                use_prefetch_factor = config.training.prefetch_factor if use_num_workers > 0 else None

                # Create data loaders
                dataloader_kwargs = {
                    'batch_size': config.training.batch_size,
                    'num_workers': use_num_workers,
                    'pin_memory': use_pin_memory,
                    'persistent_workers': use_persistent_workers
                }
                if use_prefetch_factor is not None:
                    dataloader_kwargs['prefetch_factor'] = use_prefetch_factor

                train_loader = DataLoader(
                    TensorDataset(train_data),
                    shuffle=True,
                    **dataloader_kwargs
                )
                val_loader = DataLoader(
                    TensorDataset(val_data),
                    shuffle=False,
                    **dataloader_kwargs
                )
                test_loader = DataLoader(
                    TensorDataset(test_data),
                    shuffle=False,
                    **dataloader_kwargs
                )

                # Create model
                with st.spinner("Creating model..."):
                    seq_len = st.session_state['data'].shape[2]

                    model = UCLTSCModel(
                        input_channels=3,
                        d_z=config.model.d_z,
                        num_clusters=config.model.num_clusters,
                        use_hybrid_encoder=config.model.use_hybrid_encoder,
                        seq_length=seq_len,
                        encoder_hidden_channels=config.model.encoder_hidden_channels,
                        projection_hidden_dim=config.model.projection_hidden_dim,
                        fusion_hidden_dim=config.model.fusion_hidden_dim,
                        use_projection_bottleneck=config.model.use_projection_bottleneck
                    )

                    n_params = sum(p.numel() for p in model.parameters())
                    st.success(f"✅ Model created: {n_params:,} parameters")

                # Create trainer
                trainer = TwoStageTrainer(model, config, device=device)

                # Training progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.empty()

                # Custom callback
                class StreamlitCallback:
                    def __init__(self, progress_bar, status_text, metrics_container, config):
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.metrics_container = metrics_container
                        self.stage1_epochs = config.training.max_epochs_stage1
                        self.stage2_epochs = config.training.max_epochs_stage2

                    def on_epoch_end(self, stage, epoch, total_epochs, metrics):
                        if stage == 1:
                            progress = epoch / (self.stage1_epochs + self.stage2_epochs)
                            self.status_text.markdown(f"Stage 1: Epoch {epoch}/{total_epochs}")
                        else:
                            progress = (self.stage1_epochs + epoch) / (self.stage1_epochs + self.stage2_epochs)
                            self.status_text.markdown(f"Stage 2: Epoch {epoch}/{total_epochs}")

                        self.progress_bar.progress(min(progress, 1.0))

                        cols = self.metrics_container.columns(len(metrics))
                        for i, (key, value) in enumerate(metrics.items()):
                            with cols[i]:
                                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")

                callback = StreamlitCallback(progress_bar, status_text, metrics_container, config)

                # Prepare dataset info for console logging
                dataset_info = {
                    'total_windows': len(st.session_state['data']),
                    'window_length': st.session_state['data'].shape[2],
                    'num_channels': st.session_state['data'].shape[1]
                }

                # Train
                with st.spinner(f"Training {config_entry['name']}..."):
                    history = trainer.train(
                        train_loader,
                        val_loader,
                        test_loader,
                        callback=callback,
                        stage1_only=config_entry['end_after_stage1'],
                        dataset_info=dataset_info
                    )

                progress_bar.progress(0.95)

                # Calibrate confidence scores
                st.info("Calibrating confidence scores...")
                try:
                    calibration_results = calibrate_confidence_scores(model, val_loader, device)
                    if calibration_results['passed']:
                        st.success(f"[PASS] Confidence calibration successful! Best gamma: {calibration_results['best_gamma']}, R²: {calibration_results['best_r2']:.3f}")
                    else:
                        st.warning(f"[WARN] Confidence calibration below threshold (R² = {calibration_results['best_r2']:.3f} < 0.7)")
                except Exception as e:
                    st.warning(f"Confidence calibration failed: {e}")

                progress_bar.progress(1.0)
                st.success(f"✅ {config_entry['name']} training complete!")

                # Collect dataset info
                dataset_info = {
                    'total_windows': len(st.session_state['data']),
                    'window_length': st.session_state['data'].shape[2],
                    'num_channels': st.session_state['data'].shape[1]
                }

                # Save training summary
                save_training_summary(
                    data_package=st.session_state.get('data_file', 'unknown'),
                    config=config,
                    history=history,
                    end_after_stage1=config_entry['end_after_stage1'],
                    device=device,
                    train_split=config_entry['train_split'],
                    val_split=config_entry['val_split'],
                    test_split=config_entry['test_split'],
                    dataset_info=dataset_info
                )

                # Save model automatically
                model_dir = Path("models/trained")
                model_dir.mkdir(parents=True, exist_ok=True)
                model_name = f"{config_entry['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_path = model_dir / f"{model_name}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'history': history
                }, model_path)
                st.info(f"💾 Model saved: {model_path.name}")

            except Exception as e:
                st.error(f"❌ Training failed for {config_entry['name']}: {e}")
                st.exception(e)

                # Ask if user wants to continue
                if config_idx < total_configs - 1:
                    st.warning("Continuing with next configuration...")

        # Update overall progress
        overall_progress.progress((config_idx + 1) / total_configs)

    # All done
    overall_status.markdown("### 🎉 All Configurations Complete!")
    overall_progress.progress(1.0)
    st.balloons()

    # Clear the queue
    if st.button("Clear Completed Queue"):
        st.session_state['config_queue'] = []
        st.rerun()


def load_training_summaries():
    """Load all training summaries from JSON files."""
    summary_dir = Path("data/training_logs")
    if not summary_dir.exists():
        return []

    summaries = []
    for json_file in sorted(summary_dir.glob("training_summary_*.json"), reverse=True):
        try:
            with open(json_file, 'r') as f:
                summary = json.load(f)
                summaries.append(summary)
        except Exception as e:
            st.warning(f"Failed to load {json_file.name}: {e}")

    return summaries


def create_full_export_dataframe(summaries):
    """
    Create a comprehensive DataFrame with ALL configuration options for AI analysis.
    Uses clear column naming and proper data types for easy analysis.
    """
    rows = []

    for summary in summaries:
        row = {
            # Metadata
            'run_id': summary.get('timestamp', ''),
            'datetime': summary.get('datetime', ''),
            'data_package': summary.get('data_package', ''),
            'device': summary.get('device', ''),
            'stage1_only': summary.get('end_after_stage1', False),

            # Dataset Properties (for AI analysis of data scale impact)
            'dataset_total_windows': summary.get('dataset_total_windows', None),
            'dataset_window_length_bars': summary.get('dataset_window_length', None),
            'dataset_num_feature_channels': summary.get('dataset_num_channels', None),

            # Data splits
            'split_train_pct': summary.get('train_split', 0),
            'split_val_pct': summary.get('val_split', 0),
            'split_test_pct': summary.get('test_split', 0),

            # Model Architecture
            'model_d_z': summary.get('model', {}).get('d_z', 0),
            'model_num_clusters': summary.get('model', {}).get('num_clusters', 0),
            'model_tau': summary.get('model', {}).get('tau', 0),
            'model_use_hybrid_encoder': summary.get('model', {}).get('use_hybrid_encoder', False),
            'model_seq_length': summary.get('model', {}).get('seq_length', 0),
            'model_encoder_hidden_channels': summary.get('model', {}).get('encoder_hidden_channels', 0),
            'model_projection_hidden_dim': summary.get('model', {}).get('projection_hidden_dim', 0),
            'model_fusion_hidden_dim': summary.get('model', {}).get('fusion_hidden_dim', 0),
            'model_use_projection_bottleneck': summary.get('model', {}).get('use_projection_bottleneck', False),

            # Training Hyperparameters
            'train_batch_size': summary.get('training', {}).get('batch_size', 0),
            'train_learning_rate': summary.get('training', {}).get('learning_rate', 0),
            'train_max_epochs_stage1': summary.get('training', {}).get('max_epochs_stage1', 0),
            'train_max_epochs_stage2': summary.get('training', {}).get('max_epochs_stage2', 0),
            'train_lr_warmup_epochs': summary.get('training', {}).get('lr_warmup_epochs', 0),
            'train_stage2_lr_factor': summary.get('training', {}).get('stage2_lr_factor', 0),
            'train_early_stopping_patience': summary.get('training', {}).get('early_stopping_patience', 0),

            # Training Optimizations
            'train_grad_accum_stage1': summary.get('training', {}).get('gradient_accumulation_steps_stage1', 1),
            'train_grad_accum_stage2': summary.get('training', {}).get('gradient_accumulation_steps_stage2', 1),
            'train_use_mixed_precision': summary.get('training', {}).get('use_mixed_precision', False),
            'train_num_workers': summary.get('training', {}).get('num_workers', 0),
            'train_pin_memory': summary.get('training', {}).get('pin_memory', False),
            'train_preload_to_gpu': summary.get('training', {}).get('preload_to_gpu', False),
            'train_persistent_workers': summary.get('training', {}).get('persistent_workers', False),
            'train_prefetch_factor': summary.get('training', {}).get('prefetch_factor', 2),
            'train_use_fused_optimizer': summary.get('training', {}).get('use_fused_optimizer', False),
            'train_use_torch_compile': summary.get('training', {}).get('use_torch_compile', False),
            'train_compile_mode': summary.get('training', {}).get('compile_mode', 'default'),
            'train_use_channels_last': summary.get('training', {}).get('use_channels_last', False),

            # Stage 2 Specific
            'train_centroid_normalize_freq': summary.get('training', {}).get('centroid_normalize_every_n_batches', 10),
            'train_lambda_start': summary.get('training', {}).get('lambda_start', 0),
            'train_lambda_end': summary.get('training', {}).get('lambda_end', 0),
            'train_lambda_warmup_epochs': summary.get('training', {}).get('lambda_warmup_epochs', 0),

            # Augmentation Settings
            'aug_jitter_sigma': summary.get('augmentation', {}).get('jitter_sigma', 0),
            'aug_scale_min': summary.get('augmentation', {}).get('scale_range', [1.0, 1.0])[0],
            'aug_scale_max': summary.get('augmentation', {}).get('scale_range', [1.0, 1.0])[1],
            'aug_mask_max_length_pct': summary.get('augmentation', {}).get('mask_max_length_pct', 0),
            'aug_apply_jitter': summary.get('augmentation', {}).get('apply_jitter', False),
            'aug_apply_scaling': summary.get('augmentation', {}).get('apply_scaling', False),
            'aug_apply_masking': summary.get('augmentation', {}).get('apply_masking', False),

            # Stage 1 Results
            'result_s1_epochs_completed': summary.get('results', {}).get('stage1_epochs_completed', 0),
            'result_s1_best_val_loss': summary.get('results', {}).get('stage1_best_val_loss', None),
            'result_s1_final_train_loss': summary.get('results', {}).get('stage1_final_train_loss', None),
            'result_s1_final_val_loss': summary.get('results', {}).get('stage1_final_val_loss', None),

            # Stage 1 Quality Metrics
            'metric_variance': summary.get('results', {}).get('variance', None),
            'metric_variance_pass': 1 if summary.get('results', {}).get('variance_pass', False) else 0,
            'metric_effective_rank': summary.get('results', {}).get('effective_rank', None),
            'metric_effective_rank_pass': 1 if summary.get('results', {}).get('effective_rank_pass', False) else 0,
            'metric_alignment': summary.get('results', {}).get('alignment', None),
            'metric_alignment_pass': 1 if summary.get('results', {}).get('alignment_pass', False) else 0,
            'metric_uniformity': summary.get('results', {}).get('uniformity', None),
            'metric_uniformity_pass': 1 if summary.get('results', {}).get('uniformity_pass', False) else 0,
            'metric_knn_accuracy': summary.get('results', {}).get('knn_accuracy', None),
            'metric_knn_accuracy_pass': 1 if summary.get('results', {}).get('knn_accuracy_pass', False) else 0,
            'metric_overall_score': summary.get('results', {}).get('overall_score', None),
            'metric_n_passed': summary.get('results', {}).get('n_passed', 0),
            'metric_n_total': summary.get('results', {}).get('n_total', 0),

            # Stage 2 Results (if available)
            'result_s2_epochs_completed': summary.get('results', {}).get('stage2_epochs_completed', None),
            'result_s2_best_cluster_loss': summary.get('results', {}).get('stage2_best_cluster_loss', None),
            'result_s2_final_train_loss': summary.get('results', {}).get('stage2_final_train_loss', None),
            'result_s2_final_val_loss': summary.get('results', {}).get('stage2_final_val_loss', None),
            'result_s2_final_cluster_loss': summary.get('results', {}).get('stage2_final_cluster_loss', None),
            'result_s2_final_contrastive_loss': summary.get('results', {}).get('stage2_final_contrastive_loss', None),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by datetime (most recent first)
    if 'datetime' in df.columns:
        df = df.sort_values('datetime', ascending=False)

    return df


def show_training_summaries():
    """Display comparison table of all training runs."""
    summaries = load_training_summaries()

    if not summaries:
        st.info("No training summaries found. Train a model to see results here.")
        return

    st.markdown(f"**Total training runs:** {len(summaries)}")

    # Define available columns for configuration
    config_columns = {
        'timestamp': 'Timestamp',
        'data_package': 'Data Package',
        'end_after_stage1': 'Stage 1 Only',
        'device': 'Device',

        # Dataset properties
        'dataset_total_windows': 'Total Windows',
        'dataset_window_length': 'Window Size',
        'dataset_num_channels': 'Channels',

        # Model config
        'model.d_z': 'd_z',
        'model.num_clusters': 'Clusters',
        'model.tau': 'Tau',
        'model.use_hybrid_encoder': 'Hybrid',
        'model.encoder_hidden_channels': 'Enc Hidden Ch',
        'model.projection_hidden_dim': 'Proj Hidden',
        'model.fusion_hidden_dim': 'Fusion Hidden',
        'model.use_projection_bottleneck': 'Proj Bottleneck',

        # Training config
        'training.batch_size': 'Batch Size',
        'training.learning_rate': 'LR',
        'training.max_epochs_stage1': 'Max Epochs S1',
        'training.max_epochs_stage2': 'Max Epochs S2',
        'training.lr_warmup_epochs': 'LR Warmup',
        'training.early_stopping_patience': 'Patience',
        'training.gradient_accumulation_steps_stage1': 'Grad Accum S1',
        'training.gradient_accumulation_steps_stage2': 'Grad Accum S2',
        'training.use_mixed_precision': 'FP16',
        'training.use_fused_optimizer': 'Fused Opt',
        'training.use_torch_compile': 'Compile',
        'training.compile_mode': 'Compile Mode',
        'training.use_channels_last': 'Ch Last',

        # Augmentation
        'augmentation.jitter_sigma': 'Jitter σ',
        'augmentation.mask_max_length_pct': 'Mask %',
    }

    result_columns = {
        'results.stage1_epochs_completed': 'S1 Epochs',
        'results.stage1_best_val_loss': 'S1 Best Val',
        'results.stage1_final_val_loss': 'S1 Final Val',
        'results.variance': 'Variance',
        'results.variance_pass': 'Var ✓',
        'results.effective_rank': 'Eff Rank',
        'results.effective_rank_pass': 'Rank ✓',
        'results.alignment': 'Alignment',
        'results.alignment_pass': 'Align ✓',
        'results.uniformity': 'Uniformity',
        'results.uniformity_pass': 'Unif ✓',
        'results.knn_accuracy': 'k-NN Acc',
        'results.knn_accuracy_pass': 'k-NN ✓',
        'results.overall_score': 'Overall %',
        'results.n_passed': 'Passed',
    }

    # Column selector
    with st.expander("🔧 Select Columns to Display", expanded=False):
        st.markdown("**Choose which parameters to show in the table:**")

        # Create columns for better layout
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Basic**")
            selected_cols = []
            if st.checkbox("Timestamp", value=True, key="col_timestamp"):
                selected_cols.append('timestamp')
            if st.checkbox("Data Package", value=True, key="col_data_package"):
                selected_cols.append('data_package')
            if st.checkbox("Stage 1 Only", value=True, key="col_stage1_only"):
                selected_cols.append('end_after_stage1')
            if st.checkbox("Device", value=False, key="col_device"):
                selected_cols.append('device')

            st.markdown("**Dataset Properties**")
            if st.checkbox("Total Windows", value=False, key="col_total_windows"):
                selected_cols.append('dataset_total_windows')
            if st.checkbox("Window Size", value=False, key="col_window_size"):
                selected_cols.append('dataset_window_length')
            if st.checkbox("Channels", value=False, key="col_channels"):
                selected_cols.append('dataset_num_channels')

            st.markdown("**Model Architecture**")
            if st.checkbox("d_z", value=True, key="col_d_z"):
                selected_cols.append('model.d_z')
            if st.checkbox("Clusters", value=True, key="col_clusters"):
                selected_cols.append('model.num_clusters')
            if st.checkbox("Tau", value=True, key="col_tau"):
                selected_cols.append('model.tau')
            if st.checkbox("Hybrid Encoder", value=False, key="col_hybrid"):
                selected_cols.append('model.use_hybrid_encoder')
            if st.checkbox("Encoder Hidden Ch", value=False, key="col_enc_hidden"):
                selected_cols.append('model.encoder_hidden_channels')
            if st.checkbox("Projection Hidden", value=False, key="col_proj_hidden"):
                selected_cols.append('model.projection_hidden_dim')
            if st.checkbox("Fusion Hidden", value=False, key="col_fusion"):
                selected_cols.append('model.fusion_hidden_dim')

        with col2:
            st.markdown("**Training Parameters**")
            if st.checkbox("Batch Size", value=True, key="col_batch"):
                selected_cols.append('training.batch_size')
            if st.checkbox("Learning Rate", value=True, key="col_lr"):
                selected_cols.append('training.learning_rate')
            if st.checkbox("Max Epochs S1", value=False, key="col_max_s1"):
                selected_cols.append('training.max_epochs_stage1')
            if st.checkbox("LR Warmup", value=False, key="col_warmup"):
                selected_cols.append('training.lr_warmup_epochs')
            if st.checkbox("Patience", value=False, key="col_patience"):
                selected_cols.append('training.early_stopping_patience')
            if st.checkbox("Grad Accum S1", value=True, key="col_grad_s1"):
                selected_cols.append('training.gradient_accumulation_steps_stage1')
            if st.checkbox("FP16", value=False, key="col_fp16"):
                selected_cols.append('training.use_mixed_precision')
            if st.checkbox("Fused Optimizer", value=False, key="col_fused"):
                selected_cols.append('training.use_fused_optimizer')
            if st.checkbox("Torch Compile", value=False, key="col_compile"):
                selected_cols.append('training.use_torch_compile')

        with col3:
            st.markdown("**Augmentation**")
            if st.checkbox("Jitter Sigma", value=True, key="col_jitter"):
                selected_cols.append('augmentation.jitter_sigma')
            if st.checkbox("Mask %", value=False, key="col_mask"):
                selected_cols.append('augmentation.mask_max_length_pct')

        # Result columns selection
        selected_result_cols = []
        with col4:
            st.markdown("**Stage 1 Results**")
            if st.checkbox("S1 Epochs", value=True, key="col_s1_epochs"):
                selected_result_cols.append('results.stage1_epochs_completed')
            if st.checkbox("S1 Best Val", value=True, key="col_s1_best_val"):
                selected_result_cols.append('results.stage1_best_val_loss')
            if st.checkbox("S1 Final Val", value=False, key="col_s1_final_val"):
                selected_result_cols.append('results.stage1_final_val_loss')

            st.markdown("**Quality Metrics**")
            if st.checkbox("Variance", value=True, key="col_variance"):
                selected_result_cols.append('results.variance')
            if st.checkbox("Var [PASS]", value=False, key="col_variance_pass"):
                selected_result_cols.append('results.variance_pass')
            if st.checkbox("Eff Rank", value=True, key="col_eff_rank"):
                selected_result_cols.append('results.effective_rank')
            if st.checkbox("Rank [PASS]", value=False, key="col_eff_rank_pass"):
                selected_result_cols.append('results.effective_rank_pass')
            if st.checkbox("Alignment", value=True, key="col_alignment"):
                selected_result_cols.append('results.alignment')
            if st.checkbox("Align [PASS]", value=False, key="col_alignment_pass"):
                selected_result_cols.append('results.alignment_pass')
            if st.checkbox("Uniformity", value=True, key="col_uniformity"):
                selected_result_cols.append('results.uniformity')
            if st.checkbox("Unif [PASS]", value=False, key="col_unif_pass"):
                selected_result_cols.append('results.uniformity_pass')
            if st.checkbox("k-NN Acc", value=True, key="col_knn"):
                selected_result_cols.append('results.knn_accuracy')
            if st.checkbox("k-NN [PASS]", value=False, key="col_knn_pass"):
                selected_result_cols.append('results.knn_accuracy_pass')
            if st.checkbox("Overall %", value=True, key="col_overall"):
                selected_result_cols.append('results.overall_score')
            if st.checkbox("Passed/Total", value=False, key="col_passed"):
                selected_result_cols.append('results.n_passed')

    # Build dataframe (all runs for display)
    rows = []
    for summary in summaries:
        row = {}

        # Add selected configuration columns
        for col in selected_cols:
            parts = col.split('.')
            value = summary
            for part in parts:
                value = value.get(part, None)
                if value is None:
                    break
            row[config_columns[col]] = value

        # Add selected result columns only
        for col in selected_result_cols:
            if col not in result_columns:
                continue

            label = result_columns[col]
            parts = col.split('.')
            value = summary
            for part in parts:
                value = value.get(part, None)
                if value is None:
                    break

            # Format based on type
            if value is not None:
                if isinstance(value, bool):
                    row[label] = '[PASS]' if value else '[FAIL]'
                elif isinstance(value, float):
                    row[label] = f"{value:.4f}"
                else:
                    row[label] = value
            else:
                row[label] = '-'

        rows.append(row)

    df = pd.DataFrame(rows)

    # Display table with row selection enabled
    event = st.dataframe(
        df,
        use_container_width=True,
        height=400,
        selection_mode="multi-row",
        on_select="rerun",
        key="training_table",
        column_config={
            # Dataset Properties
            "Total Windows": st.column_config.NumberColumn(
                "Total Windows",
                help="Total number of time series windows in the prepackaged dataset",
                format="%d"
            ),
            "Window Size": st.column_config.NumberColumn(
                "Window Size",
                help="Length of each time series window (sequence length)",
                format="%d"
            ),
            "Channels": st.column_config.NumberColumn(
                "Channels",
                help="Number of feature channels (e.g., OHLCV = 5 channels)",
                format="%d"
            ),

            # Stage 1 Result Columns
            "S1 Epochs": st.column_config.NumberColumn(
                "S1 Epochs",
                help="Number of epochs completed in Stage 1"
            ),
            "S1 Best Val": st.column_config.NumberColumn(
                "S1 Best Val",
                help="Best validation loss achieved during Stage 1",
                format="%.4f"
            ),
            "S1 Final Val": st.column_config.NumberColumn(
                "S1 Final Val",
                help="Final validation loss at end of Stage 1",
                format="%.4f"
            ),

            # Quality Metrics with Pass/Fail Thresholds
            "Variance": st.column_config.NumberColumn(
                "Variance",
                help="Embedding variance - measures feature spread. PASS: >0.25 (good spread), FAIL: ≤0.25 (feature collapse)",
                format="%.4f"
            ),
            "Var ✓": st.column_config.TextColumn(
                "Var ✓",
                help="PASS if variance >0.25, FAIL otherwise"
            ),
            "Eff Rank": st.column_config.NumberColumn(
                "Eff Rank",
                help="Effective rank - measures dimensionality usage. PASS: >60% of d_z, FAIL: ≤60% of d_z",
                format="%.2f"
            ),
            "Rank ✓": st.column_config.TextColumn(
                "Rank ✓",
                help="PASS if effective rank >60% of d_z, FAIL otherwise"
            ),
            "Alignment": st.column_config.NumberColumn(
                "Alignment",
                help="Alignment - how close are positive pairs. PASS: <0.5 (pairs close), FAIL: ≥0.5 (pairs too far)",
                format="%.4f"
            ),
            "Align ✓": st.column_config.TextColumn(
                "Align ✓",
                help="PASS if alignment <0.5, FAIL otherwise"
            ),
            "Uniformity": st.column_config.NumberColumn(
                "Uniformity",
                help="Uniformity - how evenly embeddings spread on hypersphere. PASS: <-1.5 (uniform), FAIL: ≥-1.5 (clustered)",
                format="%.4f"
            ),
            "Unif ✓": st.column_config.TextColumn(
                "Unif ✓",
                help="PASS if uniformity <-1.5, FAIL otherwise"
            ),
            "k-NN Acc": st.column_config.NumberColumn(
                "k-NN Acc",
                help="k-NN accuracy - augmentation invariance proxy. PASS: >0.75 (strong invariance), FAIL: ≤0.75 (weak invariance)",
                format="%.4f"
            ),
            "k-NN ✓": st.column_config.TextColumn(
                "k-NN ✓",
                help="PASS if k-NN accuracy >0.75, FAIL otherwise"
            ),
            "Overall %": st.column_config.NumberColumn(
                "Overall %",
                help="Overall quality score. EXCELLENT: ≥80%, GOOD: ≥60%, FAIR: ≥40%, POOR: <40%",
                format="%.1f"
            ),
            "Passed": st.column_config.NumberColumn(
                "Passed",
                help="Number of metrics that passed out of 5 total (variance, effective rank, alignment, uniformity, k-NN accuracy)"
            ),
        }
    )

    # Get selected row indices
    selected_rows = event.selection.rows if hasattr(event.selection, 'rows') else []

    # Action buttons
    st.divider()
    if selected_rows:
        st.info(f"📊 {len(selected_rows)} row(s) selected")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Delete Selected", type="secondary"):
                # Delete corresponding JSON files
                summary_dir = Path("data/training_logs")
                deleted_count = 0
                for idx in sorted(selected_rows, reverse=True):
                    if idx < len(summaries):
                        timestamp = summaries[idx].get('timestamp', '')
                        json_file = summary_dir / f"training_summary_{timestamp}.json"
                        if json_file.exists():
                            json_file.unlink()
                            deleted_count += 1

                st.success(f"✅ Deleted {deleted_count} training run(s)")
                st.rerun()

        with col2:
            # Export Display View
            selected_summaries = [summaries[idx] for idx in selected_rows if idx < len(summaries)]

            # Build filtered dataframe for display view export
            filtered_rows = []
            for idx in selected_rows:
                if idx < len(df):
                    filtered_rows.append(rows[idx])

            df_filtered = pd.DataFrame(filtered_rows)
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label=f"📥 Export Display View ({len(selected_rows)} runs)",
                data=csv,
                file_name=f"training_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_display"
            )

        with col3:
            # Export Full AI-friendly format
            full_export = create_full_export_dataframe(selected_summaries)
            csv_full = full_export.to_csv(index=False)
            st.download_button(
                label=f"📥 Export Full - AI-friendly ({len(selected_rows)} runs)",
                data=csv_full,
                file_name=f"training_full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_full"
            )
    else:
        st.info("💡 Select rows in the table above to export or delete training runs")


def show_training_results(history):
    """Display training results."""
    st.subheader("📈 Training Results")

    # Check if Stage 2 was completed
    has_stage2 = ('stage2' in history and
                  history['stage2']['train_loss'] and
                  history['stage2']['cluster_loss'])

    # Create plots
    if has_stage2:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Stage 1: Contrastive Loss',
                'Stage 2: Total Loss',
                'Stage 2: Clustering Loss',
                'Stage 2: Contrastive Loss'
            )
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Stage 1: Contrastive Loss',)
        )

    # Stage 1 losses
    epochs_s1 = list(range(1, len(history['stage1']['train_loss']) + 1))
    fig.add_trace(
        go.Scatter(x=epochs_s1, y=history['stage1']['train_loss'], name='Train', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs_s1, y=history['stage1']['val_loss'], name='Val', line=dict(color='orange')),
        row=1, col=1
    )

    if has_stage2:
        # Stage 2 losses
        epochs_s2 = list(range(1, len(history['stage2']['train_loss']) + 1))
        fig.add_trace(
            go.Scatter(x=epochs_s2, y=history['stage2']['train_loss'], name='Train', line=dict(color='blue'), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs_s2, y=history['stage2']['val_loss'], name='Val', line=dict(color='orange'), showlegend=False),
            row=1, col=2
        )

        # Clustering loss
        fig.add_trace(
            go.Scatter(x=epochs_s2, y=history['stage2']['cluster_loss'], name='Cluster', line=dict(color='green')),
            row=2, col=1
        )

        # Contrastive loss
        fig.add_trace(
            go.Scatter(x=epochs_s2, y=history['stage2']['contrastive_loss'], name='Contrastive', line=dict(color='red')),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(height=600 if has_stage2 else 400, showlegend=True)

    st.plotly_chart(fig, width='stretch', key='training_losses_chart')

    # Summary metrics - Two column layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Training Summary")

        # Basic training metrics
        if has_stage2:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Stage 1 Epochs",
                len(history['stage1']['train_loss'])
            )
        with col2:
            st.metric(
                "Best S1 Val Loss",
                f"{min(history['stage1']['val_loss']):.4f}"
            )

        if has_stage2:
            with col3:
                st.metric(
                    "Stage 2 Epochs",
                    len(history['stage2']['train_loss'])
                )
            with col4:
                st.metric(
                    "Best Cluster Loss",
                    f"{min(history['stage2']['cluster_loss']):.4f}"
                )

    with col_right:
        st.markdown("### Stage 1 Quality Metrics")

        # Check if metrics are available
        stage1_metrics = history.get('stage1', {}).get('metrics', {})

        if stage1_metrics:
            # Display metrics in a compact table format
            metrics_data = []

            # Variance
            var_val = stage1_metrics.get('variance', 0)
            var_pass = stage1_metrics.get('variance_pass', False)
            metrics_data.append({
                'Metric': 'Variance',
                'Value': f"{var_val:.4f}",
                'Status': '[PASS]' if var_pass else '[FAIL]'
            })

            # Effective Rank
            rank_val = stage1_metrics.get('effective_rank', 0)
            rank_pass = stage1_metrics.get('effective_rank_pass', False)
            metrics_data.append({
                'Metric': 'Effective Rank',
                'Value': f"{rank_val:.2f}",
                'Status': '[PASS]' if rank_pass else '[FAIL]'
            })

            # Alignment
            align_val = stage1_metrics.get('alignment', 0)
            align_pass = stage1_metrics.get('alignment_pass', False)
            metrics_data.append({
                'Metric': 'Alignment',
                'Value': f"{align_val:.4f}",
                'Status': '[PASS]' if align_pass else '[FAIL]'
            })

            # Uniformity
            unif_val = stage1_metrics.get('uniformity', 0)
            unif_pass = stage1_metrics.get('uniformity_pass', False)
            metrics_data.append({
                'Metric': 'Uniformity',
                'Value': f"{unif_val:.4f}",
                'Status': '[PASS]' if unif_pass else '[FAIL]'
            })

            # k-NN Accuracy
            knn_val = stage1_metrics.get('knn_accuracy', 0)
            knn_pass = stage1_metrics.get('knn_accuracy_pass', False)
            metrics_data.append({
                'Metric': 'k-NN Accuracy',
                'Value': f"{knn_val:.4f}",
                'Status': '[PASS]' if knn_pass else '[FAIL]'
            })

            # Overall score
            overall = stage1_metrics.get('overall_score', 0)
            n_passed = stage1_metrics.get('n_passed', 0)
            metrics_data.append({
                'Metric': 'Overall',
                'Value': f"{overall:.1f}% ({n_passed}/5)",
                'Status': '[PASS]' if n_passed >= 4 else '[FAIL]'
            })

            # Display as dataframe
            import pandas as pd
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, hide_index=True, use_container_width=True)
        else:
            st.info("No quality metrics available. Metrics are computed after Stage 1 training.")


def show_evaluation():
    """Step 4: Evaluate trained model."""
    st.header("Step 4: Evaluate Model")
    st.markdown("Analyze model performance and cluster assignments")

    if not st.session_state.get('training_complete', False):
        st.warning("⚠️ Please train a model first (Step 3)")
        return

    model = st.session_state['model']

    # Data source selection
    st.subheader("📊 Select Data Source")

    data_source = st.radio(
        "Choose data to evaluate:",
        ["Training Data", "Preprocessed Package"],
        help="Use training data or load a new preprocessed package for evaluation"
    )

    data = None

    if data_source == "Training Data":
        if 'data' not in st.session_state:
            st.warning("⚠️ No training data found. Please load data first (Step 1)")
            return
        data = st.session_state['data']
        st.info(f"Using training data: {len(data):,} samples")

    else:  # Preprocessed Package
        # File selection
        data_dir = Path("data/preprocessed")
        if not data_dir.exists():
            st.warning("No preprocessed data directory found.")
            st.info("📁 Expected directory: `data/preprocessed/`")
            return

        # Get available HDF5 files
        hdf5_files = list(data_dir.glob("*.h5"))

        if not hdf5_files:
            st.warning("No HDF5 files found in data/preprocessed/")
            st.info("💡 Run the preprocessing pipeline first (Page 20)")
            return

        # File selector
        selected_file = st.selectbox(
            "Select preprocessed package:",
            options=hdf5_files,
            format_func=lambda x: x.name,
            key="eval_package_selector"
        )

        if selected_file:
            try:
                # Load data info
                with h5py.File(selected_file, 'r') as f:
                    n_samples = f['windows'].shape[0]
                    seq_len = f['windows'].shape[1]
                    n_channels = f['windows'].shape[2]

                    # Get metadata
                    metadata = {}
                    if 'window_metadata' in f.attrs:
                        metadata = json.loads(f.attrs['window_metadata'])
                    elif 'metadata' in f.attrs:
                        metadata = json.loads(f.attrs['metadata'])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", f"{n_samples:,}")
                with col2:
                    st.metric("Sequence Length", seq_len)
                with col3:
                    st.metric("Channels", n_channels)

                # Load button
                if st.button("Load Package for Evaluation", type="primary", key="load_eval_package"):
                    with st.spinner("Loading package..."):
                        with h5py.File(selected_file, 'r') as f:
                            windows = f['windows'][:]

                        # Convert to torch tensor
                        windows_tensor = torch.from_numpy(windows).float()

                        # Transpose from (N, T, C) to (N, C, T)
                        windows_tensor = windows_tensor.permute(0, 2, 1)

                        # Store temporarily for evaluation
                        st.session_state['eval_data'] = windows_tensor
                        st.session_state['eval_package_name'] = selected_file.name

                        st.success(f"✅ Loaded {n_samples:,} windows from {selected_file.name}")
                        st.rerun()

                # Use loaded evaluation data if available
                if 'eval_data' in st.session_state:
                    data = st.session_state['eval_data']
                    st.info(f"Using package: {st.session_state['eval_package_name']} ({len(data):,} samples)")
                else:
                    return  # Wait for user to load the package

            except Exception as e:
                st.error(f"Error loading file: {e}")
                return

    if data is None:
        return

    st.divider()
    st.subheader("🎯 Cluster Analysis")

    # Get predictions
    with st.spinner("Computing predictions..."):
        model.eval()
        with torch.no_grad():
            # Move data to same device as model
            device = next(model.parameters()).device
            data = data.to(device)
            z_norm, cluster_ids = model(data)

        cluster_ids = cluster_ids.cpu().numpy()

    # Cluster distribution
    unique, counts = np.unique(cluster_ids, return_counts=True)
    cluster_df = pd.DataFrame({
        'Cluster': unique,
        'Count': counts,
        'Percentage': counts / len(cluster_ids) * 100
    })

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(cluster_df, use_container_width=True)

    with col2:
        fig = go.Figure(data=[
            go.Bar(x=cluster_df['Cluster'], y=cluster_df['Count'])
        ])
        fig.update_layout(
            title="Cluster Distribution",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Samples"
        )
        st.plotly_chart(fig, width='stretch', key='cluster_distribution_chart')

    # Save predictions
    if st.button("💾 Save Predictions"):
        output_dir = Path("data/predictions")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"predictions_{timestamp}.npz"

        np.savez(
            output_file,
            cluster_ids=cluster_ids,
            latent_vectors=z_norm.cpu().numpy()
        )

        st.success(f"✅ Predictions saved to: {output_file}")


def show_model_management():
    """Step 5: Save/load models."""
    st.header("Step 5: Model Management")
    st.markdown("Save and load trained models")

    # Save model section
    st.subheader("💾 Save Model")

    if st.session_state.get('training_complete', False):
        model_name = st.text_input(
            "Model Name",
            value=f"ucl_tsc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        if st.button("Save Model", type="primary"):
            model_dir = Path("models/trained")
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / f"{model_name}.pt"

            torch.save({
                'model_state_dict': st.session_state['model'].state_dict(),
                'config': st.session_state['config'],
                'history': st.session_state['history']
            }, model_path)

            st.success(f"✅ Model saved to: {model_path}")
    else:
        st.info("Train a model first to save it")

    # Load model section
    st.divider()
    st.subheader("📂 Load Model")

    model_dir = Path("models/trained")
    if model_dir.exists():
        model_files = list(model_dir.glob("*.pt"))

        if model_files:
            selected_model = st.selectbox(
                "Select model to load:",
                options=model_files,
                format_func=lambda x: x.name
            )

            if st.button("Load Model"):
                with st.spinner("Loading model..."):
                    checkpoint = torch.load(selected_model)

                    # Create model from config
                    config = checkpoint['config']
                    seq_len = config.model.seq_length if hasattr(config.model, 'seq_length') else 127

                    # Get architecture parameters with backward compatibility
                    encoder_hidden_channels = getattr(config.model, 'encoder_hidden_channels', 128)
                    projection_hidden_dim = getattr(config.model, 'projection_hidden_dim', 512)
                    fusion_hidden_dim = getattr(config.model, 'fusion_hidden_dim', 256)
                    use_projection_bottleneck = getattr(config.model, 'use_projection_bottleneck', False)

                    model = UCLTSCModel(
                        input_channels=3,
                        d_z=config.model.d_z,
                        num_clusters=config.model.num_clusters,
                        use_hybrid_encoder=config.model.use_hybrid_encoder,
                        seq_length=seq_len,
                        encoder_hidden_channels=encoder_hidden_channels,
                        projection_hidden_dim=projection_hidden_dim,
                        fusion_hidden_dim=fusion_hidden_dim,
                        use_projection_bottleneck=use_projection_bottleneck
                    )

                    # Load weights (strict=False to handle architecture changes)
                    # This allows loading models trained with different sequence lengths
                    missing_keys, unexpected_keys = model.load_state_dict(
                        checkpoint['model_state_dict'],
                        strict=False
                    )

                    if missing_keys:
                        st.warning(f"⚠️ Missing keys in checkpoint (model has more layers than saved): {len(missing_keys)} keys")
                        with st.expander("View missing keys"):
                            st.code('\n'.join(missing_keys[:20]))  # Show first 20

                    if unexpected_keys:
                        st.warning(f"⚠️ Unexpected keys in checkpoint (saved model has extra layers): {len(unexpected_keys)} keys")
                        with st.expander("View unexpected keys"):
                            st.code('\n'.join(unexpected_keys[:20]))  # Show first 20

                    # Store in session
                    st.session_state['model'] = model
                    st.session_state['config'] = config
                    st.session_state['history'] = checkpoint['history']
                    st.session_state['training_complete'] = True

                    st.success(f"✅ Model loaded: {selected_model.name}")
        else:
            st.info("No saved models found")
    else:
        st.info("No models directory found")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Model Training",
        page_icon="🚀",
        layout="wide"
    )
    main()
