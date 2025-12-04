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
from src.config.config import Config, ModelConfig, TrainingConfig, AugmentationConfig, save_preset_config, load_preset_config
from src.training.trainer import TwoStageTrainer
from src.evaluation import ConfidenceCalibrator
import torch.nn.functional as F


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

    # Create config object
    custom_config = Config(
        model=ModelConfig(
            d_z=d_z,
            num_clusters=num_clusters,
            tau=tau,
            use_hybrid_encoder=use_hybrid
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
            mask_max_length_pct=mask_pct
        )
    )

    # Save config button
    col1, col2 = st.columns([1, 3])
    with col1:
        save_button = st.button("💾 Save to Preset", type="primary")
    with col2:
        use_button = st.button("✅ Use Configuration", type="secondary")

    if save_button:
        # Save to the selected preset
        save_preset_config(preset, custom_config)
        st.session_state['config'] = custom_config
        st.success(f"✅ Configuration saved to preset: **{preset}**")
        st.info("👉 Proceed to **Step 3: Train**")

    if use_button:
        # Just use the config without saving to disk
        st.session_state['config'] = custom_config
        st.success("✅ Configuration loaded (not saved to preset)")
        st.info("👉 Proceed to **Step 3: Train**")

    # Show current config
    if 'config' in st.session_state:
        st.divider()
        with st.expander("📋 Current Configuration"):
            st.text(str(st.session_state['config']))


def show_training():
    """Step 3: Train the model."""
    st.header("Step 3: Train Model")
    st.markdown("Run two-stage training: Contrastive pre-training → Joint fine-tuning")

    # Check prerequisites
    if 'data' not in st.session_state:
        st.warning("⚠️ Please load data first (Step 1)")
        return

    if 'config' not in st.session_state:
        st.warning("⚠️ Please configure model first (Step 2)")
        return

    # Training settings
    st.subheader("🎯 Training Setup")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_split = st.slider("Train %", 50, 80, 70)
    with col2:
        val_split = st.slider("Validation %", 10, 30, 15)
    with col3:
        test_split = 100 - train_split - val_split
        st.metric("Test %", test_split)

    # Device selection
    device = st.selectbox(
        "Device",
        options=['cuda' if torch.cuda.is_available() else 'cpu', 'cpu'],
        index=0
    )

    if device == 'cuda' and not torch.cuda.is_available():
        st.error("CUDA not available. Falling back to CPU.")
        device = 'cpu'

    # Show training info
    config = st.session_state['config']
    data = st.session_state['data']

    st.info(f"""
    **Training Configuration:**
    - Total Samples: {len(data):,}
    - Train/Val/Test: {train_split}/{val_split}/{test_split}%
    - Batch Size: {config.training.batch_size}
    - Device: {device.upper()}
    - Stage 1 Epochs: {config.training.max_epochs_stage1}
    - Stage 2 Epochs: {config.training.max_epochs_stage2}
    """)

    # Train button
    if st.button("🚀 Start Training", type="primary", key="start_training"):
        # Create train/val/test split
        n = len(data)
        train_size = int(n * train_split / 100)
        val_size = int(n * val_split / 100)

        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]

        # Handle GPU preloading if enabled
        if config.training.preload_to_gpu and device == 'cuda':
            with st.spinner("Preloading dataset to GPU..."):
                try:
                    # Check available GPU memory
                    if torch.cuda.is_available():
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        dataset_size = train_data.element_size() * train_data.nelement()
                        dataset_size += val_data.element_size() * val_data.nelement()
                        dataset_size += test_data.element_size() * test_data.nelement()

                        if dataset_size > free_memory * 0.8:  # Use 80% threshold for safety
                            st.warning(f"Dataset ({dataset_size/1e9:.2f}GB) may not fit in available GPU memory ({free_memory/1e9:.2f}GB). Falling back to standard loading.")
                            config.training.preload_to_gpu = False
                        else:
                            train_data = train_data.to(device)
                            val_data = val_data.to(device)
                            test_data = test_data.to(device)
                            st.success(f"✅ Dataset preloaded to GPU ({dataset_size/1e9:.2f}GB)")
                except Exception as e:
                    st.warning(f"Failed to preload to GPU: {e}. Falling back to standard loading.")
                    config.training.preload_to_gpu = False

        # Adjust DataLoader settings based on whether data is on GPU
        use_pin_memory = config.training.pin_memory and not config.training.preload_to_gpu and device == 'cuda'
        use_num_workers = 0 if config.training.preload_to_gpu else config.training.num_workers
        use_persistent_workers = config.training.persistent_workers and use_num_workers > 0
        use_prefetch_factor = config.training.prefetch_factor if use_num_workers > 0 else None

        # Create data loaders
        # Build DataLoader kwargs (exclude prefetch_factor if None to avoid PyTorch error)
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
            # Detect sequence length from data
            seq_len = st.session_state['data'].shape[2]  # (N, C, T) -> T

            model = UCLTSCModel(
                input_channels=3,
                d_z=config.model.d_z,
                num_clusters=config.model.num_clusters,
                use_hybrid_encoder=config.model.use_hybrid_encoder,
                seq_length=seq_len
            )

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            st.success(f"✅ Model created: {n_params:,} parameters (seq_length={seq_len})")

        # Create trainer
        trainer = TwoStageTrainer(model, config, device=device)

        # Training progress containers
        st.divider()
        st.subheader("📊 Training Progress")

        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        chart_container = st.empty()

        # Custom training loop with UI updates
        try:
            # Create callback for progress updates
            class StreamlitCallback:
                def __init__(self, progress_bar, status_text, metrics_container):
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                    self.metrics_container = metrics_container
                    self.stage1_epochs = config.training.max_epochs_stage1
                    self.stage2_epochs = config.training.max_epochs_stage2

                def on_epoch_end(self, stage, epoch, total_epochs, metrics):
                    if stage == 1:
                        progress = epoch / (self.stage1_epochs + self.stage2_epochs)
                        self.status_text.markdown(f"### Stage 1: Contrastive Pre-training - Epoch {epoch}/{total_epochs}")
                    else:
                        progress = (self.stage1_epochs + epoch) / (self.stage1_epochs + self.stage2_epochs)
                        self.status_text.markdown(f"### Stage 2: Joint Fine-tuning - Epoch {epoch}/{total_epochs}")

                    self.progress_bar.progress(min(progress, 1.0))

                    # Show metrics
                    cols = self.metrics_container.columns(len(metrics))
                    for i, (key, value) in enumerate(metrics.items()):
                        with cols[i]:
                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")

            callback = StreamlitCallback(progress_bar, status_text, metrics_container)

            with st.spinner("Training..."):
                history = trainer.train(
                    train_loader,
                    val_loader,
                    test_loader,
                    callback=callback
                )

            # Store results
            st.session_state['model'] = model
            st.session_state['trainer'] = trainer
            st.session_state['history'] = history
            st.session_state['training_complete'] = True
            st.session_state['train_loader'] = train_loader
            st.session_state['val_loader'] = val_loader
            st.session_state['test_loader'] = test_loader

            progress_bar.progress(0.95)

            # NEW: Calibrate confidence scores
            st.info("Calibrating confidence scores on validation set...")
            try:
                calibration_results = calibrate_confidence_scores(
                    model, val_loader, device
                )
                st.session_state['calibration'] = calibration_results

                # Display calibration results
                if calibration_results['passed']:
                    st.success(
                        f"[PASS] Confidence calibration successful! "
                        f"Best gamma: {calibration_results['best_gamma']}, "
                        f"R²: {calibration_results['best_r2']:.3f}"
                    )
                else:
                    st.warning(
                        f"[WARN] Confidence calibration below threshold "
                        f"(R² = {calibration_results['best_r2']:.3f} < 0.7). "
                        f"Model may need retraining."
                    )
            except Exception as e:
                st.warning(f"Confidence calibration failed: {e}. Continuing without calibration.")
                st.session_state['calibration'] = None

            progress_bar.progress(1.0)
            st.success("🎉 Training Complete!")

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)

    # Show existing training results if available
    if st.session_state.get('training_complete', False):
        st.divider()
        show_training_results(st.session_state['history'])


def show_training_results(history):
    """Display training results."""
    st.subheader("📈 Training Results")

    # Create plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Stage 1: Contrastive Loss',
            'Stage 2: Total Loss',
            'Stage 2: Clustering Loss',
            'Stage 2: Contrastive Loss'
        )
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
    fig.update_layout(height=600, showlegend=True)

    st.plotly_chart(fig, width='stretch', key='training_losses_chart')

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Stage 1 Epochs",
            len(history['stage1']['train_loss'])
        )
    with col2:
        st.metric(
            "Stage 2 Epochs",
            len(history['stage2']['train_loss'])
        )
    with col3:
        st.metric(
            "Best S1 Val Loss",
            f"{min(history['stage1']['val_loss']):.4f}"
        )
    with col4:
        st.metric(
            "Best Cluster Loss",
            f"{min(history['stage2']['cluster_loss']):.4f}"
        )


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
                    model = UCLTSCModel(
                        input_channels=3,
                        d_z=config.model.d_z,
                        num_clusters=config.model.num_clusters,
                        use_hybrid_encoder=config.model.use_hybrid_encoder,
                        seq_length=seq_len
                    )

                    # Load weights
                    model.load_state_dict(checkpoint['model_state_dict'])

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
