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
from src.config.config import Config, ModelConfig, TrainingConfig, AugmentationConfig
from src.training.trainer import TwoStageTrainer


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

                    # Convert to torch tensor
                    windows_tensor = torch.from_numpy(windows).float()

                    # Transpose from (N, T, C) to (N, C, T)
                    windows_tensor = windows_tensor.permute(0, 2, 1)

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
    preset = st.selectbox(
        "Choose a preset:",
        ["Custom", "Default", "Small (Fast)", "Large (Best Quality)"],
        index=1
    )

    # Initialize config based on preset
    if preset == "Default":
        from src.config.config import get_default_config
        config = get_default_config()
    elif preset == "Small (Fast)":
        from src.config.config import get_small_config
        config = get_small_config()
    elif preset == "Large (Best Quality)":
        from src.config.config import get_large_config
        config = get_large_config()
    else:
        config = Config()

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
            min_value=5, max_value=15, value=config.model.num_clusters,
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
            help="Enable 3rd encoder for intermediate patterns (~40 bars)"
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
        batch_size = st.selectbox(
            "Batch Size",
            options=[32,64,128, 256, 512, 1024],
            index=2  # 512
        )

    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Optimization**")
            early_stopping = st.number_input(
                "Early Stopping Patience",
                min_value=5, max_value=20, value=config.training.early_stopping_patience
            )
            use_mixed_precision = st.checkbox(
                "Mixed Precision (FP16)",
                value=config.training.use_mixed_precision,
                help="Faster training on CUDA, 50% memory reduction"
            )

        with col2:
            st.markdown("**Data Augmentation**")
            jitter_sigma = st.slider(
                "Jitter Sigma",
                min_value=0.005, max_value=0.05, value=config.augmentation.jitter_sigma,
                format="%.3f"
            )
            mask_pct = st.slider(
                "Time Mask %",
                min_value=0.05, max_value=0.2, value=config.augmentation.mask_max_length_pct,
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
            use_mixed_precision=use_mixed_precision
        ),
        augmentation=AugmentationConfig(
            jitter_sigma=jitter_sigma,
            mask_max_length_pct=mask_pct
        )
    )

    # Save config button
    if st.button("Save Configuration", type="primary"):
        st.session_state['config'] = custom_config
        st.success("✅ Configuration saved!")
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

        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=False
        )
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=False
        )
        test_loader = DataLoader(
            TensorDataset(test_data),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=False
        )

        # Create model
        with st.spinner("Creating model..."):
            model = UCLTSCModel(
                input_channels=3,
                d_z=config.model.d_z,
                num_clusters=config.model.num_clusters,
                use_hybrid_encoder=config.model.use_hybrid_encoder
            )

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            st.success(f"✅ Model created: {n_params:,} parameters")

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
    data = st.session_state['data']

    st.subheader("🎯 Cluster Analysis")

    # Get predictions
    with st.spinner("Computing predictions..."):
        model.eval()
        with torch.no_grad():
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
                    model = UCLTSCModel(
                        input_channels=3,
                        d_z=config.model.d_z,
                        num_clusters=config.model.num_clusters,
                        use_hybrid_encoder=config.model.use_hybrid_encoder
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
