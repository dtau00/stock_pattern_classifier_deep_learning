"""
Model Comparison and Benchmarking Page

Compare multiple trained models side-by-side:
- Training metrics comparison
- Cluster quality metrics
- Performance benchmarks
- Model selection guidance
"""

import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ucl_tsc_model import UCLTSCModel
from src.config.config import Config, ModelConfig, TrainingConfig, AugmentationConfig, DataConfig

# Register safe globals for torch.load (PyTorch 2.6+)
torch.serialization.add_safe_globals([Config, ModelConfig, TrainingConfig, AugmentationConfig, DataConfig])


def main():
    st.title("📊 Model Comparison & Benchmarking")
    st.markdown("Compare and analyze multiple trained models")

    # Check for saved models
    model_dir = Path("models/trained")
    if not model_dir.exists() or not list(model_dir.glob("*.pt")):
        st.warning("⚠️ No trained models found. Train some models first!")
        st.info("📌 Go to **Model Training** page to train models")
        return

    # Get all model files
    model_files = list(model_dir.glob("*.pt"))

    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Selection")
        selected_models = st.multiselect(
            "Select models to compare:",
            options=model_files,
            format_func=lambda x: x.stem,
            default=model_files[:min(3, len(model_files))]  # Select first 3 by default
        )

    if not selected_models:
        st.info("👈 Select models from the sidebar to compare")
        return

    # Load model information
    models_info = []
    for model_path in selected_models:
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['config']
            history = checkpoint['history']

            # Get model stats
            info = {
                'name': model_path.stem,
                'path': model_path,
                'config': config,
                'history': history,
                'd_z': config.model.d_z,
                'num_clusters': config.model.num_clusters,
                'tau': config.model.tau,
                'use_hybrid': config.model.use_hybrid_encoder,
                'batch_size': config.training.batch_size,
                'lr': config.training.learning_rate,
                'stage1_epochs': len(history['stage1']['train_loss']),
                'stage2_epochs': len(history['stage2']['train_loss']),
                'best_s1_val_loss': min(history['stage1']['val_loss']),
                'best_cluster_loss': min(history['stage2']['cluster_loss']),
                'final_val_loss': history['stage2']['val_loss'][-1],
            }
            models_info.append(info)
        except Exception as e:
            st.error(f"Error loading {model_path.name}: {e}")

    if not models_info:
        st.error("Failed to load any models")
        return

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Training Metrics",
        "🎯 Model Comparison",
        "⚙️ Configuration",
        "🏆 Best Model"
    ])

    # Tab 1: Training Metrics Comparison
    with tab1:
        show_training_metrics_comparison(models_info)

    # Tab 2: Model Comparison Table
    with tab2:
        show_model_comparison_table(models_info)

    # Tab 3: Configuration Comparison
    with tab3:
        show_configuration_comparison(models_info)

    # Tab 4: Best Model Selection
    with tab4:
        show_best_model_selection(models_info)


def show_training_metrics_comparison(models_info):
    """Display training metrics comparison charts."""
    st.header("Training Metrics Comparison")

    # Create comparison plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Stage 1: Validation Loss',
            'Stage 2: Total Loss',
            'Stage 2: Clustering Loss',
            'Stage 2: Contrastive Loss'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, model_info in enumerate(models_info):
        color = colors[i % len(colors)]
        name = model_info['name']
        history = model_info['history']

        # Stage 1 validation loss
        epochs_s1 = list(range(1, len(history['stage1']['val_loss']) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs_s1,
                y=history['stage1']['val_loss'],
                name=name,
                line=dict(color=color),
                legendgroup=name,
            ),
            row=1, col=1
        )

        # Stage 2 total loss
        epochs_s2 = list(range(1, len(history['stage2']['val_loss']) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs_s2,
                y=history['stage2']['val_loss'],
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False
            ),
            row=1, col=2
        )

        # Clustering loss
        fig.add_trace(
            go.Scatter(
                x=epochs_s2,
                y=history['stage2']['cluster_loss'],
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False
            ),
            row=2, col=1
        )

        # Contrastive loss
        fig.add_trace(
            go.Scatter(
                x=epochs_s2,
                y=history['stage2']['contrastive_loss'],
                name=name,
                line=dict(color=color),
                legendgroup=name,
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(height=700, showlegend=True, legend=dict(x=1.05, y=1))

    st.plotly_chart(fig, use_container_width=True)


def show_model_comparison_table(models_info):
    """Display model comparison table."""
    st.header("Model Comparison Table")

    # Create comparison dataframe
    comparison_data = []
    for info in models_info:
        comparison_data.append({
            'Model': info['name'],
            'd_z (Latent Dim)': info['d_z'],
            'Clusters': info['num_clusters'],
            'τ (Temperature)': f"{info['tau']:.2f}",
            'Hybrid Encoder': 'Yes' if info['use_hybrid'] else 'No',
            'Batch Size': info['batch_size'],
            'Learning Rate': f"{info['lr']:.4f}",
            'S1 Epochs': info['stage1_epochs'],
            'S2 Epochs': info['stage2_epochs'],
            'Best S1 Val Loss': f"{info['best_s1_val_loss']:.4f}",
            'Best Cluster Loss': f"{info['best_cluster_loss']:.4f}",
            'Final Val Loss': f"{info['final_val_loss']:.4f}",
        })

    df = pd.DataFrame(comparison_data)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Performance metrics
    st.subheader("Performance Ranking")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Best Stage 1 Performance**")
        best_s1 = min(models_info, key=lambda x: x['best_s1_val_loss'])
        st.success(f"🥇 {best_s1['name']}")
        st.metric("Val Loss", f"{best_s1['best_s1_val_loss']:.4f}")

    with col2:
        st.markdown("**Best Clustering Quality**")
        best_cluster = min(models_info, key=lambda x: x['best_cluster_loss'])
        st.success(f"🥇 {best_cluster['name']}")
        st.metric("Cluster Loss", f"{best_cluster['best_cluster_loss']:.4f}")

    with col3:
        st.markdown("**Best Final Performance**")
        best_final = min(models_info, key=lambda x: x['final_val_loss'])
        st.success(f"🥇 {best_final['name']}")
        st.metric("Final Val Loss", f"{best_final['final_val_loss']:.4f}")


def show_configuration_comparison(models_info):
    """Display configuration comparison."""
    st.header("Configuration Comparison")

    # Model architecture comparison
    st.subheader("Model Architecture")

    arch_data = []
    for info in models_info:
        arch_data.append({
            'Model': info['name'],
            'Latent Dimension (d_z)': info['d_z'],
            'Number of Clusters': info['num_clusters'],
            'Temperature (τ)': info['tau'],
            'Hybrid Encoder': info['use_hybrid'],
        })

    df_arch = pd.DataFrame(arch_data)
    st.dataframe(df_arch, use_container_width=True, hide_index=True)

    # Training configuration comparison
    st.subheader("Training Configuration")

    train_data = []
    for info in models_info:
        train_data.append({
            'Model': info['name'],
            'Batch Size': info['batch_size'],
            'Learning Rate': info['lr'],
            'Stage 1 Max Epochs': info['config'].training.max_epochs_stage1,
            'Stage 2 Max Epochs': info['config'].training.max_epochs_stage2,
            'Early Stopping Patience': info['config'].training.early_stopping_patience,
            'Mixed Precision': info['config'].training.use_mixed_precision,
        })

    df_train = pd.DataFrame(train_data)
    st.dataframe(df_train, use_container_width=True, hide_index=True)

    # Augmentation configuration
    st.subheader("Augmentation Configuration")

    aug_data = []
    for info in models_info:
        aug_config = info['config'].augmentation
        aug_data.append({
            'Model': info['name'],
            'Jitter Sigma': aug_config.jitter_sigma,
            'Scale Range': f"{aug_config.scale_range}",
            'Mask %': aug_config.mask_max_length_pct,
            'Apply Jitter': aug_config.apply_jitter,
            'Apply Scaling': aug_config.apply_scaling,
            'Apply Masking': aug_config.apply_masking,
        })

    df_aug = pd.DataFrame(aug_data)
    st.dataframe(df_aug, use_container_width=True, hide_index=True)


def show_best_model_selection(models_info):
    """Help user select the best model."""
    st.header("Best Model Selection")

    st.markdown("""
    Use the criteria below to select the best model for your use case:

    **Metrics to Consider:**
    - **Stage 1 Val Loss**: Lower = Better feature representation
    - **Cluster Loss**: Lower = More compact and well-separated clusters
    - **Final Val Loss**: Overall model performance
    - **Training Efficiency**: Fewer epochs = Faster convergence
    """)

    # Calculate scores
    scored_models = []
    for info in models_info:
        # Normalize metrics (lower is better)
        s1_loss = info['best_s1_val_loss']
        cluster_loss = info['best_cluster_loss']
        final_loss = info['final_val_loss']

        # Simple scoring: inverse of losses
        score = 1.0 / (s1_loss + cluster_loss + final_loss)

        scored_models.append({
            'Model': info['name'],
            'Stage 1 Loss': s1_loss,
            'Cluster Loss': cluster_loss,
            'Final Loss': final_loss,
            'Combined Score': score,
            'S1 Epochs': info['stage1_epochs'],
            'S2 Epochs': info['stage2_epochs'],
            'Total Epochs': info['stage1_epochs'] + info['stage2_epochs'],
            'path': info['path']
        })

    # Sort by score
    scored_models.sort(key=lambda x: x['Combined Score'], reverse=True)

    # Display ranking
    st.subheader("🏆 Overall Ranking")

    for i, model in enumerate(scored_models, 1):
        with st.expander(f"#{i} - {model['Model']}", expanded=(i == 1)):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Stage 1 Loss", f"{model['Stage 1 Loss']:.4f}")
            with col2:
                st.metric("Cluster Loss", f"{model['Cluster Loss']:.4f}")
            with col3:
                st.metric("Final Loss", f"{model['Final Loss']:.4f}")
            with col4:
                st.metric("Total Epochs", model['Total Epochs'])

            if i == 1:
                st.success("⭐ Recommended Model")

                if st.button("🔄 Load This Model to Session", key=f"load_{i}"):
                    try:
                        checkpoint = torch.load(model['path'], map_location='cpu')
                        config = checkpoint['config']

                        # Get architecture parameters with backward compatibility
                        seq_len = config.model.seq_length if hasattr(config.model, 'seq_length') else 127
                        encoder_hidden_channels = getattr(config.model, 'encoder_hidden_channels', 128)
                        projection_hidden_dim = getattr(config.model, 'projection_hidden_dim', 512)
                        fusion_hidden_dim = getattr(config.model, 'fusion_hidden_dim', 256)
                        use_projection_bottleneck = getattr(config.model, 'use_projection_bottleneck', False)

                        # Create model
                        loaded_model = UCLTSCModel(
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
                        missing_keys, unexpected_keys = loaded_model.load_state_dict(
                            checkpoint['model_state_dict'],
                            strict=False
                        )

                        if missing_keys or unexpected_keys:
                            st.warning(f"⚠️ Model architecture mismatch: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")

                        # Store in session
                        st.session_state['model'] = loaded_model
                        st.session_state['config'] = config
                        st.session_state['history'] = checkpoint['history']
                        st.session_state['training_complete'] = True

                        st.success(f"✅ Model loaded: {model['Model']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")

    # Export comparison
    st.divider()
    st.subheader("📤 Export Comparison")

    if st.button("Export Comparison as CSV"):
        df = pd.DataFrame(scored_models)
        df = df.drop('path', axis=1)

        output_dir = Path("data/model_comparisons")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"model_comparison_{timestamp}.csv"

        df.to_csv(output_file, index=False)
        st.success(f"✅ Comparison exported to: {output_file}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Model Comparison",
        page_icon="📊",
        layout="wide"
    )
    main()
