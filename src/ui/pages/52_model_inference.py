"""
Model Inference & Pattern Recognition Page

Use trained models to:
- Classify new patterns
- Analyze pattern clusters
- Visualize latent space
- Generate pattern reports
"""

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import h5py
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ucl_tsc_model import UCLTSCModel


def main():
    st.title("🔮 Model Inference & Pattern Recognition")
    st.markdown("Use trained models to classify and analyze stock patterns")

    # Check if model is loaded
    if 'model' not in st.session_state or not st.session_state.get('training_complete', False):
        st.warning("⚠️ No model loaded. Please train or load a model first.")

        # Quick load option
        st.subheader("Quick Load Model")
        model_dir = Path("models/trained")
        if model_dir.exists():
            model_files = list(model_dir.glob("*.pt"))
            if model_files:
                selected = st.selectbox(
                    "Select model to load:",
                    options=model_files,
                    format_func=lambda x: x.stem
                )

                if st.button("Load Model"):
                    load_model(selected)
                    st.rerun()
            else:
                st.info("No trained models found")
        return

    # Tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Pattern Classification",
        "📊 Cluster Analysis",
        "🗺️ Latent Space Visualization",
        "📋 Pattern Report"
    ])

    with tab1:
        show_pattern_classification()

    with tab2:
        show_cluster_analysis()

    with tab3:
        show_latent_space_visualization()

    with tab4:
        show_pattern_report()


def load_model(model_path):
    """Load a model checkpoint into session."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']

        model = UCLTSCModel(
            input_channels=3,
            d_z=config.model.d_z,
            num_clusters=config.model.num_clusters,
            use_hybrid_encoder=config.model.use_hybrid_encoder
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        st.session_state['model'] = model
        st.session_state['config'] = config
        st.session_state['history'] = checkpoint['history']
        st.session_state['training_complete'] = True

        st.success(f"✅ Model loaded: {model_path.name}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")


def show_pattern_classification():
    """Classify patterns from uploaded data."""
    st.header("Pattern Classification")

    st.markdown("""
    Upload preprocessed data to classify patterns using the loaded model.
    The model will assign each window to one of the learned pattern clusters.
    """)

    # Data source selection
    source = st.radio(
        "Data Source:",
        ["Use Session Data", "Upload New Data"],
        horizontal=True
    )

    data = None

    if source == "Use Session Data":
        if 'data' in st.session_state:
            data = st.session_state['data']
            st.success(f"✅ Using {len(data):,} windows from session")
        else:
            st.warning("No data in session. Please load data first or upload new data.")
            return

    else:  # Upload New Data
        st.subheader("Upload Preprocessed Windows")

        uploaded_file = st.file_uploader(
            "Select HDF5 file:",
            type=['h5', 'hdf5'],
            help="Upload preprocessed windows from the preprocessing pipeline"
        )

        if uploaded_file:
            try:
                with h5py.File(uploaded_file, 'r') as f:
                    windows = f['windows'][:]

                # Convert to torch tensor
                data = torch.from_numpy(windows).float()
                # Transpose from (N, T, C) to (N, C, T)
                data = data.permute(0, 2, 1)

                st.success(f"✅ Loaded {len(data):,} windows")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return

    if data is None:
        return

    # Classify button
    if st.button("🚀 Classify Patterns", type="primary"):
        with st.spinner("Classifying patterns..."):
            model = st.session_state['model']
            model.eval()

            with torch.no_grad():
                # Get predictions
                z_norm, cluster_ids = model(data)

            cluster_ids = cluster_ids.cpu().numpy()
            z_norm = z_norm.cpu().numpy()

            # Store results
            st.session_state['inference_results'] = {
                'cluster_ids': cluster_ids,
                'latent_vectors': z_norm,
                'data': data
            }

            st.success("✅ Classification complete!")

    # Display results
    if 'inference_results' in st.session_state:
        results = st.session_state['inference_results']
        cluster_ids = results['cluster_ids']

        st.divider()
        st.subheader("Classification Results")

        # Cluster distribution
        unique, counts = np.unique(cluster_ids, return_counts=True)
        cluster_df = pd.DataFrame({
            'Cluster': unique,
            'Count': counts,
            'Percentage': (counts / len(cluster_ids) * 100).round(2)
        })

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)

            # Summary stats
            st.metric("Total Windows", f"{len(cluster_ids):,}")
            st.metric("Unique Clusters", len(unique))
            st.metric("Largest Cluster", f"#{unique[np.argmax(counts)]}")

        with col2:
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=cluster_df['Cluster'],
                    y=cluster_df['Count'],
                    text=cluster_df['Percentage'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Cluster Distribution",
                xaxis_title="Cluster ID",
                yaxis_title="Number of Windows",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Export predictions
        if st.button("💾 Export Predictions"):
            output_dir = Path("data/predictions")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"predictions_{timestamp}.npz"

            np.savez_compressed(
                output_file,
                cluster_ids=cluster_ids,
                latent_vectors=results['latent_vectors']
            )

            st.success(f"✅ Predictions saved to: {output_file}")


def show_cluster_analysis():
    """Analyze cluster characteristics."""
    st.header("Cluster Analysis")

    if 'inference_results' not in st.session_state:
        st.info("👈 Run pattern classification first")
        return

    results = st.session_state['inference_results']
    cluster_ids = results['cluster_ids']
    latent_vectors = results['latent_vectors']
    data = results['data']

    st.subheader("Cluster Statistics")

    # Compute cluster statistics
    unique_clusters = np.unique(cluster_ids)

    cluster_stats = []
    for cluster_id in unique_clusters:
        mask = cluster_ids == cluster_id
        cluster_vectors = latent_vectors[mask]

        # Compute stats
        centroid = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)

        cluster_stats.append({
            'Cluster': cluster_id,
            'Size': mask.sum(),
            'Mean Distance': distances.mean(),
            'Std Distance': distances.std(),
            'Max Distance': distances.max(),
            'Min Distance': distances.min(),
        })

    df_stats = pd.DataFrame(cluster_stats)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # Cluster quality metrics
    st.subheader("Cluster Quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Compactness (average intra-cluster distance)
        avg_compactness = df_stats['Mean Distance'].mean()
        st.metric(
            "Average Compactness",
            f"{avg_compactness:.4f}",
            help="Lower = More compact clusters"
        )

    with col2:
        # Balance (coefficient of variation of cluster sizes)
        sizes = df_stats['Size'].values
        cv = sizes.std() / sizes.mean()
        st.metric(
            "Balance (CV)",
            f"{cv:.4f}",
            help="Lower = More balanced cluster sizes"
        )

    with col3:
        # Consistency (average std of distances)
        avg_consistency = df_stats['Std Distance'].mean()
        st.metric(
            "Consistency",
            f"{avg_consistency:.4f}",
            help="Lower = More consistent within clusters"
        )

    # Individual cluster inspection
    st.subheader("Inspect Clusters")

    selected_cluster = st.selectbox(
        "Select cluster to inspect:",
        options=unique_clusters,
        format_func=lambda x: f"Cluster {x} ({(cluster_ids == x).sum()} windows)"
    )

    if selected_cluster is not None:
        show_cluster_details(selected_cluster, cluster_ids, data)


def show_cluster_details(cluster_id, cluster_ids, data):
    """Show details for a specific cluster."""
    mask = cluster_ids == cluster_id
    cluster_data = data[mask]

    st.markdown(f"### Cluster {cluster_id}")

    # Show sample windows
    n_samples = min(5, len(cluster_data))
    sample_indices = np.random.choice(len(cluster_data), n_samples, replace=False)

    st.markdown(f"**Sample Windows** (showing {n_samples} random windows)")

    for i, idx in enumerate(sample_indices):
        window = cluster_data[idx].cpu().numpy()  # Shape: (C, T)

        with st.expander(f"Sample {i+1}", expanded=(i == 0)):
            # Plot the window
            fig = go.Figure()

            channel_names = ['Returns', 'Volume/OBV', 'Volatility/NATR']
            for c in range(window.shape[0]):
                fig.add_trace(go.Scatter(
                    y=window[c],
                    name=channel_names[c],
                    mode='lines'
                ))

            fig.update_layout(
                title=f"Window {idx} - Cluster {cluster_id}",
                xaxis_title="Time Step",
                yaxis_title="Normalized Value",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def show_latent_space_visualization():
    """Visualize latent space using dimensionality reduction."""
    st.header("Latent Space Visualization")

    if 'inference_results' not in st.session_state:
        st.info("👈 Run pattern classification first")
        return

    results = st.session_state['inference_results']
    cluster_ids = results['cluster_ids']
    latent_vectors = results['latent_vectors']

    st.markdown("""
    Visualize the latent space using dimensionality reduction techniques.
    This shows how the model has organized patterns into clusters.
    """)

    # Method selection
    method = st.selectbox(
        "Dimensionality Reduction Method:",
        ["PCA (Fast)", "t-SNE (Better Visualization)"],
        help="PCA is faster but t-SNE often gives better visual separation"
    )

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider(
            "Number of samples to visualize:",
            min_value=100,
            max_value=min(10000, len(latent_vectors)),
            value=min(1000, len(latent_vectors)),
            help="Large values may be slow"
        )

    # Sample data
    if n_samples < len(latent_vectors):
        indices = np.random.choice(len(latent_vectors), n_samples, replace=False)
        sampled_vectors = latent_vectors[indices]
        sampled_labels = cluster_ids[indices]
    else:
        sampled_vectors = latent_vectors
        sampled_labels = cluster_ids

    # Compute embedding
    if st.button("🎨 Generate Visualization", type="primary"):
        with st.spinner(f"Computing {method}..."):
            if "PCA" in method:
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(sampled_vectors)
                explained_var = reducer.explained_variance_ratio_.sum()
                st.info(f"📊 Explained variance: {explained_var*100:.2f}%")
            else:  # t-SNE
                with col2:
                    perplexity = st.slider(
                        "Perplexity:",
                        min_value=5,
                        max_value=50,
                        value=30,
                        help="Higher = focus on global structure"
                    )
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                embedding = reducer.fit_transform(sampled_vectors)

            # Plot
            fig = go.Figure()

            for cluster_id in np.unique(sampled_labels):
                mask = sampled_labels == cluster_id
                fig.add_trace(go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(size=5, opacity=0.6)
                ))

            fig.update_layout(
                title=f"Latent Space Visualization ({method})",
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                height=600,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Save embedding
            st.session_state['latent_embedding'] = embedding
            st.session_state['embedding_labels'] = sampled_labels


def show_pattern_report():
    """Generate pattern recognition report."""
    st.header("Pattern Recognition Report")

    if 'inference_results' not in st.session_state:
        st.info("👈 Run pattern classification first")
        return

    results = st.session_state['inference_results']
    cluster_ids = results['cluster_ids']
    latent_vectors = results['latent_vectors']

    st.markdown("""
    Generate a comprehensive report of pattern recognition results.
    """)

    # Report sections
    st.subheader("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patterns", f"{len(cluster_ids):,}")

    with col2:
        st.metric("Unique Clusters", len(np.unique(cluster_ids)))

    with col3:
        dominant_cluster = np.argmax(np.bincount(cluster_ids))
        st.metric("Dominant Pattern", f"Cluster {dominant_cluster}")

    with col4:
        dominant_pct = (cluster_ids == dominant_cluster).sum() / len(cluster_ids) * 100
        st.metric("Dominant %", f"{dominant_pct:.1f}%")

    # Cluster distribution
    st.subheader("Cluster Distribution")

    unique, counts = np.unique(cluster_ids, return_counts=True)
    percentages = counts / len(cluster_ids) * 100

    cluster_summary = pd.DataFrame({
        'Cluster ID': unique,
        'Count': counts,
        'Percentage': percentages.round(2),
        'Status': ['Dominant' if c == dominant_cluster else
                  'Large' if p > 15 else
                  'Medium' if p > 5 else
                  'Small' for c, p in zip(unique, percentages)]
    })

    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    # Export report
    if st.button("📄 Export Full Report"):
        output_dir = Path("data/reports")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"pattern_report_{timestamp}.csv"

        # Save cluster summary
        cluster_summary.to_csv(report_file, index=False)

        st.success(f"✅ Report exported to: {report_file}")

        # Also save detailed results
        detailed_file = output_dir / f"pattern_details_{timestamp}.npz"
        np.savez_compressed(
            detailed_file,
            cluster_ids=cluster_ids,
            latent_vectors=latent_vectors
        )

        st.success(f"✅ Detailed results saved to: {detailed_file}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Model Inference",
        page_icon="🔮",
        layout="wide"
    )
    main()
