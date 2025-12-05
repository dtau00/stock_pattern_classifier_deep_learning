"""
Find Optimal Shape Clusters

Interactive UI for running optimal cluster analysis and applying shape-based clustering.
Helps determine the best number of shape clusters for your price pattern data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import h5py

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.shape_features import extract_windows_shape_features, cluster_by_shape, get_shape_cluster_stats
from src.preprocessing.segmentation import load_preprocessed_package


def plot_clustering_metrics_plotly(df_results: pd.DataFrame):
    """
    Create interactive Plotly visualization of clustering metrics.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Silhouette Score (Higher = Better Separation)',
            'Calinski-Harabasz Score (Higher = Better)',
            'Davies-Bouldin Score (Lower = Better)',
            'Cluster Size Distribution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Silhouette Score
    fig.add_trace(
        go.Scatter(
            x=df_results['n_clusters'],
            y=df_results['silhouette'],
            mode='lines+markers',
            name='Silhouette',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    # Mark best silhouette
    best_idx = df_results['silhouette'].idxmax()
    best_n = df_results.loc[best_idx, 'n_clusters']
    best_score = df_results.loc[best_idx, 'silhouette']
    fig.add_trace(
        go.Scatter(
            x=[best_n],
            y=[best_score],
            mode='markers',
            name=f'Best: n={int(best_n)}',
            marker=dict(size=20, color='red', symbol='star')
        ),
        row=1, col=1
    )

    # Calinski-Harabasz Score
    fig.add_trace(
        go.Scatter(
            x=df_results['n_clusters'],
            y=df_results['calinski_harabasz'],
            mode='lines+markers',
            name='Calinski-Harabasz',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            showlegend=False
        ),
        row=1, col=2
    )

    # Davies-Bouldin Score
    fig.add_trace(
        go.Scatter(
            x=df_results['n_clusters'],
            y=df_results['davies_bouldin'],
            mode='lines+markers',
            name='Davies-Bouldin',
            line=dict(color='orange', width=2),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=1
    )

    # Mark best davies-bouldin
    best_idx = df_results['davies_bouldin'].idxmin()
    best_n = df_results.loc[best_idx, 'n_clusters']
    best_score = df_results.loc[best_idx, 'davies_bouldin']
    fig.add_trace(
        go.Scatter(
            x=[best_n],
            y=[best_score],
            mode='markers',
            name=f'Best: n={int(best_n)}',
            marker=dict(size=20, color='red', symbol='star'),
            showlegend=False
        ),
        row=2, col=1
    )

    # Cluster size distribution
    fig.add_trace(
        go.Scatter(
            x=df_results['n_clusters'],
            y=df_results['avg_cluster_size'],
            mode='lines+markers',
            name='Average',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=2
    )

    # Add min-max range
    fig.add_trace(
        go.Scatter(
            x=df_results['n_clusters'].tolist() + df_results['n_clusters'].tolist()[::-1],
            y=df_results['max_cluster_size'].tolist() + df_results['min_cluster_size'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Min-Max Range',
            showlegend=False
        ),
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
    fig.update_xaxes(title_text="Number of Clusters", row=2, col=1)
    fig.update_xaxes(title_text="Number of Clusters", row=2, col=2)

    fig.update_yaxes(title_text="Silhouette Score", row=1, col=1)
    fig.update_yaxes(title_text="CH Score", row=1, col=2)
    fig.update_yaxes(title_text="DB Score", row=2, col=1)
    fig.update_yaxes(title_text="Cluster Size", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )

    return fig


def evaluate_clustering_range(
    shape_features: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    step: int,
    progress_bar
):
    """
    Evaluate clustering quality for different numbers of clusters.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    results = []
    cluster_range = list(range(min_clusters, max_clusters + 1, step))

    for idx, n_clusters in enumerate(cluster_range):
        progress_bar.progress((idx + 1) / len(cluster_range),
                             text=f"Testing n={n_clusters}...")

        # Cluster
        labels, _ = cluster_by_shape(
            shape_features,
            n_clusters=n_clusters,
            method='kmeans',
            random_state=42
        )

        # Compute metrics
        try:
            silhouette = silhouette_score(shape_features, labels)
            calinski = calinski_harabasz_score(shape_features, labels)
            davies = davies_bouldin_score(shape_features, labels)
        except Exception as e:
            st.warning(f"Error for n={n_clusters}: {e}")
            continue

        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)

        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'min_cluster_size': counts.min(),
            'max_cluster_size': counts.max(),
            'avg_cluster_size': counts.mean(),
            'std_cluster_size': counts.std(),
        })

    return pd.DataFrame(results)


def apply_shape_clustering(
    shape_features: np.ndarray,
    n_clusters: int,
    method: str,
    windows_df: pd.DataFrame,
    metadata: list,
    output_dir: Path,
    data_name: str
):
    """
    Apply shape clustering and save results.
    """
    # Cluster
    labels, clusterer = cluster_by_shape(
        shape_features,
        n_clusters=n_clusters,
        method=method,
        random_state=42
    )

    # Get cluster statistics
    df_stats = get_shape_cluster_stats(labels, windows_df, metadata)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{data_name}_shape_clusters.npz"
    np.savez(
        results_file,
        shape_labels=labels,
        n_shape_clusters=n_clusters,
        shape_features=shape_features
    )

    stats_file = output_dir / f"{data_name}_shape_clusters_stats.csv"
    df_stats.to_csv(stats_file, index=False)

    # Save sample indices for visualization
    samples = {}
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        # Sample up to 20 windows from each cluster
        n_samples = min(20, len(cluster_indices))
        sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        samples[str(cluster_id)] = sample_indices.tolist()

    samples_file = output_dir / f"{data_name}_shape_clusters_samples.json"
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)

    return results_file, stats_file, samples_file, df_stats


def main():
    st.set_page_config(page_title="Find Optimal Shape Clusters", layout="wide")

    st.title("🔍 Find Optimal Shape Clusters")

    st.markdown("""
    This tool helps you find the optimal number of **shape-based clusters** for your price pattern data.

    ### What are shape clusters?
    - Groups price windows with **visually similar patterns**
    - Normalizes by price level (pattern at $100 looks same as at $10,000)
    - First stage before training the prediction model

    ### Workflow:
    1. **Analyze**: Test different cluster counts and view quality metrics
    2. **Choose**: Select optimal number based on metrics and interpretation
    3. **Apply**: Create clusters with your chosen settings
    4. **Visualize**: View patterns in each cluster (page 35)
    """)

    # Sidebar: Data selection
    st.sidebar.header("📁 Data Selection")

    # Find preprocessed files
    preprocessed_dir = Path("data/preprocessed")
    if not preprocessed_dir.exists():
        st.error(f"Preprocessed directory not found: {preprocessed_dir}")
        st.info("Run the preprocessing step first (page 20)")
        return

    h5_files = list(preprocessed_dir.glob("*_preprocessed.h5"))
    if not h5_files:
        st.error("No preprocessed files found")
        st.info("Run the preprocessing step first (page 20)")
        return

    selected_h5 = st.sidebar.selectbox(
        "Select preprocessed data",
        h5_files,
        format_func=lambda x: x.stem.replace("_preprocessed", "")
    )

    # Find corresponding package CSV
    package_files = list(Path("data/packages").glob("*.csv"))

    # Try to match by name
    matched_csv = None
    for csv_file in package_files:
        if csv_file.stem in selected_h5.stem:
            matched_csv = csv_file
            break

    if matched_csv is None and package_files:
        matched_csv = package_files[0]

    selected_csv = st.sidebar.selectbox(
        "Select OHLCV data",
        package_files,
        index=package_files.index(matched_csv) if matched_csv else 0,
        format_func=lambda x: x.name
    )

    # Sidebar: Analysis parameters
    st.sidebar.header("⚙️ Analysis Parameters")

    min_clusters = st.sidebar.number_input(
        "Minimum clusters",
        min_value=2,
        max_value=50,
        value=5,
        step=1,
        help="Minimum number of clusters to test"
    )

    max_clusters = st.sidebar.number_input(
        "Maximum clusters",
        min_value=min_clusters,
        max_value=100,
        value=25,
        step=1,
        help="Maximum number of clusters to test"
    )

    step = st.sidebar.number_input(
        "Step size",
        min_value=1,
        max_value=5,
        value=1,
        help="Increment between cluster counts"
    )

    # Load data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading preprocessed windows..."):
            windows, metadata, norm_stats = load_preprocessed_package(str(selected_h5))
            st.session_state.windows = windows
            st.session_state.metadata = metadata
            st.session_state.norm_stats = norm_stats
            st.session_state.data_name = selected_h5.stem.replace("_preprocessed", "")

        with st.spinner("Loading OHLCV data..."):
            df_ohlcv = pd.read_csv(selected_csv)
            st.session_state.df_ohlcv = df_ohlcv

        with st.spinner("Extracting shape features..."):
            window_size = metadata[0]['end_idx'] - metadata[0]['start_idx'] + 1
            shape_features = extract_windows_shape_features(
                df_ohlcv,
                window_size,
                metadata,
                ohlc_cols=True
            )
            st.session_state.shape_features = shape_features
            st.session_state.window_size = window_size

        st.session_state.data_loaded = True
        st.success("✓ Data loaded successfully!")

    # Show data info if loaded
    if st.session_state.data_loaded:
        st.sidebar.success("✓ Data loaded")
        st.sidebar.metric("Windows", len(st.session_state.metadata))
        st.sidebar.metric("Window Size", st.session_state.window_size)
        st.sidebar.metric("Shape Features", st.session_state.shape_features.shape[1])

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Run Analysis", "📈 View Results", "✅ Apply Clustering"])

    # Tab 1: Run Analysis
    with tab1:
        st.header("Run Optimal Cluster Analysis")

        if not st.session_state.data_loaded:
            st.info("👈 Load data from the sidebar first")
        else:
            st.markdown(f"""
            **Configuration:**
            - Testing cluster range: **{min_clusters}** to **{max_clusters}** (step: {step})
            - Total tests: **{len(range(min_clusters, max_clusters + 1, step))}**
            - Windows to analyze: **{len(st.session_state.metadata)}**
            """)

            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Starting analysis...")

                with st.spinner("Evaluating different cluster counts..."):
                    df_results = evaluate_clustering_range(
                        st.session_state.shape_features,
                        min_clusters,
                        max_clusters,
                        step,
                        progress_bar
                    )

                st.session_state.analysis_results = df_results
                progress_bar.empty()
                st.success("✓ Analysis complete!")
                st.rerun()

    # Tab 2: View Results
    with tab2:
        st.header("Analysis Results")

        if 'analysis_results' not in st.session_state:
            st.info("Run the analysis first (Tab 1)")
        else:
            df_results = st.session_state.analysis_results

            # Show recommendations
            st.subheader("🎯 Recommendations")

            col1, col2, col3 = st.columns(3)

            with col1:
                best_silhouette = df_results.loc[df_results['silhouette'].idxmax()]
                st.metric(
                    "Best by Silhouette",
                    f"n = {int(best_silhouette['n_clusters'])}",
                    f"Score: {best_silhouette['silhouette']:.3f}",
                    help="Measures how well-separated clusters are (0.5+ is good)"
                )

            with col2:
                best_davies = df_results.loc[df_results['davies_bouldin'].idxmin()]
                st.metric(
                    "Best by Davies-Bouldin",
                    f"n = {int(best_davies['n_clusters'])}",
                    f"Score: {best_davies['davies_bouldin']:.3f}",
                    help="Lower is better (below 1.0 is good)"
                )

            with col3:
                best_calinski = df_results.loc[df_results['calinski_harabasz'].idxmax()]
                st.metric(
                    "Best by Calinski-Harabasz",
                    f"n = {int(best_calinski['n_clusters'])}",
                    f"Score: {best_calinski['calinski_harabasz']:.0f}",
                    help="Higher is better (relative metric)"
                )

            # Interpretation guide
            with st.expander("💡 How to interpret these metrics"):
                st.markdown("""
                **Silhouette Score (0 to 1):**
                - Measures how similar windows are to their own cluster vs other clusters
                - > 0.5 = Good separation
                - > 0.7 = Excellent separation
                - Look for the highest value or where it plateaus

                **Davies-Bouldin Score (lower is better):**
                - Ratio of within-cluster to between-cluster distances
                - < 1.0 = Good clustering
                - < 0.5 = Excellent clustering
                - Look for the lowest value

                **Calinski-Harabasz Score (higher is better):**
                - Ratio of between-cluster to within-cluster variance
                - Relative metric (compare within your data)
                - Look for peaks or where it plateaus

                **General advice:**
                - Look for "elbow points" where metrics plateau
                - Balance quality with interpretability (fewer clusters = easier to understand)
                - Consider cluster size distribution (avoid very small clusters)
                """)

            # Visualization
            st.subheader("📈 Metrics Visualization")

            fig = plot_clustering_metrics_plotly(df_results)
            st.plotly_chart(fig, use_container_width=True)

            # Full results table
            st.subheader("📋 Full Results")

            st.dataframe(
                df_results.style.format({
                    'silhouette': '{:.4f}',
                    'calinski_harabasz': '{:.2f}',
                    'davies_bouldin': '{:.4f}',
                    'avg_cluster_size': '{:.1f}',
                    'std_cluster_size': '{:.1f}'
                }).background_gradient(subset=['silhouette'], cmap='Greens')
                  .background_gradient(subset=['davies_bouldin'], cmap='Reds_r'),
                use_container_width=True
            )

            # Download results
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv_data,
                f"optimal_clusters_analysis_{st.session_state.data_name}.csv",
                "text/csv"
            )

    # Tab 3: Apply Clustering
    with tab3:
        st.header("Apply Shape Clustering")

        if not st.session_state.data_loaded:
            st.info("👈 Load data from the sidebar first")
        else:
            st.markdown("""
            Apply shape-based clustering with your chosen parameters.
            This will create cluster labels that can be used for training.
            """)

            col1, col2 = st.columns(2)

            with col1:
                n_clusters = st.number_input(
                    "Number of clusters",
                    min_value=2,
                    max_value=100,
                    value=15,
                    help="Choose based on analysis results"
                )

            with col2:
                method = st.selectbox(
                    "Clustering method",
                    ["kmeans", "hierarchical"],
                    help="K-means is faster, hierarchical creates dendrograms"
                )

            output_dir = Path("data/clustering")

            st.info(f"Results will be saved to: `{output_dir}/`")

            if st.button("✅ Apply Clustering", type="primary", use_container_width=True):
                with st.spinner("Applying shape clustering..."):
                    results_file, stats_file, samples_file, df_stats = apply_shape_clustering(
                        st.session_state.shape_features,
                        n_clusters,
                        method,
                        st.session_state.df_ohlcv,
                        st.session_state.metadata,
                        output_dir,
                        st.session_state.data_name
                    )

                st.success("✓ Clustering applied successfully!")

                st.markdown(f"""
                **Files created:**
                - `{results_file.name}` - Cluster labels and features
                - `{stats_file.name}` - Cluster statistics
                - `{samples_file.name}` - Sample indices for visualization
                """)

                # Show cluster statistics
                st.subheader("📊 Cluster Statistics")

                st.dataframe(
                    df_stats.style.format({
                        'avg_price_change': '{:.4f}',
                        'std_price_change': '{:.4f}',
                        'median_price_change': '{:.4f}'
                    }),
                    use_container_width=True
                )

                st.info("👉 Go to **page 35** to visualize these clusters!")


if __name__ == "__main__":
    main()
