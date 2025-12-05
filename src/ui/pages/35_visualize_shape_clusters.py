"""
Shape Cluster Visualization

Visualize the results of hierarchical shape clustering.
Shows samples from each shape cluster to understand what price patterns
are grouped together.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_shape_clustering_results(results_path: str):
    """Load shape clustering results."""
    data = np.load(results_path)

    return {
        'shape_labels': data['shape_labels'],
        'n_shape_clusters': int(data['n_shape_clusters']),
        'shape_features': data.get('shape_features', None)
    }


def load_cluster_stats(stats_path: str):
    """Load cluster statistics."""
    return pd.read_csv(stats_path)


def load_sample_indices(samples_path: str):
    """Load sample window indices for each cluster."""
    with open(samples_path, 'r') as f:
        return json.load(f)


def plot_window(df: pd.DataFrame, start_idx: int, end_idx: int, title: str = ""):
    """Plot a single price window."""
    window = df.iloc[start_idx:end_idx + 1]

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=list(range(len(window))),
        open=window['open'],
        high=window['high'],
        low=window['low'],
        close=window['close'],
        name='OHLC'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Bar Index',
        yaxis_title='Price',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        xaxis_rangeslider_visible=False
    )

    return fig


def plot_normalized_window(df: pd.DataFrame, start_idx: int, end_idx: int, title: str = ""):
    """Plot a single price window normalized to start at 0."""
    window = df.iloc[start_idx:end_idx + 1].copy()

    # Normalize to percentage change from start
    first_close = window['close'].iloc[0]
    window['close_norm'] = (window['close'] - first_close) / first_close * 100
    window['high_norm'] = (window['high'] - first_close) / first_close * 100
    window['low_norm'] = (window['low'] - first_close) / first_close * 100

    fig = go.Figure()

    # Plot normalized price
    fig.add_trace(go.Scatter(
        x=list(range(len(window))),
        y=window['close_norm'],
        mode='lines+markers',
        name='Close',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))

    # Add high/low range
    fig.add_trace(go.Scatter(
        x=list(range(len(window))),
        y=window['high_norm'],
        mode='lines',
        name='High',
        line=dict(color='lightblue', width=1, dash='dash'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(window))),
        y=window['low_norm'],
        mode='lines',
        name='Low',
        line=dict(color='lightblue', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.2)',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Bar Index',
        yaxis_title='% Change from Start',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    return fig


def main():
    st.set_page_config(page_title="Shape Cluster Visualization", layout="wide")

    st.title("📊 Shape Cluster Visualization")

    st.markdown("""
    This page shows the results of **hierarchical shape clustering**.
    Each cluster groups windows with visually similar price patterns.
    """)

    # File selection
    st.sidebar.header("Data Selection")

    clustering_dir = Path("data/clustering")
    if not clustering_dir.exists():
        st.error(f"Clustering directory not found: {clustering_dir}")
        st.info("Run: `python scripts/apply_hierarchical_clustering.py` first")
        return

    # Find available clustering results
    result_files = list(clustering_dir.glob("*_shape_clusters.npz"))

    if not result_files:
        st.error("No clustering results found")
        st.info("Run: `python scripts/apply_hierarchical_clustering.py` first")
        return

    selected_file = st.sidebar.selectbox(
        "Select clustering results",
        result_files,
        format_func=lambda x: x.stem
    )

    # Load results
    with st.spinner("Loading clustering results..."):
        results = load_shape_clustering_results(str(selected_file))

        stats_file = selected_file.parent / f"{selected_file.stem}_stats.csv"
        samples_file = selected_file.parent / f"{selected_file.stem}_samples.json"

        if not stats_file.exists():
            st.error(f"Statistics file not found: {stats_file}")
            return

        if not samples_file.exists():
            st.error(f"Samples file not found: {samples_file}")
            return

        df_stats = load_cluster_stats(str(stats_file))

        # Add pct_of_total column
        total_windows = df_stats['num_windows'].sum()
        df_stats['pct_of_total'] = (df_stats['num_windows'] / total_windows) * 100

        samples = load_sample_indices(str(samples_file))

    # Load original OHLCV data
    st.sidebar.header("OHLCV Data")

    package_files = list(Path("data/packages").glob("*.csv"))
    if not package_files:
        st.error("No package CSV files found in data/packages/")
        return

    selected_package = st.sidebar.selectbox(
        "Select OHLCV data",
        package_files,
        format_func=lambda x: x.name
    )

    with st.spinner("Loading OHLCV data..."):
        df_ohlcv = pd.read_csv(selected_package)

    # Display summary
    st.header("📈 Clustering Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Windows", len(results['shape_labels']))
    with col2:
        st.metric("Shape Clusters", results['n_shape_clusters'])
    with col3:
        avg_cluster_size = len(results['shape_labels']) / results['n_shape_clusters']
        st.metric("Avg Cluster Size", f"{avg_cluster_size:.0f}")

    # Display cluster statistics
    st.header("📊 Cluster Statistics")

    st.dataframe(
        df_stats.style.format({
            'pct_of_total': '{:.1f}%',
            'avg_price_change': '{:.4f}',
            'std_price_change': '{:.4f}',
            'median_price_change': '{:.4f}'
        }),
        use_container_width=True
    )

    # Cluster visualization
    st.header("🔍 Cluster Patterns")

    st.markdown("""
    Below you can see sample windows from each shape cluster.
    Windows in the same cluster should have visually similar price patterns.
    """)

    # Cluster selection
    cluster_id = st.selectbox(
        "Select shape cluster to visualize",
        options=sorted(samples.keys(), key=int),
        format_func=lambda x: f"Shape Cluster {x} ({df_stats[df_stats['cluster_id']==int(x)]['num_windows'].values[0]} windows)"
    )

    cluster_id_int = int(cluster_id)

    # Get cluster statistics
    cluster_row = df_stats[df_stats['cluster_id'] == cluster_id_int].iloc[0]

    st.subheader(f"Shape Cluster {cluster_id} Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Windows", f"{cluster_row['num_windows']:.0f}")
    with col2:
        st.metric("% of Total", f"{cluster_row['pct_of_total']:.1f}%")
    with col3:
        st.metric("Avg Price Change", f"{cluster_row['avg_price_change']:.2%}")
    with col4:
        st.metric("Std Price Change", f"{cluster_row['std_price_change']:.4f}")

    # Display mode selection
    display_mode = st.radio(
        "Display mode",
        ["Normalized (% change)", "Absolute prices"],
        horizontal=True
    )

    # Number of samples to show
    n_samples = st.slider("Number of samples to show", 4, 20, 12, step=4)

    # Get sample indices for this cluster
    cluster_samples = samples[str(cluster_id)]
    n_to_show = min(n_samples, len(cluster_samples))
    sample_indices = cluster_samples[:n_to_show]

    # Load metadata to get window boundaries
    preprocessed_file = Path("data/preprocessed") / f"{selected_package.stem}_preprocessed.h5"
    if not preprocessed_file.exists():
        st.error(f"Preprocessed file not found: {preprocessed_file}")
        return

    import h5py
    with h5py.File(preprocessed_file, 'r') as f:
        metadata = json.loads(f.attrs['window_metadata'])

    # Plot samples in grid
    st.subheader(f"Sample Windows from Cluster {cluster_id}")

    # Create grid layout
    cols_per_row = 3
    for i in range(0, n_to_show, cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < n_to_show:
                sample_idx = sample_indices[i + j]
                meta = metadata[sample_idx]

                start_idx = meta['start_idx']
                end_idx = meta['end_idx']

                with col:
                    if display_mode == "Normalized (% change)":
                        fig = plot_normalized_window(
                            df_ohlcv,
                            start_idx,
                            end_idx,
                            title=f"Window {sample_idx}"
                        )
                    else:
                        fig = plot_window(
                            df_ohlcv,
                            start_idx,
                            end_idx,
                            title=f"Window {sample_idx}"
                        )

                    st.plotly_chart(fig, use_container_width=True)

    # Pattern interpretation
    st.header("💡 Pattern Interpretation")

    st.markdown("""
    ### How to interpret clusters:

    **Good clustering** = Windows in the same cluster look visually similar
    - Same general direction (up, down, or sideways)
    - Similar shapes and turning points
    - Similar volatility characteristics

    **What to look for:**
    - **Cluster 0-3**: Often strong trends (up or down)
    - **Middle clusters**: Consolidation or range-bound patterns
    - **Last clusters**: Reversals or volatile patterns

    ### Next steps:

    1. **Identify useful patterns**: Which clusters have predictive value?
    2. **Adjust n_clusters**: Too few = mixed patterns, too many = overfitting
    3. **Apply Stage 2**: Use contrastive learning to refine within shapes
    """)


if __name__ == "__main__":
    main()
