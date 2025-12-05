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
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.ucl_tsc_model import UCLTSCModel
from src.config.config import Config, ModelConfig, TrainingConfig, AugmentationConfig, DataConfig

# Register safe globals for torch.load (PyTorch 2.6+)
torch.serialization.add_safe_globals([Config, ModelConfig, TrainingConfig, AugmentationConfig, DataConfig])


def load_ohlcv_data_from_metadata(uploaded_file):
    """
    Load the original OHLCV data that corresponds to the preprocessed windows.

    Args:
        uploaded_file: The uploaded HDF5 file or file path

    Returns:
        DataFrame with OHLCV data and timestamps, or None if not found
    """
    try:
        # Read metadata from HDF5
        if isinstance(uploaded_file, str):
            file_to_open = uploaded_file
        else:
            file_to_open = uploaded_file

        with h5py.File(file_to_open, 'r') as f:
            additional_metadata = json.loads(f.attrs['additional_metadata'])

        # Extract info
        source_package = additional_metadata.get('source_package')
        symbol = additional_metadata.get('symbol')
        interval = additional_metadata.get('interval')

        if not source_package:
            return None

        # Try to find the source OHLCV file
        ohlcv_path = Path("data/packages") / source_package

        if not ohlcv_path.exists():
            st.warning(f"Source OHLCV file not found: {ohlcv_path}")
            return None

        # Load OHLCV data
        df = pd.read_csv(ohlcv_path)

        # Convert timestamp column to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

        return df

    except Exception as e:
        st.error(f"Error loading OHLCV data: {e}")
        return None


def get_window_ohlcv(ohlcv_df, start_timestamp, end_timestamp):
    """
    Extract OHLCV data for a specific window based on timestamps.

    Args:
        ohlcv_df: DataFrame with OHLCV data and timestamps
        start_timestamp: Start timestamp in milliseconds
        end_timestamp: End timestamp in milliseconds

    Returns:
        DataFrame slice with OHLCV data for the window
    """
    # Convert timestamps to datetime
    start_dt = pd.to_datetime(start_timestamp, unit='ms')
    end_dt = pd.to_datetime(end_timestamp, unit='ms')

    # Filter data
    mask = (ohlcv_df['timestamp'] >= start_dt) & (ohlcv_df['timestamp'] <= end_dt)
    return ohlcv_df[mask]


def create_candlestick_chart(window_ohlcv, cluster_id, window_idx, show_volume=True, height=500):
    """
    Create a candlestick chart from OHLCV data.

    Args:
        window_ohlcv: DataFrame with OHLCV data
        cluster_id: Cluster ID for the title
        window_idx: Window index for the title
        show_volume: Whether to show volume subplot
        height: Chart height in pixels

    Returns:
        Plotly figure with candlestick chart
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price', 'Volume')
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=window_ohlcv['timestamp'],
            open=window_ohlcv['open'],
            high=window_ohlcv['high'],
            low=window_ohlcv['low'],
            close=window_ohlcv['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Volume bars (only if show_volume is True)
    if show_volume:
        colors = ['red' if close < open else 'green'
                  for close, open in zip(window_ohlcv['close'], window_ohlcv['open'])]

        fig.add_trace(
            go.Bar(
                x=window_ohlcv['timestamp'],
                y=window_ohlcv['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

    layout_kwargs = {
        'title': f"Window {window_idx} - Cluster {cluster_id}",
        'xaxis_rangeslider_visible': False,
        'height': height,
        'yaxis_title': "Price"
    }

    if show_volume:
        layout_kwargs['xaxis2_title'] = "Time"
        layout_kwargs['yaxis2_title'] = "Volume"
    else:
        layout_kwargs['xaxis_title'] = "Time"

    fig.update_layout(**layout_kwargs)

    return fig


def main():
    st.title("🔮 Model Inference & Pattern Recognition")
    st.markdown("Use trained models to classify and analyze stock patterns")

    # Model and Data Selection Section
    st.subheader("🔧 Setup")

    col_model, col_data = st.columns(2)

    with col_model:
        st.markdown("**Model Selection**")

        # Display loaded model info or selection
        if 'model' in st.session_state and st.session_state.get('training_complete', False):
            config = st.session_state.get('config')

            st.success("✅ Model Loaded")

            if 'loaded_model_path' in st.session_state:
                model_name = st.session_state['loaded_model_path'].stem
                st.caption(f"📁 `{model_name}`")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latent Dim", config.model.d_z)
                st.metric("Num Clusters", config.model.num_clusters)
            with col2:
                encoder_type = "Hybrid" if config.model.use_hybrid_encoder else "CNN"
                st.metric("Encoder", encoder_type)
                seq_len = getattr(config.model, 'seq_length', 127)
                st.metric("Seq Length", seq_len)

            if st.button("📂 Change Model", use_container_width=True):
                st.session_state['show_model_selector'] = True
                st.rerun()
        else:
            st.warning("⚠️ No model loaded")
            st.session_state['show_model_selector'] = True

        # Model selector
        if st.session_state.get('show_model_selector', False):
            model_dir = Path("models/trained")
            if model_dir.exists():
                model_files = sorted(list(model_dir.glob("*.pt")), key=lambda x: x.stat().st_mtime, reverse=True)
                if model_files:
                    selected = st.selectbox(
                        "Select model:",
                        options=model_files,
                        format_func=lambda x: f"{x.stem} ({x.stat().st_size / 1024 / 1024:.1f} MB)"
                    )

                    if st.button("✅ Load Model", type="primary", use_container_width=True):
                        load_model(selected)
                        st.session_state['show_model_selector'] = False
                        st.rerun()
                else:
                    st.info("No trained models found in `models/trained/`")
            else:
                st.error("Model directory not found: `models/trained/`")

    with col_data:
        st.markdown("**Data Selection**")

        # Display loaded data info or selection
        if 'data' in st.session_state and 'window_metadata' in st.session_state:
            st.success("✅ Data Loaded")

            if 'loaded_data_filename' in st.session_state:
                st.caption(f"📁 `{st.session_state['loaded_data_filename']}`")

            metadata = st.session_state['window_metadata']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Windows", f"{len(st.session_state['data']):,}")
                if len(metadata) > 0:
                    symbol = metadata[0].get('symbol', 'Unknown')
                    st.metric("Symbol", symbol)
            with col2:
                if len(metadata) > 0:
                    interval = metadata[0].get('interval', 'Unknown')
                    st.metric("Interval", interval)
                has_ohlcv = 'ohlcv_data' in st.session_state and st.session_state['ohlcv_data'] is not None
                st.metric("OHLCV", "✓" if has_ohlcv else "✗")

            if st.button("📂 Change Data", use_container_width=True):
                st.session_state['show_data_selector'] = True
                st.rerun()
        else:
            st.warning("⚠️ No data loaded")
            st.session_state['show_data_selector'] = True

        # Data selector
        if st.session_state.get('show_data_selector', False):
            data_dir = Path("data/preprocessed")
            if data_dir.exists():
                data_files = sorted(list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5")),
                                   key=lambda x: x.stat().st_mtime, reverse=True)
                if data_files:
                    selected_data = st.selectbox(
                        "Select preprocessed data:",
                        options=data_files,
                        format_func=lambda x: f"{x.stem} ({x.stat().st_size / 1024 / 1024:.1f} MB)"
                    )

                    if st.button("✅ Load Data", type="primary", use_container_width=True):
                        load_data_file(selected_data)
                        st.session_state['show_data_selector'] = False
                        st.rerun()
                else:
                    st.info("No preprocessed data found in `data/preprocessed/`")
            else:
                st.error("Data directory not found: `data/preprocessed/`")

    st.divider()

    # Check if both model and data are loaded before showing tabs
    if 'model' not in st.session_state or not st.session_state.get('training_complete', False):
        st.info("👆 Please load a model to continue")
        return

    if 'data' not in st.session_state:
        st.info("👆 Please load preprocessed data to continue")
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


def load_data_file(file_path):
    """Load preprocessed data from HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            windows = f['windows'][:]
            # Load metadata
            window_metadata = json.loads(f.attrs['window_metadata'])

        # Convert to torch tensor
        data = torch.from_numpy(windows).float()
        # Transpose from (N, T, C) to (N, C, T)
        data = data.permute(0, 2, 1)

        # Load corresponding OHLCV data
        ohlcv_df = load_ohlcv_data_from_metadata(str(file_path))

        # Store metadata and OHLCV data
        st.session_state['window_metadata'] = window_metadata
        st.session_state['ohlcv_data'] = ohlcv_df
        st.session_state['loaded_data_filename'] = file_path.name
        st.session_state['data'] = data

        if ohlcv_df is not None:
            st.success(f"✅ Loaded {len(data):,} windows with OHLCV data from {file_path.name}")
        else:
            st.success(f"✅ Loaded {len(data):,} windows from {file_path.name}")
    except Exception as e:
        st.error(f"Failed to load data: {e}")


def load_model(model_path):
    """Load a model checkpoint into session."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']

        # Get architecture parameters with backward compatibility
        seq_len = config.model.seq_length if hasattr(config.model, 'seq_length') else 127
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
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False
        )

        if missing_keys or unexpected_keys:
            st.warning(f"⚠️ Model architecture mismatch: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")

        model.eval()

        st.session_state['model'] = model
        st.session_state['config'] = config
        st.session_state['history'] = checkpoint['history']
        st.session_state['training_complete'] = True
        st.session_state['loaded_model_path'] = model_path

        st.success(f"✅ Model loaded: {model_path.name}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")


def show_pattern_classification():
    """Classify patterns from uploaded data."""
    st.header("Pattern Classification")

    st.markdown("""
    Classify patterns using the loaded model and data.
    The model will assign each window to one of the learned pattern clusters.
    """)

    # Get data from session (already loaded in main)
    data = st.session_state['data']

    # Classify button
    if st.button("🚀 Classify Patterns", type="primary"):
        with st.spinner("Classifying patterns..."):
            model = st.session_state['model']
            model.eval()

            # Get calibrated gamma (if available)
            calibration = st.session_state.get('calibration', {})
            gamma = calibration.get('best_gamma', 5.0)

            with torch.no_grad():
                # NEW: Use predict_with_confidence instead of forward
                cluster_ids, confidence_scores, metrics = model.predict_with_confidence(
                    data, gamma=gamma
                )

            cluster_ids = cluster_ids.cpu().numpy()
            confidence_scores = confidence_scores.cpu().numpy()
            z_norm = metrics['z_normalized'].cpu().numpy()

            # Store results
            st.session_state['inference_results'] = {
                'cluster_ids': cluster_ids,
                'confidence_scores': confidence_scores,
                'latent_vectors': z_norm,
                'data': data,
                'gamma': gamma
            }

            st.success(f"✅ Classification complete! (using gamma={gamma:.1f})")

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

        # NEW: Confidence Score Visualization
        if 'confidence_scores' in results:
            st.divider()
            st.subheader("🎯 Confidence Scores")

            confidence_scores = results['confidence_scores']
            threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Samples above this threshold are considered high-confidence"
            )

            col1, col2 = st.columns([1, 2])

            with col1:
                # Confidence statistics
                high_conf_mask = confidence_scores >= threshold
                n_high = high_conf_mask.sum()
                n_low = len(confidence_scores) - n_high

                st.metric("High Confidence", f"{n_high:,}",
                         delta=f"{n_high/len(confidence_scores)*100:.1f}%")
                st.metric("Low Confidence", f"{n_low:,}",
                         delta=f"{n_low/len(confidence_scores)*100:.1f}%")
                st.metric("Mean Confidence", f"{confidence_scores.mean():.3f}")
                st.metric("Median Confidence", f"{np.median(confidence_scores):.3f}")

                # Calibration info
                gamma = results.get('gamma', 'N/A')
                st.caption(f"Calibrated gamma: {gamma}")

            with col2:
                # Confidence histogram
                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=confidence_scores,
                    nbinsx=50,
                    name='Confidence Scores',
                    marker_color='lightblue'
                ))

                # Add threshold line
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Threshold ({threshold:.2f})",
                    annotation_position="top right"
                )

                fig.update_layout(
                    title="Confidence Score Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Count",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            # Show high/low confidence breakdown by cluster
            st.subheader("Confidence by Cluster")

            conf_by_cluster = []
            for cluster_id in unique:
                mask = cluster_ids == cluster_id
                cluster_conf = confidence_scores[mask]
                high_conf_pct = (cluster_conf >= threshold).sum() / len(cluster_conf) * 100

                conf_by_cluster.append({
                    'Cluster': cluster_id,
                    'Count': len(cluster_conf),
                    'Mean Confidence': cluster_conf.mean(),
                    'High Conf %': high_conf_pct
                })

            conf_df = pd.DataFrame(conf_by_cluster)
            conf_df['Mean Confidence'] = conf_df['Mean Confidence'].round(3)
            conf_df['High Conf %'] = conf_df['High Conf %'].round(1)

            st.dataframe(conf_df, use_container_width=True, hide_index=True)

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

    # Get indices in the original data
    original_indices = np.where(mask)[0]

    st.markdown(f"### Cluster {cluster_id}")

    # Grid layout configuration
    st.markdown("**Display Configuration**")
    col_config1, col_config2, col_config3, col_config4 = st.columns(4)

    with col_config1:
        n_cols = st.selectbox(
            "Columns",
            options=[1, 2, 3, 4],
            index=3,  # Default to 4 columns
            key=f"cols_{cluster_id}"
        )

    with col_config2:
        n_rows = st.selectbox(
            "Rows",
            options=list(range(4, 11)),  # 4-10 rows
            index=0,  # Default to 4 rows
            key=f"rows_{cluster_id}"
        )

    with col_config3:
        chart_height = st.selectbox(
            "Chart Height",
            options=[200, 300, 400, 500, 600, 800, 1000],
            index=3,  # Default to 500
            key=f"height_{cluster_id}",
            format_func=lambda x: f"{x}px"
        )

    with col_config4:
        show_volume = st.checkbox(
            "Show Volume",
            value=True,
            key=f"volume_{cluster_id}"
        )

    # Calculate samples per page
    samples_per_page = n_cols * n_rows
    total_samples = len(cluster_data)
    total_pages = (total_samples + samples_per_page - 1) // samples_per_page

    # Pagination controls
    st.markdown(f"**Sample Windows** ({total_samples:,} total windows)")

    # Initialize page number in session state
    page_key = f"page_{cluster_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    # Pagination buttons
    col_prev, col_pages, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button("◀ Prev", key=f"prev_{cluster_id}", disabled=(st.session_state[page_key] == 0)):
            st.session_state[page_key] = max(0, st.session_state[page_key] - 1)
            st.rerun()

    with col_next:
        if st.button("Next ▶", key=f"next_{cluster_id}", disabled=(st.session_state[page_key] >= total_pages - 1)):
            st.session_state[page_key] = min(total_pages - 1, st.session_state[page_key] + 1)
            st.rerun()

    with col_pages:
        # Page number buttons
        current_page = st.session_state[page_key]

        # Show up to 10 page numbers
        start_page = max(0, current_page - 4)
        end_page = min(total_pages, start_page + 10)

        page_cols = st.columns(min(10, end_page - start_page))
        for i, page_num in enumerate(range(start_page, end_page)):
            with page_cols[i]:
                if page_num == current_page:
                    st.markdown(f"**{page_num + 1}**")
                else:
                    if st.button(str(page_num + 1), key=f"page_{cluster_id}_{page_num}"):
                        st.session_state[page_key] = page_num
                        st.rerun()

    st.caption(f"Page {current_page + 1} of {total_pages}")

    # Get samples for current page
    start_idx = current_page * samples_per_page
    end_idx = min(start_idx + samples_per_page, total_samples)
    page_indices = list(range(start_idx, end_idx))

    # Check if OHLCV data is available
    has_ohlcv = ('ohlcv_data' in st.session_state and
                 st.session_state['ohlcv_data'] is not None and
                 'window_metadata' in st.session_state)

    # Display samples in grid layout
    for row in range(n_rows):
        cols = st.columns(n_cols)

        for col in range(n_cols):
            sample_num = row * n_cols + col

            if sample_num >= len(page_indices):
                break

            idx = page_indices[sample_num]
            window = cluster_data[idx].cpu().numpy()  # Shape: (C, T)
            original_idx = original_indices[idx]

            with cols[col]:
                # Try to show OHLCV candlestick chart
                chart_displayed = False

                if has_ohlcv:
                    try:
                        # Get metadata for this window
                        window_meta = st.session_state['window_metadata'][original_idx]
                        start_ts = window_meta.get('start_timestamp')
                        end_ts = window_meta.get('end_timestamp')

                        if start_ts and end_ts:
                            # Get OHLCV data for this window
                            ohlcv_df = st.session_state['ohlcv_data']
                            window_ohlcv = get_window_ohlcv(ohlcv_df, start_ts, end_ts)

                            if len(window_ohlcv) > 0:
                                # Create and display candlestick chart with user-configured settings
                                fig = create_candlestick_chart(
                                    window_ohlcv,
                                    cluster_id,
                                    original_idx,
                                    show_volume=show_volume,
                                    height=chart_height
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{cluster_id}_{original_idx}")
                                chart_displayed = True
                    except Exception as e:
                        # Silently fall back to normalized features
                        pass

                # Fallback: show normalized features as line chart
                if not chart_displayed:
                    fig = go.Figure()

                    channel_names = ['Returns', 'Volume/OBV', 'Volatility/NATR']
                    for c in range(window.shape[0]):
                        fig.add_trace(go.Scatter(
                            y=window[c],
                            name=channel_names[c],
                            mode='lines'
                        ))

                    fig.update_layout(
                        title=f"Window {original_idx}",
                        xaxis_title="Time Step",
                        yaxis_title="Normalized Value",
                        height=chart_height,
                        showlegend=(n_cols <= 2)  # Only show legend for wider layouts
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{cluster_id}_{original_idx}")


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
