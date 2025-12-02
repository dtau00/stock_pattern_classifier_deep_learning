"""
Normalized Model Inputs & Window Inspector

Combined view for exploring preprocessed data:
- Load and view metadata about preprocessed packages
- Overview of data distribution and statistics
- Individual window inspection with visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.data_viz import plot_single_window


def show():
    """Display normalized model inputs and window inspector"""
    st.title("📊 Preprocessed Data Explorer")
    st.markdown("---")

    preprocessed_dir = project_root / "data" / "preprocessed"

    # Check if directory exists
    if not preprocessed_dir.exists():
        st.warning(f"Preprocessed data directory not found: `{preprocessed_dir}`")
        st.info("""
        **How to create a preprocessed package:**
        1. Go to **🔧 Create Preprocessed** page
        2. Select an OHLCV data package
        3. Run the preprocessing pipeline
        4. The package will be saved to `data/preprocessed/`
        """)
        return

    # List available packages
    packages = [f for f in os.listdir(preprocessed_dir) if f.endswith('.h5')]

    if len(packages) == 0:
        st.warning("No preprocessed packages found.")
        st.info("""
        **How to create a preprocessed package:**
        1. Go to **🔧 Create Preprocessed** page
        2. Select an OHLCV data package
        3. Configure preprocessing settings
        4. Click "Run Preprocessing Pipeline"
        5. Return here to visualize the normalized data
        """)
        return

    # Package selection
    st.subheader("📦 Select Preprocessed Package")
    selected_package = st.selectbox(
        "Choose a package to explore",
        packages,
        key="package_selector"
    )

    package_path = preprocessed_dir / selected_package

    # Load package button or auto-load
    col1, col2 = st.columns([3, 1])

    with col1:
        load_button = st.button("📂 Load Package", type="primary", use_container_width=True)

    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            # Clear session state to force reload
            if 'loaded_package_name' in st.session_state:
                del st.session_state['loaded_package_name']
            if 'preprocessed_windows' in st.session_state:
                del st.session_state['preprocessed_windows']
            if 'package_metadata' in st.session_state:
                del st.session_state['package_metadata']
            st.rerun()

    # Load package if button clicked or if already loaded
    should_load = load_button or st.session_state.get('loaded_package_name') == selected_package

    if should_load:
        with st.spinner(f"Loading {selected_package}..."):
            try:
                import h5py

                with h5py.File(package_path, 'r') as f:
                    windows = f['windows'][:]

                    # Load metadata
                    metadata = {
                        'num_windows': windows.shape[0],
                        'sequence_length': windows.shape[1],
                        'num_channels': windows.shape[2],
                    }

                    # Load additional metadata from HDF5 attributes
                    if 'window_metadata' in f.attrs:
                        window_metadata = json.loads(f.attrs['window_metadata'])
                        metadata['window_metadata'] = window_metadata

                    if 'normalization_stats' in f.attrs:
                        normalization_stats = json.loads(f.attrs['normalization_stats'])
                        metadata['normalization_stats'] = normalization_stats

                    if 'additional_metadata' in f.attrs:
                        additional_metadata = json.loads(f.attrs['additional_metadata'])
                        metadata.update(additional_metadata)

                    # Store in session state
                    st.session_state['preprocessed_windows'] = windows
                    st.session_state['package_metadata'] = metadata
                    st.session_state['loaded_package_name'] = selected_package

                st.success(f"✅ Loaded {metadata['num_windows']:,} windows")

            except Exception as e:
                st.error(f"❌ Error loading package: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                return

    # Display loaded package info
    if 'preprocessed_windows' not in st.session_state:
        st.info("👆 Click 'Load Package' to begin exploring the data")
        return

    windows = st.session_state['preprocessed_windows']
    metadata = st.session_state['package_metadata']

    # Display package overview
    st.markdown("---")
    st.subheader("📊 Package Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Windows", f"{metadata['num_windows']:,}")

    with col2:
        st.metric("Window Length", f"{metadata['sequence_length']} bars")

    with col3:
        st.metric("Channels", metadata['num_channels'])

    with col4:
        file_size_mb = os.path.getsize(package_path) / (1024 * 1024)
        st.metric("File Size", f"{file_size_mb:.2f} MB")

    # Additional metadata
    if metadata.get('symbol') or metadata.get('interval'):
        col1, col2 = st.columns(2)

        with col1:
            if metadata.get('symbol'):
                st.metric("Symbol", metadata['symbol'])

        with col2:
            if metadata.get('interval'):
                st.metric("Interval", metadata['interval'])

    # Distribution statistics
    st.markdown("---")
    st.subheader("📈 Data Distribution")

    channel_names = ['Returns', 'Volume/Liquidity', 'Volatility/Risk']

    # Calculate statistics across all windows
    tabs = st.tabs(channel_names)

    for i, (tab, channel_name) in enumerate(zip(tabs, channel_names)):
        with tab:
            channel_data = windows[:, :, i].flatten()

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Mean", f"{channel_data.mean():.4f}")

            with col2:
                st.metric("Std Dev", f"{channel_data.std():.4f}")

            with col3:
                st.metric("Min", f"{channel_data.min():.4f}")

            with col4:
                st.metric("Max", f"{channel_data.max():.4f}")

            with col5:
                st.metric("Median", f"{np.median(channel_data):.4f}")

            # Histogram
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=channel_data,
                nbinsx=100,
                name=channel_name,
                marker_color='#1f77b4'
            ))

            fig.update_layout(
                title=f"{channel_name} Distribution (All Windows)",
                xaxis_title="Normalized Value",
                yaxis_title="Count",
                height=400,
                template='plotly_white',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    # Normalization stats
    if 'normalization_stats' in metadata:
        with st.expander("🔢 Normalization Statistics"):
            st.json(metadata['normalization_stats'])

    # Source metadata
    if metadata.get('source_bars') or metadata.get('gaps_detected'):
        with st.expander("📋 Source Data Info"):
            info_cols = st.columns(4)

            if metadata.get('source_bars'):
                info_cols[0].metric("Source Bars", f"{metadata['source_bars']:,}")

            if metadata.get('gaps_detected') is not None:
                info_cols[1].metric("Gaps Detected", metadata['gaps_detected'])

            if metadata.get('excluded_bars'):
                info_cols[2].metric("Excluded Bars", f"{metadata['excluded_bars']:,}")

            if metadata.get('source_package'):
                st.write(f"**Source Package:** `{metadata['source_package']}`")

    # Window Inspector
    st.markdown("---")
    st.subheader("🔍 Window Inspector")
    st.caption("Explore individual windows in detail")

    # Window selection controls
    col1, col2 = st.columns([4, 1])

    with col1:
        window_idx = st.slider(
            "Select window",
            0,
            len(windows) - 1,
            0,
            key="window_slider"
        )

    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        random_window = st.button("🎲 Random", use_container_width=True)

        if random_window:
            window_idx = np.random.randint(0, len(windows))
            st.session_state['window_slider'] = window_idx
            st.rerun()

    # Display window info
    st.write(f"**Window {window_idx + 1} of {len(windows):,}**")

    # Get window metadata if available
    if 'window_metadata' in metadata and window_idx < len(metadata['window_metadata']):
        win_meta = metadata['window_metadata'][window_idx]

        meta_cols = st.columns(4)

        if 'start_idx' in win_meta:
            meta_cols[0].caption(f"Start Index: {win_meta['start_idx']}")

        if 'end_idx' in win_meta:
            meta_cols[1].caption(f"End Index: {win_meta['end_idx']}")

        if 'start_timestamp' in win_meta:
            from datetime import datetime
            start_dt = datetime.fromtimestamp(win_meta['start_timestamp'] / 1000)
            meta_cols[2].caption(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M')}")

        if 'end_timestamp' in win_meta:
            from datetime import datetime
            end_dt = datetime.fromtimestamp(win_meta['end_timestamp'] / 1000)
            meta_cols[3].caption(f"End: {end_dt.strftime('%Y-%m-%d %H:%M')}")

    # Visualize window
    try:
        window = windows[window_idx]
        fig = plot_single_window(window, window_idx=window_idx)
        st.plotly_chart(fig, use_container_width=True)

        # Window statistics
        with st.expander("📊 Window Statistics"):
            st.write(f"**Window Index:** {window_idx}")
            st.write(f"**Window Shape:** {window.shape}")
            st.markdown("---")

            for i, channel_name in enumerate(channel_names):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.write(f"**{channel_name}**")
                with col2:
                    st.metric("Mean", f"{window[:, i].mean():.4f}")
                with col3:
                    st.metric("Std", f"{window[:, i].std():.4f}")
                with col4:
                    st.metric("Range", f"[{window[:, i].min():.2f}, {window[:, i].max():.2f}]")

        # Raw data table
        with st.expander("📄 Raw Window Data"):
            df_window = pd.DataFrame(
                window,
                columns=channel_names
            )
            df_window.insert(0, 'Timestep', range(len(df_window)))
            st.dataframe(df_window, use_container_width=True, height=400)

    except Exception as e:
        st.error(f"❌ Error visualizing window: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())


# Execute the main function
show()
