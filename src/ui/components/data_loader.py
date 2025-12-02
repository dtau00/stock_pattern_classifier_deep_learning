"""
Reusable UI Components for Data Loading

This module contains reusable Streamlit components for loading and filtering data packages.
"""
import streamlit as st
import pandas as pd
import os


def load_package_with_date_range(packages_dir, session_key, key_prefix,
                                   file_extension='.csv',
                                   package_type="OHLCV"):
    """
    Reusable component for loading data packages with date range selection.

    Parameters:
    -----------
    packages_dir : str
        Directory containing the data packages
    session_key : str
        Key to store the loaded data in session state
    key_prefix : str
        Prefix for Streamlit widget keys to ensure uniqueness
    file_extension : str
        File extension to filter packages (default: '.csv')
    package_type : str
        Type of package for display messages (default: 'OHLCV')

    Returns:
    --------
    bool
        True if data was loaded successfully, False otherwise
    """
    if not os.path.exists(packages_dir):
        st.warning(f"{package_type} packages directory not found: {packages_dir}")
        st.info(f"Please download {package_type} data first using the OHLCV Manager page.")
        return False

    # List available packages
    packages = [f for f in os.listdir(packages_dir) if f.endswith(file_extension)]

    if len(packages) == 0:
        st.warning(f"No {package_type} packages found.")
        st.info(f"Please download {package_type} data first using the OHLCV Manager page.")
        return False

    selected_package = st.selectbox(
        f"Select {package_type} package",
        packages,
        key=f"{key_prefix}_package_selector"
    )

    # Store selected package name in session state for metadata extraction (use different key)
    st.session_state[f"{key_prefix}_selected_package"] = selected_package

    # Load package initially to get date range
    try:
        package_path = os.path.join(packages_dir, selected_package)

        # Handle different file types
        if file_extension == '.csv':
            df_temp = pd.read_csv(package_path)
        elif file_extension == '.h5':
            import h5py
            # For HDF5, we'll need custom handling - return False for now
            st.warning("HDF5 loading not implemented in this component yet.")
            return False
        else:
            st.error(f"Unsupported file extension: {file_extension}")
            return False

        # Convert timestamp if present
        if 'timestamp' in df_temp.columns:
            try:
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
            except:
                try:
                    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
                except:
                    st.warning("Could not parse timestamp column")

        # Date range selection if timestamp available
        if 'timestamp' in df_temp.columns and df_temp['timestamp'].notna().any():
            min_date = df_temp['timestamp'].min().date()
            max_date = df_temp['timestamp'].max().date()

            col1, col2, col3 = st.columns(3)

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{key_prefix}_start_date"
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{key_prefix}_end_date"
                )

            with col3:
                # Show estimated bars in range
                mask = (df_temp['timestamp'].dt.date >= start_date) & (df_temp['timestamp'].dt.date <= end_date)
                bars_in_range = mask.sum()
                st.metric("Bars in selected range", f"{bars_in_range:,}")

            # Validate date range
            if start_date > end_date:
                st.error("❌ Start date must be before end date")
                return False
            else:
                # Load button with date filtering
                if st.button("Load & Visualize", type="primary", key=f"{key_prefix}_load"):
                    # Filter data by selected date range
                    df_filtered = df_temp[mask].copy()

                    if len(df_filtered) == 0:
                        st.error("❌ No data in selected date range")
                        return False
                    else:
                        st.session_state[session_key] = df_filtered
                        return True
        else:
            st.warning("⚠️ No timestamp column found - using entire dataset")
            if st.button("📊 Load & Visualize", type="primary", key=f"{key_prefix}_load_no_date"):
                st.session_state[session_key] = df_temp
                st.success(f"✓ Loaded {len(df_temp):,} bars")
                return True

    except Exception as e:
        st.error(f"Error loading package: {e}")
        return False

    return False
