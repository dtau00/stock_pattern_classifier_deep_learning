"""
Manage Packages Page
View and manage downloaded data packages
"""
import streamlit as st

def show():
    """Manage downloaded data packages"""
    st.title("📦 Manage Packages")
    st.markdown("---")

    st.info("**Section Purpose:** View, delete, export, and reload downloaded data packages")

    # Placeholder content
    st.markdown("""
    ### Features to be implemented:
    - Interactive table of all downloaded packages
    - Package metadata display (symbol, timeframe, date range, bars, size)
    - Delete/archive functionality
    - Export to CSV/HDF5
    - Reload/refresh packages
    - Search and filter capabilities
    """)

    st.warning("⚠️ This section will be implemented in Phase 2")

    # Preview of what the UI will look like
    with st.expander("📋 UI Preview"):
        st.markdown("""
        **Package Management Table:**
        ```
        | Symbol   | Timeframe | Date Range              | Bars  | Size   | Downloaded      | Actions   |
        |----------|-----------|-------------------------|-------|--------|-----------------|-----------|
        | BTCUSDT  | 1h        | 2024-01-01 - 2024-12-31 | 8,760 | 2.5 MB | 2024-12-01 10:30| [Details] |
        | ETHUSDT  | 1h        | 2024-01-01 - 2024-12-31 | 8,760 | 2.5 MB | 2024-12-01 11:15| [Details] |
        ```
        
        **Actions:**
        - View Details
        - Delete Package
        - Export to CSV/HDF5
        - Reload Data
        """)
