"""
Visualize Data Page
Explore normalized preprocessed data
"""
import streamlit as st

def show():
    """Visualize normalized preprocessed data"""
    st.title("📊 Visualize Data")
    st.markdown("---")

    st.info("**Section Purpose:** Explore normalized feature channels in interactive charts")

    # Placeholder content
    st.markdown("""
    ### Features to be implemented:
    - Multi-pane interactive chart (Plotly)
    - Three channels: Returns, Volume/Liquidity, Volatility/Risk
    - Zoom, pan, hover tooltips
    - Time range selector
    - Full dataset or selected window display
    - Export chart as image
    """)

    st.warning("⚠️ This section will be implemented in Phase 6")

    # Preview of what the UI will look like
    with st.expander("📋 UI Preview"):
        st.markdown("""
        **Interactive Multi-Pane Chart:**
        ```
        Pane 1: Returns Channel
        - Log Returns (normalized)
        - Centered around 0
        - Stationary signal
        
        Pane 2: Volume/Liquidity Channel
        - First-order differenced OBV
        - EMA smoothed (period: 20)
        - Normalized
        
        Pane 3: Volatility/Risk Channel
        - NATR (Normalized ATR)
        - Period: 14
        - Regime-agnostic measure
        
        Controls:
        - Zoom: Click and drag
        - Pan: Shift + Click and drag
        - Reset: Double click
        - Hover: Show exact values
        - Range slider: Select time window
        ```
        """)
