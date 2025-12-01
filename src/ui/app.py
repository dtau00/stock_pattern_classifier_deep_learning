"""
Stock Pattern Classifier - Data Manager UI
Main Streamlit Application
"""

import streamlit as st
from pages import ohlcv_manager
from pages import visualize_data, ta_verification

# Page configuration
st.set_page_config(
    page_title="Stock Pattern Classifier - Data Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stRadio > label {
        font-weight: 600;
    }
    /* Make sidebar navigation more prominent */
    .css-1d391kg {
        padding-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📈 Data Manager")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 OHLCV Manager",
        "📈 Visualize Data",
        "🔧 TA Verification"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("**Stock Pattern Classifier**\nDeep Learning-Based Pattern Recognition")
st.sidebar.markdown("**Phase:** Data Pipeline & Preprocessing")

# Main content area - route to selected page
if page == "🏠 Home":
    st.title("Stock Pattern Classifier - Data Manager")
    st.markdown("---")

    st.markdown("""
    ## Welcome to the Data Manager

    This tool helps you manage historical market data for training the
    stock pattern classifier model.

    ### Navigation

    Use the **Navigation** radio buttons in the sidebar (left panel) to switch between pages:

    1. **📊 OHLCV Manager** - Download and validate OHLCV data from Binance
       - **Download OHLCV Tab:** Configure and download historical data
         - Select symbol (e.g., BTCUSDT, ETHUSDT)
         - Choose timeframe (1m, 5m, 15m, 1h, 4h, 1d)
         - Specify date range
         - View estimated bar count
       - **Validate & Preview Tab:** Check data quality and preview contents
         - View data quality metrics and scores
         - Detect gaps and missing bars
         - Preview OHLCV data in multiple views
         - Interactive candlestick and volume charts

    2. **📈 Visualize Data** - Explore normalized features
       - Interactive multi-pane charts
       - View Returns, Volume, Volatility channels
       - Zoom, pan, and hover for details

    3. **🔧 TA Verification** - Verify technical indicators
       - Candlestick charts with overlays
       - Verify SMA, Bollinger Bands, RSI, MACD
       - Compare against raw price data

    ---

    ### System Information
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Framework", "PyTorch")

    with col2:
        st.metric("Target GPU", "RTX 5060 Ti 16GB")

    with col3:
        st.metric("Data Source", "Binance API")

    st.markdown("---")
    st.success("✅ Phase 0: Streamlit GUI Foundation - Complete")
    st.info("📋 Next: Implement data download functionality (Phase 1)")

    with st.expander("📖 Implementation Progress"):
        st.markdown("""
        **Completed:**
        - ✅ Step 0.1: Project structure and dependencies
        - ✅ Step 0.2: Main app with sidebar navigation
        - ⏳ Step 0.3: Page templates (in progress)
        - ⏳ Step 0.4: Navigation testing

        **Upcoming:**
        - Phase 1: Data sourcing and retrieval
        - Phase 2: Data Manager UI components
        - Phase 3: Feature engineering
        - Phase 4: Preprocessing pipeline
        - Phase 5: Data splitting
        - Phase 6: Visualization tools
        """)

elif page == "📊 OHLCV Manager":
    ohlcv_manager.show()

elif page == "📈 Visualize Data":
    visualize_data.show()

elif page == "🔧 TA Verification":
    ta_verification.show()
