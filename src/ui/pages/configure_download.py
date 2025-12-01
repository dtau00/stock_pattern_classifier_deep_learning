"""
Configure Download Page
Download OHLCV data from Binance
"""
import streamlit as st

def show():
    """Configure and download OHLCV data packages"""
    st.title("⬇️ Configure Download")
    st.markdown("---")

    st.info("**Section Purpose:** Configure parameters and download historical OHLCV data from Binance")

    # Placeholder content
    st.markdown("""
    ### Features to be implemented:
    - Symbol selection (e.g., BTCUSDT, ETHUSDT)
    - Timeframe selection (1m, 5m, 15m, 1h, 4h, 1d)
    - Date range picker
    - Estimated bar count calculator
    - Download progress tracking
    - Data validation after download
    """)

    st.warning("⚠️ This section will be implemented in Phase 1 & Phase 2")

    # Preview of what the UI will look like
    with st.expander("📋 UI Preview"):
        st.markdown("""
        **Download Configuration Panel:**
        ```
        Symbol: [Dropdown: BTCUSDT, ETHUSDT, ...]
        Timeframe: [Dropdown: 1m, 5m, 15m, 1h, 4h, 1d]
        Start Date: [Date Picker]
        End Date: [Date Picker]
        
        Estimated Bars: 8,760 bars
        Estimated Size: ~2.5 MB
        
        [Download Button]
        ```
        """)
