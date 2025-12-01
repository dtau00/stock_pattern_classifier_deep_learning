# Data Manager UI

Streamlit-based GUI for managing training data.

## Status
🚧 **In Development** - Phase 0 Implementation

## Completed
- ✅ Step 0.1: Directory structure created
- ✅ Step 0.1: Requirements updated

## Next Steps
- ⏳ Step 0.2: Create main app with sidebar navigation
- ⏳ Step 0.3: Create page templates
- ⏳ Step 0.4: Test navigation and app launch

## Launch (After Implementation)
```bash
# From project root:
./run_data_manager.sh

# Or directly:
streamlit run src/ui/app.py
```

## Navigation
- **Home**: Overview and quick start
- **OHLCV Manager**: Download and manage OHLCV data from Binance
  - Download OHLCV tab: Configure and download historical data
  - Manage Data Packages tab: View and manage downloaded packages
- **Validate & Preview**: Check data quality
- **Visualize Data**: Explore normalized features
- **TA Verification**: Verify technical indicators
