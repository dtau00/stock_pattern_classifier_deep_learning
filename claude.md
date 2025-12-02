# Stock Pattern Classifier (Deep Learning)

Deep learning system for classifying stock price patterns using PyTorch.

## Project Overview

**Goal:** Train a deep learning model to identify chart patterns in financial markets and predict movements based on historical price data.

**Data Source:** Binance API for cryptocurrency OHLCV data
**Framework:** PyTorch
**Target Hardware:** RTX 980 Ti 6GB (later change to RTX 5060 Ti 16GB)
**UI:** Streamlit-based data manager

## Architecture

### Data Pipeline
1. **Data Download** - Fetch OHLCV data from Binance API ([src/data/](src/data/))
2. **Feature Engineering** - Three core channels ([src/features/](src/features/)):
   - Returns: Log returns `log(P_t / P_{t-1})`
   - Volume/Liquidity: OBV differenced + EMA smoothed
   - Volatility/Risk: Normalized ATR (NATR)
3. **Preprocessing** ([src/preprocessing/](src/preprocessing/)):
   - Data cleaning and gap detection
   - Normalization (z-score per channel)
   - Sliding window segmentation (127-bar windows, 50% overlap)
4. **Data Splitting** - Train/val/test split with diversity sampling
5. **Model Training** - (TODO: not yet implemented)

### File Structure
```
src/
├── data/              # Binance client, data fetching, metadata
├── features/          # Feature engineering (returns, OBV, NATR)
├── preprocessing/     # Cleaning, normalization, segmentation, splitting
├── visualization/     # Data visualization and TradingView charts
└── ui/                # Streamlit app and pages
    ├── app.py         # Main app with navigation
    ├── pages/         # OHLCV manager, visualizations, TA verification
    └── components/    # Reusable UI components

data/                  # Raw and processed datasets (gitignored)
models/                # Model checkpoints (gitignored)
```

## Key Technical Details

### Feature Engineering ([src/features/feature_engineering.py](src/features/feature_engineering.py))
- **Returns Channel:** Stationary log returns for price movements
- **Volume/Liquidity:** OBV → diff → EMA(20) for trend-following volume signal
- **Volatility/Risk:** NATR (ATR/Close) for regime-agnostic volatility

Features are validated for independence (correlation < 0.8 warning, < 0.9 critical).

### Preprocessing Pipeline
1. **Data Cleaning** ([src/preprocessing/data_cleaning.py](src/preprocessing/data_cleaning.py))
   - Gap detection and exclusion flagging
   - Removes windows overlapping with gaps

2. **Normalization** ([src/preprocessing/normalization.py](src/preprocessing/normalization.py))
   - Z-score normalization per channel
   - Tracks statistics for inverse transform

3. **Segmentation** ([src/preprocessing/segmentation.py](src/preprocessing/segmentation.py))
   - Sliding windows: 127 bars, 50% overlap (step=63)
   - Excludes windows with gaps or NaN
   - Output shape: `(num_windows, 127, 3)` for 3 channels
   - Saves to HDF5 with compression

4. **Data Splitting** ([src/preprocessing/data_splitting.py](src/preprocessing/data_splitting.py))
   - Train/val/test split with diversity sampling
   - Custom batch sampler for balanced training

### Streamlit UI ([src/ui/app.py](src/ui/app.py))
- **OHLCV Manager** - Download and validate Binance data
- **Feature Visualization** - View raw OHLCV, normalized inputs, window inspection
- **TA Verification** - Verify technical indicator calculations

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit UI
streamlit run src/ui/app.py

# Or use the shell script
./run_data_manager.sh
```

## Key Dependencies

- `torch>=2.0.0` - Deep learning framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `streamlit>=1.29.0` - UI framework
- `python-binance>=1.0.19` - Binance API client
- `h5py>=3.10.0` - HDF5 file format for preprocessed data
- `matplotlib`, `plotly` - Visualization

## Notes for AI Assistance

- Data is stored in `data/` directory (gitignored)
- HDF5 format preferred for preprocessed windows (compression enabled)
- Windows are 127 bars (prime number for FFT efficiency)
- All features are normalized and validated for independence
- Gap detection is critical - windows overlapping gaps are excluded
- UI is modular - pages are in `src/ui/pages/`, components in `src/ui/components/`

## Development and Code Generation Guidelines

- Only create documentation unless necessary.  
- Keep features short and concise, unless specified.  Don't overengineer features.
- Update this claude.md file with important and critial information as we build out this project.
- Stick to SOLID principles
- Follow the same patterns and styles as other pages when possible
