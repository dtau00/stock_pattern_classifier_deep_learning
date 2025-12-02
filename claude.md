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
- **Validation Tests** - Run pre-flight tests and view validation metrics

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

## Validation & Testing

**CRITICAL:** This project has TWO types of validation - preprocessing and model.

### 1. Preprocessing Validation (Available Now)
Validates the data pipeline BEFORE model training.

**Tests in:** `src/validation/preprocessing_tests.py`
**UI Page:** Page 13 - Preprocessing Validation

**What it tests:**
- ✓ Gap detection in OHLCV data
- ✓ Z-score normalization (mean≈0, std≈1)
- ✓ Sliding window segmentation (127 bars, 50% overlap)
- ✓ Train/val/test splitting (70/15/15)
- ✓ HDF5 save/load integrity
- ✓ Full pipeline end-to-end

**Run via Streamlit UI:**
```bash
streamlit run src/ui/app.py
# Navigate to: Page 13 - Preprocessing Validation
```

**Run via CLI:**
```bash
python src/validation/preprocessing_tests.py
```

### 2. Model Architecture Validation (Future - Not Yet Implemented)
Validates the PyTorch model BEFORE training on real data.

**Tests in:** `tests/` (will skip until model is implemented)
**UI Page:** Page 12 - Model Architecture Validation

**What it tests:**
- Test A: Temporal causality - no future leakage
- Test A2: Batch normalization - no cross-sample leakage
- Test D: Synthetic data - model can learn simple patterns (ARI >= 0.95)
- Tests E-H: Architecture tests (projection head, centroids, lambda, L2 norm)

**Run when model is ready:**
```bash
# Run all model pre-flight tests
python -m pytest tests/ -v

# Or run individually
python tests/test_causality.py
python tests/test_batch_norm.py
python tests/test_synthetic_data.py
python tests/test_architecture.py
```

### Validation Modules (For Future Training Integration)
Located in `src/validation/`:
- **Preprocessing Tests:** Validate data pipeline (IMPLEMENTED)
- **Contrastive Metrics:** Track alignment & uniformity during Stage 1 (TODO)
- **Training Visualizer:** Real-time loss and metric plots (TODO)
- **Stability Test:** Multi-run reproducibility (ARI >= 0.85) (TODO)
- **Latent Space Viz:** UMAP/t-SNE visualization (TODO)

### Integration Example
```python
# In training loop - Stage 1
from src.validation.contrastive_metrics import ContrastiveMetricsTracker

tracker = ContrastiveMetricsTracker()
for epoch in range(stage1_epochs):
    # ... training ...

    # Track contrastive learning quality
    with torch.no_grad():
        z1_val, z2_val = get_validation_latents()
        alignment, uniformity = tracker.update(epoch, z1_val, z2_val)
        # Warns automatically if collapse detected
```

**See:** [docs/validation_implementation_status.md](docs/validation_implementation_status.md) for full details.

## Current Status

### ✓ Completed (Preprocessing Pipeline)
- Data fetching from Binance API
- Feature engineering (returns, OBV, NATR)
- Data cleaning and gap detection
- Normalization (z-score per channel)
- Sliding window segmentation (127 bars, 50% overlap)
- Train/val/test splitting with temporal diversity
- Streamlit UI with visualization tools
- **Preprocessing validation tests**

### ⏳ TODO (Model Training)
- PyTorch model architecture (TCN encoder, projection head)
- Contrastive learning Stage 1
- Clustering Stage 2
- Model architecture validation tests
- Training pipeline and checkpointing

## Notes for AI Assistance

- Data is stored in `data/` directory (gitignored)
- HDF5 format preferred for preprocessed windows (compression enabled)
- Windows are 127 bars (prime number for FFT efficiency)
- All features are normalized and validated for independence
- Gap detection is critical - windows overlapping gaps are excluded
- UI is modular - pages are in `src/ui/pages/`, components in `src/ui/components/`
- **Run preprocessing validation BEFORE creating windows** - Page 13 in Streamlit UI
- **Run model validation BEFORE training** - Page 12 (will skip until model exists)

## Development and Code Generation Guidelines

- Only create documentation unless necessary.
- Keep features short and concise, unless specified.  Don't overengineer features.
- Update this claude.md file with important and critial information as we build out this project.
- Stick to SOLID principles
- Follow the same patterns and styles as other pages when possible

## Known Issues & Best Practices

### Windows Console Encoding (cp1252)
**Issue:** Windows console uses cp1252 encoding which doesn't support Unicode characters like ✓, ✗, →, etc.

**Solution:** Use ASCII-safe alternatives in console output:
- Use `[PASS]` and `[FAIL]` instead of ✓ and ✗
- Use `->` instead of → for arrows
- Use `...` instead of … for ellipsis

**Affected modules:**
- Validation tests and CLI output
- Progress indicators
- Any console/terminal output

**Note:** Streamlit UI and web interfaces support full Unicode - only console output needs ASCII fallbacks.
