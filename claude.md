# Stock Pattern Classifier (Deep Learning)

PyTorch-based deep learning system for stock pattern classification using contrastive learning.

## Project Structure

```
src/
├── data/              # Binance API client and data fetching
├── features/          # Feature engineering (returns, OBV, NATR)
├── preprocessing/     # Cleaning, normalization, segmentation
├── models/            # PyTorch model architectures
├── training/          # Two-stage training pipeline
├── clustering/        # DEC clustering for Stage 2
├── validation/        # Testing and metrics
└── ui/                # Streamlit interface
    ├── app.py         # Main entry point
    └── pages/         # UI pages (OHLCV, training, inference)

data/                  # Raw/processed data (gitignored)
models/                # Model checkpoints (gitignored)
```

## Key Technical Specs

**Data Pipeline:**
- 3 feature channels: Returns (log), Volume (OBV), Volatility (NATR)
- Sliding windows: 127 bars, 50% overlap (step=63)
- Z-score normalization per channel
- Gap detection excludes invalid windows
- HDF5 storage with compression

**Model Architecture:**
- Stage 1: Contrastive learning (NT-Xent loss)
  - Auto-evaluates quality metrics after training (variance, effective rank, k-NN, etc.)
- Stage 2: DEC clustering + contrastive
- Encoders: CNN, TCN, or Hybrid options
- Output: Cluster assignments for pattern recognition

**Running:**
```bash
streamlit run src/ui/app.py
```

## Development Guidelines

- Keep code concise - no over-engineering
- Follow SOLID principles
- Use existing UI patterns from `src/ui/pages/`
- Don't create docs unless prompted
- ASCII-only for console output (Windows cp1252): use `[PASS]`/`[FAIL]` not ✓/✗

## Critical Notes

- Windows are 127 bars (prime for FFT)
- Gap detection is critical - must exclude overlapping windows
- HDF5 format required for preprocessed data
- Run validation before training (Page 13 in UI)
- use the virtual env in: .\venv

## Training Quality Metrics

**Stage 1 evaluation (automatic after training):**
- Embedding Variance: >0.25 (detects feature collapse)
- Effective Rank: >60% of d_z (dimensionality usage)
- Alignment: <0.5 (positive pairs close)
- Uniformity: <-1.5 (embeddings spread on hypersphere)
- k-NN Accuracy: >0.75 (augmentation invariance)

**Batch size matters:** Contrastive learning quality depends on number of negative samples
- Batch 8: Only 7 negatives (poor quality, artificially low loss)
- Batch 128-256: 127-255 negatives (good quality)
- Loss values NOT comparable across batch sizes (higher batch = higher loss is normal)
