# Setup Instructions

## Prerequisites
- Python 3.12.3 (verified)
- pip (Python package manager)

## Installation Steps

### 1. Install pip (if not already installed)
```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install python3-pip

# Verify installation
python3 -m pip --version
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Note: TA-Lib may require system dependencies
# For Ubuntu/Debian:
# sudo apt-get install build-essential
# wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# tar -xzf ta-lib-0.4.0-src.tar.gz
# cd ta-lib/
# ./configure --prefix=/usr
# make
# sudo make install
# pip install TA-Lib
```

### 4. Verify Streamlit Installation
```bash
streamlit --version
# Expected output: Streamlit, version 1.29.0+
```

### 5. Test Streamlit
```bash
# Run Streamlit hello app (built-in demo)
streamlit hello
```

## Directory Structure
The following directories have been created:

```
stock_pattern_classifier_deep_learning/
├── src/
│   ├── ui/
│   │   ├── app.py                    # Main Streamlit app (to be created)
│   │   ├── pages/                    # Page modules
│   │   │   └── __init__.py
│   │   └── components/               # Reusable UI components
│   │       └── __init__.py
│   ├── data/                         # Data fetching modules
│   ├── features/                     # Feature engineering
│   ├── preprocessing/                # Preprocessing pipeline
│   └── visualization/                # Plotting utilities
├── data/
│   ├── packages/                     # Downloaded OHLCV data
│   ├── metadata/                     # Package metadata JSON
│   ├── normalization_stats/          # Saved μ and σ
│   └── preprocessed/                 # Preprocessed windows
└── config/
    └── preprocessing_config.yaml     # Configuration file (to be created)
```

## Next Steps
After successful installation:
1. Run `streamlit hello` to verify Streamlit works
2. Proceed to Phase 0, Step 0.2: Create Main App with Sidebar Navigation

## Troubleshooting

### pip not found
If `pip` is not installed, install it using:
```bash
sudo apt install python3-pip
```

### Permission errors during installation
Use `--user` flag:
```bash
pip install --user -r requirements.txt
```

Or use a virtual environment (recommended).

### TA-Lib installation issues
If TA-Lib installation fails, use pandas-ta as an alternative:
```bash
# Already included in requirements.txt
pip install pandas-ta
```
