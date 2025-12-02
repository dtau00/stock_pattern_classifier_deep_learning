"""
Quick test to verify Streamlit UI components can be imported.
Run this before launching the full Streamlit app.
"""
import sys
import io
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("Testing Streamlit UI Components")
print("=" * 60)

# Test 1: Import Streamlit
print("\n[1/6] Testing Streamlit import...")
try:
    import streamlit as st
    print(f"  ✓ Streamlit {st.__version__} imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import Streamlit: {e}")
    sys.exit(1)

# Test 2: Import Plotly
print("\n[2/6] Testing Plotly import...")
try:
    import plotly
    print(f"  ✓ Plotly {plotly.__version__} imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import Plotly: {e}")
    sys.exit(1)

# Test 3: Import data modules
print("\n[3/6] Testing data modules...")
try:
    from data import OHLCVDataFetcher, MetadataManager, estimate_bar_count
    print("  ✓ Data modules imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import data modules: {e}")
    sys.exit(1)

# Test 4: Import UI pages
print("\n[4/6] Testing UI pages import...")
try:
    sys.path.insert(0, str(Path(__file__).parent / 'src' / 'ui'))
    from pages import configure_download, validate_preview
    print("  ✓ UI pages imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import UI pages: {e}")
    sys.exit(1)

# Test 5: Check data directories
print("\n[5/6] Checking data directories...")
data_dirs = [
    'data/packages',
    'data/metadata',
    'data/normalization_stats',
    'data/preprocessed'
]

for dir_path in data_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"  ✓ {dir_path} exists")
    else:
        print(f"  ! {dir_path} does not exist (will be created on first use)")

# Test 6: Test metadata manager
print("\n[6/6] Testing MetadataManager...")
try:
    manager = MetadataManager()
    count = manager.get_package_count()
    print(f"  ✓ MetadataManager initialized")
    print(f"  ✓ Found {count} existing package(s)")
except Exception as e:
    print(f"  ✗ MetadataManager test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print("\nYou can now launch the Streamlit app with:")
print("  streamlit run src/ui/app.py")
print("\nOr use the browser that automatically opens at:")
print("  http://localhost:8501")
print("=" * 60)
