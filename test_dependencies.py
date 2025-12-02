"""
Test script to verify all Phase 4 dependencies are installed and working.
"""

import sys

def test_imports():
    """Test all required imports for Phase 4."""
    results = []

    # Test pandas
    try:
        import pandas as pd
        results.append(f"[OK] pandas: {pd.__version__}")
    except ImportError as e:
        results.append(f"[FAIL] pandas: {e}")

    # Test numpy
    try:
        import numpy as np
        results.append(f"[OK] numpy: {np.__version__}")
    except ImportError as e:
        results.append(f"[FAIL] numpy: {e}")

    # Test plotly
    try:
        import plotly
        results.append(f"[OK] plotly: {plotly.__version__}")
    except ImportError as e:
        results.append(f"[FAIL] plotly: {e}")

    # Test PyTorch
    try:
        import torch
        results.append(f"[OK] torch: {torch.__version__}")
        results.append(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            results.append(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        results.append(f"[FAIL] torch: {e}")

    # Test h5py
    try:
        import h5py
        results.append(f"[OK] h5py: {h5py.__version__}")
    except ImportError as e:
        results.append(f"[FAIL] h5py: {e}")

    # Test scipy (for statistical functions)
    try:
        import scipy
        results.append(f"[OK] scipy: {scipy.__version__}")
    except ImportError as e:
        results.append(f"[FAIL] scipy: {e}")

    # Test yaml (for config loading)
    try:
        import yaml
        results.append(f"[OK] pyyaml: {yaml.__version__}")
    except ImportError as e:
        results.append(f"[FAIL] pyyaml: {e}")

    return results

def test_basic_operations():
    """Test basic operations with the libraries."""
    import pandas as pd
    import numpy as np

    results = []

    # Test numpy array creation
    try:
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        results.append("[OK] NumPy array operations work")
    except Exception as e:
        results.append(f"[FAIL] NumPy operations failed: {e}")

    # Test pandas DataFrame
    try:
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        assert df['a'].mean() == 2.0
        results.append("[OK] Pandas DataFrame operations work")
    except Exception as e:
        results.append(f"[FAIL] Pandas operations failed: {e}")

    # Test h5py basic operation
    try:
        import h5py
        import tempfile
        import os

        # Create a temporary HDF5 file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            tmp_path = tmp.name

        # Write test data
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('test', data=np.array([1, 2, 3]))

        # Read test data
        with h5py.File(tmp_path, 'r') as f:
            data = f['test'][:]
            assert np.array_equal(data, np.array([1, 2, 3]))

        # Clean up
        os.unlink(tmp_path)
        results.append("[OK] HDF5 read/write operations work")
    except Exception as e:
        results.append(f"[FAIL] HDF5 operations failed: {e}")

    return results

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Dependency Check")
    print("=" * 60)
    print()

    print("Testing imports...")
    print("-" * 60)
    for result in test_imports():
        print(result)
    print()

    print("Testing basic operations...")
    print("-" * 60)
    for result in test_basic_operations():
        print(result)
    print()

    print("=" * 60)
    print("Dependency check complete!")
    print("=" * 60)
