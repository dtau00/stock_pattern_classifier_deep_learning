"""
Model Architecture Validation Dashboard

IMPORTANT: This page tests the MODEL ARCHITECTURE (encoder, projection head, etc.)
These tests will SKIP until the model is implemented.

For PREPROCESSING validation (data cleaning, normalization, etc.), use:
- Page 13: Preprocessing Validation

This page provides:
- One-click test runner for all model pre-flight tests
- Real-time test results with pass/fail status
- Detailed error messages and troubleshooting guidance
- Test history tracking
"""

import streamlit as st
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(page_title="Validation Tests", page_icon="✓", layout="wide")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = PROJECT_ROOT / "data" / "validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Title
st.title("🧪 Model Architecture Validation")
st.markdown("""
Run critical pre-flight tests to ensure **model architecture** correctness before training.

**NOTE:** These tests check the **PyTorch model** (encoder, projection head, etc.)
They will **skip** until the model is implemented.

**For preprocessing validation** (data cleaning, normalization, segmentation), see:
→ **Page 13: Preprocessing Validation**

**IMPORTANT:** All model tests must pass before training on real data.
""")

# Test definitions
TESTS = {
    "Test A: Temporal Causality": {
        "file": "test_causality.py",
        "description": "Verifies encoder does not leak future information to past timesteps",
        "critical": True,
        "category": "Architecture"
    },
    "Test A2: Batch Normalization": {
        "file": "test_batch_norm.py",
        "description": "Verifies no cross-sample information leakage through batch statistics",
        "critical": True,
        "category": "Architecture"
    },
    "Test D: Synthetic Data": {
        "file": "test_synthetic_data.py",
        "description": "Verifies model can learn to separate simple patterns (ARI >= 0.95)",
        "critical": True,
        "category": "Learning Capability"
    },
    "Test E-H: Architecture": {
        "file": "test_architecture.py",
        "description": "Tests projection head, centroids, lambda schedule, L2 normalization",
        "critical": True,
        "category": "Architecture"
    }
}


def run_test(test_file: str) -> dict:
    """
    Run a single test file and return results.

    Args:
        test_file: Name of test file (e.g., 'test_causality.py')

    Returns:
        dict with keys: success, output, error, duration
    """
    test_path = TESTS_DIR / test_file
    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT)
        )

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": "Test timed out after 60 seconds",
            "duration": 60.0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "duration": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }


def save_test_results(results: dict):
    """Save test results to JSON file."""
    results_file = RESULTS_DIR / "test_results.json"

    # Load existing history
    history = []
    if results_file.exists():
        with open(results_file, 'r') as f:
            history = json.load(f)

    # Add new results
    history.append({
        "timestamp": datetime.now().isoformat(),
        "results": results
    })

    # Keep last 50 runs
    history = history[-50:]

    # Save
    with open(results_file, 'w') as f:
        json.dump(history, f, indent=2)


def load_test_history():
    """Load test history from JSON file."""
    results_file = RESULTS_DIR / "test_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return []


# Main UI
st.markdown("---")

# Test runner section
st.header("🚀 Run Tests")

col1, col2 = st.columns([1, 3])

with col1:
    run_all = st.button("▶️ Run All Tests", type="primary", use_container_width=True)

with col2:
    st.info("Run all critical pre-flight tests before training. Tests will skip if model components are not yet implemented.")

if run_all:
    st.markdown("---")
    st.subheader("Test Execution")

    results = {}
    all_passed = True

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (test_name, test_info) in enumerate(TESTS.items()):
        status_text.text(f"Running {test_name}...")

        # Run test
        result = run_test(test_info["file"])
        results[test_name] = result

        if not result["success"]:
            all_passed = False

        # Update progress
        progress_bar.progress((idx + 1) / len(TESTS))

    status_text.text("All tests completed!")
    progress_bar.empty()

    # Save results
    save_test_results(results)

    # Display results
    st.markdown("---")
    st.subheader("Test Results")

    if all_passed:
        st.success("✓ All tests passed! Model architecture is ready for training.")
    else:
        st.error("✗ Some tests failed. Please review errors below before training.")

    # Display each test result
    for test_name, test_info in TESTS.items():
        result = results[test_name]

        with st.expander(
            f"{'✓' if result['success'] else '✗'} {test_name} "
            f"({'PASS' if result['success'] else 'FAIL'}) - {result['duration']:.2f}s",
            expanded=not result['success']
        ):
            # Test description
            st.markdown(f"**Description:** {test_info['description']}")
            st.markdown(f"**Category:** {test_info['category']}")

            # Output
            if result['output']:
                st.markdown("**Output:**")
                st.code(result['output'], language="text")

            # Error
            if result['error']:
                st.markdown("**Error:**")
                st.code(result['error'], language="text")

            # Troubleshooting
            if not result['success']:
                if "not yet implemented" in result['error'].lower() or "skiptest" in result['output'].lower():
                    st.info("This test was skipped because model components are not yet implemented. This is expected during development.")
                else:
                    st.warning("This test failed. Review the error message above for troubleshooting guidance.")


# Test history section
st.markdown("---")
st.header("📊 Test History")

history = load_test_history()

if history:
    # Convert to DataFrame for display
    history_data = []
    for run in history[-10:]:  # Last 10 runs
        timestamp = run['timestamp']
        passed = sum(1 for r in run['results'].values() if r['success'])
        total = len(run['results'])

        history_data.append({
            "Timestamp": timestamp,
            "Passed": passed,
            "Total": total,
            "Status": "✓ All Pass" if passed == total else f"✗ {total - passed} Failed"
        })

    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Show detailed results for last run
    if st.checkbox("Show details for last run"):
        last_run = history[-1]
        st.json(last_run)
else:
    st.info("No test history yet. Run tests to see history.")


# Documentation section
st.markdown("---")
st.header("📚 Test Documentation")

with st.expander("What are pre-flight tests?"):
    st.markdown("""
Pre-flight tests are critical validation checks that must pass before training begins. They verify:

1. **Architectural Correctness:** Encoder causality, normalization, routing
2. **Learning Capability:** Model can learn simple synthetic patterns
3. **Implementation Details:** Lambda schedule, L2 normalization, centroid updates

**Why are they critical?**
- Catch bugs early before expensive training
- Ensure reproducibility and stability
- Verify mathematical correctness
    """)

with st.expander("Test Descriptions"):
    for test_name, test_info in TESTS.items():
        st.markdown(f"**{test_name}**")
        st.markdown(f"- {test_info['description']}")
        st.markdown(f"- Category: {test_info['category']}")
        st.markdown("")

with st.expander("What if tests fail?"):
    st.markdown("""
**Test Skipped (Expected during development):**
- If you see "not yet implemented" or "skipTest", this is normal
- Tests will automatically pass once model components are implemented

**Test Failed (Action Required):**
- Review the error message for troubleshooting guidance
- Each test provides specific recommendations for fixing issues
- Common issues:
  - Temporal causality: Check causal padding in Conv1d layers
  - Batch normalization: Replace BatchNorm with LayerNorm/GroupNorm
  - Synthetic data: Check learning rate, loss function, augmentation
  - Architecture: Verify normalization, routing, and schedules

**Need Help?**
- See [docs/validation_implementation_status.md](../../docs/validation_implementation_status.md)
- See [docs/validation_implementation.md](../../docs/validation_implementation.md)
    """)

with st.expander("Integration with Training"):
    st.markdown("""
### How to use validation during training:

```python
# Before training - run all pre-flight tests
# Use this page or run from command line:
python -m pytest tests/ -v

# During Stage 1 training - track contrastive metrics
from src.validation.contrastive_metrics import ContrastiveMetricsTracker

tracker = ContrastiveMetricsTracker()
for epoch in range(stage1_epochs):
    # ... training code ...

    # Track alignment & uniformity
    with torch.no_grad():
        z1_val, z2_val = get_validation_latents()
        alignment, uniformity = tracker.update(epoch, z1_val, z2_val)
        # Automatic collapse detection warnings
```

### After training:
- Run stability test (multiple seeds, ARI >= 0.85)
- Visualize latent space (UMAP/t-SNE)
- Check cluster interpretability

See CLAUDE.md for full integration examples.
    """)


# Footer
st.markdown("---")
st.caption("Validation tests ensure model correctness before training. Always run before training on real data!")
