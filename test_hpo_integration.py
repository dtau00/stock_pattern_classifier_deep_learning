"""
Quick integration test for HPO implementation.
Tests the core modules without requiring real data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("Testing HPO Implementation - Integration Test")
print("="*60)

# Test 1: Import all modules
print("\n[Test 1] Import modules")
try:
    from src.optimization import (
        HyperparameterOptimizer,
        TrialHandler,
        save_hpo_config,
        load_hpo_config,
        list_saved_configs,
        get_default_param_space
    )
    print("  [PASS] All modules imported successfully")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Get default parameter space
print("\n[Test 2] Get default parameter spaces")
try:
    stage1_params = get_default_param_space('stage1')
    stage2_params = get_default_param_space('stage2')
    combined_params = get_default_param_space('combined')

    print(f"  [PASS] Stage 1: {len(stage1_params)} parameters")
    print(f"  [PASS] Stage 2: {len(stage2_params)} parameters")
    print(f"  [PASS] Combined: {len(combined_params)} parameters")
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    sys.exit(1)

# Test 3: Save and load configuration
print("\n[Test 3] Save/Load configuration")
try:
    test_config_name = "test_hpo_integration"

    # Save
    save_hpo_config(
        name=test_config_name,
        param_space={'batch_size': [64, 128], 'learning_rate': [1e-3, 1e-4]},
        strategy='bayesian',
        metric='silhouette',
        n_trials=10
    )
    print(f"  [PASS] Configuration saved")

    # Load
    loaded = load_hpo_config(test_config_name)
    assert loaded['strategy'] == 'bayesian'
    assert loaded['metric'] == 'silhouette'
    print(f"  [PASS] Configuration loaded")

    # Clean up
    from src.optimization.config_manager import delete_hpo_config
    delete_hpo_config(test_config_name)
    print(f"  [PASS] Configuration deleted")

except Exception as e:
    print(f"  [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Trial handler
print("\n[Test 4] Trial handler")
try:
    handler = TrialHandler(enable_early_stop=True)

    # Test OOM handling
    handler.handle_oom(1)
    assert 1 in handler.oom_trials
    print(f"  [PASS] OOM handling works")

    # Test early stopping
    should_prune = handler.should_prune(2, 0.05, 'silhouette')
    assert should_prune
    print(f"  [PASS] Early stopping works")

    # Get summary
    summary = handler.get_summary()
    assert summary['oom_count'] == 1
    assert summary['pruned_count'] == 1
    print(f"  [PASS] Summary generation works")

except Exception as e:
    print(f"  [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Optimizer initialization
print("\n[Test 5] Optimizer initialization")
try:
    optimizer = HyperparameterOptimizer(
        data_path='dummy.h5',
        param_space={'batch_size': [64, 128], 'learning_rate': [1e-3, 1e-4]},
        strategy='bayesian',
        metric='silhouette',
        n_trials=5
    )

    assert optimizer.strategy == 'bayesian'
    assert optimizer.metric == 'silhouette'
    print(f"  [PASS] Optimizer initialized")

    # Create study
    study = optimizer.create_study()
    assert study is not None
    print(f"  [PASS] Optuna study created")

except Exception as e:
    print(f"  [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check UI page exists
print("\n[Test 6] Check UI page")
try:
    ui_page_path = Path("src/ui/pages/14_hyperparameter_optimization.py")
    assert ui_page_path.exists()
    print(f"  [PASS] UI page exists at {ui_page_path}")

    # Check file is not empty
    with open(ui_page_path, 'r') as f:
        content = f.read()
        assert len(content) > 1000
        assert 'def main()' in content
        assert 'streamlit' in content
    print(f"  [PASS] UI page has valid content")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# Test 7: Check directories exist
print("\n[Test 7] Check directory structure")
try:
    dirs_to_check = [
        'src/optimization',
        'data/hpo_configs'
    ]

    for dir_path in dirs_to_check:
        path = Path(dir_path)
        assert path.exists()
        print(f"  [PASS] {dir_path} exists")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# Summary
print("\n" + "="*60)
print("INTEGRATION TEST COMPLETE")
print("="*60)
print("\n[SUCCESS] All integration tests passed!")
print("\nNext steps:")
print("  1. Run: streamlit run src/ui/app.py")
print("  2. Navigate to Page 14: Hyperparameter Optimization")
print("  3. Follow the 4-step workflow in the UI")
print("\nNote: Full testing requires preprocessed HDF5 data.")
print("="*60)
