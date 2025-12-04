"""
Hyperparameter Optimization Page

This page provides automated hyperparameter tuning using Optuna for both
Stage 1 (contrastive learning) and Stage 2 (DEC clustering) training.

Features:
- Grid search and Bayesian optimization (TPE)
- Parameter space configuration
- Save/load configurations
- Real-time progress monitoring
- Results visualization
- Parameter importance analysis
"""

import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import h5py
from datetime import datetime
import time

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.optimization import (
    HyperparameterOptimizer,
    TrialHandler,
    save_hpo_config,
    load_hpo_config,
    list_saved_configs,
    get_default_param_space,
    get_param_space_summary,
    validate_param_space
)
from src.optimization.objective import (
    stage1_objective,
    stage2_objective,
    combined_objective
)


def main():
    st.title("⚙️ Hyperparameter Optimization (HPO)")
    st.markdown("Automated hyperparameter tuning using Optuna")

    # Sidebar for navigation
    with st.sidebar:
        st.header("HPO Steps")
        step = st.radio(
            "Select Step:",
            [
                "1. Select Data",
                "2. Configure Parameters",
                "3. Run Optimization",
                "4. View Results"
            ],
            index=0
        )

    # Step 1: Select Data
    if step == "1. Select Data":
        show_data_selection()

    # Step 2: Configure Parameters
    elif step == "2. Configure Parameters":
        show_parameter_configuration()

    # Step 3: Run Optimization
    elif step == "3. Run Optimization":
        show_optimization()

    # Step 4: View Results
    elif step == "4. View Results":
        show_results()


def show_data_selection():
    """Step 1: Select preprocessed data file."""
    st.header("Step 1: Select Preprocessed Data")
    st.markdown("Choose HDF5 file with preprocessed windows for optimization.")

    # File selection
    data_dir = Path("data/preprocessed")
    if not data_dir.exists():
        st.warning("No preprocessed data directory found.")
        st.info("📁 Expected directory: `data/preprocessed/`")
        return

    # Get available HDF5 files
    hdf5_files = list(data_dir.glob("*.h5"))

    if not hdf5_files:
        st.warning("No HDF5 files found in data/preprocessed/")
        st.info("💡 Run the preprocessing pipeline first (Page 20)")
        return

    # File selector
    file_names = [f.name for f in hdf5_files]
    selected_file = st.selectbox(
        "Select preprocessed data file:",
        file_names
    )

    if selected_file:
        file_path = data_dir / selected_file

        # Load and display data info
        try:
            with h5py.File(file_path, 'r') as f:
                train_shape = f['train'].shape
                val_shape = f['val'].shape
                test_shape = f['test'].shape

                st.success(f"✓ Loaded: {selected_file}")

                # Display data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train Windows", f"{train_shape[0]:,}")
                with col2:
                    st.metric("Val Windows", f"{val_shape[0]:,}")
                with col3:
                    st.metric("Test Windows", f"{test_shape[0]:,}")

                st.info(
                    f"📊 Shape: {train_shape[1]} channels x "
                    f"{train_shape[2]} timesteps"
                )

                # Save to session state
                st.session_state.hpo_data_path = str(file_path)
                st.session_state.hpo_data_info = {
                    'train_size': train_shape[0],
                    'val_size': val_shape[0],
                    'test_size': test_shape[0],
                    'n_channels': train_shape[1],
                    'seq_length': train_shape[2]
                }

        except Exception as e:
            st.error(f"Error loading file: {e}")


def show_parameter_configuration():
    """Step 2: Configure hyperparameter search space."""
    st.header("Step 2: Configure Parameters")

    # Check if data is selected
    if 'hpo_data_path' not in st.session_state:
        st.warning("⚠ Please select data first (Step 1)")
        return

    # Stage selection
    stage = st.radio(
        "Select training stage:",
        ["Stage 1 (Contrastive)", "Stage 2 (Clustering)", "Combined (Both Stages)"],
        horizontal=True
    )

    stage_key = {
        "Stage 1 (Contrastive)": "stage1",
        "Stage 2 (Clustering)": "stage2",
        "Combined (Both Stages)": "combined"
    }[stage]

    st.session_state.hpo_stage = stage_key

    # Load or create parameter space
    st.subheader("Parameter Space")

    # Configuration management
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        config_action = st.radio(
            "Configuration:",
            ["Use Default", "Load Saved", "Create Custom"],
            horizontal=True
        )

    # Default parameters
    if config_action == "Use Default":
        param_space = get_default_param_space(stage_key)
        st.info(f"Using default parameter space for {stage_key}")

    # Load saved configuration
    elif config_action == "Load Saved":
        saved_configs = list_saved_configs()

        if not saved_configs:
            st.warning("No saved configurations found")
            param_space = get_default_param_space(stage_key)
        else:
            selected_config = st.selectbox("Select configuration:", saved_configs)

            if selected_config:
                config_data = load_hpo_config(selected_config)
                param_space = config_data['param_space']
                st.success(f"✓ Loaded configuration: {selected_config}")
                st.json(config_data)

    # Create custom configuration
    else:
        st.markdown("**Customize parameter ranges:**")

        # Start with default
        param_space = get_default_param_space(stage_key)

        # Allow editing each parameter
        edited_param_space = {}

        for param_name, default_values in param_space.items():
            with st.expander(f"📌 {param_name}"):
                # Show current values
                st.write(f"Current values: {default_values}")

                # Allow editing
                values_str = st.text_input(
                    f"Values (comma-separated):",
                    value=",".join(map(str, default_values)),
                    key=f"param_{param_name}"
                )

                try:
                    # Parse values
                    if '.' in values_str or 'e' in values_str.lower():
                        # Float values
                        values = [float(v.strip()) for v in values_str.split(',')]
                    else:
                        # Int values
                        values = [int(v.strip()) for v in values_str.split(',')]

                    edited_param_space[param_name] = values
                except Exception as e:
                    st.error(f"Invalid values: {e}")
                    edited_param_space[param_name] = default_values

        param_space = edited_param_space

    # Validate parameter space
    is_valid, error_msg = validate_param_space(param_space)
    if not is_valid:
        st.error(f"❌ Invalid parameter space: {error_msg}")
        return

    # Display parameter space summary
    st.subheader("Parameter Space Summary")
    summary = get_param_space_summary(param_space)
    st.code(summary)

    # Save configuration
    st.subheader("Save Configuration")

    col1, col2 = st.columns([3, 1])

    with col1:
        config_name = st.text_input(
            "Configuration name:",
            value=f"hpo_{stage_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("💾 Save Config", use_container_width=True):
            if config_name:
                try:
                    save_hpo_config(
                        name=config_name,
                        param_space=param_space,
                        strategy='bayesian',  # Default
                        metric='silhouette',  # Default
                        n_trials=50,
                        stage=stage_key
                    )
                    st.success(f"✓ Configuration saved: {config_name}")
                except Exception as e:
                    st.error(f"Error saving config: {e}")

    # Store in session state
    st.session_state.hpo_param_space = param_space


def show_optimization():
    """Step 3: Run hyperparameter optimization."""
    st.header("Step 3: Run Optimization")

    # Check prerequisites
    if 'hpo_data_path' not in st.session_state:
        st.warning("⚠ Please select data first (Step 1)")
        return

    if 'hpo_param_space' not in st.session_state:
        st.warning("⚠ Please configure parameters first (Step 2)")
        return

    # Optimization settings
    st.subheader("Optimization Strategy")

    col1, col2 = st.columns(2)

    with col1:
        strategy = st.radio(
            "Strategy:",
            ["Bayesian (TPE)", "Grid Search"],
            help="Bayesian is recommended for large parameter spaces"
        )

        strategy_key = "bayesian" if strategy == "Bayesian (TPE)" else "grid"

    with col2:
        metric = st.selectbox(
            "Target Metric:",
            ["Silhouette Score", "Davies-Bouldin Index", "Validation Loss"],
            help="Metric to optimize"
        )

        metric_key = {
            "Silhouette Score": "silhouette",
            "Davies-Bouldin Index": "davies_bouldin",
            "Validation Loss": "val_loss"
        }[metric]

    # Number of trials (for Bayesian)
    if strategy_key == "bayesian":
        n_trials = st.slider(
            "Number of trials:",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
    else:
        # Calculate total combinations for grid search
        param_space = st.session_state.hpo_param_space
        total_combinations = 1
        for values in param_space.values():
            total_combinations *= len(values)

        st.info(f"📊 Grid search will run {total_combinations:,} trials (all combinations)")
        n_trials = None

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            subset_fraction = st.slider(
                "Data subset fraction:",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Fraction of data to use (lower = faster trials)"
            )

            enable_early_stop = st.checkbox(
                "Enable early stopping",
                value=True,
                help="Prune poor-performing trials early"
            )

        with col2:
            if st.session_state.hpo_stage == "stage1":
                max_epochs = st.number_input(
                    "Max epochs (Stage 1):",
                    min_value=10,
                    max_value=100,
                    value=30
                )
            elif st.session_state.hpo_stage == "stage2":
                max_epochs = st.number_input(
                    "Max epochs (Stage 2):",
                    min_value=10,
                    max_value=100,
                    value=30
                )
            else:
                max_epochs_s1 = st.number_input(
                    "Max epochs (Stage 1):",
                    min_value=10,
                    max_value=100,
                    value=30
                )
                max_epochs_s2 = st.number_input(
                    "Max epochs (Stage 2):",
                    min_value=10,
                    max_value=100,
                    value=30
                )

    # Start optimization button
    st.subheader("Execute")

    if st.button("▶ Start Optimization", type="primary", use_container_width=True):
        run_optimization(
            strategy=strategy_key,
            metric=metric_key,
            n_trials=n_trials or total_combinations,
            subset_fraction=subset_fraction,
            enable_early_stop=enable_early_stop
        )


def run_optimization(
    strategy: str,
    metric: str,
    n_trials: int,
    subset_fraction: float,
    enable_early_stop: bool
):
    """Execute the optimization process."""

    # Create trial handler
    trial_handler = TrialHandler(enable_early_stop=enable_early_stop)

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        data_path=st.session_state.hpo_data_path,
        param_space=st.session_state.hpo_param_space,
        strategy=strategy,
        metric=metric,
        n_trials=n_trials,
        stage=st.session_state.hpo_stage,
        subset_fraction=subset_fraction
    )

    # Create study
    optimizer.create_study()

    # Prepare objective function
    stage = st.session_state.hpo_stage

    def objective_wrapper(trial):
        if stage == "stage1":
            return stage1_objective(
                trial,
                st.session_state.hpo_data_path,
                st.session_state.hpo_param_space,
                metric,
                trial_handler,
                subset_fraction
            )
        elif stage == "stage2":
            # Need Stage 1 checkpoint
            if 'stage1_checkpoint' not in st.session_state:
                st.error("Stage 2 requires a Stage 1 checkpoint. Please train Stage 1 first.")
                raise ValueError("Missing Stage 1 checkpoint")

            return stage2_objective(
                trial,
                st.session_state.hpo_data_path,
                st.session_state.hpo_param_space,
                metric,
                trial_handler,
                st.session_state.stage1_checkpoint,
                subset_fraction
            )
        else:  # combined
            return combined_objective(
                trial,
                st.session_state.hpo_data_path,
                st.session_state.hpo_param_space,
                metric,
                trial_handler,
                subset_fraction
            )

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()

    logs = []

    def callback(trial_number, trial_value, best_value):
        """Callback for progress updates."""
        progress = (trial_number + 1) / n_trials
        progress_bar.progress(progress)

        if trial_value is not None:
            log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Trial {trial_number}: {metric}={trial_value:.4f}"
            if best_value is not None:
                log_msg += f" | Best: {best_value:.4f}"
        else:
            log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Trial {trial_number}: FAILED"

        logs.append(log_msg)

        # Update status
        status_text.markdown(f"**Progress:** {trial_number + 1}/{n_trials} trials")

        # Update log
        log_container.code("\n".join(logs[-20:]))  # Show last 20 logs

    # Run optimization
    with st.spinner("Running optimization..."):
        try:
            results = optimizer.optimize(objective_wrapper, callback=callback)

            # Show results
            st.success("✓ Optimization complete!")

            st.subheader("Best Parameters")
            st.json(results['best_params'])

            st.metric(
                f"Best {metric}",
                f"{results['best_value']:.4f}",
                help=f"Achieved in trial {results['best_trial_number']}"
            )

            # Save results to session state
            st.session_state.hpo_optimizer = optimizer
            st.session_state.hpo_results = results

            # Show trial handler summary
            st.subheader("Optimization Summary")
            summary = trial_handler.get_summary()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trials", results['n_trials'])
            with col2:
                st.metric("Pruned Trials", summary['pruned_count'])
            with col3:
                st.metric("OOM Errors", summary['oom_count'])

        except Exception as e:
            st.error(f"Optimization failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def show_results():
    """Step 4: View optimization results."""
    st.header("Step 4: View Results")

    if 'hpo_optimizer' not in st.session_state:
        st.warning("⚠ No optimization results available. Run optimization first (Step 3)")
        return

    optimizer = st.session_state.hpo_optimizer
    results = st.session_state.hpo_results

    # Best parameters
    st.subheader("🏆 Best Parameters")
    st.json(results['best_params'])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best Value", f"{results['best_value']:.4f}")
    with col2:
        st.metric("Best Trial", results['best_trial_number'])

    # All trials table
    st.subheader("📊 All Trials")
    df = optimizer.get_results_dataframe()
    st.dataframe(df, use_container_width=True)

    # Download results
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇ Download Results (CSV)",
        data=csv,
        file_name=f"hpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # Visualization
    st.subheader("📈 Visualization")

    tab1, tab2, tab3 = st.tabs([
        "Optimization History",
        "Parameter Importance",
        "Parallel Coordinate"
    ])

    with tab1:
        st.plotly_chart(
            optimizer.plot_optimization_history(),
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            optimizer.plot_param_importance(),
            use_container_width=True
        )

    with tab3:
        try:
            st.plotly_chart(
                optimizer.plot_parallel_coordinate(),
                use_container_width=True
            )
        except Exception as e:
            st.info("Parallel coordinate plot not available")

    # Copy best config
    st.subheader("📋 Copy Best Configuration")
    st.markdown("Use these parameters for full training:")

    config_code = f"""
# Best HPO Configuration
# Trial: {results['best_trial_number']}
# {optimizer.metric}: {results['best_value']:.4f}

best_params = {json.dumps(results['best_params'], indent=4)}
"""

    st.code(config_code, language="python")


if __name__ == "__main__":
    main()
