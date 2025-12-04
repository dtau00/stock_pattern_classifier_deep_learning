"""
Hyperparameter Optimizer

Core Optuna-based optimizer for hyperparameter search.
"""

import optuna
from optuna.samplers import GridSampler, TPESampler
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Callable
import logging


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna.

    Supports both grid search and Bayesian optimization (TPE) strategies.
    """

    def __init__(
        self,
        data_path: str,
        param_space: Dict[str, Any],
        strategy: str = 'bayesian',
        metric: str = 'silhouette',
        n_trials: int = 50,
        stage: str = 'combined',
        subset_fraction: float = 0.3,
        stage1_checkpoint: Optional[str] = None,
        study_name: Optional[str] = None
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            data_path: Path to HDF5 preprocessed data
            param_space: Dict of parameters and their search ranges
            strategy: 'grid' or 'bayesian'
            metric: Target metric ('silhouette', 'davies_bouldin', 'val_loss')
            n_trials: Number of trials (for Bayesian)
            stage: Training stage ('stage1', 'stage2', 'combined')
            subset_fraction: Fraction of data to use (0.0-1.0)
            stage1_checkpoint: Path to Stage 1 checkpoint (for stage2)
            study_name: Optional name for the study
        """
        self.data_path = data_path
        self.param_space = param_space
        self.strategy = strategy
        self.metric = metric
        self.n_trials = n_trials
        self.stage = stage
        self.subset_fraction = subset_fraction
        self.stage1_checkpoint = stage1_checkpoint
        self.study_name = study_name or f"hpo_{stage}_{metric}"

        self.study = None
        self.logger = logging.getLogger(__name__)

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate initialization parameters."""
        assert self.strategy in ['grid', 'bayesian'], \
            f"strategy must be 'grid' or 'bayesian', got {self.strategy}"

        assert self.metric in ['silhouette', 'davies_bouldin', 'val_loss'], \
            f"metric must be 'silhouette', 'davies_bouldin', or 'val_loss', got {self.metric}"

        assert self.stage in ['stage1', 'stage2', 'combined'], \
            f"stage must be 'stage1', 'stage2', or 'combined', got {self.stage}"

        assert 0.0 < self.subset_fraction <= 1.0, \
            f"subset_fraction must be in (0.0, 1.0], got {self.subset_fraction}"

        if self.stage == 'stage2' and not self.stage1_checkpoint:
            raise ValueError("stage1_checkpoint required for stage2 optimization")

    def create_study(self) -> optuna.Study:
        """
        Create Optuna study with appropriate sampler.

        Returns:
            Optuna study object
        """
        # Determine optimization direction
        if self.metric == 'silhouette':
            direction = 'maximize'
        else:  # davies_bouldin or val_loss
            direction = 'minimize'

        # Create sampler
        if self.strategy == 'grid':
            # Grid search requires all parameters to be categorical
            search_space = {}
            for param_name, values in self.param_space.items():
                if not isinstance(values, list):
                    raise ValueError(f"Grid search requires list of values for '{param_name}'")
                search_space[param_name] = values

            sampler = GridSampler(search_space)
            self.logger.info(f"Created GridSampler with {len(search_space)} parameters")
        else:
            # Bayesian optimization with TPE
            sampler = TPESampler(seed=42)
            self.logger.info(f"Created TPESampler for Bayesian optimization")

        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=sampler
        )

        self.logger.info(
            f"Created study '{self.study_name}' - "
            f"Direction: {direction}, Strategy: {self.strategy}"
        )

        return self.study

    def optimize(
        self,
        objective_fn: Callable,
        callback: Optional[Callable] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            objective_fn: Objective function to optimize
            callback: Optional callback function(trial_number, trial_value, best_value)
            n_jobs: Number of parallel jobs (default: 1, -1 for all CPUs)

        Returns:
            Dictionary with best parameters and metrics
        """
        if self.study is None:
            self.create_study()

        # Determine number of trials
        if self.strategy == 'grid':
            # Grid search runs all combinations
            n_trials = None
        else:
            n_trials = self.n_trials

        self.logger.info(f"Starting optimization with {n_trials or 'all'} trials")

        # Define callback wrapper
        def optuna_callback(study, trial):
            if callback is not None:
                callback(
                    trial.number,
                    trial.value if trial.value is not None else None,
                    study.best_value if study.best_trial else None
                )

        # Run optimization
        try:
            self.study.optimize(
                objective_fn,
                n_trials=n_trials,
                callbacks=[optuna_callback] if callback else None,
                n_jobs=n_jobs,
                show_progress_bar=False  # We'll handle progress in UI
            )
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")

        # Get results
        best_trial = self.study.best_trial
        results = {
            'best_params': best_trial.params,
            'best_value': best_trial.value,
            'best_trial_number': best_trial.number,
            'n_trials': len(self.study.trials),
            'study': self.study
        }

        self.logger.info(
            f"Optimization complete - Best {self.metric}: {best_trial.value:.4f} "
            f"(Trial {best_trial.number})"
        )

        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Convert trials to pandas DataFrame.

        Returns:
            DataFrame with trial results
        """
        if self.study is None:
            return pd.DataFrame()

        # Get trials dataframe from Optuna
        df = self.study.trials_dataframe()

        # Rename columns for clarity
        df = df.rename(columns={
            'value': self.metric,
            'state': 'status'
        })

        # Extract parameter columns
        param_cols = [col for col in df.columns if col.startswith('params_')]
        other_cols = [col for col in df.columns if not col.startswith('params_')]

        # Reorder: number, status, metric, then parameters
        ordered_cols = ['number', 'status', self.metric] + param_cols
        ordered_cols = [col for col in ordered_cols if col in df.columns]

        df = df[ordered_cols]

        # Sort by metric value
        if self.metric == 'silhouette':
            df = df.sort_values(self.metric, ascending=False)
        else:
            df = df.sort_values(self.metric, ascending=True)

        return df

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best trial parameters.

        Returns:
            Dictionary of best parameters
        """
        if self.study is None or self.study.best_trial is None:
            return {}

        return self.study.best_trial.params

    def plot_optimization_history(self) -> go.Figure:
        """
        Create plotly figure of optimization history.

        Returns:
            Plotly figure
        """
        if self.study is None:
            return go.Figure()

        # Get completed trials
        trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not trials:
            return go.Figure()

        trial_numbers = [t.number for t in trials]
        trial_values = [t.value for t in trials]

        # Calculate running best
        running_best = []
        current_best = trial_values[0]
        for val in trial_values:
            if self.metric == 'silhouette':
                current_best = max(current_best, val)
            else:
                current_best = min(current_best, val)
            running_best.append(current_best)

        # Create figure
        fig = go.Figure()

        # Add trial values
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=trial_values,
            mode='markers',
            name='Trial Value',
            marker=dict(size=8, color='lightblue')
        ))

        # Add running best
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=running_best,
            mode='lines',
            name='Best Value',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=f'Optimization History - {self.metric}',
            xaxis_title='Trial Number',
            yaxis_title=self.metric.replace('_', ' ').title(),
            hovermode='x unified'
        )

        return fig

    def plot_param_importance(self, n_params: int = 10) -> go.Figure:
        """
        Create plotly figure of parameter importance.

        Args:
            n_params: Number of top parameters to show

        Returns:
            Plotly figure
        """
        if self.study is None:
            return go.Figure()

        try:
            # Calculate parameter importance
            importance = optuna.importance.get_param_importances(self.study)

            # Get top N parameters
            params = list(importance.keys())[:n_params]
            values = [importance[p] for p in params]

            # Create figure
            fig = go.Figure(go.Bar(
                x=values,
                y=params,
                orientation='h',
                marker=dict(color='steelblue')
            ))

            fig.update_layout(
                title=f'Parameter Importance (Top {len(params)})',
                xaxis_title='Importance',
                yaxis_title='Parameter',
                height=max(400, len(params) * 40)
            )

            return fig

        except Exception as e:
            self.logger.warning(f"Could not compute parameter importance: {e}")
            return go.Figure()

    def plot_parallel_coordinate(self) -> go.Figure:
        """
        Create parallel coordinate plot for parameter relationships.

        Returns:
            Plotly figure
        """
        if self.study is None:
            return go.Figure()

        try:
            fig = optuna.visualization.plot_parallel_coordinate(
                self.study,
                params=list(self.param_space.keys())[:10]  # Limit to 10 params
            )
            return fig
        except Exception as e:
            self.logger.warning(f"Could not create parallel coordinate plot: {e}")
            return go.Figure()

    def save_study(self, filepath: str):
        """
        Save study to file using pickle.

        Args:
            filepath: Path to save file
        """
        import pickle

        if self.study is None:
            raise ValueError("No study to save")

        with open(filepath, 'wb') as f:
            pickle.dump(self.study, f)

        self.logger.info(f"Study saved to {filepath}")

    def load_study(self, filepath: str):
        """
        Load study from file.

        Args:
            filepath: Path to study file
        """
        import pickle

        with open(filepath, 'rb') as f:
            self.study = pickle.load(f)

        self.logger.info(f"Study loaded from {filepath}")


# Test function
def test_optimizer():
    """Test optimizer functionality."""
    print("Testing Hyperparameter Optimizer...")

    # Test 1: Initialization
    print("\n[Test 1] Initialize optimizer")
    param_space = {
        'batch_size': [64, 128, 256],
        'learning_rate': [1e-4, 1e-3, 1e-2]
    }

    optimizer = HyperparameterOptimizer(
        data_path='dummy.h5',
        param_space=param_space,
        strategy='bayesian',
        metric='silhouette',
        n_trials=10
    )
    assert optimizer.strategy == 'bayesian'
    assert optimizer.metric == 'silhouette'
    print("  [PASS] Optimizer initialized")

    # Test 2: Create study
    print("\n[Test 2] Create study")
    study = optimizer.create_study()
    assert study is not None
    assert optimizer.study is not None
    print(f"  [PASS] Study created: {optimizer.study_name}")

    # Test 3: Mock optimization (simple objective)
    print("\n[Test 3] Mock optimization")

    def mock_objective(trial):
        x = trial.suggest_categorical('batch_size', [64, 128, 256])
        y = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2])
        # Simple quadratic function
        return (x - 128)**2 + (y - 1e-3)**2

    results = optimizer.optimize(mock_objective, n_jobs=1)
    assert 'best_params' in results
    assert 'best_value' in results
    print(f"  [PASS] Optimization complete - Best params: {results['best_params']}")

    # Test 4: Get results dataframe
    print("\n[Test 4] Results dataframe")
    df = optimizer.get_results_dataframe()
    assert len(df) > 0
    print(f"  [PASS] DataFrame created with {len(df)} trials")

    # Test 5: Get best params
    print("\n[Test 5] Get best parameters")
    best_params = optimizer.get_best_params()
    assert 'batch_size' in best_params
    assert 'learning_rate' in best_params
    print(f"  [PASS] Best params: {best_params}")

    # Test 6: Plot optimization history
    print("\n[Test 6] Plot optimization history")
    fig = optimizer.plot_optimization_history()
    assert fig is not None
    print("  [PASS] Optimization history plot created")

    # Test 7: Plot parameter importance
    print("\n[Test 7] Plot parameter importance")
    fig = optimizer.plot_param_importance()
    assert fig is not None
    print("  [PASS] Parameter importance plot created")

    print("\n[SUCCESS] All optimizer tests passed!")


if __name__ == '__main__':
    test_optimizer()
