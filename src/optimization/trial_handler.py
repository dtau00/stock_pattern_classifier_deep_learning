"""
Trial Handler

Handles OOM errors, early stopping, and trial monitoring for Optuna optimization.
"""

import logging
from typing import Dict, List, Optional
import torch


class TrialHandler:
    """Handles OOM errors and early stopping for optimization trials."""

    def __init__(
        self,
        enable_early_stop: bool = True,
        early_stop_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize trial handler.

        Args:
            enable_early_stop: Enable early stopping for poor trials
            early_stop_thresholds: Custom thresholds for early stopping
                                  Format: {'metric_name': threshold_value}
        """
        self.enable_early_stop = enable_early_stop
        self.oom_trials: List[int] = []
        self.pruned_trials: List[int] = []

        # Default early stopping thresholds
        self.thresholds = {
            'silhouette': 0.1,  # Min acceptable silhouette score
            'davies_bouldin': 5.0,  # Max acceptable Davies-Bouldin index
            'val_loss': 2.0,  # Max acceptable validation loss
        }

        # Update with custom thresholds if provided
        if early_stop_thresholds:
            self.thresholds.update(early_stop_thresholds)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def handle_oom(self, trial_number: int) -> None:
        """
        Handle OOM error during trial.

        Args:
            trial_number: Trial number that encountered OOM
        """
        self.oom_trials.append(trial_number)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.logger.warning(
            f"Trial {trial_number}: OOM occurred. "
            f"GPU cache cleared. Trial will be pruned."
        )

    def should_prune(
        self,
        trial_number: int,
        metric_value: float,
        metric_name: str
    ) -> bool:
        """
        Determine if trial should be pruned early.

        Args:
            trial_number: Current trial number
            metric_value: Current metric value
            metric_name: Name of metric ('silhouette', 'davies_bouldin', 'val_loss')

        Returns:
            True if trial should be pruned, False otherwise
        """
        if not self.enable_early_stop:
            return False

        should_prune = False
        reason = ""

        # Check thresholds based on metric type
        if metric_name == 'silhouette':
            if metric_value < self.thresholds['silhouette']:
                should_prune = True
                reason = f"Silhouette score {metric_value:.4f} < {self.thresholds['silhouette']}"

        elif metric_name == 'davies_bouldin':
            if metric_value > self.thresholds['davies_bouldin']:
                should_prune = True
                reason = f"Davies-Bouldin index {metric_value:.4f} > {self.thresholds['davies_bouldin']}"

        elif metric_name == 'val_loss':
            if metric_value > self.thresholds['val_loss']:
                should_prune = True
                reason = f"Validation loss {metric_value:.4f} > {self.thresholds['val_loss']}"

        if should_prune:
            self.pruned_trials.append(trial_number)
            self.logger.info(
                f"Trial {trial_number}: Early stopping triggered. Reason: {reason}"
            )

        return should_prune

    def log_trial_start(self, trial_number: int, params: Dict) -> None:
        """
        Log trial start with parameters.

        Args:
            trial_number: Trial number
            params: Trial parameters
        """
        self.logger.info(f"Trial {trial_number}: Starting with params: {params}")

    def log_trial_complete(
        self,
        trial_number: int,
        metric_value: float,
        metric_name: str
    ) -> None:
        """
        Log trial completion.

        Args:
            trial_number: Trial number
            metric_value: Final metric value
            metric_name: Metric name
        """
        self.logger.info(
            f"Trial {trial_number}: COMPLETE - "
            f"{metric_name}={metric_value:.4f}"
        )

    def log_trial_pruned(self, trial_number: int, reason: str = "") -> None:
        """
        Log trial pruning.

        Args:
            trial_number: Trial number
            reason: Reason for pruning
        """
        msg = f"Trial {trial_number}: PRUNED"
        if reason:
            msg += f" - {reason}"
        self.logger.info(msg)

    def log_trial_failed(self, trial_number: int, error: str) -> None:
        """
        Log trial failure.

        Args:
            trial_number: Trial number
            error: Error message
        """
        self.logger.error(f"Trial {trial_number}: FAILED - {error}")

    def get_summary(self) -> Dict:
        """
        Get summary of trial handling.

        Returns:
            Dictionary with trial statistics
        """
        return {
            'oom_count': len(self.oom_trials),
            'pruned_count': len(self.pruned_trials),
            'oom_trials': self.oom_trials,
            'pruned_trials': self.pruned_trials,
            'thresholds': self.thresholds,
        }

    def print_summary(self) -> None:
        """Print formatted summary of trial handling."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("Trial Handler Summary")
        print("=" * 50)
        print(f"  OOM Errors: {summary['oom_count']}")
        print(f"  Pruned Trials: {summary['pruned_count']}")

        if summary['oom_trials']:
            print(f"  OOM Trial Numbers: {summary['oom_trials']}")

        if summary['pruned_trials']:
            print(f"  Pruned Trial Numbers: {summary['pruned_trials']}")

        print("\nEarly Stopping Thresholds:")
        for metric, threshold in summary['thresholds'].items():
            print(f"  {metric}: {threshold}")

        print("=" * 50)

    def reset(self) -> None:
        """Reset trial tracking."""
        self.oom_trials.clear()
        self.pruned_trials.clear()
        self.logger.info("Trial handler reset")


# Test function
def test_trial_handler():
    """Test trial handler functionality."""
    print("Testing Trial Handler...")

    # Test 1: Initialization
    print("\n[Test 1] Initialize handler")
    handler = TrialHandler(enable_early_stop=True)
    assert handler.enable_early_stop
    assert len(handler.oom_trials) == 0
    assert len(handler.pruned_trials) == 0
    print("  [PASS] Handler initialized")

    # Test 2: OOM handling
    print("\n[Test 2] Handle OOM")
    handler.handle_oom(1)
    assert 1 in handler.oom_trials
    print("  [PASS] OOM handled")

    # Test 3: Early stopping - silhouette
    print("\n[Test 3] Early stopping - silhouette")
    should_prune = handler.should_prune(2, 0.05, 'silhouette')
    assert should_prune
    assert 2 in handler.pruned_trials
    print("  [PASS] Low silhouette score triggers pruning")

    # Test 4: Early stopping - Davies-Bouldin
    print("\n[Test 4] Early stopping - Davies-Bouldin")
    should_prune = handler.should_prune(3, 6.0, 'davies_bouldin')
    assert should_prune
    assert 3 in handler.pruned_trials
    print("  [PASS] High Davies-Bouldin triggers pruning")

    # Test 5: No pruning for good metrics
    print("\n[Test 5] No pruning for good metrics")
    should_prune = handler.should_prune(4, 0.8, 'silhouette')
    assert not should_prune
    assert 4 not in handler.pruned_trials
    print("  [PASS] Good silhouette score doesn't trigger pruning")

    # Test 6: Summary
    print("\n[Test 6] Get summary")
    summary = handler.get_summary()
    assert summary['oom_count'] == 1
    assert summary['pruned_count'] == 2
    print("  [PASS] Summary generated")
    handler.print_summary()

    # Test 7: Custom thresholds
    print("\n[Test 7] Custom thresholds")
    custom_handler = TrialHandler(
        enable_early_stop=True,
        early_stop_thresholds={'silhouette': 0.3}
    )
    assert custom_handler.thresholds['silhouette'] == 0.3
    print("  [PASS] Custom thresholds applied")

    # Test 8: Disable early stopping
    print("\n[Test 8] Disable early stopping")
    no_stop_handler = TrialHandler(enable_early_stop=False)
    should_prune = no_stop_handler.should_prune(5, 0.01, 'silhouette')
    assert not should_prune
    print("  [PASS] Early stopping disabled")

    # Test 9: Reset
    print("\n[Test 9] Reset handler")
    handler.reset()
    assert len(handler.oom_trials) == 0
    assert len(handler.pruned_trials) == 0
    print("  [PASS] Handler reset")

    # Test 10: Logging
    print("\n[Test 10] Logging functions")
    handler.log_trial_start(10, {'batch_size': 64, 'lr': 0.001})
    handler.log_trial_complete(10, 0.75, 'silhouette')
    handler.log_trial_pruned(11, "Poor performance")
    handler.log_trial_failed(12, "Configuration error")
    print("  [PASS] All logging functions work")

    print("\n[SUCCESS] All trial handler tests passed!")


if __name__ == '__main__':
    test_trial_handler()
