"""
Temporal Diversity Logging Module.

This module provides logging and monitoring of temporal diversity metrics
during training to ensure batches maintain sufficient temporal spread.
"""

import numpy as np
from typing import List, Dict, Optional
import logging


class TemporalDiversityLogger:
    """
    Logger for monitoring temporal diversity during training.

    Tracks temporal diversity metrics across epochs and provides warnings
    when diversity falls below target thresholds.

    Args:
        dataset_size: Total size of the training dataset
        target_threshold: Target diversity as fraction of dataset (default: 0.3)
        warning_threshold: Warning threshold as fraction of dataset (default: 0.2)
        warning_epochs: Number of consecutive low-diversity epochs before warning (default: 5)

    Example:
        >>> logger = TemporalDiversityLogger(dataset_size=1000)
        >>> logger.log_epoch(epoch=1, batch_stds=[300, 310, 295, ...])
        >>> logger.check_diversity_warnings()
    """

    def __init__(
        self,
        dataset_size: int,
        target_threshold: float = 0.3,
        warning_threshold: float = 0.2,
        warning_epochs: int = 5
    ):
        self.dataset_size = dataset_size
        self.target_threshold = target_threshold
        self.warning_threshold = warning_threshold
        self.warning_epochs = warning_epochs

        # Calculate target values
        self.target_std = target_threshold * dataset_size
        self.warning_std = warning_threshold * dataset_size

        # Tracking metrics
        self.epoch_metrics = []
        self.low_diversity_streak = 0

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def log_epoch(
        self,
        epoch: int,
        batch_stds: List[float],
        verbose: bool = True
    ) -> Dict:
        """
        Log temporal diversity metrics for an epoch.

        Args:
            epoch: Current epoch number
            batch_stds: List of temporal standard deviations for each batch
            verbose: Whether to print verbose output (default: True)

        Returns:
            Dictionary with epoch metrics
        """
        if len(batch_stds) == 0:
            self.logger.warning(f"Epoch {epoch}: No batch diversity metrics available")
            return {}

        # Calculate metrics
        mean_std = float(np.mean(batch_stds))
        min_std = float(np.min(batch_stds))
        max_std = float(np.max(batch_stds))
        median_std = float(np.median(batch_stds))

        # Check if diversity meets targets
        meets_target = mean_std >= self.target_std
        meets_warning = mean_std >= self.warning_std

        # Store metrics
        metrics = {
            'epoch': epoch,
            'mean_std': mean_std,
            'min_std': min_std,
            'max_std': max_std,
            'median_std': median_std,
            'num_batches': len(batch_stds),
            'meets_target': meets_target,
            'meets_warning': meets_warning,
            'target_std': self.target_std,
            'warning_std': self.warning_std
        }

        self.epoch_metrics.append(metrics)

        # Update low diversity streak
        if not meets_warning:
            self.low_diversity_streak += 1
        else:
            self.low_diversity_streak = 0

        # Print metrics if verbose
        if verbose:
            self._print_epoch_metrics(metrics)

        return metrics

    def _print_epoch_metrics(self, metrics: Dict):
        """Print epoch metrics in a formatted way."""
        epoch = metrics['epoch']
        mean_std = metrics['mean_std']
        meets_target = metrics['meets_target']
        meets_warning = metrics['meets_warning']

        # Determine status symbol
        if meets_target:
            status = "[OK]"
        elif meets_warning:
            status = "[WARN]"
        else:
            status = "[LOW]"

        print(f"Epoch {epoch}: {status} Mean Temporal Std = {mean_std:.1f} "
              f"(target > {self.target_std:.1f}, warning > {self.warning_std:.1f})")

        if not meets_target:
            print(f"  Min: {metrics['min_std']:.1f}, Max: {metrics['max_std']:.1f}, "
                  f"Median: {metrics['median_std']:.1f}")

    def check_diversity_warnings(self) -> Optional[str]:
        """
        Check for diversity warnings and return warning message if applicable.

        Returns:
            Warning message string if warning conditions met, None otherwise
        """
        if self.low_diversity_streak >= self.warning_epochs:
            warning_msg = (
                f"WARNING: Low temporal diversity detected for {self.low_diversity_streak} "
                f"consecutive epochs (std < {self.warning_std:.1f}). "
                f"Consider increasing num_temporal_bins or checking batch sampler configuration."
            )
            self.logger.warning(warning_msg)
            return warning_msg

        return None

    def get_summary(self) -> Dict:
        """
        Get summary statistics across all logged epochs.

        Returns:
            Dictionary with summary statistics
        """
        if len(self.epoch_metrics) == 0:
            return {}

        mean_stds = [m['mean_std'] for m in self.epoch_metrics]
        meets_target_pct = sum(m['meets_target'] for m in self.epoch_metrics) / len(self.epoch_metrics) * 100

        summary = {
            'total_epochs': len(self.epoch_metrics),
            'overall_mean_std': float(np.mean(mean_stds)),
            'overall_min_std': float(np.min(mean_stds)),
            'overall_max_std': float(np.max(mean_stds)),
            'meets_target_pct': meets_target_pct,
            'max_low_diversity_streak': self.low_diversity_streak,
            'target_std': self.target_std,
            'warning_std': self.warning_std
        }

        return summary

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()

        if not summary:
            print("No diversity metrics logged yet.")
            return

        print("\n" + "=" * 60)
        print("Temporal Diversity Summary")
        print("=" * 60)
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Overall Mean Std: {summary['overall_mean_std']:.1f}")
        print(f"  Min: {summary['overall_min_std']:.1f}")
        print(f"  Max: {summary['overall_max_std']:.1f}")
        print(f"Epochs Meeting Target: {summary['meets_target_pct']:.1f}%")
        print(f"Max Low Diversity Streak: {summary['max_low_diversity_streak']}")
        print(f"Target Std: {summary['target_std']:.1f}")
        print(f"Warning Std: {summary['warning_std']:.1f}")
        print("=" * 60)

    def reset(self):
        """Reset all metrics."""
        self.epoch_metrics = []
        self.low_diversity_streak = 0


if __name__ == "__main__":
    # Test the temporal diversity logger
    print("=" * 60)
    print("Testing Temporal Diversity Logger")
    print("=" * 60)

    # Test 1: Basic logging
    print("\n--- Test 1: Basic Logging ---")
    logger = TemporalDiversityLogger(dataset_size=1000, target_threshold=0.3, warning_threshold=0.2)

    # Simulate good diversity
    batch_stds = [300, 310, 295, 305, 290, 315, 300, 308, 302, 297]
    metrics = logger.log_epoch(epoch=1, batch_stds=batch_stds)

    assert metrics['meets_target'] == True, "Should meet target"
    assert metrics['meets_warning'] == True, "Should meet warning threshold"
    print("[PASS] Test 1 passed: Basic logging works")

    # Test 2: Low diversity detection
    print("\n--- Test 2: Low Diversity Detection ---")
    # Simulate low diversity
    low_batch_stds = [150, 160, 155, 158, 152, 162, 157, 159, 153, 156]
    metrics = logger.log_epoch(epoch=2, batch_stds=low_batch_stds)

    assert metrics['meets_target'] == False, "Should not meet target"
    assert metrics['meets_warning'] == False, "Should not meet warning threshold"
    assert logger.low_diversity_streak == 1, "Should have 1 epoch streak"
    print("[PASS] Test 2 passed: Low diversity detected")

    # Test 3: Warning after consecutive low diversity epochs
    print("\n--- Test 3: Warning After Consecutive Low Diversity ---")
    for epoch in range(3, 8):
        logger.log_epoch(epoch=epoch, batch_stds=low_batch_stds, verbose=False)

    assert logger.low_diversity_streak >= 5, f"Should have 5+ epoch streak, got {logger.low_diversity_streak}"

    warning = logger.check_diversity_warnings()
    assert warning is not None, "Should generate warning"
    print(f"Warning message: {warning}")
    print("[PASS] Test 3 passed: Warning generated after 5 low epochs")

    # Test 4: Recovery from low diversity
    print("\n--- Test 4: Recovery from Low Diversity ---")
    # Simulate recovery
    logger.log_epoch(epoch=8, batch_stds=batch_stds, verbose=False)
    assert logger.low_diversity_streak == 0, "Streak should reset after good epoch"
    print("[PASS] Test 4 passed: Streak resets after recovery")

    # Test 5: Summary statistics
    print("\n--- Test 5: Summary Statistics ---")
    summary = logger.get_summary()

    print(f"Summary: {summary}")
    assert summary['total_epochs'] == 8, "Should have 8 epochs logged"
    assert 'overall_mean_std' in summary, "Should have mean std"
    print("[PASS] Test 5 passed: Summary statistics work")

    # Test 6: Print summary
    print("\n--- Test 6: Print Summary ---")
    logger.print_summary()
    print("[PASS] Test 6 passed: Summary printed successfully")

    # Test 7: Reset functionality
    print("\n--- Test 7: Reset Functionality ---")
    logger.reset()
    assert len(logger.epoch_metrics) == 0, "Metrics should be cleared"
    assert logger.low_diversity_streak == 0, "Streak should be reset"
    print("[PASS] Test 7 passed: Reset works correctly")

    # Test 8: Edge case - empty batch stds
    print("\n--- Test 8: Edge Case - Empty Batch Stds ---")
    metrics = logger.log_epoch(epoch=1, batch_stds=[], verbose=False)
    assert metrics == {}, "Should return empty dict for empty input"
    print("[PASS] Test 8 passed: Empty input handled correctly")

    # Test 9: Custom thresholds
    print("\n--- Test 9: Custom Thresholds ---")
    custom_logger = TemporalDiversityLogger(
        dataset_size=1000,
        target_threshold=0.4,
        warning_threshold=0.25,
        warning_epochs=3
    )

    assert custom_logger.target_std == 400, "Custom target should be 400"
    assert custom_logger.warning_std == 250, "Custom warning should be 250"
    print("[PASS] Test 9 passed: Custom thresholds work")

    # Test 10: Integration with batch sampler
    print("\n--- Test 10: Integration Test ---")
    from batch_sampler import StratifiedTemporalBatchSampler
    from torch.utils.data import TensorDataset
    import torch

    # Create dataset and sampler
    data = torch.randn(1000, 127, 3)
    dataset = TensorDataset(data)
    sampler = StratifiedTemporalBatchSampler(dataset, batch_size=100, num_temporal_bins=10)

    # Create logger
    diversity_logger = TemporalDiversityLogger(dataset_size=1000)

    # Simulate one epoch
    for batch_indices in sampler:
        pass  # Just iterate through

    # Log metrics
    batch_stds = sampler.epoch_temporal_stds
    metrics = diversity_logger.log_epoch(epoch=1, batch_stds=batch_stds, verbose=False)

    print(f"Integration test metrics: mean_std={metrics['mean_std']:.1f}")
    assert 'mean_std' in metrics, "Should have metrics from sampler"
    print("[PASS] Test 10 passed: Integration with batch sampler works")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
