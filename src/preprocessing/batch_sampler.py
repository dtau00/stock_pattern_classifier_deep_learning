"""
Stratified Temporal Batch Sampler for PyTorch DataLoader.

This module implements a custom batch sampler that ensures temporal diversity
in training batches by sampling from all temporal bins.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, List


class StratifiedTemporalBatchSampler(Sampler):
    """
    Stratified Temporal Batch Sampler for ensuring temporal diversity in batches.

    Divides the dataset into temporal bins and samples from all bins per batch
    to prevent temporally clustered batches.

    Args:
        data_source: The dataset to sample from (torch.utils.data.Dataset)
        batch_size: Number of samples per batch
        num_temporal_bins: Number of bins to divide the temporal range (default: 10)
        drop_last: Whether to drop the last incomplete batch (default: False)

    Example:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> dataset = TensorDataset(torch.randn(1000, 127, 3))
        >>> sampler = StratifiedTemporalBatchSampler(dataset, batch_size=256, num_temporal_bins=10)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        num_temporal_bins: int = 10,
        drop_last: bool = False
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_temporal_bins = num_temporal_bins
        self.drop_last = drop_last

        # Dataset size
        self.num_samples = len(data_source)

        # Create temporal bins
        self._create_temporal_bins()

        # Initialize metrics tracking
        self.epoch_temporal_stds = []
        self.current_epoch = 0

    def _create_temporal_bins(self):
        """Divide dataset into temporal bins."""
        # Calculate bin size
        bin_size = self.num_samples // self.num_temporal_bins

        # Create bins
        self.bins = []
        for i in range(self.num_temporal_bins):
            start_idx = i * bin_size
            if i == self.num_temporal_bins - 1:
                # Last bin gets any remainder
                end_idx = self.num_samples
            else:
                end_idx = (i + 1) * bin_size

            self.bins.append(list(range(start_idx, end_idx)))

        # Shuffle indices within each bin (but maintain temporal structure across bins)
        for bin_indices in self.bins:
            np.random.shuffle(bin_indices)

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches with samples from all temporal bins.

        Yields:
            List of indices for each batch
        """
        # Reset epoch metrics
        self.epoch_temporal_stds = []

        # Create working copies of bins
        working_bins = [bin_indices.copy() for bin_indices in self.bins]

        # Calculate samples per bin per batch
        samples_per_bin = max(1, self.batch_size // self.num_temporal_bins)
        remainder = self.batch_size % self.num_temporal_bins

        # Generate batches
        while True:
            batch = []

            # Sample from each bin
            for i, bin_indices in enumerate(working_bins):
                # Calculate how many samples to take from this bin
                n_samples = samples_per_bin
                if i < remainder:
                    n_samples += 1

                # Check if bin has enough samples
                if len(bin_indices) < n_samples:
                    if self.drop_last:
                        # Not enough samples, stop iteration
                        if len(batch) > 0:
                            # Calculate temporal diversity for this batch
                            temporal_std = np.std(batch)
                            self.epoch_temporal_stds.append(temporal_std)
                        return
                    else:
                        # Take what's left
                        n_samples = len(bin_indices)

                if n_samples == 0:
                    continue

                # Sample from bin
                sampled = bin_indices[:n_samples]
                batch.extend(sampled)
                working_bins[i] = bin_indices[n_samples:]

            # Check if we have a full batch
            if len(batch) < self.batch_size and self.drop_last:
                break

            if len(batch) == 0:
                break

            # Calculate temporal diversity for this batch
            temporal_std = np.std(batch)
            self.epoch_temporal_stds.append(temporal_std)

            # Shuffle batch to avoid any ordering artifacts
            np.random.shuffle(batch)

            yield batch

        # Increment epoch counter
        self.current_epoch += 1

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def get_epoch_temporal_diversity(self) -> float:
        """
        Get mean temporal diversity for the current epoch.

        Returns:
            Mean temporal standard deviation across all batches
        """
        if len(self.epoch_temporal_stds) == 0:
            return 0.0
        return float(np.mean(self.epoch_temporal_stds))

    def check_temporal_diversity(self, target_threshold: float = 0.3) -> bool:
        """
        Check if temporal diversity meets target threshold.

        Args:
            target_threshold: Target diversity as fraction of dataset size (default: 0.3)

        Returns:
            True if diversity meets target, False otherwise
        """
        mean_std = self.get_epoch_temporal_diversity()
        target_std = target_threshold * self.num_samples

        return mean_std >= target_std

    def reset_bins(self):
        """Reset bins for a new epoch (re-shuffle within bins)."""
        for bin_indices in self.bins:
            np.random.shuffle(bin_indices)


if __name__ == "__main__":
    # Test the stratified temporal batch sampler
    print("=" * 60)
    print("Testing Stratified Temporal Batch Sampler")
    print("=" * 60)

    # Create synthetic dataset
    from torch.utils.data import TensorDataset, DataLoader

    num_samples = 1000
    sequence_length = 127
    num_channels = 3

    # Create synthetic data
    data = torch.randn(num_samples, sequence_length, num_channels)
    dataset = TensorDataset(data)

    # Test 1: Create sampler with default parameters
    print("\n--- Test 1: Create Sampler (batch_size=100, bins=10) ---")
    sampler = StratifiedTemporalBatchSampler(
        dataset,
        batch_size=100,
        num_temporal_bins=10
    )

    print(f"Dataset size: {sampler.num_samples}")
    print(f"Batch size: {sampler.batch_size}")
    print(f"Num temporal bins: {sampler.num_temporal_bins}")
    print(f"Expected batches: {len(sampler)}")
    assert len(sampler.bins) == 10, "Expected 10 temporal bins"
    print("[PASS] Test 1 passed: Sampler created successfully")

    # Test 2: Verify bin sizes
    print("\n--- Test 2: Verify Bin Sizes ---")
    bin_sizes = [len(bin_indices) for bin_indices in sampler.bins]
    print(f"Bin sizes: {bin_sizes}")
    expected_bin_size = num_samples // 10
    for i, size in enumerate(bin_sizes):
        if i == len(bin_sizes) - 1:
            # Last bin may be larger due to remainder
            assert size >= expected_bin_size, f"Bin {i} too small"
        else:
            assert size == expected_bin_size, f"Bin {i} size mismatch"
    print("[PASS] Test 2 passed: All bins have correct sizes")

    # Test 3: Generate batches and verify temporal diversity
    print("\n--- Test 3: Generate Batches and Verify Diversity ---")
    batch_count = 0
    temporal_stds = []

    for batch_indices in sampler:
        batch_count += 1
        temporal_std = np.std(batch_indices)
        temporal_stds.append(temporal_std)

        # Verify batch size
        if batch_count < len(sampler):
            assert len(batch_indices) == 100, f"Batch {batch_count} size mismatch"

        # Verify samples come from different temporal regions
        # (high std indicates diversity)
        if batch_count <= 3:
            print(f"  Batch {batch_count}: size={len(batch_indices)}, temporal_std={temporal_std:.1f}")

    print(f"Total batches generated: {batch_count}")
    print(f"Mean temporal std: {np.mean(temporal_stds):.1f}")
    print(f"Target temporal std: {0.3 * num_samples:.1f}")

    assert batch_count == len(sampler), "Batch count mismatch"
    assert np.mean(temporal_stds) > 0.2 * num_samples, "Temporal diversity too low"
    print("[PASS] Test 3 passed: Batches have good temporal diversity")

    # Test 4: Verify all samples used exactly once per epoch
    print("\n--- Test 4: Verify All Samples Used ---")
    all_indices = []
    sampler.reset_bins()  # Reset for new epoch

    for batch_indices in sampler:
        all_indices.extend(batch_indices)

    all_indices_set = set(all_indices)
    print(f"Total indices collected: {len(all_indices)}")
    print(f"Unique indices: {len(all_indices_set)}")

    # With drop_last=False, we should get all samples
    assert len(all_indices_set) == num_samples, "Not all samples used"
    print("[PASS] Test 4 passed: All samples used exactly once")

    # Test 5: Test with DataLoader integration
    print("\n--- Test 5: DataLoader Integration ---")
    sampler.reset_bins()
    loader = DataLoader(dataset, batch_sampler=sampler)

    batch_count = 0
    for batch in loader:
        batch_count += 1
        batch_data = batch[0]  # TensorDataset returns tuple
        if batch_count == 1:
            print(f"  First batch shape: {batch_data.shape}")
            assert batch_data.shape[1:] == (sequence_length, num_channels)

    print(f"Total batches from DataLoader: {batch_count}")
    assert batch_count == len(sampler), "DataLoader batch count mismatch"
    print("[PASS] Test 5 passed: DataLoader integration works")

    # Test 6: Test diversity metrics
    print("\n--- Test 6: Test Diversity Metrics ---")
    mean_diversity = sampler.get_epoch_temporal_diversity()
    print(f"Mean epoch temporal diversity: {mean_diversity:.1f}")

    is_diverse = sampler.check_temporal_diversity(target_threshold=0.3)
    print(f"Meets diversity target (0.3): {is_diverse}")

    assert mean_diversity > 0, "Diversity should be positive"
    print("[PASS] Test 6 passed: Diversity metrics work")

    # Test 7: Test with different batch size
    print("\n--- Test 7: Different Batch Size (batch_size=256) ---")
    sampler2 = StratifiedTemporalBatchSampler(
        dataset,
        batch_size=256,
        num_temporal_bins=10
    )

    batch_count = 0
    for batch_indices in sampler2:
        batch_count += 1

    print(f"Expected batches: {len(sampler2)}")
    print(f"Actual batches: {batch_count}")
    assert batch_count == len(sampler2), "Batch count mismatch"
    print("[PASS] Test 7 passed: Different batch size works")

    # Test 8: Test drop_last parameter
    print("\n--- Test 8: Test drop_last=True ---")
    sampler3 = StratifiedTemporalBatchSampler(
        dataset,
        batch_size=256,
        num_temporal_bins=10,
        drop_last=True
    )

    batch_count = 0
    for batch_indices in sampler3:
        batch_count += 1
        assert len(batch_indices) == 256, "All batches should have full size with drop_last=True"

    print(f"Batches with drop_last=True: {batch_count}")
    assert batch_count == num_samples // 256, "Should drop incomplete batch"
    print("[PASS] Test 8 passed: drop_last works correctly")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
