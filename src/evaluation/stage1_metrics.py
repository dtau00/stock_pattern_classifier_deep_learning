"""
Stage 1 Evaluation Metrics

This module provides quality metrics for evaluating Stage 1 (contrastive pre-training)
representations WITHOUT requiring Stage 2 clustering.

Metrics:
1. Embedding Variance - Detects feature collapse
2. Effective Rank - Measures dimensionality usage
3. Alignment & Uniformity - Contrastive learning quality
4. k-NN Accuracy - Self-supervised proxy metric

Reference: Wang & Isola (2020) "Understanding Contrastive Representation Learning"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


def embedding_variance(model, dataloader, device) -> float:
    """
    Compute average standard deviation across embedding dimensions.

    Higher values indicate embeddings use the full latent space.
    Lower values (<0.2) indicate feature collapse.

    Args:
        model: Trained model with encoder
        dataloader: Validation data loader
        device: Device (cuda/cpu)

    Returns:
        Average std dev across dimensions

    Target: 0.3-0.5 (good), <0.1 (collapsed)
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            z = model.encoder(x)
            z_norm = F.normalize(z, p=2, dim=1)
            embeddings.append(z_norm.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    # Std dev per dimension, then average
    std_per_dim = embeddings.std(dim=0)
    avg_std = std_per_dim.mean().item()

    return avg_std


def effective_rank(model, dataloader, device) -> float:
    """
    Compute effective rank of embedding matrix via singular values.

    Measures how many dimensions are actually used.
    Higher is better (max = d_z).

    Args:
        model: Trained model with encoder
        dataloader: Validation data loader
        device: Device (cuda/cpu)

    Returns:
        Effective rank (exponential of entropy of singular values)

    Target: >60% of d_z (e.g., >76 for d_z=128)
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)
            z = model.encoder(x)
            z_norm = F.normalize(z, p=2, dim=1)
            embeddings.append(z_norm.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    # SVD to get singular values
    _, S, _ = torch.svd(embeddings)

    # Effective rank: exp(entropy of normalized singular values)
    S_normalized = S / S.sum()
    entropy = -(S_normalized * torch.log(S_normalized + 1e-12)).sum()
    eff_rank = torch.exp(entropy).item()

    return eff_rank


def alignment_uniformity(model, dataloader, device, augmentation) -> Tuple[float, float]:
    """
    Compute alignment and uniformity metrics for contrastive learning.

    Alignment: How close are positive pairs (augmented views)?
    Uniformity: How evenly distributed are embeddings on hypersphere?

    Args:
        model: Trained model with encoder
        dataloader: Validation data loader
        device: Device (cuda/cpu)
        augmentation: TimeSeriesAugmentation instance

    Returns:
        Tuple of (alignment, uniformity)

    Targets:
        Alignment: <0.5 (lower = positive pairs closer)
        Uniformity: <-1.5 (lower = more uniform distribution)
    """
    model.eval()

    alignments = []
    embeddings = []

    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)

            # Create two augmented views
            x1, x2 = augmentation(x)

            z1 = model.encoder(x1)
            z2 = model.encoder(x2)

            z1_norm = F.normalize(z1, p=2, dim=1)
            z2_norm = F.normalize(z2, p=2, dim=1)

            # Alignment: L2 distance between positive pairs
            alignment = (z1_norm - z2_norm).pow(2).sum(dim=1).mean()
            alignments.append(alignment.cpu())

            embeddings.append(z1_norm.cpu())

    # Average alignment
    avg_alignment = torch.stack(alignments).mean().item()

    # Uniformity: log of average pairwise Gaussian potential
    embeddings = torch.cat(embeddings, dim=0)

    # Sample for efficiency (all pairs is O(n^2))
    n = min(5000, len(embeddings))
    indices = torch.randperm(len(embeddings))[:n]
    sample = embeddings[indices]

    # Pairwise squared L2 distances
    dists = torch.pdist(sample, p=2).pow(2)

    # Uniformity: log(mean(exp(-2 * dist^2)))
    uniformity = dists.mul(-2).exp().mean().log().item()

    return avg_alignment, uniformity


def knn_accuracy(model, dataloader, device, augmentation, k: int = 5) -> float:
    """
    Compute k-NN accuracy using augmentations as pseudo-labels.

    For each sample, check if its augmented version is among its k nearest neighbors.
    This is a self-supervised proxy for representation quality.

    Args:
        model: Trained model with encoder
        dataloader: Validation data loader
        device: Device (cuda/cpu)
        augmentation: TimeSeriesAugmentation instance
        k: Number of nearest neighbors to check

    Returns:
        Fraction of samples where augmented pair is in top-k neighbors

    Target: >0.8 (good), <0.5 (poor)
    """
    model.eval()

    embeddings = []
    aug_embeddings = []

    with torch.no_grad():
        for (x,) in dataloader:
            x = x.to(device)

            # Original
            z = model.encoder(x)
            z_norm = F.normalize(z, p=2, dim=1)
            embeddings.append(z_norm.cpu())

            # Augmented
            x_aug, _ = augmentation(x)
            z_aug = model.encoder(x_aug)
            z_aug_norm = F.normalize(z_aug, p=2, dim=1)
            aug_embeddings.append(z_aug_norm.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    aug_embeddings = torch.cat(aug_embeddings, dim=0).numpy()

    # Compute cosine similarity for each augmented embedding to all originals
    # Then check if the corresponding original is in top-k
    from sklearn.metrics.pairwise import cosine_similarity

    # Similarity matrix: aug[i] vs emb[j]
    sim_matrix = cosine_similarity(aug_embeddings, embeddings)

    # For each row, get indices of top-k+1 (including self)
    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k+1]

    # Check if index i is in the top-k neighbors of aug[i]
    correct = 0
    for i in range(len(embeddings)):
        if i in top_k_indices[i]:
            correct += 1

    accuracy = correct / len(embeddings)
    return accuracy


def evaluate_stage1(model, val_loader, device, augmentation, config) -> Dict:
    """
    Comprehensive Stage 1 quality evaluation.

    Runs all metrics and provides interpretable output.

    Args:
        model: Trained model with encoder
        val_loader: Validation data loader
        device: Device (cuda/cpu)
        augmentation: TimeSeriesAugmentation instance
        config: Config object with model.d_z

    Returns:
        Dictionary with all metrics and pass/fail flags
    """
    print("\n" + "="*60)
    print("Stage 1 Quality Evaluation (Validation Set)")
    print("="*60)

    # 1. Embedding Variance
    print("\n[1/4] Computing embedding variance...")
    var = embedding_variance(model, val_loader, device)
    var_pass = var > 0.25
    print(f"      Embedding Variance: {var:.4f}")
    print(f"      Target: >0.25 | {'[PASS] Good spread' if var_pass else '[FAIL] Feature collapse detected'}")

    # 2. Effective Rank
    print("\n[2/4] Computing effective rank...")
    eff_rank = effective_rank(model, val_loader, device)
    rank_threshold = config.model.d_z * 0.6
    rank_pass = eff_rank > rank_threshold
    print(f"      Effective Rank: {eff_rank:.1f} / {config.model.d_z}")
    print(f"      Target: >{rank_threshold:.0f} | {'[PASS] Good dimensionality usage' if rank_pass else '[FAIL] Low-dimensional subspace'}")

    # 3. Alignment & Uniformity
    print("\n[3/4] Computing alignment & uniformity...")
    alignment, uniformity = alignment_uniformity(model, val_loader, device, augmentation)
    align_pass = alignment < 0.5
    uniform_pass = uniformity < -1.5
    print(f"      Alignment: {alignment:.4f}")
    print(f"      Target: <0.5 | {'[PASS] Positive pairs close' if align_pass else '[FAIL] Positive pairs too far'}")
    print(f"      Uniformity: {uniformity:.4f}")
    print(f"      Target: <-1.5 | {'[PASS] Embeddings spread uniformly' if uniform_pass else '[FAIL] Embeddings clustered'}")

    # 4. k-NN Accuracy
    print("\n[4/4] Computing k-NN accuracy...")
    knn_acc = knn_accuracy(model, val_loader, device, augmentation, k=5)
    knn_pass = knn_acc > 0.75
    print(f"      k-NN Accuracy (k=5): {knn_acc:.3f}")
    print(f"      Target: >0.75 | {'[PASS] Strong augmentation invariance' if knn_pass else '[FAIL] Weak augmentation invariance'}")

    # Overall score
    scores = [var_pass, rank_pass, align_pass, uniform_pass, knn_pass]
    n_passed = sum(scores)
    overall_pct = (n_passed / len(scores)) * 100

    print(f"\n{'='*60}")
    print(f"Overall: {n_passed}/{len(scores)} metrics passed ({overall_pct:.0f}%)")

    if overall_pct >= 80:
        print("Quality Assessment: [EXCELLENT] Ready for Stage 2")
    elif overall_pct >= 60:
        print("Quality Assessment: [GOOD] Proceed to Stage 2")
    elif overall_pct >= 40:
        print("Quality Assessment: [FAIR] Consider retraining with larger batch")
    else:
        print("Quality Assessment: [POOR] Retrain with larger batch size required")

    print("="*60 + "\n")

    return {
        'variance': var,
        'variance_pass': var_pass,
        'effective_rank': eff_rank,
        'effective_rank_pass': rank_pass,
        'alignment': alignment,
        'alignment_pass': align_pass,
        'uniformity': uniformity,
        'uniformity_pass': uniform_pass,
        'knn_accuracy': knn_acc,
        'knn_accuracy_pass': knn_pass,
        'overall_score': overall_pct,
        'n_passed': n_passed,
        'n_total': len(scores)
    }


def test_stage1_metrics():
    """Test Stage 1 metrics with synthetic data."""
    print("Testing Stage 1 Evaluation Metrics...")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.ucl_tsc_model import UCLTSCModel
    from training.augmentation import TimeSeriesAugmentation
    from config.config import get_small_config
    from torch.utils.data import DataLoader, TensorDataset

    # Create synthetic data
    config = get_small_config()
    n_samples = 500
    x = torch.randn(n_samples, 3, 127)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Create model
    model = UCLTSCModel(
        input_channels=3,
        d_z=config.model.d_z,
        num_clusters=config.model.num_clusters,
        seq_length=127
    )
    device = 'cpu'
    model.to(device)

    # Create augmentation
    augmentation = TimeSeriesAugmentation()

    # Run evaluation
    print("\n[Test] Running Stage 1 evaluation on untrained model...")
    metrics = evaluate_stage1(model, loader, device, augmentation, config)

    print("\n[Test] Metrics returned:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n[SUCCESS] Stage 1 metrics test passed!")
    return True


if __name__ == '__main__':
    test_stage1_metrics()
