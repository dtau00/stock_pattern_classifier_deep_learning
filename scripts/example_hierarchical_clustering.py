"""
Example script showing how to integrate hierarchical clustering
with your existing training pipeline.

This demonstrates the recommended workflow:
1. Load preprocessed windows
2. Train Stage 1 (contrastive learning) - your existing pipeline
3. Extract latent vectors from trained model
4. Apply two-stage hierarchical clustering (NEW)
5. Visualize results
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.hierarchical_clustering import apply_hierarchical_clustering
from src.features.shape_features import extract_shape_features
from src.preprocessing.segmentation import load_preprocessed_package


def load_trained_model(model_path: str, device: str = 'cpu'):
    """
    Load your trained contrastive learning model.

    Replace this with your actual model loading code.
    """
    from src.models.ucl_tsc_model import UCLTSCModel

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model with saved config
    model = UCLTSCModel(
        input_channels=checkpoint.get('input_channels', 3),
        d_z=checkpoint.get('d_z', 128),
        num_clusters=checkpoint.get('num_clusters', 30),
        use_hybrid_encoder=checkpoint.get('use_hybrid_encoder', False),
        seq_length=checkpoint.get('seq_length', 127)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def extract_latent_vectors(model, windows: np.ndarray, batch_size: int = 32, device: str = 'cpu'):
    """
    Extract latent vectors from trained model for all windows.

    Args:
        model: Trained UCLTSCModel
        windows: Array of shape (num_windows, seq_len, num_channels)
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        Latent vectors of shape (num_windows, d_z)
    """
    latent_vectors = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]

            # Convert to tensor (channels_first format)
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).to(device)

            # Get latent vectors
            z = model.encoder(batch_tensor)

            # Normalize
            z_norm = F.normalize(z, p=2, dim=1)

            latent_vectors.append(z_norm.cpu().numpy())

    latent_vectors = np.vstack(latent_vectors)

    return latent_vectors


def example_with_existing_model():
    """
    Example: Apply hierarchical clustering using an existing trained model.
    """
    print("=" * 80)
    print("HIERARCHICAL CLUSTERING WITH EXISTING MODEL")
    print("=" * 80)

    # Configuration
    preprocessed_path = 'data/processed/preprocessed_windows.h5'
    model_path = 'models/stage1_best.pt'
    output_path = 'models/hierarchical_clustering.npz'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Step 1: Load preprocessed windows
    print(f"\n[Step 1] Loading preprocessed windows from {preprocessed_path}")

    try:
        windows, metadata, norm_stats = load_preprocessed_package(preprocessed_path)
        print(f"  Loaded {len(windows)} windows")
        print(f"  Window shape: {windows.shape}")
    except FileNotFoundError:
        print(f"  ⚠️  File not found: {preprocessed_path}")
        print("  Please run preprocessing first or update the path")
        return

    # Step 2: Load original OHLCV data for shape features
    print("\n[Step 2] Loading original OHLCV data")

    try:
        # Update this path to your actual data file
        df = pd.read_parquet('data/processed/ohlcv_data.parquet')
        print(f"  Loaded {len(df)} bars")
    except FileNotFoundError:
        print("  ⚠️  Original OHLCV data not found")
        print("  Hierarchical clustering requires original DataFrame for shape features")
        return

    # Step 3: Load trained model
    print(f"\n[Step 3] Loading trained model from {model_path}")

    try:
        model = load_trained_model(model_path, device=device)
        print(f"  Model loaded successfully")
    except FileNotFoundError:
        print(f"  ⚠️  Model not found: {model_path}")
        print("  Please train Stage 1 first or update the path")
        return

    # Step 4: Extract latent vectors
    print("\n[Step 4] Extracting latent vectors from trained model")

    latent_vectors = extract_latent_vectors(model, windows, batch_size=64, device=device)
    print(f"  Extracted latent vectors: {latent_vectors.shape}")

    # Step 5: Apply hierarchical clustering
    print("\n[Step 5] Applying two-stage hierarchical clustering")

    window_size = windows.shape[1]

    results = apply_hierarchical_clustering(
        windows_df=df,
        metadata=metadata,
        latent_vectors=latent_vectors,
        window_size=window_size,
        n_shape_clusters=10,
        n_stat_clusters_per_shape=3,
        save_path=output_path
    )

    # Step 6: Analyze results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    shape_labels = results['shape_labels']
    hierarchical_labels = results['hierarchical_labels']
    cluster_stats = results['cluster_stats']

    print(f"\nShape clusters: {len(np.unique(shape_labels))}")
    print(f"Total hierarchical clusters: {len(np.unique(hierarchical_labels))}")

    print("\n📊 Cluster Statistics:")
    print(cluster_stats.to_string(index=False))

    # Save cluster assignments for later use
    assignments_path = 'models/cluster_assignments.npz'
    np.savez(
        assignments_path,
        shape_labels=shape_labels,
        stat_labels=results['stat_labels'],
        hierarchical_labels=hierarchical_labels
    )
    print(f"\n✓ Saved cluster assignments to {assignments_path}")

    return results


def example_standalone_shape_clustering():
    """
    Example: Apply only shape clustering without contrastive model.

    Useful if you want to see shape clusters before training.
    """
    print("=" * 80)
    print("STANDALONE SHAPE CLUSTERING (NO MODEL NEEDED)")
    print("=" * 80)

    # Load original data
    try:
        df = pd.read_parquet('data/processed/ohlcv_data.parquet')
        print(f"\nLoaded {len(df)} bars")
    except FileNotFoundError:
        print("\n⚠️  OHLCV data not found")
        return

    # Add shape features
    print("\nAdding shape features...")
    df_with_shapes = extract_shape_features(df, window_size=20)

    shape_cols = [col for col in df_with_shapes.columns if col.startswith('shape_')]
    print(f"Added {len(shape_cols)} shape feature columns:")
    for col in shape_cols:
        print(f"  - {col}")

    # Create windows with shape features
    from src.preprocessing.segmentation import create_sliding_windows

    print("\nCreating windows with shape features...")

    # Use shape features as normalized channels
    windows, metadata = create_sliding_windows(
        df_with_shapes,
        normalized_channels=shape_cols,
        sequence_length=20,
        overlap=0.5
    )

    print(f"Created {len(windows)} windows with shape features")

    # Cluster by shape
    from src.features.shape_features import cluster_by_shape, extract_windows_shape_features

    print("\nExtracting shape vectors...")
    shape_features = extract_windows_shape_features(
        df_with_shapes,
        window_size=20,
        metadata=metadata,
        ohlc_cols=True
    )

    print(f"Shape features: {shape_features.shape}")

    print("\nClustering by shape...")
    shape_labels, _ = cluster_by_shape(
        shape_features,
        n_clusters=10,
        method='kmeans'
    )

    # Print distribution
    unique, counts = np.unique(shape_labels, return_counts=True)
    print("\nShape cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Shape {cluster_id}: {count} windows ({count/len(shape_labels)*100:.1f}%)")

    print("\n✓ Shape clustering complete!")
    print("  Next step: Train your contrastive model and apply statistical refinement")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical Clustering Examples')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'shape_only'],
        default='full',
        help='Mode: "full" (with trained model) or "shape_only" (shape clustering only)'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        example_with_existing_model()
    elif args.mode == 'shape_only':
        example_standalone_shape_clustering()
