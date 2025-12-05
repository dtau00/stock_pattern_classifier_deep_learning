"""
Apply hierarchical clustering to your existing preprocessed data.

This script:
1. Loads your preprocessed windows from HDF5
2. Loads original OHLCV data for shape features
3. Applies two-stage hierarchical clustering
4. Saves results for visualization and training

Usage:
    python scripts/apply_hierarchical_clustering.py --data_file data/preprocessed/BTCUSDT_1h_preprocessed.h5
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.shape_features import extract_windows_shape_features, cluster_by_shape
from src.preprocessing.segmentation import load_preprocessed_package


def load_original_ohlcv(package_csv_path: str) -> pd.DataFrame:
    """Load original OHLCV data from package CSV."""
    print(f"\nLoading original OHLCV data from {package_csv_path}")

    df = pd.read_csv(package_csv_path)

    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        else:
            raise ValueError("No timestamp or open_time column found")

    print(f"  Loaded {len(df)} bars")
    print(f"  Columns: {list(df.columns)}")

    return df


def apply_shape_clustering_only(
    windows_path: str,
    package_csv_path: str,
    n_shape_clusters: int = 15,
    output_path: str = None
):
    """
    Apply shape-based clustering only (no trained model needed).

    This is useful to:
    - See what shape clusters look like before training
    - Decide on optimal n_shape_clusters
    - Understand your data distribution
    """
    print("=" * 80)
    print("SHAPE-BASED CLUSTERING (Stage 1 Only)")
    print("=" * 80)

    # Load preprocessed windows
    print(f"\n[1/4] Loading preprocessed windows...")
    windows, metadata, norm_stats = load_preprocessed_package(windows_path)
    print(f"  Windows: {windows.shape}")
    print(f"  Metadata: {len(metadata)} entries")

    # Load original OHLCV data
    print(f"\n[2/4] Loading original OHLCV data...")
    df = load_original_ohlcv(package_csv_path)

    # Extract window size from metadata
    window_size = metadata[0]['end_idx'] - metadata[0]['start_idx'] + 1
    print(f"  Window size: {window_size} bars")

    # Extract shape features for all windows
    print(f"\n[3/4] Extracting shape features...")
    shape_features = extract_windows_shape_features(
        df,
        window_size,
        metadata,
        ohlc_cols=True
    )
    print(f"  Shape features: {shape_features.shape}")
    print(f"  Feature dimension: {shape_features.shape[1]}")

    # Cluster by shape
    print(f"\n[4/4] Clustering by shape (n={n_shape_clusters})...")
    shape_labels, shape_clusterer = cluster_by_shape(
        shape_features,
        n_clusters=n_shape_clusters,
        method='kmeans',
        random_state=42
    )

    # Print distribution
    print(f"\nShape cluster distribution:")
    unique, counts = np.unique(shape_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = count / len(shape_labels) * 100
        print(f"  Shape {cluster_id:2d}: {count:5d} windows ({pct:5.1f}%)")

    # Compute statistics per shape cluster
    print(f"\nComputing shape cluster statistics...")
    cluster_stats = []

    for cluster_id in unique:
        cluster_mask = shape_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Compute price statistics
        price_changes = []
        volatilities = []

        for idx in cluster_indices[:100]:  # Sample first 100 to speed up
            meta = metadata[idx]
            window = df.iloc[meta['start_idx']:meta['end_idx'] + 1]

            # Price change
            if len(window) > 0:
                pct_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
                price_changes.append(pct_change)

                # Volatility
                vol = window['close'].pct_change().std()
                volatilities.append(vol)

        stats = {
            'shape_cluster': cluster_id,
            'num_windows': len(cluster_indices),
            'pct_of_total': len(cluster_indices) / len(shape_labels) * 100,
            'avg_price_change': np.mean(price_changes) if price_changes else 0,
            'std_price_change': np.std(price_changes) if price_changes else 0,
            'avg_volatility': np.mean(volatilities) if volatilities else 0,
        }
        cluster_stats.append(stats)

    df_stats = pd.DataFrame(cluster_stats)

    print("\n" + "=" * 80)
    print("SHAPE CLUSTER STATISTICS")
    print("=" * 80)
    print(df_stats.to_string(index=False))

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cluster labels
        np.savez(
            output_path,
            shape_labels=shape_labels,
            n_shape_clusters=n_shape_clusters,
            shape_features=shape_features
        )
        print(f"\n✓ Saved shape clustering results to {output_path}")

        # Save statistics
        stats_path = output_path.parent / f"{output_path.stem}_stats.csv"
        df_stats.to_csv(stats_path, index=False)
        print(f"✓ Saved statistics to {stats_path}")

        # Save sample window indices for visualization
        samples_path = output_path.parent / f"{output_path.stem}_samples.json"
        samples = {}
        for cluster_id in unique:
            cluster_mask = shape_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Sample random windows from this cluster
            n_samples = min(20, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)
            samples[int(cluster_id)] = sample_indices.tolist()

        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"✓ Saved sample indices to {samples_path}")

    return {
        'shape_labels': shape_labels,
        'shape_features': shape_features,
        'cluster_stats': df_stats,
        'clusterer': shape_clusterer
    }


def main():
    parser = argparse.ArgumentParser(
        description='Apply hierarchical clustering to preprocessed data'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='data/preprocessed/BTCUSDT_1h_preprocessed.h5',
        help='Path to preprocessed HDF5 file'
    )
    parser.add_argument(
        '--package_csv',
        type=str,
        default='data/packages/BTCUSDT_1h_2015-11-05_2025-12-05.csv',
        help='Path to original package CSV file'
    )
    parser.add_argument(
        '--n_shape_clusters',
        type=int,
        default=15,
        help='Number of shape-based clusters (Stage 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/clustering/shape_clusters.npz',
        help='Output path for clustering results'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['shape_only', 'hierarchical'],
        default='shape_only',
        help='Clustering mode: shape_only (no model) or hierarchical (requires trained model)'
    )

    args = parser.parse_args()

    if args.mode == 'shape_only':
        # Apply shape clustering only
        results = apply_shape_clustering_only(
            windows_path=args.data_file,
            package_csv_path=args.package_csv,
            n_shape_clusters=args.n_shape_clusters,
            output_path=args.output
        )

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Visualize shape clusters:")
        print(f"   - Use the UI page: Hierarchical Clustering Visualization")
        print(f"   - Sample indices saved to: {Path(args.output).parent}/{Path(args.output).stem}_samples.json")

        print("\n2. Train your contrastive model:")
        print("   - Use existing training pipeline")
        print("   - Or train within shape clusters for better results")

        print("\n3. Apply statistical refinement (Stage 2):")
        print("   - After training, extract latent vectors")
        print("   - Run with --mode hierarchical")

    elif args.mode == 'hierarchical':
        print("\n⚠️  Hierarchical mode requires a trained model.")
        print("Please train your model first, then use this script with --mode hierarchical")
        print("For now, try --mode shape_only to see shape clusters")

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
