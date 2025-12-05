"""
Hierarchical Two-Stage Clustering Module

Combines shape-based pre-clustering with statistical microstructure refinement.

Stage 1 (Shape Pre-clustering):
    - Groups windows by visual price pattern similarity
    - Uses shape features: normalized price curves, trends, turning points
    - Creates N_shape clusters (e.g., 10-15 shape groups)

Stage 2 (Statistical Refinement):
    - Within each shape cluster, applies contrastive learning model
    - Groups by statistical microstructure (volume, volatility, momentum)
    - Creates M_stat sub-clusters per shape (e.g., 3-5 sub-clusters)
    - Total clusters: N_shape × M_stat

Benefits:
    - Visually similar patterns grouped together (human interpretable)
    - Statistical properties refined within each shape (predictive power)
    - Best of both worlds: visual consistency + statistical rigor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from pathlib import Path

# Import shape features module
try:
    from src.features.shape_features import (
        extract_windows_shape_features,
        cluster_by_shape,
        get_shape_cluster_stats
    )
except ImportError:
    from features.shape_features import (
        extract_windows_shape_features,
        cluster_by_shape,
        get_shape_cluster_stats
    )


class HierarchicalClusteringPipeline:
    """
    Two-stage hierarchical clustering pipeline.

    Usage:
        1. Fit on training data to learn both shape and statistical clusters
        2. Save cluster assignments and models
        3. Transform new data using learned clusters
    """

    def __init__(self,
                 n_shape_clusters: int = 10,
                 n_stat_clusters_per_shape: int = 3,
                 shape_method: str = 'kmeans',
                 random_state: int = 42):
        """
        Initialize hierarchical clustering pipeline.

        Args:
            n_shape_clusters: Number of shape-based clusters (Stage 1)
            n_stat_clusters_per_shape: Number of statistical sub-clusters per shape (Stage 2)
            shape_method: Clustering method for shapes ('kmeans' or 'hierarchical')
            random_state: Random seed
        """
        self.n_shape_clusters = n_shape_clusters
        self.n_stat_clusters_per_shape = n_stat_clusters_per_shape
        self.shape_method = shape_method
        self.random_state = random_state

        # Fitted models (populated during fit)
        self.shape_clusterer = None
        self.stat_clusterers = {}  # One per shape cluster
        self.shape_scaler = None

        # Cluster assignments
        self.shape_labels = None
        self.stat_labels = None
        self.hierarchical_labels = None  # Combined shape+stat labels

    def fit_stage1_shapes(self,
                         windows_df: pd.DataFrame,
                         metadata: List[dict],
                         window_size: int) -> np.ndarray:
        """
        Stage 1: Cluster by price shape.

        Args:
            windows_df: Original DataFrame with OHLCV data
            metadata: List of window metadata
            window_size: Window size in bars

        Returns:
            Array of shape cluster labels
        """
        print(f"\n[Stage 1] Clustering by price shape...")
        print(f"  Creating {self.n_shape_clusters} shape-based clusters")

        # Extract shape features for all windows
        shape_features = extract_windows_shape_features(
            windows_df,
            window_size,
            metadata,
            ohlc_cols=True
        )

        print(f"  Extracted {shape_features.shape[1]} shape features per window")

        # Cluster by shape
        shape_labels, shape_clusterer = cluster_by_shape(
            shape_features,
            n_clusters=self.n_shape_clusters,
            method=self.shape_method,
            random_state=self.random_state
        )

        # Store fitted model
        self.shape_clusterer = shape_clusterer
        self.shape_labels = shape_labels

        # Print cluster distribution
        unique, counts = np.unique(shape_labels, return_counts=True)
        print(f"  Shape cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"    Shape {cluster_id}: {count} windows ({count/len(shape_labels)*100:.1f}%)")

        return shape_labels

    def fit_stage2_statistics(self,
                             latent_vectors: np.ndarray,
                             shape_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: Within each shape cluster, cluster by statistical properties.

        Args:
            latent_vectors: Latent vectors from contrastive model (N, d_z)
            shape_labels: Shape cluster labels from Stage 1

        Returns:
            Tuple of (statistical_labels, hierarchical_labels)
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        print(f"\n[Stage 2] Refining with statistical clustering...")
        print(f"  Creating {self.n_stat_clusters_per_shape} sub-clusters per shape")

        stat_labels = np.zeros(len(latent_vectors), dtype=int)
        hierarchical_labels = np.zeros(len(latent_vectors), dtype=int)

        current_hierarchical_id = 0

        # For each shape cluster, perform statistical sub-clustering
        for shape_id in range(self.n_shape_clusters):
            # Get windows belonging to this shape cluster
            shape_mask = shape_labels == shape_id
            shape_indices = np.where(shape_mask)[0]

            if len(shape_indices) == 0:
                continue

            # Get latent vectors for this shape cluster
            shape_latents = latent_vectors[shape_indices]

            # Handle small clusters
            n_stat = min(self.n_stat_clusters_per_shape, len(shape_indices))

            if n_stat <= 1:
                # Too few samples for sub-clustering
                stat_labels[shape_indices] = 0
                hierarchical_labels[shape_indices] = current_hierarchical_id
                current_hierarchical_id += 1
                print(f"  Shape {shape_id}: {len(shape_indices)} windows → 1 cluster (too small)")
                continue

            # Normalize latent vectors
            scaler = StandardScaler()
            shape_latents_scaled = scaler.fit_transform(shape_latents)

            # K-means on latent vectors
            kmeans = KMeans(
                n_clusters=n_stat,
                init='k-means++',
                n_init=10,
                random_state=self.random_state
            )
            sub_labels = kmeans.fit_predict(shape_latents_scaled)

            # Store statistical cluster labels (relative to shape cluster)
            stat_labels[shape_indices] = sub_labels

            # Assign hierarchical labels (unique across all clusters)
            for sub_id in range(n_stat):
                sub_mask = sub_labels == sub_id
                sub_indices = shape_indices[sub_mask]
                hierarchical_labels[sub_indices] = current_hierarchical_id
                current_hierarchical_id += 1

            # Store fitted model
            self.stat_clusterers[shape_id] = (kmeans, scaler)

            # Print distribution
            unique, counts = np.unique(sub_labels, return_counts=True)
            print(f"  Shape {shape_id}: {len(shape_indices)} windows → {len(unique)} sub-clusters")
            for sub_id, count in zip(unique, counts):
                print(f"    └─ Sub-cluster {sub_id}: {count} windows")

        self.stat_labels = stat_labels
        self.hierarchical_labels = hierarchical_labels

        print(f"\n  Total hierarchical clusters: {current_hierarchical_id}")

        return stat_labels, hierarchical_labels

    def get_cluster_summary(self,
                           windows_df: pd.DataFrame,
                           metadata: List[dict]) -> pd.DataFrame:
        """
        Generate summary statistics for all hierarchical clusters.

        Args:
            windows_df: Original DataFrame with OHLCV data
            metadata: Window metadata

        Returns:
            DataFrame with cluster statistics
        """
        if self.hierarchical_labels is None:
            raise ValueError("Must call fit_stage2_statistics first")

        cluster_stats = []

        unique_clusters = np.unique(self.hierarchical_labels)

        for cluster_id in unique_clusters:
            cluster_mask = self.hierarchical_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Find which shape cluster this belongs to
            shape_id = self.shape_labels[cluster_indices[0]]
            stat_id = self.stat_labels[cluster_indices[0]]

            # Compute price statistics
            price_changes = []
            volatilities = []

            for idx in cluster_indices:
                meta = metadata[idx]
                window = windows_df.iloc[meta['start_idx']:meta['end_idx'] + 1]

                # Price change
                pct_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
                price_changes.append(pct_change)

                # Volatility
                vol = window['close'].pct_change().std()
                volatilities.append(vol)

            stats = {
                'hierarchical_cluster': cluster_id,
                'shape_cluster': shape_id,
                'stat_cluster': stat_id,
                'num_windows': len(cluster_indices),
                'avg_price_change': np.mean(price_changes),
                'std_price_change': np.std(price_changes),
                'avg_volatility': np.mean(volatilities),
            }
            cluster_stats.append(stats)

        df_stats = pd.DataFrame(cluster_stats)
        df_stats = df_stats.sort_values(['shape_cluster', 'stat_cluster'])

        return df_stats

    def save(self, filepath: str):
        """
        Save fitted clustering models and labels.

        Args:
            filepath: Path to save file (will create .npz and .pkl files)
        """
        import pickle
        from pathlib import Path

        # Create directory
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save cluster labels as numpy
        np.savez(
            filepath,
            shape_labels=self.shape_labels,
            stat_labels=self.stat_labels,
            hierarchical_labels=self.hierarchical_labels,
            n_shape_clusters=self.n_shape_clusters,
            n_stat_clusters_per_shape=self.n_stat_clusters_per_shape
        )

        # Save sklearn models as pickle
        models_path = str(filepath).replace('.npz', '_models.pkl')
        with open(models_path, 'wb') as f:
            pickle.dump({
                'shape_clusterer': self.shape_clusterer,
                'stat_clusterers': self.stat_clusterers,
            }, f)

        print(f"✓ Saved hierarchical clustering to {filepath}")

    def load(self, filepath: str):
        """
        Load fitted clustering models and labels.

        Args:
            filepath: Path to saved file
        """
        import pickle

        # Load cluster labels
        data = np.load(filepath)
        self.shape_labels = data['shape_labels']
        self.stat_labels = data['stat_labels']
        self.hierarchical_labels = data['hierarchical_labels']
        self.n_shape_clusters = int(data['n_shape_clusters'])
        self.n_stat_clusters_per_shape = int(data['n_stat_clusters_per_shape'])

        # Load sklearn models
        models_path = str(filepath).replace('.npz', '_models.pkl')
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
            self.shape_clusterer = models['shape_clusterer']
            self.stat_clusterers = models['stat_clusterers']

        print(f"✓ Loaded hierarchical clustering from {filepath}")


def apply_hierarchical_clustering(windows_df: pd.DataFrame,
                                  metadata: List[dict],
                                  latent_vectors: np.ndarray,
                                  window_size: int,
                                  n_shape_clusters: int = 10,
                                  n_stat_clusters_per_shape: int = 3,
                                  save_path: Optional[str] = None) -> Dict:
    """
    Convenience function to apply full two-stage hierarchical clustering.

    Args:
        windows_df: Original DataFrame with OHLCV data
        metadata: Window metadata
        latent_vectors: Latent vectors from contrastive model
        window_size: Window size in bars
        n_shape_clusters: Number of shape clusters (Stage 1)
        n_stat_clusters_per_shape: Number of stat sub-clusters per shape (Stage 2)
        save_path: Optional path to save clustering results

    Returns:
        Dictionary with:
            - 'shape_labels': Shape cluster labels
            - 'stat_labels': Statistical sub-cluster labels
            - 'hierarchical_labels': Combined hierarchical labels
            - 'cluster_stats': DataFrame with cluster statistics
            - 'pipeline': Fitted HierarchicalClusteringPipeline
    """
    print("=" * 60)
    print("HIERARCHICAL TWO-STAGE CLUSTERING")
    print("=" * 60)

    # Initialize pipeline
    pipeline = HierarchicalClusteringPipeline(
        n_shape_clusters=n_shape_clusters,
        n_stat_clusters_per_shape=n_stat_clusters_per_shape
    )

    # Stage 1: Shape clustering
    shape_labels = pipeline.fit_stage1_shapes(
        windows_df,
        metadata,
        window_size
    )

    # Stage 2: Statistical refinement
    stat_labels, hierarchical_labels = pipeline.fit_stage2_statistics(
        latent_vectors,
        shape_labels
    )

    # Get cluster statistics
    cluster_stats = pipeline.get_cluster_summary(windows_df, metadata)

    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    print(cluster_stats.to_string(index=False))

    # Save if requested
    if save_path:
        pipeline.save(save_path)

    return {
        'shape_labels': shape_labels,
        'stat_labels': stat_labels,
        'hierarchical_labels': hierarchical_labels,
        'cluster_stats': cluster_stats,
        'pipeline': pipeline
    }


if __name__ == '__main__':
    print("Testing hierarchical clustering...")

    # Create synthetic data
    np.random.seed(42)

    # Create synthetic OHLCV data
    n_bars = 500
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n_bars) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(n_bars) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'volume': np.random.randint(1000, 10000, n_bars)
    })
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))

    # Create synthetic windows
    window_size = 20
    n_windows = 100
    metadata = [
        {'start_idx': i * 4, 'end_idx': i * 4 + window_size - 1}
        for i in range(n_windows)
    ]

    # Create synthetic latent vectors (simulating contrastive model output)
    latent_dim = 128
    latent_vectors = np.random.randn(n_windows, latent_dim)

    # Apply hierarchical clustering
    results = apply_hierarchical_clustering(
        windows_df=df,
        metadata=metadata,
        latent_vectors=latent_vectors,
        window_size=window_size,
        n_shape_clusters=5,
        n_stat_clusters_per_shape=2,
        save_path=None
    )

    print("\n✓ Hierarchical clustering test completed!")
    print(f"  Shape clusters: {len(np.unique(results['shape_labels']))}")
    print(f"  Total hierarchical clusters: {len(np.unique(results['hierarchical_labels']))}")
