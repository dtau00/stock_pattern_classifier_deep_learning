"""
Find optimal number of shape clusters using elbow method and silhouette analysis.

Usage:
    python scripts/find_optimal_clusters.py --data_file data/preprocessed/BTCUSDT_1h_preprocessed.h5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.shape_features import extract_windows_shape_features, cluster_by_shape
from src.preprocessing.segmentation import load_preprocessed_package


def evaluate_clustering_range(
    shape_features: np.ndarray,
    min_clusters: int = 5,
    max_clusters: int = 30,
    step: int = 1
):
    """
    Evaluate clustering quality for different numbers of clusters.

    Returns:
        DataFrame with metrics for each cluster count
    """
    results = []

    cluster_range = range(min_clusters, max_clusters + 1, step)

    print(f"Evaluating {len(list(cluster_range))} different cluster counts...")

    for n_clusters in cluster_range:
        print(f"  Testing n={n_clusters}...", end='')

        # Cluster
        labels, _ = cluster_by_shape(
            shape_features,
            n_clusters=n_clusters,
            method='kmeans',
            random_state=42
        )

        # Compute metrics
        try:
            silhouette = silhouette_score(shape_features, labels)
            calinski = calinski_harabasz_score(shape_features, labels)
            davies = davies_bouldin_score(shape_features, labels)
        except Exception as e:
            print(f" Error: {e}")
            continue

        # Check cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        min_cluster_size = counts.min()
        max_cluster_size = counts.max()
        avg_cluster_size = counts.mean()
        std_cluster_size = counts.std()

        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'min_cluster_size': min_cluster_size,
            'max_cluster_size': max_cluster_size,
            'avg_cluster_size': avg_cluster_size,
            'std_cluster_size': std_cluster_size,
        })

        print(f" ✓ (silhouette={silhouette:.3f})")

    return pd.DataFrame(results)


def plot_clustering_metrics(df_results: pd.DataFrame, output_path: str = None):
    """
    Plot clustering quality metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Silhouette Score (higher is better)
    ax = axes[0, 0]
    ax.plot(df_results['n_clusters'], df_results['silhouette'], 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score (Higher = Better Separation)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    # Mark best
    best_idx = df_results['silhouette'].idxmax()
    best_n = df_results.loc[best_idx, 'n_clusters']
    best_score = df_results.loc[best_idx, 'silhouette']
    ax.scatter([best_n], [best_score], color='red', s=200, zorder=5, marker='*',
               label=f'Best: n={int(best_n)}')
    ax.legend()

    # Calinski-Harabasz Score (higher is better)
    ax = axes[0, 1]
    ax.plot(df_results['n_clusters'], df_results['calinski_harabasz'], 'o-',
            linewidth=2, markersize=6, color='green')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Calinski-Harabasz Score', fontsize=12)
    ax.set_title('Calinski-Harabasz Score (Higher = Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Davies-Bouldin Score (lower is better)
    ax = axes[1, 0]
    ax.plot(df_results['n_clusters'], df_results['davies_bouldin'], 'o-',
            linewidth=2, markersize=6, color='orange')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Score', fontsize=12)
    ax.set_title('Davies-Bouldin Score (Lower = Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mark best
    best_idx = df_results['davies_bouldin'].idxmin()
    best_n = df_results.loc[best_idx, 'n_clusters']
    best_score = df_results.loc[best_idx, 'davies_bouldin']
    ax.scatter([best_n], [best_score], color='red', s=200, zorder=5, marker='*',
               label=f'Best: n={int(best_n)}')
    ax.legend()

    # Cluster size distribution
    ax = axes[1, 1]
    ax.plot(df_results['n_clusters'], df_results['avg_cluster_size'], 'o-',
            linewidth=2, markersize=6, label='Average', color='blue')
    ax.fill_between(df_results['n_clusters'],
                     df_results['min_cluster_size'],
                     df_results['max_cluster_size'],
                     alpha=0.2, color='blue', label='Min-Max Range')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Cluster Size (windows)', fontsize=12)
    ax.set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal number of shape clusters'
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
        '--min_clusters',
        type=int,
        default=5,
        help='Minimum number of clusters to test'
    )
    parser.add_argument(
        '--max_clusters',
        type=int,
        default=25,
        help='Maximum number of clusters to test'
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default='data/clustering/optimal_clusters_analysis.png',
        help='Output path for plot'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OPTIMAL CLUSTER ANALYSIS")
    print("=" * 80)

    # Load preprocessed windows
    print(f"\n[1/3] Loading preprocessed windows...")
    windows, metadata, norm_stats = load_preprocessed_package(args.data_file)
    print(f"  Windows: {windows.shape}")

    # Load original OHLCV data
    print(f"\n[2/3] Loading original OHLCV data...")
    df = pd.read_csv(args.package_csv)

    # Extract window size from metadata
    window_size = metadata[0]['end_idx'] - metadata[0]['start_idx'] + 1
    print(f"  Window size: {window_size} bars")

    # Extract shape features
    print(f"\n[2/3] Extracting shape features...")
    shape_features = extract_windows_shape_features(
        df,
        window_size,
        metadata,
        ohlc_cols=True
    )
    print(f"  Shape features: {shape_features.shape}")

    # Evaluate different cluster counts
    print(f"\n[3/3] Evaluating cluster range {args.min_clusters}-{args.max_clusters}...")
    df_results = evaluate_clustering_range(
        shape_features,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        step=1
    )

    # Print results
    print("\n" + "=" * 80)
    print("CLUSTERING QUALITY METRICS")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # Find recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_silhouette = df_results.loc[df_results['silhouette'].idxmax()]
    best_davies = df_results.loc[df_results['davies_bouldin'].idxmin()]
    best_calinski = df_results.loc[df_results['calinski_harabasz'].idxmax()]

    print(f"\n✓ Best by Silhouette Score: n={int(best_silhouette['n_clusters'])} (score={best_silhouette['silhouette']:.3f})")
    print(f"✓ Best by Davies-Bouldin: n={int(best_davies['n_clusters'])} (score={best_davies['davies_bouldin']:.3f})")
    print(f"✓ Best by Calinski-Harabasz: n={int(best_calinski['n_clusters'])} (score={best_calinski['calinski_harabasz']:.1f})")

    # Look for elbow in silhouette
    print("\n💡 Interpretation:")
    print("  - Silhouette: Measures how well-separated clusters are (0.5+ is good)")
    print("  - Davies-Bouldin: Lower is better (below 1.0 is good)")
    print("  - Calinski-Harabasz: Higher is better (relative metric)")
    print("\n  → Look for where metrics plateau (elbow point)")
    print("  → Balance between quality and interpretability")

    # Plot
    print(f"\n[4/4] Creating visualization...")
    Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
    plot_clustering_metrics(df_results, args.output_plot)

    # Save CSV
    csv_path = str(args.output_plot).replace('.png', '.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics to {csv_path}")

    print("\n✓ Done! Review the plot to choose optimal number of clusters.")


if __name__ == '__main__':
    main()
