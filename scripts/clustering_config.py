"""
Clustering Configuration File

Edit these settings to control how clustering works.
Then run: python scripts/apply_hierarchical_clustering.py
"""

# ============================================================================
# BASIC SETTINGS - Most Important
# ============================================================================

# Number of shape-based clusters (Stage 1)
# Guidelines:
#   - Small datasets (<5k windows): 8-12
#   - Medium datasets (5k-20k): 12-20
#   - Large datasets (>20k): 20-30
N_SHAPE_CLUSTERS = 12

# Input files
DATA_FILE = 'data/preprocessed/BTCUSDT_1h_preprocessed.h5'
PACKAGE_CSV = 'data/packages/BTCUSDT_1h_2015-11-05_2025-12-05.csv'

# Output path
OUTPUT_PATH = 'data/clustering/BTCUSDT_1h_2015-11-05_2025-12-05_shape_clusters.npz'

# ============================================================================
# ADVANCED SETTINGS - For Fine-Tuning
# ============================================================================

# Clustering method
# Options: 'kmeans' or 'hierarchical'
#   - kmeans: Faster, good for most cases
#   - hierarchical: Better for nested patterns, slower
CLUSTERING_METHOD = 'kmeans'

# Random seed (for reproducibility)
RANDOM_SEED = 42

# Number of samples to save per cluster (for visualization)
N_SAMPLES_PER_CLUSTER = 20

# ============================================================================
# STAGE 2 SETTINGS - For Statistical Refinement (Future)
# ============================================================================

# Number of statistical sub-clusters within each shape cluster
# Only used if you run with --mode hierarchical
N_STAT_CLUSTERS_PER_SHAPE = 3

# ============================================================================
# HELP & TIPS
# ============================================================================

"""
HOW TO USE THIS FILE:
=====================
1. Edit the settings above
2. Run: python scripts/apply_hierarchical_clustering.py
3. View results in UI: Shape Cluster Visualization page

OR use command-line arguments to override:
python scripts/apply_hierarchical_clustering.py --n_shape_clusters 15


HOW TO CHOOSE N_SHAPE_CLUSTERS:
================================
Method 1 - Quick Visual Check:
  - Start with 10-15
  - View in UI
  - If clusters look too mixed → increase number
  - If clusters look too similar → decrease number

Method 2 - Systematic Analysis:
  - Run: python scripts/find_optimal_clusters.py
  - Look at the generated plot
  - Choose number where metrics plateau (elbow point)

Method 3 - Domain Knowledge:
  - Think about how many distinct price patterns you expect
  - Common patterns: strong up, strong down, sideways, volatile, etc.
  - Typical range: 8-20 clusters


CLUSTERING METHOD COMPARISON:
==============================
K-Means:
  + Faster
  + Works well for spherical clusters
  + Good for most use cases
  - Assumes similar cluster sizes

Hierarchical:
  + Can capture nested patterns
  + No assumption about cluster size/shape
  + Better for complex patterns
  - Slower on large datasets


WHAT MAKES GOOD CLUSTERS:
==========================
1. Visual Similarity: Windows in same cluster should look similar
2. Balanced Sizes: No cluster should be too small (<1%) or too large (>30%)
3. Distinct Patterns: Different clusters should look different
4. Statistical Coherence: Similar price changes and volatility within cluster
"""
