"""
UCL-TSC Model: Unsupervised Contrastive Learning for Time Series Clustering

This module implements the complete UCL-TSC model architecture, integrating:
- Multi-view encoders (CNN, TCN, CNN-TCN hybrid)
- Adaptive fusion layer
- Projection head for contrastive learning
- Clustering head with learnable centroids

The model supports two-stage training:
- Stage 1: Contrastive pre-training with NT-Xent loss
- Stage 2: Joint fine-tuning with clustering + contrastive loss

Reference: Implementation Guide Sections 2-4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoders import ResidualSpatialEncoder, TCNTemporalEncoder, CNNTCNHybridEncoder
    from .fusion import AdaptiveFusionLayer
    from .projection import ProjectionHead
except ImportError:
    from encoders import ResidualSpatialEncoder, TCNTemporalEncoder, CNNTCNHybridEncoder
    from fusion import AdaptiveFusionLayer
    from projection import ProjectionHead


class MultiViewEncoder(nn.Module):
    """
    Multi-view encoder combining specialized encoders with adaptive fusion.

    This is the core encoder module that processes input time series through
    multiple specialized encoders and fuses their outputs.

    Args:
        input_channels (int): Number of input feature channels (default: 3)
        d_z (int): Latent dimension (default: 128)
        num_clusters (int): Number of clusters for clustering head (default: 8)
        use_hybrid_encoder (bool): If True, use 3 encoders; else 2 (default: False)

    Shape:
        - Input: (batch, 3, 127)
        - Output (latent): (batch, d_z)
        - Output (projection): (batch, d_h)

    Example:
        >>> model = MultiViewEncoder(input_channels=3, d_z=128, num_clusters=8)
        >>> x = torch.randn(32, 3, 127)
        >>> z = model(x)
        >>> z.shape
        torch.Size([32, 128])
    """

    def __init__(
        self,
        input_channels: int = 3,
        d_z: int = None,
        latent_dim: int = None,  # Alias for d_z (for test compatibility)
        num_clusters: int = 8,
        use_hybrid_encoder: bool = False,
        seq_length: int = 127
    ):
        super().__init__()

        # Handle both d_z and latent_dim parameter names
        if latent_dim is not None:
            d_z = latent_dim
        elif d_z is None:
            d_z = 128  # Default value

        self.input_channels = input_channels
        self.d_z = d_z
        self.latent_dim = d_z  # Alias for compatibility
        self.num_clusters = num_clusters
        self.use_hybrid_encoder = use_hybrid_encoder
        self.seq_length = seq_length

        # Encoder modules
        self.encoder_spatial = ResidualSpatialEncoder(
            input_channels=input_channels,
            d_z=d_z,
            seq_length=seq_length
        )

        self.encoder_temporal = TCNTemporalEncoder(
            input_channels=input_channels,
            d_z=d_z,
            seq_length=seq_length
        )

        # Alias for test compatibility
        self.encoder_tcn = self.encoder_temporal

        if use_hybrid_encoder:
            self.encoder_hybrid = CNNTCNHybridEncoder(
                input_channels=input_channels,
                d_z=d_z,
                seq_length=seq_length
            )
        else:
            self.encoder_hybrid = None

        # Fusion layer
        self.fusion = AdaptiveFusionLayer(
            d_z=d_z,
            use_hybrid_encoder=use_hybrid_encoder
        )

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            d_z=d_z,
            d_h=d_z,  # Same dimension for simplicity
            use_bottleneck=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoders and fusion.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Fused latent vector z of shape (batch, d_z)

        Notes:
            - This is the default forward method
            - Returns latent z (used for clustering)
            - Does NOT return projection h (use forward_projection for that)
        """
        # Encode through specialized encoders
        z_spatial = self.encoder_spatial(x)  # (batch, d_z)
        z_temporal = self.encoder_temporal(x)  # (batch, d_z)

        if self.use_hybrid_encoder:
            z_fused_intermediate = self.encoder_hybrid(x)  # (batch, d_z)
        else:
            z_fused_intermediate = None

        # Fuse encoder outputs
        z = self.fusion(z_spatial, z_temporal, z_fused_intermediate)  # (batch, d_z)

        return z

    def forward_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning latent vector z.

        This is an alias for forward() for clarity.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Latent vector z of shape (batch, d_z)
        """
        return self.forward(x)

    def forward_projection(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning both latent z and projection h.

        Used during contrastive pre-training (Stage 1).

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tuple of (z, h):
                - z: Latent vector of shape (batch, d_z)
                - h: Projection vector of shape (batch, d_h)
        """
        # Get latent vector
        z = self.forward(x)

        # Project to contrastive space
        h = self.projection_head(z)

        return z, h

    def get_encoder_outputs(self, x: torch.Tensor) -> dict:
        """
        Get individual encoder outputs for analysis/visualization.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Dictionary containing:
                - 'z_spatial': Spatial encoder output
                - 'z_temporal': Temporal encoder output
                - 'z_hybrid': Hybrid encoder output (if enabled)
                - 'z_fused': Final fused output
                - 'attention_weights': Fusion attention weights
        """
        # Encode through specialized encoders
        z_spatial = self.encoder_spatial(x)
        z_temporal = self.encoder_temporal(x)

        if self.use_hybrid_encoder:
            z_hybrid = self.encoder_hybrid(x)
        else:
            z_hybrid = None

        # Get fusion output
        z_fused = self.fusion(z_spatial, z_temporal, z_hybrid)

        # Get attention weights
        attention_weights = self.fusion.get_attention_weights(
            z_spatial, z_temporal, z_hybrid
        )

        return {
            'z_spatial': z_spatial,
            'z_temporal': z_temporal,
            'z_hybrid': z_hybrid,
            'z_fused': z_fused,
            'attention_weights': attention_weights
        }


class UCLTSCModel(nn.Module):
    """
    Complete UCL-TSC model with clustering head.

    This extends MultiViewEncoder with a clustering head containing
    learnable centroids for Stage 2 training.

    Args:
        input_channels (int): Number of input feature channels (default: 3)
        d_z (int): Latent dimension (default: 128)
        num_clusters (int): Number of clusters (default: 8)
        use_hybrid_encoder (bool): If True, use 3 encoders (default: False)

    Shape:
        - Input: (batch, 3, 127)
        - Latent output: (batch, d_z)
        - Cluster assignments: (batch,) - indices
        - Cluster distances: (batch, num_clusters)

    Example:
        >>> model = UCLTSCModel(input_channels=3, d_z=128, num_clusters=8)
        >>> x = torch.randn(32, 3, 127)
        >>> z, cluster_ids = model(x)
        >>> z.shape, cluster_ids.shape
        (torch.Size([32, 128]), torch.Size([32]))
    """

    def __init__(
        self,
        input_channels: int = 3,
        d_z: int = 128,
        num_clusters: int = 8,
        use_hybrid_encoder: bool = False,
        seq_length: int = 127
    ):
        super().__init__()

        # Encoder
        self.encoder = MultiViewEncoder(
            input_channels=input_channels,
            d_z=d_z,
            num_clusters=num_clusters,
            use_hybrid_encoder=use_hybrid_encoder,
            seq_length=seq_length
        )

        # Clustering head: learnable centroids
        # Initialized randomly, will be set via K-Means++ before Stage 2
        self.centroids = nn.Parameter(torch.randn(num_clusters, d_z))

        # Store parameters
        self.input_channels = input_channels
        self.d_z = d_z
        self.num_clusters = num_clusters
        self.use_hybrid_encoder = use_hybrid_encoder
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through encoder and clustering.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tuple of (z, cluster_ids):
                - z: L2-normalized latent vector of shape (batch, d_z)
                - cluster_ids: Cluster assignments of shape (batch,)
        """
        # Get latent vector
        z = self.encoder(x)

        # L2 normalize latent vectors
        z_normalized = F.normalize(z, p=2, dim=1)

        # Get cluster assignments
        cluster_ids = self.get_cluster_assignment(z_normalized)

        return z_normalized, cluster_ids

    def get_cluster_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute cluster assignments from latent vectors.

        Args:
            z: Latent vectors of shape (batch, d_z)
                (Should be L2-normalized for correct distances)

        Returns:
            Cluster assignments of shape (batch,)
        """
        # L2 normalize centroids
        centroids_normalized = F.normalize(self.centroids, p=2, dim=1)

        # Compute Euclidean distances to all centroids
        # (batch, num_clusters)
        distances = torch.cdist(z, centroids_normalized, p=2)

        # Assign to nearest centroid
        cluster_ids = distances.argmin(dim=1)  # (batch,)

        return cluster_ids

    def get_cluster_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute distances to all cluster centroids.

        Args:
            z: Latent vectors of shape (batch, d_z)
                (Should be L2-normalized for correct distances)

        Returns:
            Distances to all centroids of shape (batch, num_clusters)
        """
        # L2 normalize centroids
        centroids_normalized = F.normalize(self.centroids, p=2, dim=1)

        # Compute Euclidean distances
        distances = torch.cdist(z, centroids_normalized, p=2)

        return distances

    def normalize_centroids(self):
        """
        Normalize centroids to unit hypersphere (L2 norm = 1).

        CRITICAL: This must be called after EVERY optimizer step during Stage 2.

        This ensures centroids remain on the unit hypersphere, maintaining
        geometric consistency with L2-normalized latent vectors.
        """
        with torch.no_grad():
            self.centroids.data = F.normalize(self.centroids.data, p=2, dim=1)

    def initialize_centroids(self, z: torch.Tensor):
        """
        Initialize centroids using K-Means++ algorithm.

        This should be called before Stage 2 training using latent vectors
        from the best Stage 1 checkpoint.

        Args:
            z: Latent vectors from training data of shape (N, d_z)
                (Should be L2-normalized)

        Notes:
            - Uses K-Means++ for intelligent initialization
            - Centroids are automatically normalized after initialization
        """
        from sklearn.cluster import KMeans

        # Convert to numpy
        z_np = z.detach().cpu().numpy()

        # K-Means++ initialization
        kmeans = KMeans(
            n_clusters=self.num_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(z_np)

        # Set centroids
        centroids_init = torch.from_numpy(kmeans.cluster_centers_).float()
        self.centroids.data = centroids_init.to(self.centroids.device)

        # Normalize centroids
        self.normalize_centroids()

        print(f"Initialized {self.num_clusters} centroids using K-Means++")

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        gamma: float = 5.0
    ) -> tuple:
        """
        Predict cluster assignments with confidence scores.

        This is the primary inference method that returns both cluster assignments
        and confidence scores, as specified in Design Document Section 4.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)
            gamma: Calibrated gamma parameter from confidence calibration
                  (should be loaded from saved calibration results)

        Returns:
            Tuple of (cluster_ids, confidence_scores, metrics):
                - cluster_ids: Cluster assignments of shape (batch,)
                - confidence_scores: Confidence scores in [0, 1] of shape (batch,)
                - metrics: Dictionary with:
                    - 'z_normalized': L2-normalized latent vectors
                    - 'D_assigned': Distances to assigned centroids
                    - 'D_sep': Distances to second-closest centroids
                    - 'margin': D_sep - D_assigned

        Example:
            >>> model = UCLTSCModel(...)
            >>> # After training and calibration
            >>> x_new = torch.randn(10, 3, 127)
            >>> cluster_ids, confidence_scores, metrics = model.predict_with_confidence(x_new, gamma=5.0)
            >>> print(f"High confidence: {(confidence_scores >= 0.7).sum().item()}/{len(confidence_scores)}")
        """
        # Import here to avoid circular dependency
        try:
            from src.evaluation.confidence_scoring import ConfidenceScorer
        except ImportError:
            # Try relative import for standalone testing
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.evaluation.confidence_scoring import ConfidenceScorer

        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            # Get latent vector (normalized)
            z = self.encoder(x)
            z_normalized = F.normalize(z, p=2, dim=1)

            # Normalize centroids
            centroids_normalized = F.normalize(self.centroids, p=2, dim=1)

            # Compute confidence scores
            scorer = ConfidenceScorer(gamma=gamma)
            cluster_ids, confidence_scores, conf_metrics = scorer.compute_confidence(
                z_normalized,
                centroids_normalized
            )

            # Add latent vectors to metrics
            conf_metrics['z_normalized'] = z_normalized

        return cluster_ids, confidence_scores, conf_metrics


def test_ucl_tsc_model():
    """
    Unit tests for UCL-TSC model.

    Tests:
    1. MultiViewEncoder output shapes
    2. Projection head integration
    3. Full model with clustering
    4. Centroid normalization
    5. K-Means++ initialization
    """
    print("Testing UCL-TSC Model...")

    batch_size, channels, seq_len = 32, 3, 127
    d_z, num_clusters = 128, 8

    # Test 1: MultiViewEncoder (2-encoder mode)
    print("\n[Test 1] MultiViewEncoder (2 encoders)")
    encoder_2 = MultiViewEncoder(
        input_channels=channels,
        d_z=d_z,
        num_clusters=num_clusters,
        use_hybrid_encoder=False
    )

    x = torch.randn(batch_size, channels, seq_len)
    z = encoder_2(x)

    assert z.shape == (batch_size, d_z), f"Shape mismatch: {z.shape}"
    print(f"  [PASS] Output shape: {z.shape}")

    # Test 2: Projection head integration
    print("\n[Test 2] Projection head integration")
    z, h = encoder_2.forward_projection(x)

    assert z.shape == (batch_size, d_z), f"Latent shape mismatch: {z.shape}"
    assert h.shape == (batch_size, d_z), f"Projection shape mismatch: {h.shape}"
    print(f"  [PASS] Latent z shape: {z.shape}")
    print(f"  [PASS] Projection h shape: {h.shape}")

    # Test 3: MultiViewEncoder (3-encoder mode)
    print("\n[Test 3] MultiViewEncoder (3 encoders)")
    encoder_3 = MultiViewEncoder(
        input_channels=channels,
        d_z=d_z,
        num_clusters=num_clusters,
        use_hybrid_encoder=True
    )

    z_3 = encoder_3(x)
    assert z_3.shape == (batch_size, d_z), f"Shape mismatch: {z_3.shape}"
    print(f"  [PASS] Output shape: {z_3.shape}")

    # Test 4: Get encoder outputs
    print("\n[Test 4] Individual encoder outputs")
    outputs = encoder_3.get_encoder_outputs(x)

    assert 'z_spatial' in outputs
    assert 'z_temporal' in outputs
    assert 'z_hybrid' in outputs
    assert 'z_fused' in outputs
    assert 'attention_weights' in outputs

    print(f"  [PASS] z_spatial shape: {outputs['z_spatial'].shape}")
    print(f"  [PASS] z_temporal shape: {outputs['z_temporal'].shape}")
    print(f"  [PASS] z_hybrid shape: {outputs['z_hybrid'].shape}")
    print(f"  [PASS] z_fused shape: {outputs['z_fused'].shape}")
    print(f"  [PASS] attention_weights shape: {outputs['attention_weights'].shape}")
    print(f"  [INFO] Mean attention weights: {outputs['attention_weights'].mean(dim=0)}")

    # Test 5: Full UCL-TSC model
    print("\n[Test 5] Full UCL-TSC model")
    model = UCLTSCModel(
        input_channels=channels,
        d_z=d_z,
        num_clusters=num_clusters,
        use_hybrid_encoder=False
    )

    z_norm, cluster_ids = model(x)

    assert z_norm.shape == (batch_size, d_z), f"Latent shape mismatch: {z_norm.shape}"
    assert cluster_ids.shape == (batch_size,), f"Cluster IDs shape mismatch: {cluster_ids.shape}"
    print(f"  [PASS] Latent z shape: {z_norm.shape}")
    print(f"  [PASS] Cluster IDs shape: {cluster_ids.shape}")
    print(f"  [INFO] Unique clusters: {cluster_ids.unique().tolist()}")

    # Test 6: L2 normalization
    print("\n[Test 6] L2 normalization")
    norms = torch.norm(z_norm, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), \
        "Latent vectors not L2-normalized"
    print(f"  [PASS] Latent norms (should be 1.0): mean={norms.mean():.6f}, std={norms.std():.6f}")

    # Test 7: Centroid normalization
    print("\n[Test 7] Centroid normalization")
    model.normalize_centroids()
    centroid_norms = torch.norm(model.centroids, p=2, dim=1)
    assert torch.allclose(centroid_norms, torch.ones(num_clusters), atol=1e-5), \
        "Centroids not L2-normalized"
    print(f"  [PASS] Centroid norms (should be 1.0): mean={centroid_norms.mean():.6f}")

    # Test 8: K-Means++ initialization
    print("\n[Test 8] K-Means++ initialization")
    # Generate some latent vectors
    z_train = torch.randn(1000, d_z)
    z_train_norm = F.normalize(z_train, p=2, dim=1)

    model.initialize_centroids(z_train_norm)

    # Check centroids are normalized
    centroid_norms = torch.norm(model.centroids, p=2, dim=1)
    assert torch.allclose(centroid_norms, torch.ones(num_clusters), atol=1e-5), \
        "Centroids not normalized after K-Means++"
    print(f"  [PASS] Centroids initialized and normalized")

    # Test 9: Parameter count
    print("\n[Test 9] Parameter count")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    params = count_parameters(model)
    print(f"  Total parameters: {params:,}")

    print("\n[SUCCESS] All UCL-TSC model tests passed!")
    return True


if __name__ == '__main__':
    test_ucl_tsc_model()
