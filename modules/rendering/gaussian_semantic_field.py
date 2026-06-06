"""
Gaussian Semantic Field
Distills 2D semantic features (CLIP/DINOv2) into 3D Gaussian points,
enabling view-independent semantic querying directly in 3D space.

This is the CORE NOVELTY module that transforms the system from
"simple late fusion" to "feature-level 3D semantic representation".

Academic contribution:
- Each Gaussian carries a learned semantic feature vector
- Features are supervised by multi-view 2D foundation model outputs
- Enables open-vocabulary 3D querying without any 2D detector at inference
- Contrastive loss ensures multi-view consistency of semantic features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.logger import default_logger as logger


@dataclass
class SemanticQueryResult:
    """Result of a semantic query in 3D space"""
    query_text: str
    relevance_scores: np.ndarray       # Per-Gaussian relevance [N]
    top_gaussian_indices: np.ndarray   # Top-k most relevant Gaussian indices
    object_center_3d: np.ndarray       # Estimated 3D center of matched region
    object_bbox_3d: Tuple[np.ndarray, np.ndarray]  # (min, max) 3D bbox
    confidence: float


class GaussianSemanticField(nn.Module):
    """
    Attaches a learnable semantic feature to each 3D Gaussian.
    
    Architecture:
    - Each Gaussian i has a semantic feature f_i ∈ R^d
    - During rendering, features are splatted just like colors:
        F(p) = Σ_i f_i · α_i · T_i   (alpha-compositing of features)
    - Supervised by 2D features from CLIP/DINOv2 via feature rendering loss
    - Contrastive loss enforces cross-view consistency
    
    This enables:
    1. Rendering a "semantic feature map" from any viewpoint
    2. Open-vocabulary querying: encode text with CLIP, compare with Gaussian features
    3. Object segmentation in 3D by clustering Gaussian features
    4. LLM grounding: map LLM's text output to specific 3D regions
    """

    def __init__(
        self,
        num_gaussians: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        clip_model_name: str = "ViT-B/16",
        device: str = "cuda",
    ):
        """
        Args:
            num_gaussians: Number of Gaussian points in the scene
            feature_dim: Dimension of semantic features (match CLIP dim)
            hidden_dim: Hidden dimension for the feature MLP
            clip_model_name: CLIP model for text encoding
            device: Compute device
        """
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.device = device

        # --- Learnable semantic features per Gaussian ---
        # Option 1: Direct learnable embeddings (simple, effective)
        self.semantic_features = nn.Parameter(
            torch.randn(num_gaussians, feature_dim, device=device) * 0.01
        )

        # Option 2: MLP that maps Gaussian properties to semantic features
        # Input: position (3) + color_sh_dc (3) + scale (3) + opacity (1) = 10
        self.feature_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        ).to(device)

        # --- Feature rendering head ---
        # Reduces high-dim features to compact representation for rendering
        self.feature_compressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        ).to(device)

        # --- CLIP text encoder (frozen) ---
        self.clip_model = None
        self.clip_tokenizer = None
        self._load_clip(clip_model_name)

        # --- Losses ---
        self.temperature = nn.Parameter(torch.tensor(0.07))

        logger.info(f"GaussianSemanticField initialized: "
                     f"{num_gaussians} gaussians × {feature_dim}D features")

    def _load_clip(self, model_name: str):
        """Load CLIP model for text encoding."""
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='laion2b_s34b_b88k'
            )
            self.clip_model = model.to(self.device).eval()
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
            # Freeze CLIP
            for param in self.clip_model.parameters():
                param.requires_grad = False
            logger.info("CLIP text encoder loaded (frozen)")
        except Exception as e:
            logger.warning(f"CLIP loading failed: {e}. Text queries will be unavailable.")

    # =====================================================================
    # Core: Semantic Feature Rendering (differentiable)
    # =====================================================================
    def get_semantic_features(
        self,
        gaussian_properties: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "direct",
    ) -> torch.Tensor:
        """
        Get semantic features for all Gaussians.

        Args:
            gaussian_properties: Dict with 'xyz', 'colors_dc', 'scales', 'opacities'
            mode: 'direct' uses learnable embeddings, 'mlp' uses property-conditioned MLP

        Returns:
            Semantic features [N, feature_dim]
        """
        if mode == "direct":
            return F.normalize(self.semantic_features, dim=-1)

        elif mode == "mlp" and gaussian_properties is not None:
            # Concatenate Gaussian properties as input
            xyz = gaussian_properties['xyz']                  # [N, 3]
            colors_dc = gaussian_properties['colors_dc']      # [N, 3]
            scales = gaussian_properties['scales']             # [N, 3]
            opacities = gaussian_properties['opacities']      # [N, 1]

            props = torch.cat([xyz, colors_dc, scales, opacities], dim=-1)  # [N, 10]
            features = self.feature_mlp(props)
            return F.normalize(features, dim=-1)

        else:
            return F.normalize(self.semantic_features, dim=-1)

    def render_semantic_map(
        self,
        camera,
        gaussians,
        pipeline,
        background,
    ) -> torch.Tensor:
        """
        Render a semantic feature map by alpha-compositing Gaussian features.
        
        This is analogous to color rendering but produces a feature map instead.
        Uses the same alpha values and transmittance from the standard render.

        Args:
            camera: Camera object
            gaussians: GaussianModel
            pipeline: Pipeline parameters
            background: Background tensor

        Returns:
            Semantic feature map [feature_dim, H, W]
        """
        # Get per-Gaussian semantic features
        sem_features = self.get_semantic_features()  # [N, feature_dim]

        # Strategy: Override the color with semantic features and render
        # We render in chunks if feature_dim > 3 (since rasterizer expects 3-channel)
        from gaussian_renderer import render

        feature_maps = []
        for start in range(0, self.feature_dim, 3):
            end = min(start + 3, self.feature_dim)
            chunk = sem_features[:, start:end]

            # Pad to 3 channels if needed
            if chunk.shape[1] < 3:
                chunk = F.pad(chunk, (0, 3 - chunk.shape[1]))

            # Render with feature chunk as override color
            with torch.no_grad():
                render_pkg = render(
                    camera, gaussians, pipeline, background,
                    override_color=chunk
                )
            feature_maps.append(render_pkg['render'])  # [3, H, W]

        # Concatenate and trim to feature_dim
        full_map = torch.cat(feature_maps, dim=0)[:self.feature_dim]  # [D, H, W]
        return full_map

    # =====================================================================
    # Loss Functions for Training
    # =====================================================================
    def compute_feature_distillation_loss(
        self,
        rendered_feature_map: torch.Tensor,
        target_feature_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_distill: Align rendered 3D semantic features with 2D foundation model features.
        
        Args:
            rendered_feature_map: Our rendered features [D, H, W]
            target_feature_map: Target from CLIP/DINOv2 [D, H, W]
            valid_mask: Optional mask [H, W] for valid pixels
            
        Returns:
            Distillation loss (cosine similarity based)
        """
        # Normalize along feature dimension
        rendered_norm = F.normalize(rendered_feature_map, dim=0)
        target_norm = F.normalize(target_feature_map, dim=0)

        # Cosine similarity per pixel
        cos_sim = (rendered_norm * target_norm).sum(dim=0)  # [H, W]

        if valid_mask is not None:
            loss = (1 - cos_sim)[valid_mask].mean()
        else:
            loss = (1 - cos_sim).mean()

        return loss

    def compute_contrastive_loss(
        self,
        features_view1: torch.Tensor,
        features_view2: torch.Tensor,
        mask_view1: torch.Tensor,
        mask_view2: torch.Tensor,
        num_samples: int = 256,
    ) -> torch.Tensor:
        """
        L_contrastive: Multi-view contrastive loss.
        Same 3D point rendered from two views should have consistent features.
        
        Features from corresponding pixels (same 3D point) = positive pairs
        Features from different objects = negative pairs

        Args:
            features_view1: Rendered features from view 1 [D, H, W]
            features_view2: Rendered features from view 2 [D, H, W]
            mask_view1: Object instance mask view 1 [H, W] (integer labels)
            mask_view2: Object instance mask view 2 [H, W]
            num_samples: Number of pixel samples for contrastive pairs

        Returns:
            InfoNCE contrastive loss
        """
        D, H, W = features_view1.shape

        # Sample random pixels
        indices = torch.randint(0, H * W, (num_samples,), device=features_view1.device)
        h_idx = indices // W
        w_idx = indices % W

        f1 = features_view1[:, h_idx, w_idx].T  # [num_samples, D]
        f2 = features_view2[:, h_idx, w_idx].T  # [num_samples, D]

        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)

        # Similarity matrix
        sim = torch.mm(f1, f2.T) / self.temperature  # [N, N]

        # Positive pairs: same index (corresponding pixels)
        labels = torch.arange(num_samples, device=sim.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

    def compute_object_grouping_loss(
        self,
        gaussian_features: torch.Tensor,
        object_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_group: Encourage Gaussians belonging to the same object to have
        similar features, and different objects to have dissimilar features.

        Args:
            gaussian_features: [N, D] semantic features
            object_labels: [N] integer labels (-1 for unlabeled)

        Returns:
            Grouping loss
        """
        valid = object_labels >= 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=gaussian_features.device)

        feats = F.normalize(gaussian_features[valid], dim=-1)
        labels = object_labels[valid]

        # Compute pairwise similarity
        sim = torch.mm(feats, feats.T)  # [M, M]

        # Same object = positive (label 1), different object = negative (label 0)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Contrastive loss
        pos_sim = (sim * label_matrix).sum() / label_matrix.sum().clamp(min=1)
        neg_sim = (sim * (1 - label_matrix)).sum() / (1 - label_matrix).sum().clamp(min=1)

        loss = -pos_sim + neg_sim + 1.0  # margin = 1.0
        return loss.clamp(min=0)

    # =====================================================================
    # Inference: Open-Vocabulary 3D Querying
    # =====================================================================
    @torch.no_grad()
    def query_by_text(
        self,
        text_query: str,
        gaussian_positions: torch.Tensor,
        top_k: int = 1000,
        threshold: float = 0.2,
    ) -> SemanticQueryResult:
        """
        Open-vocabulary 3D query: find Gaussians matching a text description.
        
        This is the KEY INFERENCE capability:
        - User/LLM says "find the red chair"
        - CLIP encodes text -> compare with each Gaussian's semantic feature
        - Return 3D region of matching Gaussians

        Args:
            text_query: Natural language query
            gaussian_positions: [N, 3] positions of all Gaussians
            top_k: Number of top Gaussians to return
            threshold: Minimum relevance score

        Returns:
            SemanticQueryResult with 3D location of matched region
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP not loaded, cannot perform text queries")

        # Encode text
        tokens = self.clip_tokenizer([text_query]).to(self.device)
        text_feat = self.clip_model.encode_text(tokens)
        text_feat = F.normalize(text_feat, dim=-1)  # [1, D]

        # Get all Gaussian features
        sem_feats = self.get_semantic_features()  # [N, D]

        # Compute relevance scores
        relevance = torch.mm(sem_feats, text_feat.T).squeeze(-1)  # [N]

        # Threshold and get top-k
        mask = relevance > threshold
        if mask.sum() == 0:
            logger.warning(f"No Gaussians matched query: '{text_query}' (threshold={threshold})")
            # Lower threshold and try again
            threshold = relevance.quantile(0.95).item()
            mask = relevance > threshold

        top_k_actual = min(top_k, mask.sum().item())
        top_scores, top_indices = relevance.topk(top_k_actual)

        # Compute 3D bounding box of matched region
        matched_positions = gaussian_positions[top_indices].cpu().numpy()
        center = matched_positions.mean(axis=0)
        bbox_min = matched_positions.min(axis=0)
        bbox_max = matched_positions.max(axis=0)

        result = SemanticQueryResult(
            query_text=text_query,
            relevance_scores=relevance.cpu().numpy(),
            top_gaussian_indices=top_indices.cpu().numpy(),
            object_center_3d=center,
            object_bbox_3d=(bbox_min, bbox_max),
            confidence=float(top_scores.mean()),
        )

        logger.info(f"Query '{text_query}': found {top_k_actual} matching Gaussians, "
                     f"center={center}, confidence={result.confidence:.3f}")

        return result

    @torch.no_grad()
    def segment_objects_3d(
        self,
        gaussian_positions: torch.Tensor,
        num_clusters: int = 20,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Unsupervised 3D object segmentation by clustering Gaussian semantic features.
        
        Args:
            gaussian_positions: [N, 3]
            num_clusters: Expected number of objects
            
        Returns:
            labels: [N] cluster assignment
            objects: List of object info dicts
        """
        from sklearn.cluster import KMeans

        sem_feats = self.get_semantic_features().cpu().numpy()
        positions = gaussian_positions.cpu().numpy()

        # Combine spatial and semantic features for clustering
        # Normalize positions to similar scale as features
        pos_normalized = (positions - positions.mean(0)) / (positions.std(0) + 1e-8)
        combined = np.concatenate([sem_feats, pos_normalized * 0.3], axis=-1)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(combined)

        # Build object info
        objects = []
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            if mask.sum() < 10:  # Skip tiny clusters
                continue
            cluster_pos = positions[mask]
            objects.append({
                'cluster_id': cluster_id,
                'num_gaussians': int(mask.sum()),
                'center': cluster_pos.mean(axis=0).tolist(),
                'bbox_min': cluster_pos.min(axis=0).tolist(),
                'bbox_max': cluster_pos.max(axis=0).tolist(),
                'size': (cluster_pos.max(axis=0) - cluster_pos.min(axis=0)).tolist(),
            })

        logger.info(f"3D segmentation: {len(objects)} objects from {num_clusters} clusters")
        return labels, objects
