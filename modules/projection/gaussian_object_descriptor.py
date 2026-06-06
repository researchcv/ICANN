"""
Gaussian Object Descriptor (GOD)
Extracts rich object-level representations from 3DGS Gaussian attributes:
opacity-weighted centroid, PCA-based extent/orientation, shape classification,
volume estimation, and dominant color from SH DC coefficients.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from scipy.spatial import cKDTree

from ..utils.logger import default_logger as logger

# SH band-0 normalization constant: Y_0^0 = 1 / (2*sqrt(pi))
_SH_C0 = 0.28209479177387814


@dataclass
class GODDescriptor:
    """Gaussian Object Descriptor for a single detected object."""
    object_id: int
    class_name: str

    centroid: np.ndarray              # opacity-weighted centroid [3]
    extent: np.ndarray                # PCA principal axis lengths [3], descending
    orientation: np.ndarray           # 3x3 rotation matrix, columns are principal axes
    shape_type: str                   # "elongated" | "flat" | "compact"

    volume: float                     # sum of individual Gaussian ellipsoid volumes
    surface_area_approx: float        # approximate total surface area
    num_gaussians: int
    density: float                    # num_gaussians / volume

    mean_opacity: float
    dominant_color: np.ndarray        # RGB [3], range [0,1]
    color_variance: float

    gaussian_indices: np.ndarray      # indices into the global Gaussian array

    def to_dict(self) -> Dict:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "centroid": self.centroid.tolist(),
            "extent": self.extent.tolist(),
            "orientation": self.orientation.tolist(),
            "shape_type": self.shape_type,
            "volume": self.volume,
            "surface_area_approx": self.surface_area_approx,
            "num_gaussians": self.num_gaussians,
            "density": self.density,
            "mean_opacity": self.mean_opacity,
            "dominant_color": self.dominant_color.tolist(),
            "color_variance": self.color_variance,
        }

    def to_text(self) -> str:
        """Human-readable summary for LLM consumption."""
        ext = self.extent
        col = self.dominant_color
        lines = [
            f"[ID {self.object_id}] {self.class_name}:",
            f"  position: ({self.centroid[0]:.2f}, {self.centroid[1]:.2f}, {self.centroid[2]:.2f})",
            f"  size: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m",
            f"  shape: {self.shape_type}, volume: {self.volume:.3f} m3",
            f"  color: R{col[0]:.2f} G{col[1]:.2f} B{col[2]:.2f}",
            f"  gaussians: {self.num_gaussians}, density: {self.density:.1f}",
        ]
        return "\n".join(lines)


class GaussianObjectDescriptorBuilder:
    """
    Builds GOD descriptors from Gaussian indices associated with detected objects.
    Requires access to the GaussianModel to read per-Gaussian attributes.
    """

    def __init__(self, gaussians):
        self._xyz = gaussians.get_xyz.detach().cpu().numpy()
        self._opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()
        self._scaling = gaussians.get_scaling.detach().cpu().numpy()
        self._rotation = gaussians.get_rotation.detach().cpu().numpy()
        self._sh_features = gaussians.get_features.detach().cpu().numpy()
        self._tree: Optional[cKDTree] = None

    @property
    def kdtree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self._xyz)
        return self._tree

    def build(
        self,
        object_id: int,
        class_name: str,
        gaussian_indices: np.ndarray,
    ) -> Optional[GODDescriptor]:
        if gaussian_indices is None or len(gaussian_indices) == 0:
            return None

        ids = np.asarray(gaussian_indices, dtype=int)
        xyz = self._xyz[ids]
        opacity = self._opacity[ids]
        scaling = self._scaling[ids]
        sh_dc = self._sh_features[ids, :, 0]

        if len(ids) < 3:
            return None

        centroid = self._weighted_centroid(xyz, opacity)
        extent, orientation = self._pca_extent(xyz, opacity)
        shape_type = self._classify_shape(extent)
        volume, surface_area = self._estimate_volume_and_area(scaling)
        dominant_color, color_var = self._extract_color(sh_dc)
        density = len(ids) / max(volume, 1e-10)

        return GODDescriptor(
            object_id=object_id,
            class_name=class_name,
            centroid=centroid,
            extent=extent,
            orientation=orientation,
            shape_type=shape_type,
            volume=volume,
            surface_area_approx=surface_area,
            num_gaussians=len(ids),
            density=density,
            mean_opacity=float(opacity.mean()),
            dominant_color=dominant_color,
            color_variance=color_var,
            gaussian_indices=ids,
        )

    def build_batch(self, objects_3d: List) -> List[GODDescriptor]:
        """Build GOD descriptors for a list of Object3D instances."""
        descriptors = []
        for obj in objects_3d:
            indices = obj.gaussian_indices
            if indices is None:
                continue
            god = self.build(obj.object_id, obj.class_name, np.asarray(indices))
            if god is not None:
                descriptors.append(god)
        logger.info(f"Built {len(descriptors)} GOD descriptors from {len(objects_3d)} objects")
        return descriptors

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_centroid(xyz: np.ndarray, opacity: np.ndarray) -> np.ndarray:
        weights = np.maximum(opacity, 0.0)
        total = weights.sum()
        if total < 1e-12:
            return xyz.mean(axis=0)
        return np.average(xyz, axis=0, weights=weights)

    @staticmethod
    def _pca_extent(
        xyz: np.ndarray, opacity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PCA on opacity-weighted point cloud.
        Returns axis lengths (descending) and 3x3 rotation whose columns
        are the principal directions.
        """
        weights = np.maximum(opacity, 0.0)
        total = weights.sum()
        if total < 1e-12:
            weights = np.ones(len(xyz))
            total = float(len(xyz))

        centroid = np.average(xyz, axis=0, weights=weights)
        centered = xyz - centroid

        # weighted covariance
        w_norm = weights / total
        cov = (centered.T * w_norm) @ centered

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order; reverse to descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 2-sigma extent covers ~95% of the distribution
        extent = 2.0 * np.sqrt(np.maximum(eigenvalues, 0.0))
        return extent, eigenvectors

    @staticmethod
    def _classify_shape(extent: np.ndarray) -> str:
        """Classify shape based on principal axis length ratios."""
        s = np.sort(extent)[::-1]
        r1 = s[0] / (s[1] + 1e-8)
        r2 = s[1] / (s[2] + 1e-8)

        if r1 > 3.0:
            return "elongated"
        if r2 > 3.0:
            return "flat"
        return "compact"

    @staticmethod
    def _estimate_volume_and_area(scaling: np.ndarray) -> Tuple[float, float]:
        """
        Each Gaussian is an ellipsoid with semi-axes given by scaling.
        Volume = (4/3) * pi * a * b * c
        Surface approximation via Knud Thomsen's formula.
        """
        # clamp to avoid negative/zero scales
        s = np.maximum(np.abs(scaling), 1e-8)
        a, b, c = s[:, 0], s[:, 1], s[:, 2]

        volumes = (4.0 / 3.0) * np.pi * a * b * c
        total_volume = float(volumes.sum())

        # Thomsen approximation: S ~ 4*pi * ((a^p*b^p + a^p*c^p + b^p*c^p)/3)^(1/p), p=1.6075
        p = 1.6075
        ap, bp, cp = a ** p, b ** p, c ** p
        surface = 4.0 * np.pi * ((ap * bp + ap * cp + bp * cp) / 3.0) ** (1.0 / p)
        total_surface = float(surface.sum())

        return total_volume, total_surface

    @staticmethod
    def _extract_color(sh_dc: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert SH DC component to RGB and compute color variance."""
        rgb = sh_dc * _SH_C0 + 0.5
        rgb = np.clip(rgb, 0.0, 1.0)
        dominant = rgb.mean(axis=0)
        variance = float(rgb.var())
        return dominant, variance

    # ------------------------------------------------------------------
    # Spatial queries using GOD
    # ------------------------------------------------------------------

    def compute_surface_distance(
        self, god_a: GODDescriptor, god_b: GODDescriptor
    ) -> float:
        """
        Minimum distance between the Gaussian point clouds of two objects.
        More accurate than centroid-to-centroid distance.
        """
        xyz_a = self._xyz[god_a.gaussian_indices]
        xyz_b = self._xyz[god_b.gaussian_indices]

        tree_b = cKDTree(xyz_b)
        dists, _ = tree_b.query(xyz_a, k=1)
        return float(dists.min())

    def compute_vertical_contact(
        self, god_a: GODDescriptor, god_b: GODDescriptor
    ) -> Dict:
        """
        Analyse whether object A sits on top of object B by comparing
        the bottom percentile of A with the top percentile of B.
        """
        xyz_a = self._xyz[god_a.gaussian_indices]
        xyz_b = self._xyz[god_b.gaussian_indices]

        a_bottom = float(np.percentile(xyz_a[:, 1], 5))
        a_top = float(np.percentile(xyz_a[:, 1], 95))
        b_bottom = float(np.percentile(xyz_b[:, 1], 5))
        b_top = float(np.percentile(xyz_b[:, 1], 95))

        contact_gap = abs(a_bottom - b_top)

        # horizontal overlap via XZ bounding box intersection
        a_xz_min = xyz_a[:, [0, 2]].min(axis=0)
        a_xz_max = xyz_a[:, [0, 2]].max(axis=0)
        b_xz_min = xyz_b[:, [0, 2]].min(axis=0)
        b_xz_max = xyz_b[:, [0, 2]].max(axis=0)

        inter_min = np.maximum(a_xz_min, b_xz_min)
        inter_max = np.minimum(a_xz_max, b_xz_max)
        inter_area = max(0.0, inter_max[0] - inter_min[0]) * max(0.0, inter_max[1] - inter_min[1])

        a_area = max(1e-8, (a_xz_max[0] - a_xz_min[0]) * (a_xz_max[1] - a_xz_min[1]))
        b_area = max(1e-8, (b_xz_max[0] - b_xz_min[0]) * (b_xz_max[1] - b_xz_min[1]))
        overlap_ratio = inter_area / min(a_area, b_area)

        return {
            "a_bottom": a_bottom,
            "a_top": a_top,
            "b_bottom": b_bottom,
            "b_top": b_top,
            "contact_gap": contact_gap,
            "horizontal_overlap": overlap_ratio,
            "is_contact": contact_gap < 0.1 and overlap_ratio > 0.2,
        }

    def compute_directional_relation(
        self, god_a: GODDescriptor, god_b: GODDescriptor
    ) -> Dict:
        """
        Determine directional relation (left/right/front/back/above/below)
        using weighted centroids instead of crude bbox centers.
        """
        delta = god_a.centroid - god_b.centroid
        dx, dy, dz = delta

        abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
        dominant_axis = np.argmax([abs_dx, abs_dy, abs_dz])

        if dominant_axis == 1:
            direction = "above" if dy > 0 else "below"
        elif dominant_axis == 0:
            direction = "right_of" if dx > 0 else "left_of"
        else:
            direction = "behind" if dz > 0 else "in_front_of"

        return {
            "direction": direction,
            "delta": delta.tolist(),
            "distance": float(np.linalg.norm(delta)),
        }
