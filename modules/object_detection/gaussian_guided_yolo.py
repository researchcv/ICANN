"""
Gaussian-Guided YOLO Detection
Leverages 3DGS geometric priors to enhance YOLO detection through
bidirectional 3D-2D information flow: 3D proposal generation,
adaptive-threshold detection, multi-view voting, and Gaussian
attribute consistency verification.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from .yolo_detector import YOLODetector
from .detection_result import Detection, DetectionResult
from ..utils.camera_utils import CameraUtils
from ..utils.logger import default_logger as logger


@dataclass
class Proposal3D:
    """A 3D region proposal derived from Gaussian density clustering."""
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    gaussian_count: int
    mean_opacity: float


@dataclass
class VerifiedDetection:
    """Detection verified through multi-view consensus and Gaussian consistency."""
    detection: Detection
    position_3d: np.ndarray
    gaussian_indices: np.ndarray
    supporting_views: List[int] = field(default_factory=list)
    color_variance: float = 0.0
    scale_variance: float = 0.0


class GaussianGuidedYOLO:
    """
    Enhances YOLO detection by exploiting 3DGS geometric priors.

    Pipeline:
      Phase A (3D->2D): Gaussian density clustering -> 3D proposals -> project to 2D RoIs
                         -> adaptive confidence thresholds per region
      Phase B (2D->3D): Multi-view voting -> Gaussian attribute consistency check
                         -> split/merge refinement -> missed detection recovery
    """

    def __init__(
        self,
        yolo_detector: YOLODetector,
        renderer,
        *,
        proposal_eps: float = 0.5,
        proposal_min_samples: int = 50,
        opacity_floor: float = 0.1,
        roi_conf_threshold: float = 0.15,
        bg_conf_threshold: float = 0.40,
        min_supporting_views: int = 2,
        color_var_ceiling: float = 0.15,
        scale_var_ceiling: float = 0.5,
        merge_iou_threshold: float = 0.5,
    ):
        self.yolo = yolo_detector
        self.renderer = renderer

        self.proposal_eps = proposal_eps
        self.proposal_min_samples = proposal_min_samples
        self.opacity_floor = opacity_floor
        self.roi_conf_threshold = roi_conf_threshold
        self.bg_conf_threshold = bg_conf_threshold
        self.min_supporting_views = min_supporting_views
        self.color_var_ceiling = color_var_ceiling
        self.scale_var_ceiling = scale_var_ceiling
        self.merge_iou_threshold = merge_iou_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all_views(
        self,
        image_paths: List[str],
        cameras: List,
    ) -> Tuple[List[DetectionResult], List[VerifiedDetection]]:
        """
        Run the full Gaussian-guided detection pipeline across all views.

        Returns:
            enhanced_results: per-view DetectionResult with refined detections
            verified_objects: cross-view verified 3D object list
        """
        if not image_paths or not cameras:
            raise ValueError("image_paths and cameras must be non-empty")
        if len(image_paths) != len(cameras):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and cameras ({len(cameras)}) length mismatch"
            )

        gaussians = self.renderer.gaussians

        # Phase A: generate 3D proposals from Gaussian density
        proposals = self._generate_3d_proposals(gaussians)
        logger.info(f"Generated {len(proposals)} 3D proposals from Gaussian density clustering")

        # Phase A continued: per-view adaptive detection
        all_results: List[DetectionResult] = []
        all_lifted: List[Dict] = []

        for view_id, (img_path, camera) in enumerate(zip(image_paths, cameras)):
            rois_2d = self._project_proposals_to_2d(proposals, camera)
            det_result = self._detect_with_adaptive_threshold(
                img_path, camera, view_id, rois_2d
            )

            depth_map = self.renderer.render_depth_map(camera)
            lifted = self._lift_detections(det_result, depth_map, camera, view_id)

            all_results.append(det_result)
            all_lifted.extend(lifted)

        logger.info(
            f"Phase A complete: {sum(len(r) for r in all_results)} raw detections "
            f"across {len(all_results)} views"
        )

        # Phase B: multi-view voting
        verified = self._multiview_vote(all_lifted, gaussians)
        logger.info(f"Phase B voting: {len(verified)} objects pass multi-view consensus")

        # Phase B continued: Gaussian attribute consistency
        verified = self._check_gaussian_consistency(verified, gaussians)

        # Phase B continued: recover missed detections
        enhanced_results = self._recover_missed_detections(
            all_results, verified, cameras
        )

        return enhanced_results, verified

    # ------------------------------------------------------------------
    # Phase A: 3D Proposal Generation
    # ------------------------------------------------------------------

    def _generate_3d_proposals(self, gaussians) -> List[Proposal3D]:
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()
        scaling = gaussians.get_scaling.detach().cpu().numpy()

        # filter out near-transparent gaussians
        fg_mask = opacity > self.opacity_floor
        if fg_mask.sum() == 0:
            logger.warning("No foreground Gaussians above opacity floor")
            return []

        fg_xyz = xyz[fg_mask]
        fg_opacity = opacity[fg_mask]

        # adaptive eps based on scene scale
        scene_extent = np.ptp(fg_xyz, axis=0).max()
        eps = max(self.proposal_eps, scene_extent * 0.02)

        clustering = DBSCAN(eps=eps, min_samples=self.proposal_min_samples).fit(fg_xyz)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}

        proposals = []
        for label in unique_labels:
            mask = labels == label
            cluster_xyz = fg_xyz[mask]
            cluster_opacity = fg_opacity[mask]

            proposals.append(Proposal3D(
                center=np.average(cluster_xyz, axis=0, weights=cluster_opacity),
                bbox_min=cluster_xyz.min(axis=0),
                bbox_max=cluster_xyz.max(axis=0),
                gaussian_count=int(mask.sum()),
                mean_opacity=float(cluster_opacity.mean()),
            ))

        return proposals

    def _project_proposals_to_2d(
        self, proposals: List[Proposal3D], camera
    ) -> List[Tuple[int, int, int, int]]:
        """Project 3D proposal bounding boxes to 2D RoIs in a given camera view."""
        if not proposals:
            return []

        rois = []
        img_w, img_h = camera.image_width, camera.image_height

        for prop in proposals:
            corners_3d = self._bbox_corners(prop.bbox_min, prop.bbox_max)
            pts_2d = CameraUtils.project_3d_to_2d(corners_3d, camera)
            vis_mask = CameraUtils.check_point_in_view(pts_2d, corners_3d, camera)

            if vis_mask.sum() < 2:
                continue

            visible_pts = pts_2d[vis_mask]
            x1 = max(0, int(visible_pts[:, 0].min()))
            y1 = max(0, int(visible_pts[:, 1].min()))
            x2 = min(img_w, int(visible_pts[:, 0].max()))
            y2 = min(img_h, int(visible_pts[:, 1].max()))

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                rois.append((x1, y1, x2, y2))

        return rois

    # ------------------------------------------------------------------
    # Phase A: Adaptive-Threshold Detection
    # ------------------------------------------------------------------

    def _detect_with_adaptive_threshold(
        self,
        image_path: str,
        camera,
        view_id: int,
        rois_2d: List[Tuple[int, int, int, int]],
    ) -> DetectionResult:
        """
        Run YOLO with a low global threshold, then keep/discard
        detections based on whether their center falls inside an RoI.
        """
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]

        # run YOLO once with the lower threshold to maximise recall
        effective_threshold = (
            self.roi_conf_threshold if rois_2d else self.bg_conf_threshold
        )

        results = self.yolo.model(
            image,
            conf=effective_threshold,
            iou=self.yolo.iou_threshold,
            classes=self.yolo.classes,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            in_roi = any(
                r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rois_2d
            )

            # inside a 3D-proposal RoI: keep even low-confidence detections
            # outside: require higher confidence
            if not in_roi and conf < self.bg_conf_threshold:
                continue

            detections.append(Detection(
                class_name=self.yolo.class_names[cls],
                class_id=cls,
                confidence=conf,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(float(cx), float(cy)),
            ))

        from pathlib import Path
        return DetectionResult(
            image_name=Path(image_path).name,
            image_path=str(image_path),
            view_id=view_id,
            detections=detections,
            image_size=(width, height),
        )

    # ------------------------------------------------------------------
    # Phase B: Lifting & Multi-View Voting
    # ------------------------------------------------------------------

    def _lift_detections(
        self,
        det_result: DetectionResult,
        depth_map: Optional[np.ndarray],
        camera,
        view_id: int,
    ) -> List[Dict]:
        """Lift 2D detections to 3D using depth map median within bbox center region."""
        if depth_map is None:
            return []

        lifted = []
        h, w = depth_map.shape[:2]

        for det in det_result.detections:
            x1, y1, x2, y2 = det.bbox
            # use center 50% of bbox for robust depth estimation
            cx1 = int(x1 + (x2 - x1) * 0.25)
            cy1 = int(y1 + (y2 - y1) * 0.25)
            cx2 = int(x1 + (x2 - x1) * 0.75)
            cy2 = int(y1 + (y2 - y1) * 0.75)

            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(w, cx2), min(h, cy2)

            region = depth_map[cy1:cy2, cx1:cx2]
            valid = region[(region > 0) & np.isfinite(region)]
            if len(valid) == 0:
                continue

            depth = float(np.median(valid))
            center_x, center_y = det.center
            pos_3d = CameraUtils.pixel_to_3d(center_x, center_y, depth, camera)

            lifted.append({
                "detection": det,
                "position_3d": pos_3d,
                "view_id": view_id,
                "depth": depth,
            })

        return lifted

    def _multiview_vote(
        self, all_lifted: List[Dict], gaussians
    ) -> List[VerifiedDetection]:
        """Group lifted detections by class + 3D proximity, keep those with enough view support."""
        by_class: Dict[str, List[Dict]] = {}
        for item in all_lifted:
            cls = item["detection"].class_name
            by_class.setdefault(cls, []).append(item)

        verified: List[VerifiedDetection] = []

        for cls_name, items in by_class.items():
            if len(items) < self.min_supporting_views:
                continue

            positions = np.array([it["position_3d"] for it in items])
            scene_extent = np.ptp(positions, axis=0).max()
            eps = max(0.3, scene_extent * 0.05)

            clustering = DBSCAN(eps=eps, min_samples=self.min_supporting_views).fit(positions)

            for label in set(clustering.labels_) - {-1}:
                mask = clustering.labels_ == label
                cluster_items = [it for it, m in zip(items, mask) if m]
                cluster_positions = positions[mask]

                view_ids = list({it["view_id"] for it in cluster_items})
                if len(view_ids) < self.min_supporting_views:
                    continue

                # pick highest-confidence detection as representative
                best = max(cluster_items, key=lambda it: it["detection"].confidence)
                avg_pos = cluster_positions.mean(axis=0)

                # find gaussian indices near this 3D position
                gauss_indices = self._find_gaussians_near(
                    avg_pos, gaussians, radius=eps * 1.5
                )

                det = best["detection"]
                det.position_3d = tuple(avg_pos.tolist())
                det.gaussian_indices = gauss_indices.tolist() if len(gauss_indices) > 0 else None
                det.visible_views = view_ids

                verified.append(VerifiedDetection(
                    detection=det,
                    position_3d=avg_pos,
                    gaussian_indices=gauss_indices,
                    supporting_views=view_ids,
                ))

        return verified

    # ------------------------------------------------------------------
    # Phase B: Gaussian Attribute Consistency
    # ------------------------------------------------------------------

    def _check_gaussian_consistency(
        self, verified: List[VerifiedDetection], gaussians
    ) -> List[VerifiedDetection]:
        """
        Verify each detection's Gaussian attribute coherence.
        High internal variance suggests the bbox covers multiple objects.
        """
        sh_features = gaussians.get_features.detach().cpu().numpy()
        scaling = gaussians.get_scaling.detach().cpu().numpy()

        refined: List[VerifiedDetection] = []

        for vdet in verified:
            if vdet.gaussian_indices is None or len(vdet.gaussian_indices) == 0:
                refined.append(vdet)
                continue

            ids = vdet.gaussian_indices

            # SH DC component -> approximate diffuse color
            dc = sh_features[ids, :, 0]  # [M, 3]
            color_var = float(dc.var())
            scale_vals = scaling[ids]  # [M, 3]
            scale_var = float(scale_vals.var())

            vdet.color_variance = color_var
            vdet.scale_variance = scale_var

            if color_var > self.color_var_ceiling or scale_var > self.scale_var_ceiling:
                # high variance: attempt sub-clustering
                sub_verified = self._try_split(vdet, gaussians)
                refined.extend(sub_verified)
            else:
                refined.append(vdet)

        logger.info(
            f"Gaussian consistency: {len(verified)} -> {len(refined)} detections"
        )
        return refined

    def _try_split(
        self, vdet: VerifiedDetection, gaussians
    ) -> List[VerifiedDetection]:
        """
        Attempt to split a detection whose Gaussians show high attribute variance.
        Falls back to keeping the original if sub-clustering doesn't yield valid groups.
        """
        ids = vdet.gaussian_indices
        if len(ids) < 10:
            return [vdet]

        xyz = gaussians.get_xyz.detach().cpu().numpy()[ids]
        sh_dc = gaussians.get_features.detach().cpu().numpy()[ids, :, 0]

        # cluster in joint position-color space (normalised)
        pos_norm = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-8)
        col_norm = (sh_dc - sh_dc.mean(axis=0)) / (sh_dc.std(axis=0) + 1e-8)
        features = np.hstack([pos_norm, col_norm * 0.5])

        clustering = DBSCAN(eps=0.8, min_samples=5).fit(features)
        unique_labels = set(clustering.labels_) - {-1}

        if len(unique_labels) <= 1:
            return [vdet]

        results = []
        for label in unique_labels:
            sub_mask = clustering.labels_ == label
            sub_ids = ids[sub_mask]
            sub_xyz = xyz[sub_mask]

            sub_det = VerifiedDetection(
                detection=Detection(
                    class_name=vdet.detection.class_name,
                    class_id=vdet.detection.class_id,
                    confidence=vdet.detection.confidence * 0.9,
                    bbox=vdet.detection.bbox,
                    center=vdet.detection.center,
                    position_3d=tuple(sub_xyz.mean(axis=0).tolist()),
                    gaussian_indices=sub_ids.tolist(),
                    visible_views=vdet.supporting_views,
                ),
                position_3d=sub_xyz.mean(axis=0),
                gaussian_indices=sub_ids,
                supporting_views=vdet.supporting_views,
            )
            results.append(sub_det)

        return results

    # ------------------------------------------------------------------
    # Phase B: Missed Detection Recovery
    # ------------------------------------------------------------------

    def _recover_missed_detections(
        self,
        all_results: List[DetectionResult],
        verified: List[VerifiedDetection],
        cameras: List,
    ) -> List[DetectionResult]:
        """
        For each verified 3D object, check if it should be visible in views
        where it was not detected. If so, synthesise a detection from reprojection.
        """
        enhanced = [
            DetectionResult(
                image_name=r.image_name,
                image_path=r.image_path,
                view_id=r.view_id,
                detections=list(r.detections),
                image_size=r.image_size,
            )
            for r in all_results
        ]

        recovered_count = 0
        for vdet in verified:
            for view_id, camera in enumerate(cameras):
                if view_id in vdet.supporting_views:
                    continue

                pt_2d = CameraUtils.project_3d_to_2d(
                    vdet.position_3d.reshape(1, 3), camera
                )
                vis = CameraUtils.check_point_in_view(
                    pt_2d, vdet.position_3d.reshape(1, 3), camera
                )
                if not vis[0]:
                    continue

                cx, cy = float(pt_2d[0, 0]), float(pt_2d[0, 1])
                # estimate bbox size from existing detection
                ow, oh = vdet.detection.get_width_height()
                half_w, half_h = ow / 2, oh / 2
                img_w, img_h = enhanced[view_id].image_size

                x1 = max(0.0, cx - half_w)
                y1 = max(0.0, cy - half_h)
                x2 = min(float(img_w), cx + half_w)
                y2 = min(float(img_h), cy + half_h)

                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                # avoid duplicate: check IoU with existing detections
                synth = Detection(
                    class_name=vdet.detection.class_name,
                    class_id=vdet.detection.class_id,
                    confidence=vdet.detection.confidence * 0.8,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    position_3d=vdet.detection.position_3d,
                    gaussian_indices=(
                        vdet.gaussian_indices.tolist()
                        if vdet.gaussian_indices is not None
                           and len(vdet.gaussian_indices) > 0
                        else None
                    ),
                )

                has_overlap = any(
                    synth.iou(d) > self.merge_iou_threshold
                    for d in enhanced[view_id].detections
                )
                if not has_overlap:
                    enhanced[view_id].detections.append(synth)
                    recovered_count += 1

        logger.info(f"Recovered {recovered_count} missed detections via reprojection")
        return enhanced

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
        return np.array([
            [bmin[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]],
        ])

    @staticmethod
    def _find_gaussians_near(
        center: np.ndarray, gaussians, radius: float
    ) -> np.ndarray:
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        dists = np.linalg.norm(xyz - center, axis=1)
        return np.where(dists < radius)[0]
