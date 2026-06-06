"""
Multi-View Consistency Checker
Filters false detections and recovers missed detections through cross-view verification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .detection_result import Detection, DetectionResult
from ..utils.camera_utils import CameraUtils
from ..utils.logger import default_logger as logger


@dataclass
class ConsistentDetection:
    """A detection verified across multiple views"""
    class_name: str
    class_id: int
    position_3d: np.ndarray          # Triangulated 3D position
    avg_confidence: float
    view_detections: Dict[int, Detection] = field(default_factory=dict)
    consistency_score: float = 0.0    # Cross-view consistency [0, 1]
    num_supporting_views: int = 0


class MultiViewConsistencyChecker:
    """
    Enforces multi-view consistency on 2D detections by:
    1. Lifting each detection to 3D via depth
    2. Cross-view re-projection and matching
    3. Voting: keep detections supported by >= min_views
    4. Recovery: propagate strong detections to views where they were missed
    """

    def __init__(
        self,
        renderer,
        min_supporting_views: int = 2,
        reprojection_iou_threshold: float = 0.3,
        depth_consistency_threshold: float = 0.5,
        confidence_boost: float = 0.1,
        recovery_conf_threshold: float = 0.15,
    ):
        """
        Args:
            renderer: GaussianRendererWrapper for depth maps
            min_supporting_views: Minimum views that must agree
            reprojection_iou_threshold: IoU threshold for matching reprojected box
            depth_consistency_threshold: Max allowed depth deviation (meters)
            confidence_boost: Confidence boost for multi-view verified detections
            recovery_conf_threshold: Min confidence to propagate to missing views
        """
        self.renderer = renderer
        self.min_supporting_views = min_supporting_views
        self.reproj_iou_thresh = reprojection_iou_threshold
        self.depth_consistency_thresh = depth_consistency_threshold
        self.confidence_boost = confidence_boost
        self.recovery_conf_thresh = recovery_conf_threshold

    def verify_and_enhance(
        self,
        detection_results: List[DetectionResult],
        cameras: List,
    ) -> Tuple[List[DetectionResult], List[ConsistentDetection]]:
        """
        Main entry: verify detections across views, filter bad ones, recover missed ones.

        Args:
            detection_results: Per-view YOLO detection results
            cameras: Corresponding camera objects

        Returns:
            enhanced_results: Filtered + recovered per-view detections
            consistent_objects: List of multi-view verified 3D detections
        """
        logger.info("=== Multi-View Consistency Check ===")

        # --- Phase 1: Lift all detections to 3D ---
        det_3d_records = self._lift_detections_to_3d(detection_results, cameras)
        logger.info(f"  Phase 1: Lifted {len(det_3d_records)} detections to 3D")

        # --- Phase 2: Cross-view matching via 3D proximity ---
        consistent_groups = self._group_by_3d_proximity(det_3d_records)
        logger.info(f"  Phase 2: Found {len(consistent_groups)} candidate object groups")

        # --- Phase 3: Voting — filter groups by min_supporting_views ---
        verified_objects = []
        for group in consistent_groups:
            unique_views = set(r['view_id'] for r in group)
            if len(unique_views) >= self.min_supporting_views:
                obj = self._merge_group(group)
                verified_objects.append(obj)

        logger.info(f"  Phase 3: {len(verified_objects)} objects passed voting "
                     f"(min_views={self.min_supporting_views})")

        # --- Phase 4: Re-projection recovery for missed detections ---
        enhanced_results = self._recover_missed_detections(
            detection_results, cameras, verified_objects
        )

        # --- Phase 5: Remove unverified detections ---
        enhanced_results = self._filter_unverified(
            enhanced_results, verified_objects, cameras
        )

        # Statistics
        orig_count = sum(len(r.detections) for r in detection_results)
        new_count = sum(len(r.detections) for r in enhanced_results)
        logger.info(f"  Result: {orig_count} -> {new_count} detections "
                     f"({len(verified_objects)} verified 3D objects)")

        return enhanced_results, verified_objects

    # -----------------------------------------------------------------
    # Phase 1: Lift to 3D
    # -----------------------------------------------------------------
    def _lift_detections_to_3d(
        self,
        detection_results: List[DetectionResult],
        cameras: List
    ) -> List[Dict]:
        """Lift each 2D detection to 3D using depth maps."""
        records = []
        for view_id, (det_result, camera) in enumerate(zip(detection_results, cameras)):
            depth_map = self.renderer.render_depth_map(camera)
            if depth_map is None:
                continue
            for det in det_result.detections:
                pos_3d = self._get_3d_position(det, depth_map, camera)
                if pos_3d is not None:
                    records.append({
                        'detection': det,
                        'position_3d': pos_3d,
                        'view_id': view_id,
                        'camera': camera,
                    })
        return records

    def _get_3d_position(self, det: Detection, depth_map: np.ndarray, camera) -> Optional[np.ndarray]:
        """Robust 3D position estimation using median depth in bbox region."""
        x1, y1, x2, y2 = map(int, det.bbox)
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Use central 50% region to avoid boundary noise
        cx1 = x1 + (x2 - x1) // 4
        cy1 = y1 + (y2 - y1) // 4
        cx2 = x2 - (x2 - x1) // 4
        cy2 = y2 - (y2 - y1) // 4

        region = depth_map[cy1:cy2, cx1:cx2]
        valid = region[(region > 0) & ~np.isnan(region) & ~np.isinf(region)]

        if len(valid) == 0:
            return None

        median_depth = np.median(valid)
        center_x, center_y = det.center
        return CameraUtils.pixel_to_3d(center_x, center_y, median_depth, camera)

    # -----------------------------------------------------------------
    # Phase 2: Group by 3D proximity (class-aware)
    # -----------------------------------------------------------------
    def _group_by_3d_proximity(
        self,
        records: List[Dict],
        distance_threshold: float = 1.5
    ) -> List[List[Dict]]:
        """Simple class-aware greedy grouping in 3D space."""
        by_class = defaultdict(list)
        for r in records:
            by_class[r['detection'].class_name].append(r)

        all_groups = []
        for class_name, class_records in by_class.items():
            assigned = [False] * len(class_records)
            for i, rec_i in enumerate(class_records):
                if assigned[i]:
                    continue
                group = [rec_i]
                assigned[i] = True
                for j, rec_j in enumerate(class_records):
                    if assigned[j]:
                        continue
                    dist = np.linalg.norm(rec_i['position_3d'] - rec_j['position_3d'])
                    if dist < distance_threshold:
                        group.append(rec_j)
                        assigned[j] = True
                all_groups.append(group)

        return all_groups

    # -----------------------------------------------------------------
    # Phase 3: Merge group into ConsistentDetection
    # -----------------------------------------------------------------
    def _merge_group(self, group: List[Dict]) -> ConsistentDetection:
        positions = np.array([r['position_3d'] for r in group])
        avg_pos = positions.mean(axis=0)
        confidences = [r['detection'].confidence for r in group]
        unique_views = set(r['view_id'] for r in group)

        view_dets = {}
        for r in group:
            view_dets[r['view_id']] = r['detection']

        return ConsistentDetection(
            class_name=group[0]['detection'].class_name,
            class_id=group[0]['detection'].class_id,
            position_3d=avg_pos,
            avg_confidence=float(np.mean(confidences)),
            view_detections=view_dets,
            consistency_score=len(unique_views) / max(len(group), 1),
            num_supporting_views=len(unique_views),
        )

    # -----------------------------------------------------------------
    # Phase 4: Recover missed detections via re-projection
    # -----------------------------------------------------------------
    def _recover_missed_detections(
        self,
        detection_results: List[DetectionResult],
        cameras: List,
        verified_objects: List[ConsistentDetection],
    ) -> List[DetectionResult]:
        """For each verified object, check if it should be visible in a view
        but was not detected. If so, re-project and add a synthetic detection."""
        enhanced = [
            DetectionResult(
                image_name=r.image_name,
                image_path=r.image_path,
                view_id=r.view_id,
                detections=list(r.detections),
                image_size=r.image_size,
            )
            for r in detection_results
        ]

        recovered_count = 0
        for obj in verified_objects:
            for view_id, camera in enumerate(cameras):
                if view_id in obj.view_detections:
                    continue  # Already detected in this view

                # Re-project 3D center to this view
                pt_2d = CameraUtils.project_3d_to_2d(
                    obj.position_3d.reshape(1, 3), camera
                )[0]

                # Check visibility
                visible = CameraUtils.check_point_in_view(
                    pt_2d.reshape(1, 2),
                    obj.position_3d.reshape(1, 3),
                    camera
                )
                if not visible[0]:
                    continue

                # Estimate bbox size from existing detections
                existing_sizes = []
                for det in obj.view_detections.values():
                    w, h = det.get_width_height()
                    existing_sizes.append((w, h))
                if not existing_sizes:
                    continue
                avg_w, avg_h = np.mean(existing_sizes, axis=0)

                # Create synthetic detection
                cx, cy = pt_2d
                synth_bbox = (
                    float(cx - avg_w / 2), float(cy - avg_h / 2),
                    float(cx + avg_w / 2), float(cy + avg_h / 2),
                )

                # Clamp to image bounds
                img_w, img_h = enhanced[view_id].image_size
                synth_bbox = (
                    max(0, synth_bbox[0]), max(0, synth_bbox[1]),
                    min(img_w, synth_bbox[2]), min(img_h, synth_bbox[3]),
                )

                synth_det = Detection(
                    class_name=obj.class_name,
                    class_id=obj.class_id,
                    confidence=obj.avg_confidence * 0.8,  # Slightly lower confidence
                    bbox=synth_bbox,
                    center=(float(cx), float(cy)),
                )

                if synth_det.confidence >= self.recovery_conf_thresh:
                    enhanced[view_id].detections.append(synth_det)
                    recovered_count += 1

        logger.info(f"  Phase 4: Recovered {recovered_count} missed detections via re-projection")
        return enhanced

    # -----------------------------------------------------------------
    # Phase 5: Filter unverified detections
    # -----------------------------------------------------------------
    def _filter_unverified(
        self,
        enhanced_results: List[DetectionResult],
        verified_objects: List[ConsistentDetection],
        cameras: List,
    ) -> List[DetectionResult]:
        """Remove detections that don't correspond to any verified object."""
        # Build set of verified (view_id, bbox_center) for fast lookup
        verified_positions = {}  # view_id -> list of 3D positions
        for obj in verified_objects:
            for view_id, det in obj.view_detections.items():
                if view_id not in verified_positions:
                    verified_positions[view_id] = []
                verified_positions[view_id].append(obj.position_3d)

        filtered_results = []
        removed_count = 0

        for view_id, result in enumerate(enhanced_results):
            kept = []
            for det in result.detections:
                # Check if this detection is close to any verified object's reprojection
                is_verified = self._is_detection_verified(
                    det, view_id, verified_objects, cameras
                )
                if is_verified:
                    kept.append(det)
                else:
                    removed_count += 1

            filtered_results.append(DetectionResult(
                image_name=result.image_name,
                image_path=result.image_path,
                view_id=result.view_id,
                detections=kept,
                image_size=result.image_size,
            ))

        logger.info(f"  Phase 5: Removed {removed_count} unverified detections")
        return filtered_results

    def _is_detection_verified(
        self,
        det: Detection,
        view_id: int,
        verified_objects: List[ConsistentDetection],
        cameras: List,
        iou_threshold: float = 0.2,
    ) -> bool:
        """Check if a detection matches any verified object in this view."""
        camera = cameras[view_id]

        for obj in verified_objects:
            if obj.class_name != det.class_name:
                continue

            # If this detection is directly in the verified set
            if view_id in obj.view_detections:
                if obj.view_detections[view_id] is det:
                    return True

            # Check by reprojection IoU
            pt_2d = CameraUtils.project_3d_to_2d(
                obj.position_3d.reshape(1, 3), camera
            )[0]

            # Quick center distance check
            cx, cy = det.center
            dist = np.sqrt((cx - pt_2d[0])**2 + (cy - pt_2d[1])**2)
            max_dim = max(det.get_width_height())
            if dist < max_dim * 1.5:
                return True

        return False
