"""
Three-Source Evidence Generator & Fusion Engine
Generates geometric, visual, and semantic evidence for spatial relations,
then fuses them into confidence-weighted scene graph edges.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .scene_graph import SceneGraph, SpatialRelation, Evidence
from ..projection.object_3d_reconstructor import Object3D
from ..projection.gaussian_object_descriptor import GODDescriptor, GaussianObjectDescriptorBuilder
from ..object_detection.detection_result import DetectionResult
from ..utils.logger import default_logger as logger


# Relation vocabulary shared across all evidence sources
RELATION_TYPES = ("on", "above", "below", "near", "left_of", "right_of",
                  "in_front_of", "behind")


class EvidenceGenerator:
    """
    Produces evidence from three independent sources and fuses them
    to build a confidence-weighted, explainable scene graph.
    """

    def __init__(
        self,
        god_builder: GaussianObjectDescriptorBuilder,
        *,
        weight_geometry: float = 0.40,
        weight_visual: float = 0.35,
        weight_semantic: float = 0.25,
        confidence_floor: float = 0.35,
        contact_gap_threshold: float = 0.10,
        near_distance_threshold: float = 1.5,
    ):
        self.god_builder = god_builder
        self.w_geo = weight_geometry
        self.w_vis = weight_visual
        self.w_sem = weight_semantic
        self.confidence_floor = confidence_floor
        self.contact_gap = contact_gap_threshold
        self.near_dist = near_distance_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_evidence_scene_graph(
        self,
        objects_3d: List[Object3D],
        god_map: Dict[int, GODDescriptor],
        detection_results: List[DetectionResult],
        llm_client=None,
    ) -> SceneGraph:
        """
        Build a scene graph where every relation is backed by an evidence chain.

        Args:
            objects_3d: reconstructed 3D objects
            god_map: object_id -> GODDescriptor mapping
            detection_results: per-view YOLO detection results
            llm_client: optional LLM interface for semantic evidence
        """
        scene_graph = SceneGraph(objects_3d)
        obj_list = list(scene_graph.objects.values())

        pair_count = 0
        for i, obj_a in enumerate(obj_list):
            god_a = god_map.get(obj_a.object_id)
            for obj_b in obj_list[i + 1:]:
                god_b = god_map.get(obj_b.object_id)

                evidences = self._collect_all_evidence(
                    obj_a, obj_b, god_a, god_b,
                    detection_results, llm_client,
                )

                relation, confidence = self._fuse(evidences)
                if confidence < self.confidence_floor:
                    continue

                distance = float(np.linalg.norm(obj_a.position - obj_b.position))
                scene_graph.add_relation(SpatialRelation(
                    subject_id=obj_a.object_id,
                    predicate=relation,
                    object_id=obj_b.object_id,
                    distance=distance,
                    confidence=confidence,
                    evidence_chain=evidences,
                ))
                pair_count += 1

        logger.info(
            f"Evidence scene graph: {pair_count} relations from "
            f"{len(obj_list)} objects"
        )

        if llm_client is not None:
            self._llm_review_pass(scene_graph, llm_client)

        return scene_graph

    # ------------------------------------------------------------------
    # Evidence Collection
    # ------------------------------------------------------------------

    def _collect_all_evidence(
        self,
        obj_a: Object3D, obj_b: Object3D,
        god_a: Optional[GODDescriptor], god_b: Optional[GODDescriptor],
        detection_results: List[DetectionResult],
        llm_client,
    ) -> List[Evidence]:
        evidences = []

        geo = self._geometric_evidence(obj_a, obj_b, god_a, god_b)
        if geo is not None:
            evidences.append(geo)

        vis = self._visual_evidence(obj_a, obj_b, detection_results)
        if vis is not None:
            evidences.append(vis)

        if llm_client is not None and evidences:
            sem = self._semantic_evidence(
                obj_a, obj_b, evidences, llm_client
            )
            if sem is not None:
                evidences.append(sem)

        return evidences

    # ------------------------------------------------------------------
    # Source 1: Geometric Evidence (from GOD / Gaussian attributes)
    # ------------------------------------------------------------------

    def _geometric_evidence(
        self,
        obj_a: Object3D, obj_b: Object3D,
        god_a: Optional[GODDescriptor], god_b: Optional[GODDescriptor],
    ) -> Optional[Evidence]:
        if god_a is None or god_b is None:
            return self._fallback_geometric(obj_a, obj_b)

        # vertical contact analysis
        contact = self.god_builder.compute_vertical_contact(god_a, god_b)
        direction = self.god_builder.compute_directional_relation(god_a, god_b)
        surface_dist = self.god_builder.compute_surface_distance(god_a, god_b)

        # ON relation: A sits on B
        if contact["is_contact"] and contact["a_bottom"] > contact["b_bottom"]:
            score = max(0.0, 1.0 - contact["contact_gap"] * 10) * contact["horizontal_overlap"]
            score = min(1.0, score)
            detail = (
                f"{obj_a.class_name} bottom(y={contact['a_bottom']:.2f}) contacts "
                f"{obj_b.class_name} top(y={contact['b_top']:.2f}), "
                f"gap: {contact['contact_gap']:.3f}m, "
                f"horizontal overlap: {contact['horizontal_overlap']:.2f}"
            )
            return Evidence("gaussian_geometry", "on", score, detail)

        # ON relation: B sits on A (reversed)
        if contact["is_contact"] and contact["b_bottom"] > contact["a_bottom"]:
            score = max(0.0, 1.0 - contact["contact_gap"] * 10) * contact["horizontal_overlap"]
            score = min(1.0, score)
            detail = (
                f"{obj_b.class_name} bottom(y={contact['b_bottom']:.2f}) contacts "
                f"{obj_a.class_name} top(y={contact['a_top']:.2f}), "
                f"gap: {contact['contact_gap']:.3f}m, "
                f"horizontal overlap: {contact['horizontal_overlap']:.2f}"
            )
            # swap: subject is B, object is A — but we always emit A->B
            # so we use the reverse predicate
            return Evidence("gaussian_geometry", "below", score, detail)

        # NEAR relation
        if surface_dist < self.near_dist:
            score = max(0.0, 1.0 - surface_dist / self.near_dist)
            detail = f"surface distance: {surface_dist:.3f}m"
            return Evidence("gaussian_geometry", "near", score, detail)

        # directional relation
        dist = direction["distance"]
        if dist < self.near_dist * 2:
            score = max(0.2, 1.0 - dist / (self.near_dist * 2))
            detail = (
                f"direction: {direction['direction']}, "
                f"delta: ({direction['delta'][0]:.2f}, "
                f"{direction['delta'][1]:.2f}, "
                f"{direction['delta'][2]:.2f}), "
                f"distance: {dist:.2f}m"
            )
            return Evidence("gaussian_geometry", direction["direction"], score, detail)

        return None

    def _fallback_geometric(
        self, obj_a: Object3D, obj_b: Object3D
    ) -> Optional[Evidence]:
        """Centroid-based fallback when GOD is not available."""
        delta = obj_a.position - obj_b.position
        dist = float(np.linalg.norm(delta))

        if dist > self.near_dist * 3:
            return None

        abs_delta = np.abs(delta)
        dominant = int(np.argmax(abs_delta))

        if dominant == 1:
            rel = "above" if delta[1] > 0 else "below"
        elif dominant == 0:
            rel = "right_of" if delta[0] > 0 else "left_of"
        else:
            rel = "behind" if delta[2] > 0 else "in_front_of"

        score = max(0.1, 1.0 - dist / (self.near_dist * 3))
        detail = f"centroid-based fallback, distance: {dist:.2f}m"
        return Evidence("gaussian_geometry", rel, score * 0.6, detail)

    # ------------------------------------------------------------------
    # Source 2: Visual Evidence (from multi-view YOLO detections)
    # ------------------------------------------------------------------

    def _visual_evidence(
        self,
        obj_a: Object3D, obj_b: Object3D,
        detection_results: List[DetectionResult],
    ) -> Optional[Evidence]:
        co_visible_count = 0
        relation_votes: Dict[str, int] = defaultdict(int)

        for det_result in detection_results:
            dets_a = [
                d for d in det_result.detections
                if d.class_name == obj_a.class_name
            ]
            dets_b = [
                d for d in det_result.detections
                if d.class_name == obj_b.class_name
            ]

            for da in dets_a:
                for db in dets_b:
                    co_visible_count += 1
                    rel = self._bbox_spatial_relation(da.bbox, db.bbox)
                    relation_votes[rel] += 1

        if co_visible_count == 0:
            return None

        best_rel = max(relation_votes, key=relation_votes.get)
        consistency = relation_votes[best_rel] / co_visible_count

        if consistency < 0.3:
            return None

        detail = (
            f"co-visible in {co_visible_count} pairs, "
            f"{relation_votes[best_rel]}/{co_visible_count} vote '{best_rel}'"
        )
        return Evidence("yolo_multiview", best_rel, consistency, detail)

    @staticmethod
    def _bbox_spatial_relation(
        bbox_a: tuple, bbox_b: tuple
    ) -> str:
        """Infer 2D spatial relation from two bounding boxes."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        a_cx, a_cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        b_cx, b_cy = (bx1 + bx2) / 2, (by1 + by2) / 2

        dx = a_cx - b_cx
        dy = a_cy - b_cy

        # in image space, y increases downward
        if abs(dy) > abs(dx):
            if dy < 0:
                # A is higher in image -> A above B in scene (or ON)
                if ay2 > by1 and ay2 < by1 + (by2 - by1) * 0.3:
                    return "on"
                return "above"
            return "below"

        return "left_of" if dx < 0 else "right_of"

    # ------------------------------------------------------------------
    # Source 3: Semantic Evidence (from LLM common sense)
    # ------------------------------------------------------------------

    def _semantic_evidence(
        self,
        obj_a: Object3D, obj_b: Object3D,
        prior_evidences: List[Evidence],
        llm_client,
    ) -> Optional[Evidence]:
        prior_summary = "; ".join(
            f"{e.source}: {e.relation}({e.score:.2f})" for e in prior_evidences
        )

        prompt = (
            f"Two objects detected in a 3D scene:\n"
            f"  A: {obj_a.class_name}\n"
            f"  B: {obj_b.class_name}\n"
            f"Prior evidence: {prior_summary}\n\n"
            f"Based on common sense, rate how plausible the suggested spatial "
            f"relation is (0.0 to 1.0). Also state the most likely relation "
            f"from: {', '.join(RELATION_TYPES)}.\n\n"
            f"Reply in JSON: "
            f'{{ "score": <float>, "relation": "<str>", "reasoning": "<str>" }}'
        )

        try:
            response = llm_client.client.chat.completions.create(
                model=llm_client.model,
                messages=[
                    {"role": "system", "content": "You are a 3D spatial reasoning assistant. Reply only in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            score = float(result.get("score", 0.5))
            relation = result.get("relation", prior_evidences[0].relation)
            reasoning = result.get("reasoning", "")

            if relation not in RELATION_TYPES:
                relation = prior_evidences[0].relation

            return Evidence(
                "llm_reasoning", relation, min(1.0, max(0.0, score)),
                f"LLM: {reasoning[:200]}"
            )

        except Exception as exc:
            logger.debug(f"LLM semantic evidence failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Evidence Fusion
    # ------------------------------------------------------------------

    def _fuse(self, evidences: List[Evidence]) -> Tuple[str, float]:
        """
        Weighted vote across evidence sources to determine relation type
        and aggregate confidence.
        """
        if not evidences:
            return "unknown", 0.0

        weight_map = {
            "gaussian_geometry": self.w_geo,
            "yolo_multiview": self.w_vis,
            "llm_reasoning": self.w_sem,
        }

        # weighted vote per relation type
        votes: Dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for ev in evidences:
            w = weight_map.get(ev.source, 0.2)
            votes[ev.relation] += w * ev.score
            total_weight += w

        if total_weight < 1e-8:
            return "unknown", 0.0

        best_relation = max(votes, key=votes.get)
        confidence = votes[best_relation] / total_weight

        return best_relation, min(1.0, confidence)

    # ------------------------------------------------------------------
    # LLM Review Pass
    # ------------------------------------------------------------------

    def _llm_review_pass(
        self, scene_graph: SceneGraph, llm_client
    ) -> None:
        """
        Submit low-confidence relations to LLM for final adjudication.
        Relations below 0.7 confidence are reviewed; LLM can confirm,
        correct, or reject.
        """
        low_conf = [
            r for r in scene_graph.relations
            if r.confidence < 0.7 and not r.verified_by_llm
        ]

        if not low_conf:
            return

        logger.info(f"LLM review pass: {len(low_conf)} low-confidence relations")

        for rel in low_conf:
            subj = scene_graph.get_object_by_id(rel.subject_id)
            obj = scene_graph.get_object_by_id(rel.object_id)
            if subj is None or obj is None:
                continue

            evidence_text = "\n".join(
                f"  - {e.source} (score {e.score:.2f}): {e.detail}"
                for e in rel.evidence_chain
            )

            prompt = (
                f"Review this spatial relation:\n"
                f"  {subj.class_name} '{rel.predicate}' {obj.class_name} "
                f"(confidence: {rel.confidence:.2f})\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Should this relation be: confirmed, corrected, or rejected?\n"
                f"If corrected, provide the correct relation from: "
                f"{', '.join(RELATION_TYPES)}.\n\n"
                f"Reply in JSON: "
                f'{{ "action": "confirmed"|"corrected"|"rejected", '
                f'"corrected_relation": "<str or null>", '
                f'"adjusted_confidence": <float>, '
                f'"reasoning": "<str>" }}'
            )

            try:
                response = llm_client.client.chat.completions.create(
                    model=llm_client.model,
                    messages=[
                        {"role": "system", "content": "You are a 3D scene graph reviewer. Reply only in valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
                action = result.get("action", "confirmed")

                if action == "rejected":
                    scene_graph.relations.remove(rel)
                elif action == "corrected":
                    corrected = result.get("corrected_relation", rel.predicate)
                    if corrected in RELATION_TYPES:
                        rel.predicate = corrected
                    rel.confidence = float(result.get("adjusted_confidence", rel.confidence))
                else:
                    rel.confidence = float(result.get("adjusted_confidence", rel.confidence))

                rel.verified_by_llm = True

            except Exception as exc:
                logger.debug(f"LLM review failed for relation {rel.subject_id}->{rel.object_id}: {exc}")
