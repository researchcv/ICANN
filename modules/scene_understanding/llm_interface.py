"""
LLM Interface
Evidence-grounded dialogue system for 3D scene interaction.
Supports multi-turn conversation with evidence citation,
GOD-enriched object descriptions, and uncertainty expression.
"""

import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
from ..utils.logger import default_logger as logger


class LLMInterface:
    """LLM Interface with evidence-grounded 3D scene dialogue."""

    MAX_HISTORY_TURNS = 20

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4-turbo",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._conversation_history: List[Dict[str, str]] = []
        self._scene_context: Optional[str] = None

        if provider == "openai" and api_key:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized, model: {model}")
        else:
            if provider != "openai":
                logger.warning(f"Unsupported LLM provider: {provider}")
            else:
                logger.warning("API key not provided, LLM functionality unavailable")
            self.client = None
    
    # ------------------------------------------------------------------
    # Scene context injection
    # ------------------------------------------------------------------

    def set_scene_context(
        self,
        scene_graph_dict: Dict,
        god_texts: Optional[List[str]] = None,
    ) -> None:
        """
        Build and cache the system-level scene context from the scene graph
        and optional GOD descriptors. Call this once after scene graph
        construction; subsequent dialogue turns reuse the cached context.
        """
        self._scene_context = self._build_scene_context(
            scene_graph_dict, god_texts
        )
        self._conversation_history.clear()

    def generate_scene_description(
        self,
        scene_graph_dict: Dict,
        god_texts: Optional[List[str]] = None,
    ) -> str:
        if self.client is None:
            return self._generate_fallback_description(scene_graph_dict)

        self.set_scene_context(scene_graph_dict, god_texts)
        prompt = self._build_description_prompt(scene_graph_dict)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error(f"Scene description generation failed: {exc}")
            return self._generate_fallback_description(scene_graph_dict)

    # ------------------------------------------------------------------
    # Multi-turn evidence-grounded dialogue
    # ------------------------------------------------------------------

    def answer_query(
        self,
        scene_graph_dict: Dict,
        query: str,
        god_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            return {
                "answer": "LLM functionality is not enabled.",
                "evidence_cited": [],
                "highlight_objects": [],
                "camera_suggestion": None,
            }

        # lazily build scene context on first query
        if self._scene_context is None:
            self.set_scene_context(scene_graph_dict, god_texts)

        self._conversation_history.append({"role": "user", "content": query})
        self._trim_history()

        messages = [
            {"role": "system", "content": self._system_prompt()},
        ] + self._conversation_history

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            self._conversation_history.append({"role": "assistant", "content": raw})

            result = json.loads(raw)
            logger.info(f"Query answered: {query[:80]}")
            return self._normalize_response(result)

        except Exception as exc:
            logger.error(f"LLM query failed: {exc}")
            return {
                "answer": f"Query processing error: {exc}",
                "evidence_cited": [],
                "highlight_objects": [],
                "camera_suggestion": None,
            }

    def reset_conversation(self) -> None:
        self._conversation_history.clear()
    
    def suggest_viewpoint(
        self,
        scene_graph_dict: Dict,
        focus_object_id: Optional[int] = None
    ) -> Dict[str, Any]:
        scene_bounds = scene_graph_dict['scene_bounds']
        center = scene_bounds['center']
        size = scene_bounds['size']

        camera_distance = max(size) * 1.5

        suggestion = {
            'position': [
                center[0] + camera_distance * 0.5,
                center[1] + camera_distance * 0.7,
                center[2] + camera_distance * 0.5
            ],
            'target': center,
            'description': 'Overlook entire scene from above'
        }

        if focus_object_id is not None:
            for obj in scene_graph_dict['objects']:
                if obj['object_id'] == focus_object_id:
                    obj_pos = obj['position']
                    suggestion['target'] = obj_pos
                    suggestion['position'] = [
                        obj_pos[0] + 2.0,
                        obj_pos[1] + 1.5,
                        obj_pos[2] + 2.0
                    ]
                    suggestion['description'] = f'Focus on {obj["class_name"]} object'
                    break

        return suggestion
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        context = self._scene_context or "No scene data loaded."
        return (
            "You are a 3D scene dialogue assistant. Your answers MUST follow "
            "these rules:\n"
            "1. Cite specific evidence when describing spatial relations. "
            "Reference the evidence source and score.\n"
            "2. For relations with confidence below 0.7, explicitly state "
            "uncertainty.\n"
            "3. Use GOD attributes (orientation, shape, color) to enrich "
            "descriptions when available.\n"
            "4. You may suggest highlighting objects or changing the viewpoint.\n"
            "5. Reply in JSON with these fields:\n"
            "   {\n"
            '     "answer": "natural language answer",\n'
            '     "evidence_cited": [{"relation": "...", "source": "...", "score": 0.0}],\n'
            '     "uncertainty": "explain low-confidence items if any",\n'
            '     "highlight_objects": [object_id, ...],\n'
            '     "camera_suggestion": {"position": [x,y,z], "target": [x,y,z]} or null\n'
            "   }\n\n"
            f"Scene data:\n{context}"
        )

    def _build_scene_context(
        self,
        scene_graph_dict: Dict,
        god_texts: Optional[List[str]] = None,
    ) -> str:
        parts = []

        # object list with GOD descriptions
        if god_texts:
            parts.append("Objects (GOD descriptors):")
            parts.extend(god_texts)
        else:
            parts.append("Objects:")
            for obj in scene_graph_dict.get("objects", []):
                parts.append(
                    f"  [ID {obj['object_id']}] {obj['class_name']}: "
                    f"position {obj['position']}, size {obj['size']}"
                )

        # relations with evidence chains
        parts.append("\nSpatial relations (with evidence):")
        for rel in scene_graph_dict.get("relations", []):
            chain_str = ""
            if "evidence_chain" in rel:
                chain_parts = []
                for ev in rel["evidence_chain"]:
                    chain_parts.append(
                        f"{ev['source']}({ev['score']:.2f}): {ev['detail'][:120]}"
                    )
                chain_str = " | ".join(chain_parts)

            conf = rel.get("confidence", 1.0)
            conf_tag = " [LOW CONFIDENCE]" if conf < 0.7 else ""
            parts.append(
                f"  {rel['subject_id']} {rel['predicate']} {rel['object_id']} "
                f"(conf: {conf:.2f}{conf_tag})"
            )
            if chain_str:
                parts.append(f"    evidence: {chain_str}")

        # scene bounds
        bounds = scene_graph_dict.get("scene_bounds", {})
        if bounds:
            parts.append(
                f"\nScene bounds: center {bounds.get('center')}, "
                f"size {bounds.get('size')}"
            )

        return "\n".join(parts)

    def _build_description_prompt(self, scene_graph_dict: Dict) -> str:
        objects = scene_graph_dict.get("objects", [])
        relations = scene_graph_dict.get("relations", [])

        prompt = "Please describe this 3D scene in natural language.\n\n"
        prompt += f"The scene contains {len(objects)} objects:\n"
        for obj in objects:
            prompt += f"- {obj['class_name']} (ID: {obj['object_id']})\n"

        if relations:
            prompt += f"\nThere are {len(relations)} spatial relations between objects.\n"
            prompt += "Please describe the spatial layout comprehensively."

        return prompt

    def _trim_history(self) -> None:
        while len(self._conversation_history) > self.MAX_HISTORY_TURNS * 2:
            self._conversation_history.pop(0)

    @staticmethod
    def _normalize_response(result: Dict) -> Dict:
        return {
            "answer": result.get("answer", ""),
            "evidence_cited": result.get("evidence_cited", []),
            "uncertainty": result.get("uncertainty", ""),
            "highlight_objects": result.get("highlight_objects", []),
            "camera_suggestion": result.get("camera_suggestion"),
        }

    def _generate_fallback_description(self, scene_graph_dict: Dict) -> str:
        objects = scene_graph_dict.get('objects', [])
        stats = scene_graph_dict.get('statistics', {})

        num_obj = stats.get('num_objects', len(objects))
        num_cls = stats.get('num_classes', 0)
        desc = f"This scene contains {num_obj} objects across {num_cls} classes.\n\n"

        class_counts: Dict[str, int] = {}
        for obj in objects:
            cn = obj.get('class_name', 'unknown')
            class_counts[cn] = class_counts.get(cn, 0) + 1

        desc += "Object distribution:\n"
        for cn, count in class_counts.items():
            desc += f"- {count} {cn}\n"

        size = scene_graph_dict.get('scene_bounds', {}).get('size', 'unknown')
        desc += f"\nScene extent: {size} meters."
        return desc

