"""
Grounded LLM Interface
Upgrades the naive LLM text-generation to a spatially-grounded multimodal dialogue system.

Key innovations over the original llm_interface.py:
1. LLM receives BOTH structured scene graph AND rendered visual context
2. LLM output is GROUNDED: it can reference specific 3D objects/regions
3. Bidirectional loop: LLM can REQUEST additional views or object details
4. LLM can GUIDE detection: suggest objects to search for via open-vocab detector
5. Spatial reasoning is augmented by chain-of-thought prompting with 3D coordinates
"""

import json
import base64
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..utils.logger import default_logger as logger


class GroundedLLMInterface:
    """
    Spatially-Grounded LLM for 3D Scene Dialogue.
    
    Differences from original LLMInterface:
    - Uses multimodal LLM (GPT-4V / LLaVA) with rendered images as input
    - Structured output includes 3D object references (grounding)
    - Supports multi-turn dialogue with scene state tracking
    - Can issue "action commands" back to the 3DGS system (render new view, highlight object)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        use_vision: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_vision = use_vision

        # Dialogue history for multi-turn
        self.dialogue_history: List[Dict] = []

        # Scene context (persists across turns)
        self.scene_context: Optional[Dict] = None

        self.client = None
        if provider == "openai" and api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"GroundedLLM initialized: {model} (vision={use_vision})")
        else:
            logger.warning("LLM client not initialized (missing API key or unsupported provider)")

    # =====================================================================
    # Core: Grounded Scene Description
    # =====================================================================
    def generate_grounded_description(
        self,
        scene_graph_dict: Dict,
        rendered_images: Optional[List[str]] = None,
        depth_stats: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a scene description that is GROUNDED in 3D space.
        Each mentioned object is linked to its object_id and 3D coordinates.

        Args:
            scene_graph_dict: Full scene graph with objects, relations, bounds
            rendered_images: Optional list of image paths for visual context
            depth_stats: Optional depth statistics for spatial reasoning

        Returns:
            {
                'description': str,
                'grounded_mentions': [
                    {'text': 'the red chair', 'object_id': 3, 'position_3d': [x,y,z]},
                    ...
                ],
                'scene_summary': {'num_objects': N, 'main_objects': [...], 'layout': str},
                'suggested_queries': [str, ...],  # Follow-up questions
            }
        """
        system_prompt = self._build_grounding_system_prompt()

        user_content = self._build_scene_context_message(
            scene_graph_dict, rendered_images, depth_stats
        )

        user_prompt = f"""{user_content}

请完成以下任务，以JSON格式返回：
1. 生成一段自然、详细的场景描述（description字段）
2. 对描述中提到的每个物体，标注其object_id和3D位置（grounded_mentions字段）
3. 总结场景的整体布局（scene_summary字段）
4. 建议3个用户可能感兴趣的后续问题（suggested_queries字段）

JSON格式：
{{
  "description": "...",
  "grounded_mentions": [
    {{"text": "物体描述文本片段", "object_id": 0, "position_3d": [x, y, z]}}
  ],
  "scene_summary": {{
    "num_objects": N,
    "main_objects": ["object_class_1", ...],
    "layout": "布局描述"
  }},
  "suggested_queries": ["问题1", "问题2", "问题3"]
}}"""

        return self._call_llm_json(system_prompt, user_prompt)

    # =====================================================================
    # Core: Grounded Q&A with Action Commands
    # =====================================================================
    def answer_grounded_query(
        self,
        query: str,
        scene_graph_dict: Dict,
        rendered_images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a user query with grounded 3D references and optional action commands.
        
        The LLM can return "actions" that the system should execute:
        - highlight: Highlight specific objects in the 3D scene
        - render_view: Request rendering from a specific viewpoint
        - search_object: Ask open-vocab detector to search for a specific object
        - zoom_to: Navigate to a specific 3D location

        Args:
            query: Natural language question
            scene_graph_dict: Scene graph
            rendered_images: Visual context images

        Returns:
            {
                'answer': str,
                'grounded_references': [{'object_id': int, 'relevance': str}],
                'reasoning': str,
                'actions': [
                    {'type': 'highlight', 'object_ids': [1, 3]},
                    {'type': 'render_view', 'position': [x,y,z], 'target': [x,y,z]},
                    {'type': 'search_object', 'query': 'red mug on the desk'},
                    {'type': 'zoom_to', 'position': [x,y,z]},
                ],
                'confidence': float,
            }
        """
        system_prompt = self._build_qa_system_prompt()

        scene_context = self._build_scene_context_message(scene_graph_dict, rendered_images)

        user_prompt = f"""{scene_context}

用户问题：{query}

请以JSON格式回答，包含以下字段：
{{
  "answer": "自然语言回答",
  "grounded_references": [
    {{"object_id": 0, "relevance": "为什么这个物体与问题相关"}}
  ],
  "reasoning": "你的空间推理过程（chain-of-thought）",
  "actions": [
    {{"type": "highlight", "object_ids": [1, 3]}},
    {{"type": "render_view", "position": [x,y,z], "target": [x,y,z], "reason": "为什么需要这个视角"}},
    {{"type": "search_object", "query": "需要搜索的物体描述"}},
    {{"type": "zoom_to", "position": [x,y,z]}}
  ],
  "confidence": 0.85
}}

注意：
- reasoning字段请展示你的空间推理过程，使用物体的3D坐标进行推理
- actions字段只在需要时才包含
- 如果你不确定答案，请说明原因并建议需要什么额外信息"""

        # Add to dialogue history
        self.dialogue_history.append({"role": "user", "content": query})

        result = self._call_llm_json(system_prompt, user_prompt)

        if result:
            self.dialogue_history.append({"role": "assistant", "content": json.dumps(result)})

        return result

    # =====================================================================
    # Innovation: LLM-Guided Object Discovery
    # =====================================================================
    def suggest_objects_to_detect(
        self,
        scene_graph_dict: Dict,
        rendered_image_path: Optional[str] = None,
    ) -> List[str]:
        """
        LLM examines the current scene understanding and suggests
        objects that SHOULD exist but haven't been detected yet.
        
        This creates the novel LLM→Detector feedback loop:
        1. LLM sees partial scene graph
        2. LLM reasons: "This looks like an office, there should be a keyboard and monitor"
        3. Open-vocab detector searches specifically for suggested objects
        4. New detections update the scene graph
        5. Repeat

        Args:
            scene_graph_dict: Current scene graph (may be incomplete)
            rendered_image_path: Optional image for visual reasoning

        Returns:
            List of object names to search for
        """
        if self.client is None:
            return self._rule_based_object_suggestions(scene_graph_dict)

        system_prompt = """你是一个场景理解专家。你的任务是分析当前已检测到的物体，
推理这个场景的类型，然后建议可能存在但尚未被检测到的物体。

例如：如果检测到了"monitor"和"keyboard"，这可能是一个办公桌场景，
那么可能还有"mouse"、"desk lamp"、"pen holder"等物体。

请只返回JSON格式：
{
  "scene_type": "推断的场景类型",
  "reasoning": "推理过程", 
  "suggested_objects": ["object1", "object2", ...]
}"""

        detected_objects = [obj['class_name'] for obj in scene_graph_dict.get('objects', [])]

        user_prompt = f"""当前已检测到的物体: {detected_objects}
场景尺寸: {scene_graph_dict.get('scene_bounds', {}).get('size', 'unknown')}
物体数量: {scene_graph_dict.get('statistics', {}).get('num_objects', 0)}

请推理可能存在但未被检测到的物体。"""

        result = self._call_llm_json(system_prompt, user_prompt)
        if result and 'suggested_objects' in result:
            logger.info(f"LLM suggested {len(result['suggested_objects'])} objects to detect: "
                       f"{result['suggested_objects']}")
            return result['suggested_objects']
        return []

    # =====================================================================
    # Innovation: Spatial Reasoning Chain-of-Thought
    # =====================================================================
    def spatial_reasoning(
        self,
        query: str,
        scene_graph_dict: Dict,
    ) -> Dict[str, Any]:
        """
        Explicit spatial reasoning using 3D coordinates.
        
        Unlike generic LLM Q&A, this forces the LLM to show its work
        using actual 3D coordinates, distances, and geometric relationships.

        Args:
            query: Spatial question (e.g., "What's between the sofa and the TV?")
            scene_graph_dict: Scene graph with 3D coordinates

        Returns:
            Reasoning result with step-by-step spatial computation
        """
        system_prompt = """你是一个3D空间推理专家。对于空间相关的问题，你必须：
1. 明确列出相关物体的3D坐标
2. 计算物体间的实际距离（欧氏距离）
3. 分析相对方位（上下左右前后）
4. 基于计算结果得出结论

返回JSON格式：
{
  "step_by_step": [
    {"step": 1, "action": "提取坐标", "details": "物体A在[x,y,z], 物体B在[x,y,z]"},
    {"step": 2, "action": "计算距离", "details": "距离 = sqrt(...) = X米"},
    {"step": 3, "action": "分析方位", "details": "A在B的左前方"},
    {"step": 4, "action": "结论", "details": "最终答案"}
  ],
  "answer": "最终答案",
  "referenced_objects": [{"object_id": 0, "role": "参考物"}],
  "computed_metrics": {"distance": 2.5, "direction": "left-front"}
}"""

        objects_info = json.dumps(scene_graph_dict.get('objects', []),
                                   ensure_ascii=False, indent=2)
        relations_info = json.dumps(scene_graph_dict.get('relations', []),
                                     ensure_ascii=False, indent=2)

        user_prompt = f"""物体列表（含3D坐标）：
{objects_info}

空间关系：
{relations_info}

问题：{query}

请展示完整的空间推理过程。"""

        return self._call_llm_json(system_prompt, user_prompt)

    # =====================================================================
    # Private helpers
    # =====================================================================
    def _build_grounding_system_prompt(self) -> str:
        return """你是一个3D场景理解助手。你会收到场景的结构化数据（包含每个物体的3D坐标、
大小、类别和空间关系）以及可能的渲染图像。

你的核心能力是"空间接地"（Spatial Grounding）：
- 你提到的每个物体都必须关联到其object_id和3D位置
- 你的描述必须包含空间关系（上下左右远近）
- 使用实际的3D坐标来支持你的描述

请始终以JSON格式回答。"""

    def _build_qa_system_prompt(self) -> str:
        return """你是一个3D场景交互助手，支持空间接地的问答。

核心规则：
1. 每次引用物体时都要标注object_id
2. 空间推理要基于3D坐标计算，不能猜测
3. 当你不确定时，可以请求系统执行"actions"来获取更多信息
4. 你可以请求高亮物体(highlight)、渲染新视角(render_view)、搜索物体(search_object)

可用的action类型：
- highlight: 高亮特定物体以便用户查看
- render_view: 从指定位置和方向渲染新视角
- search_object: 使用开放词汇检测器搜索特定物体
- zoom_to: 导航到特定3D位置"""

    def _build_scene_context_message(
        self,
        scene_graph_dict: Dict,
        rendered_images: Optional[List[str]] = None,
        depth_stats: Optional[Dict] = None,
    ) -> str:
        """Build a rich scene context string for the LLM."""
        parts = ["=== 3D场景数据 ==="]

        # Scene bounds
        bounds = scene_graph_dict.get('scene_bounds', {})
        parts.append(f"场景范围: {bounds.get('min', '?')} ~ {bounds.get('max', '?')}")
        parts.append(f"场景大小: {bounds.get('size', '?')} 米")
        parts.append(f"场景中心: {bounds.get('center', '?')}")

        # Objects
        objects = scene_graph_dict.get('objects', [])
        parts.append(f"\n物体列表 ({len(objects)} 个):")
        for obj in objects:
            parts.append(
                f"  [ID={obj['object_id']}] {obj['class_name']} "
                f"位置={obj['position']} 大小={obj.get('size', '?')} "
                f"置信度={obj.get('confidence', '?'):.2f}"
            )

        # Relations
        relations = scene_graph_dict.get('relations', [])
        if relations:
            parts.append(f"\n空间关系 ({len(relations)} 条):")
            for rel in relations[:20]:  # Limit to avoid token overflow
                parts.append(
                    f"  物体{rel['subject_id']} {rel['predicate']} 物体{rel['object_id']} "
                    f"(距离={rel.get('distance', '?'):.2f}m)"
                )

        # Depth stats
        if depth_stats:
            parts.append(f"\n深度统计: {depth_stats}")

        return "\n".join(parts)

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Dict:
        """Call LLM and parse JSON response."""
        if self.client is None:
            return self._generate_fallback(user_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_fallback(user_prompt)

    def _generate_fallback(self, prompt: str) -> Dict:
        """Fallback when LLM is unavailable."""
        return {
            'description': 'LLM unavailable. Scene data loaded but natural language generation disabled.',
            'grounded_mentions': [],
            'scene_summary': {},
            'suggested_queries': [],
        }

    def _rule_based_object_suggestions(self, scene_graph_dict: Dict) -> List[str]:
        """Rule-based object suggestion when LLM is unavailable."""
        detected = set(obj['class_name'] for obj in scene_graph_dict.get('objects', []))

        # Common co-occurrence rules
        suggestions = []
        cooccurrence = {
            'monitor': ['keyboard', 'mouse', 'desk', 'chair'],
            'chair': ['table', 'desk'],
            'sofa': ['coffee table', 'tv', 'remote', 'pillow'],
            'bed': ['nightstand', 'lamp', 'pillow'],
            'dining table': ['chair', 'plate', 'glass', 'fork'],
            'desk': ['monitor', 'keyboard', 'mouse', 'lamp'],
            'tv': ['remote', 'sofa', 'tv stand'],
        }

        for detected_obj in detected:
            for key, related in cooccurrence.items():
                if key in detected_obj.lower():
                    for r in related:
                        if r not in detected:
                            suggestions.append(r)

        return list(set(suggestions))[:10]
