"""
Open-Vocabulary Detector
Replaces closed-set YOLO with Grounding DINO + SAM for open-vocabulary detection & segmentation.
This provides:
  1. Open-vocabulary detection (any text query, not limited to COCO 80 classes)
  2. High-quality instance masks (via SAM) for precise Gaussian-to-object assignment
  3. Academic novelty: language-guided 3D scene parsing
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .detection_result import Detection, DetectionResult
from ..utils.logger import default_logger as logger


@dataclass
class SegmentedDetection(Detection):
    """Detection with instance segmentation mask"""
    mask: Optional[np.ndarray] = None  # Binary mask [H, W]
    mask_area: int = 0
    clip_feature: Optional[np.ndarray] = None  # CLIP embedding for this region


class OpenVocabDetector:
    """
    Open-Vocabulary Detector using Grounding DINO + SAM.
    
    Academic advantages over closed-set YOLO:
    - Detects ANY object described by text (open-vocabulary)
    - Produces pixel-level masks for precise 3D Gaussian assignment
    - Language-grounded: detection is conditioned on text queries from LLM
    - CLIP features per-region enable semantic feature distillation to 3DGS
    """

    def __init__(
        self,
        grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint: str = "weights/groundingdino_swint_ogc.pth",
        sam_checkpoint: str = "weights/sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: str = "cuda",
        default_text_prompt: str = "chair . table . sofa . monitor . keyboard . lamp . book . cup . bottle . plant . person",
    ):
        """
        Args:
            grounding_dino_config: Path to GroundingDINO config
            grounding_dino_checkpoint: Path to GroundingDINO weights
            sam_checkpoint: Path to SAM weights
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
            box_threshold: Detection box confidence threshold
            text_threshold: Text matching threshold
            device: Runtime device
            default_text_prompt: Default text query (dot-separated object names)
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.default_text_prompt = default_text_prompt

        self.gdino_model = None
        self.sam_predictor = None
        self.clip_model = None
        self.clip_preprocess = None

        self._load_grounding_dino(grounding_dino_config, grounding_dino_checkpoint)
        self._load_sam(sam_checkpoint, sam_model_type)
        self._load_clip()

        logger.info("OpenVocabDetector initialized (GroundingDINO + SAM + CLIP)")

    def _load_grounding_dino(self, config_path: str, checkpoint_path: str):
        """Load Grounding DINO model."""
        try:
            from groundingdino.util.inference import load_model
            self.gdino_model = load_model(config_path, checkpoint_path, device=self.device)
            logger.info("Grounding DINO loaded successfully")
        except ImportError:
            logger.warning(
                "groundingdino not installed. Install via: "
                "pip install groundingdino-py"
            )
        except Exception as e:
            logger.warning(f"Failed to load Grounding DINO: {e}")

    def _load_sam(self, checkpoint: str, model_type: str):
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            logger.info("SAM loaded successfully")
        except ImportError:
            logger.warning(
                "segment_anything not installed. Install via: "
                "pip install segment-anything"
            )
        except Exception as e:
            logger.warning(f"Failed to load SAM: {e}")

    def _load_clip(self):
        """Load CLIP model for per-region feature extraction."""
        try:
            import open_clip
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='laion2b_s34b_b88k'
            )
            self.clip_model = self.clip_model.to(self.device).eval()
            logger.info("CLIP loaded successfully")
        except ImportError:
            logger.warning("open_clip not installed. Install via: pip install open-clip-torch")
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}")

    def detect(
        self,
        image_path: str,
        view_id: int = 0,
        text_prompt: Optional[str] = None,
        extract_masks: bool = True,
        extract_clip_features: bool = True,
    ) -> DetectionResult:
        """
        Detect objects using open-vocabulary grounding.

        Args:
            image_path: Path to image
            view_id: View identifier
            text_prompt: Text query (overrides default). Dot-separated object names.
            extract_masks: Whether to extract SAM masks
            extract_clip_features: Whether to extract CLIP features per region

        Returns:
            DetectionResult with SegmentedDetection instances
        """
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            logger.error(f"Cannot read image: {image_path}")
            return DetectionResult(
                image_name=Path(image_path).name,
                image_path=str(image_path),
                view_id=view_id,
                detections=[],
                image_size=(0, 0),
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        prompt = text_prompt or self.default_text_prompt

        # --- Step 1: Grounding DINO detection ---
        boxes, confidences, phrases = self._grounding_dino_detect(image_rgb, prompt)

        if len(boxes) == 0:
            return DetectionResult(
                image_name=Path(image_path).name,
                image_path=str(image_path),
                view_id=view_id,
                detections=[],
                image_size=(w, h),
            )

        # --- Step 2: SAM segmentation for each box ---
        masks = None
        if extract_masks and self.sam_predictor is not None:
            masks = self._sam_segment(image_rgb, boxes)

        # --- Step 3: CLIP feature extraction per region ---
        clip_features = None
        if extract_clip_features and self.clip_model is not None:
            clip_features = self._extract_clip_features(image_rgb, boxes, masks)

        # --- Build detections ---
        detections = []
        for i, (box, conf, phrase) in enumerate(zip(boxes, confidences, phrases)):
            x1, y1, x2, y2 = box
            det = SegmentedDetection(
                class_name=phrase.strip(),
                class_id=i,  # Open-vocab doesn't have fixed class IDs
                confidence=float(conf),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(float((x1 + x2) / 2), float((y1 + y2) / 2)),
                mask=masks[i] if masks is not None else None,
                mask_area=int(masks[i].sum()) if masks is not None else 0,
                clip_feature=clip_features[i] if clip_features is not None else None,
            )
            detections.append(det)

        logger.info(f"OpenVocab detected {len(detections)} objects: {image_path}")

        return DetectionResult(
            image_name=Path(image_path).name,
            image_path=str(image_path),
            view_id=view_id,
            detections=detections,
            image_size=(w, h),
        )

    def _grounding_dino_detect(
        self,
        image_rgb: np.ndarray,
        text_prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run Grounding DINO inference."""
        if self.gdino_model is None:
            logger.warning("Grounding DINO not available, returning empty")
            return np.array([]), np.array([]), []

        try:
            from groundingdino.util.inference import predict
            import groundingdino.datasets.transforms as T
            from PIL import Image

            # Prepare image
            pil_image = Image.fromarray(image_rgb)
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(pil_image, None)

            boxes, logits, phrases = predict(
                model=self.gdino_model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

            # Convert normalized boxes to pixel coordinates
            h, w = image_rgb.shape[:2]
            boxes_np = boxes.cpu().numpy()
            # GroundingDINO returns cx, cy, w, h normalized
            boxes_xyxy = np.zeros_like(boxes_np)
            boxes_xyxy[:, 0] = (boxes_np[:, 0] - boxes_np[:, 2] / 2) * w
            boxes_xyxy[:, 1] = (boxes_np[:, 1] - boxes_np[:, 3] / 2) * h
            boxes_xyxy[:, 2] = (boxes_np[:, 0] + boxes_np[:, 2] / 2) * w
            boxes_xyxy[:, 3] = (boxes_np[:, 1] + boxes_np[:, 3] / 2) * h

            return boxes_xyxy, logits.cpu().numpy(), phrases

        except Exception as e:
            logger.error(f"Grounding DINO inference failed: {e}")
            return np.array([]), np.array([]), []

    def _sam_segment(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        """Generate SAM masks for each detection box."""
        self.sam_predictor.set_image(image_rgb)
        masks = []
        for box in boxes:
            input_box = np.array(box).reshape(1, 4)
            mask, score, _ = self.sam_predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            masks.append(mask[0])  # [H, W] boolean
        return masks

    def _extract_clip_features(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        masks: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Extract CLIP visual features for each detected region."""
        from PIL import Image

        features = []
        pil_image = Image.fromarray(image_rgb)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # Crop region
            crop = pil_image.crop((x1, y1, x2, y2))

            # If mask available, apply it
            if masks is not None and masks[i] is not None:
                mask_crop = masks[i][y1:y2, x1:x2]
                crop_np = np.array(crop)
                crop_np[~mask_crop] = 0  # Zero out background
                crop = Image.fromarray(crop_np)

            # CLIP encode
            img_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(img_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy().flatten())

        return features

    def detect_with_llm_queries(
        self,
        image_path: str,
        view_id: int,
        llm_suggested_objects: List[str],
    ) -> DetectionResult:
        """
        LLM-guided detection: use LLM's scene hypothesis as text queries.
        
        This creates a closed loop:
        LLM hypothesizes objects -> GroundingDINO searches for them -> 
        Results fed back to LLM for refinement.

        Args:
            image_path: Image path
            view_id: View ID
            llm_suggested_objects: Object names suggested by LLM

        Returns:
            DetectionResult
        """
        text_prompt = " . ".join(llm_suggested_objects)
        return self.detect(image_path, view_id, text_prompt=text_prompt)
