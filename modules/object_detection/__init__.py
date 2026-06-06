"""
Object Detection Module
YOLO-based object detection and annotation
"""

from .yolo_detector import YOLODetector
from .detection_result import Detection, DetectionResult
from .multiview_consistency import MultiViewConsistencyChecker, ConsistentDetection
from .gaussian_guided_yolo import GaussianGuidedYOLO, VerifiedDetection

__all__ = [
    'YOLODetector',
    'DetectionResult',
    'Detection',
    'MultiViewConsistencyChecker',
    'ConsistentDetection',
    'GaussianGuidedYOLO',
    'VerifiedDetection',
]

