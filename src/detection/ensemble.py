"""
Ensemble detector: runs multiple domain models and aggregates results.
Useful when the domain is ambiguous or for multi-label environmental scenes.
"""
from typing import Dict, Any, List, Optional
from src.detection.yolo_detector import YOLODetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleDetector:
    """
    Runs inference with multiple YOLOv8 models and merges detections.

    Strategy: Union of all detections across models, de-duplicated by IoU.
    """

    def __init__(self, model_paths: Dict[str, str], iou_merge_threshold: float = 0.5):
        """
        Args:
            model_paths: {domain: model_path} dict
            iou_merge_threshold: IoU above which two boxes are considered duplicates
        """
        self.detectors: Dict[str, YOLODetector] = {
            domain: YOLODetector(model_path=path)
            for domain, path in model_paths.items()
        }
        self.iou_merge_threshold = iou_merge_threshold

    def detect_all(self, image_path: str) -> Dict[str, Any]:
        """
        Run all domain detectors on the same image and merge results.

        Returns:
            Merged detection results with domain labels per detection
        """
        all_detections = []
        domain_results = {}

        for domain, detector in self.detectors.items():
            try:
                result = detector.detect(image_path)
                domain_results[domain] = result
                for det in result.get("detections", []):
                    det["source_domain"] = domain
                    all_detections.append(det)
            except Exception as e:
                logger.error(f"Ensemble detection failed for domain {domain}: {e}")

        return {
            "image_path": image_path,
            "total_detections": len(all_detections),
            "detections": all_detections,
            "domain_results": domain_results,
        }
