"""
Domain Router: determines which environmental domain an image belongs to
and routes it to the correct YOLOv8 model.
"""
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

from config.settings import YOLO_MODELS, BASE_YOLO_MODEL
from src.detection.yolo_detector import YOLODetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

DOMAIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "domains.yaml"


class DomainRouter:
    """
    Routes uploaded images to the correct environmental domain detector.

    If domain is explicitly specified (by mobile app or user), routes directly.
    Otherwise, falls back to a configurable default domain.
    """

    VALID_DOMAINS = [
        "air_pollution",
        "water_quality",
        "sewage_drainage",
        "mining_dust",
        "solid_waste",
    ]

    def __init__(self):
        self._detectors: Dict[str, YOLODetector] = {}
        self._domain_config = self._load_domain_config()

    def _load_domain_config(self) -> Dict:
        with open(DOMAIN_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_detector(self, domain: str) -> YOLODetector:
        """Lazy-initialize detector for a given domain."""
        if domain not in self._detectors:
            model_path = YOLO_MODELS.get(domain, BASE_YOLO_MODEL)
            self._detectors[domain] = YOLODetector(model_path=model_path)
        return self._detectors[domain]

    def route_and_detect(
        self,
        image_path: str,
        domain: Optional[str] = None,
        save_annotated: bool = True,
        output_dir: str = "data/uploads/annotated",
    ) -> Dict[str, Any]:
        """
        Route image to correct domain detector and run inference.

        Args:
            image_path: Path to the validated image
            domain: Explicitly specified domain (or None for auto-detect)
            save_annotated: Whether to save annotated output
            output_dir: Where to save annotated images

        Returns:
            Detection results with domain metadata
        """
        if domain and domain not in self.VALID_DOMAINS:
            raise ValueError(
                f"Invalid domain: {domain}. Valid: {self.VALID_DOMAINS}"
            )

        if domain is None:
            domain = self._auto_detect_domain(image_path)
            logger.info(f"Auto-detected domain: {domain}")

        detector = self._get_detector(domain)
        detection_result = detector.detect(
            image_path, save_annotated=save_annotated, output_dir=output_dir
        )

        # Domain-Specific Enhanced Aquatic Waste Integration:
        # If user inspected water_quality and primary water detector finds 0 objects,
        # check for floating solid waste / plastic debris in the water body.
        if domain == "water_quality" and len(detection_result.get("detections", [])) == 0:
            solid_detector = self._get_detector("solid_waste")
            solid_res = solid_detector.detect(
                image_path, save_annotated=save_annotated, output_dir=output_dir
            )
            if len(solid_res.get("detections", [])) > 0:
                logger.info(
                    f"Water Quality: mapped {len(solid_res['detections'])} aquatic waste detections to floating_waste"
                )
                for det in solid_res["detections"]:
                    det["class_name"] = "floating_waste"
                    det["class_id"] = 1
                detection_result["detections"] = solid_res["detections"]
                detection_result["total_detections"] = len(solid_res["detections"])
                detection_result["annotated_image_path"] = solid_res.get("annotated_image_path")

        domain_meta = self._domain_config["domains"].get(domain, {})
        detection_result["domain"] = domain
        detection_result["domain_name"] = domain_meta.get("name", domain)
        detection_result["responding_department"] = domain_meta.get(
            "responding_department", ""
        )
        detection_result["icon"] = domain_meta.get("icon", "🌍")

        return detection_result

    def _auto_detect_domain(self, image_path: str) -> str:
        """
        Intelligent multi-domain routing: evaluates candidate models and
        selects the environmental domain with highest confidence detection.
        """
        best_domain = "solid_waste"
        best_conf = 0.0

        for candidate in self.VALID_DOMAINS:
            try:
                det = self._get_detector(candidate)
                res = det.detect(image_path, save_annotated=False)
                for obj in res.get("detections", []):
                    conf = obj.get("confidence", 0.0)
                    if conf > best_conf:
                        best_conf = conf
                        best_domain = candidate
            except Exception:
                continue

        logger.info(f"Auto-detected domain: {best_domain} (peak confidence: {best_conf:.3f})")
        return best_domain
        return "solid_waste"
