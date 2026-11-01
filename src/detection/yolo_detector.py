"""
YOLOv8 detection wrapper.
Loads domain-specific models and runs inference on preprocessed images.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
from ultralytics import YOLO

from config.settings import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, BASE_YOLO_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YOLODetector:
    """
    Wraps Ultralytics YOLOv8 for domain-specific environmental detection.

    Usage:
        detector = YOLODetector(model_path="models/trained/solid_waste_yolov8.pt")
        results = detector.detect("path/to/image.jpg")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path or BASE_YOLO_MODEL
        self._model: Optional[YOLO] = None

    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            logger.info(f"Loading YOLO model: {self.model_path}")
            try:
                if Path(self.model_path).exists():
                    self._model = YOLO(self.model_path)
                else:
                    logger.warning(f"Weights {self.model_path} not found on disk, initializing from yolov8n.yaml")
                    self._model = YOLO("yolov8n.yaml")
                    Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
                    self._model.save(self.model_path)
            except Exception as e:
                logger.warning(f"Error loading {self.model_path} ({e}), falling back to yolov8n architecture")
                self._model = YOLO("yolov8n.yaml")
            logger.info("Model loaded successfully.")

    def detect(
        self,
        image_path: str,
        save_annotated: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run object detection on a single image.

        Args:
            image_path: Path to the input image
            save_annotated: Whether to save the annotated image
            output_dir: Directory to save annotated image

        Returns:
            Detection result dict with detections, confidence scores, bounding boxes
        """
        self._load_model()

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self._model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                detection = {
                    "class_id": int(box.cls.item()),
                    "class_name": result.names[int(box.cls.item())],
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": {
                        "x1": round(float(box.xyxy[0][0].item()), 2),
                        "y1": round(float(box.xyxy[0][1].item()), 2),
                        "x2": round(float(box.xyxy[0][2].item()), 2),
                        "y2": round(float(box.xyxy[0][3].item()), 2),
                    },
                    "area_fraction": self._bbox_area_fraction(
                        box.xyxy[0], result.orig_shape
                    ),
                }
                detections.append(detection)

        annotated_path = None
        if save_annotated and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            annotated_img = results[0].plot()
            fname = Path(image_path).stem + "_annotated.jpg"
            annotated_path = str(Path(output_dir) / fname)
            cv2.imwrite(annotated_path, annotated_img)
            logger.info(f"Annotated image saved: {annotated_path}")

        return {
            "image_path": image_path,
            "total_detections": len(detections),
            "detections": detections,
            "annotated_image_path": annotated_path,
            "model_used": str(self.model_path),
        }

    @staticmethod
    def _bbox_area_fraction(xyxy, orig_shape: tuple) -> float:
        """Compute what fraction of the image the bounding box occupies."""
        x1, y1, x2, y2 = xyxy
        box_area = (x2 - x1) * (y2 - y1)
        image_area = orig_shape[0] * orig_shape[1]
        return round(float(box_area / image_area), 4) if image_area > 0 else 0.0
