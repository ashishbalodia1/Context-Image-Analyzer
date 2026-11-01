"""
Comprehensive Model Evaluation & Benchmarking Engine for City Guardian.
Generates mAP@50, mAP@50-95, precision, recall, per-class metrics, confusion matrices,
FPS / latency breakdowns, and visual prediction galleries.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Independent test evaluation engine for YOLOv8 environmental detectors.
    """

    def __init__(self, weights_path: str):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        self.model = YOLO(str(self.weights_path))

    def evaluate(
        self,
        data_yaml: str,
        split: str = "test",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Executes complete independent evaluation on the specified dataset split.
        """
        print(f"\n=======================================================")
        print(f"EVALUATING MODEL: {self.weights_path.name} on split=[{split}]")
        print(f"Data Config:     {data_yaml}")
        print(f"Confidence Gate: {conf_threshold} | IoU Threshold: {iou_threshold}")
        print(f"=======================================================\n")

        # 1. Run validation using Ultralytics
        val_results = self.model.val(
            data=data_yaml,
            split=split,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            plots=True,
            verbose=False
        )

        # 2. Extract global metrics
        box = val_results.box
        p = float(box.mp)
        r = float(box.mr)
        map50 = float(box.map50)
        map50_95 = float(box.map)
        f1 = float(2 * (p * r) / (p + r + 1e-16))

        # 3. Extract speed and throughput (FPS)
        speed = val_results.speed
        pre_ms = float(speed.get("preprocess", 0.0))
        inf_ms = float(speed.get("inference", 0.0))
        post_ms = float(speed.get("postprocess", 0.0))
        total_ms = pre_ms + inf_ms + post_ms
        fps = round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0

        # 4. Extract per-class performance
        class_names = val_results.names
        per_class_metrics = {}
        if hasattr(box, "maps") and box.maps is not None:
            for cls_id, cls_map in enumerate(box.maps):
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                per_class_metrics[cls_name] = {
                    "class_id": cls_id,
                    "mAP50_95": round(float(cls_map), 4)
                }

        # 5. Model file size
        size_mb = round(self.weights_path.stat().st_size / (1024 * 1024), 2)

        summary = {
            "model_weights": str(self.weights_path),
            "model_size_mb": size_mb,
            "split": split,
            "data_config": data_yaml,
            "global_metrics": {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
            },
            "latency": {
                "preprocess_ms": round(pre_ms, 2),
                "inference_ms": round(inf_ms, 2),
                "postprocess_ms": round(post_ms, 2),
                "total_latency_ms": round(total_ms, 2),
                "fps": fps,
                "device": device.upper()
            },
            "per_class_metrics": per_class_metrics,
            "run_dir": str(val_results.save_dir) if hasattr(val_results, "save_dir") else None
        }

        self._print_evaluation_summary(summary)
        return summary

    def generate_visual_gallery(
        self,
        test_images_dir: Path,
        output_dir: Path,
        conf_threshold: float = 0.35,
        max_images: int = 20
    ) -> List[str]:
        """
        Runs inference on independent test photos and saves annotated images
        displaying predicted bounding boxes, labels, and confidence scores.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        test_images_dir = Path(test_images_dir)

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [f for f in test_images_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
        image_files = image_files[:max_images]

        saved_gallery = []
        for img_path in image_files:
            results = self.model.predict(
                str(img_path),
                conf=conf_threshold,
                verbose=False
            )

            res = results[0]
            annotated_bgr = res.plot()  # Ultralytics built-in high-contrast visualizer

            out_path = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_path), annotated_bgr)
            saved_gallery.append(str(out_path))

        print(f"Generated {len(saved_gallery)} visual prediction gallery images in: {output_dir}")
        return saved_gallery

    @staticmethod
    def _print_evaluation_summary(summary: Dict[str, Any]):
        m = summary["global_metrics"]
        lat = summary["latency"]

        print("=" * 80)
        print("INDEPENDENT MODEL EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Model Checkpoint:  {Path(summary['model_weights']).name} ({summary['model_size_mb']} MB)")
        print(f"Evaluation Split:  {summary['split'].upper()}")
        print("-" * 80)
        print(f"Precision:         {m['precision']:.4f}")
        print(f"Recall:            {m['recall']:.4f}")
        print(f"F1-Score:          {m['f1_score']:.4f}")
        print(f"mAP@50:            {m['mAP50']:.4f}")
        print(f"mAP@50-95:         {m['mAP50_95']:.4f}")
        print("-" * 80)
        print(f"Latency Breakdown: Preprocess: {lat['preprocess_ms']}ms | Inference: {lat['inference_ms']}ms | Postprocess: {lat['postprocess_ms']}ms")
        print(f"Total Latency:     {lat['total_latency_ms']} ms/image ({lat['fps']} FPS on {lat['device']})")
        print("=" * 80)

