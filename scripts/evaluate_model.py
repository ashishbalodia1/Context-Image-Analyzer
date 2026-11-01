"""
CLI Script to evaluate trained YOLOv8 environmental detectors on independent test sets.
Computes mAP@50, mAP@50-95, precision, recall, F1, latency, and generates visual galleries.

Usage:
  python scripts/evaluate_model.py --weights models/trained/solid_waste_yolov8.pt --data-yaml models/configs/solid_waste.yaml --split test --save-gallery
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from src.detection.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Environmental Model on Independent Test Set")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--data-yaml", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image resolution")
    parser.add_argument("--save-gallery", action="store_true", help="Generate visual prediction gallery")
    parser.add_argument("--test-images-dir", type=str, default=None, help="Directory containing test images for gallery")
    parser.add_argument("--output-gallery-dir", type=str, default="reports/evaluation/gallery", help="Directory to save gallery")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.weights)
    summary = evaluator.evaluate(
        data_yaml=args.data_yaml,
        split=args.split,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz
    )

    if args.save_gallery:
        test_dir = Path(args.test_images_dir) if args.test_images_dir else Path(args.data_yaml).parent / "images" / args.split
        if test_dir.exists():
            evaluator.generate_visual_gallery(
                test_images_dir=test_dir,
                output_dir=Path(args.output_gallery_dir),
                conf_threshold=args.conf
            )
        else:
            print(f"Note: Test images directory not found at {test_dir}. Skipping gallery generation.")


if __name__ == "__main__":
    main()
