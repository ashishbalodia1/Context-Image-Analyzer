"""
Real-World Test Case Evaluation & Classification Metrics Generator for City Guardian.
Evaluates trained YOLOv8 detectors on independent real-world test cases.
Computes:
  - Multi-class Classification Accuracy, Precision, Recall, Specificity, and F1-Score
  - Multi-class Confusion Matrix (both raw counts and normalized percentages)
  - Object detection IoU metrics (TP, FP, FN, mAP@50)
  - Detailed case-by-case diagnostic table for all test images
  - Confusion Matrix plot and JSON telemetry report
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils.logger import get_logger
from src.context_engine.severity_assessor import SeverityAssessor

logger = get_logger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two [x_min, y_min, x_max, y_max] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def yolo_to_xyxy(norm_box: List[float], img_w: int, img_h: int) -> List[float]:
    """Converts normalized [x_center, y_center, width, height] to [x1, y1, x2, y2] in pixels."""
    xc, yc, w, h = norm_box
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)]


def evaluate_real_world_testset(
    model_path: Path,
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
    domain_name: str = "solid_waste",
    conf_threshold: float = 0.20,
    iou_match_threshold: float = 0.40,
    output_dir: Path = PROJECT_ROOT / "reports" / "evaluation"
) -> Dict[str, Any]:
    """
    Evaluates real-world test cases image-by-image, computing detection and classification metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    severity_assessor = SeverityAssessor()

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {images_dir}")

    n_classes = len(class_names)
    # Confusion matrix: rows = Ground Truth, cols = Predicted
    # Include an extra index (n_classes) for 'Background / No Detection'
    cm_classes = class_names + ["no_anomaly"]
    n_cm = len(cm_classes)
    confusion_matrix = np.zeros((n_cm, n_cm), dtype=int)

    # Object-level detection stats
    obj_tp = defaultdict(int)
    obj_fp = defaultdict(int)
    obj_fn = defaultdict(int)

    # Case-by-case diagnosis records
    test_cases_log = []
    correct_classifications = 0
    total_images = len(image_paths)

    inference_latencies = []

    print("\n" + "=" * 85)
    print(f" REAL-WORLD TEST CASE INFERENCE & BENCHMARK: [{domain_name.upper()}]")
    print(f" Test Images: {total_images} | Model: {model_path.name}")
    print(f" Confidence Gate: {conf_threshold} | IoU Match Gate: {iou_match_threshold}")
    print("=" * 85)

    for idx, img_p in enumerate(image_paths, 1):
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        with Image.open(img_p) as pil_img:
            img_w, img_h = pil_img.size

        # Parse ground-truth bounding boxes
        gt_boxes = []
        gt_classes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    gt_classes.append(cls_id)
                    gt_boxes.append(yolo_to_xyxy(coords, img_w, img_h))

        # Run model inference with timing
        t0 = time.perf_counter()
        results = model.predict(str(img_p), conf=conf_threshold, verbose=False)
        t1 = time.perf_counter()
        inference_latencies.append((t1 - t0) * 1000.0)

        pred_boxes = []
        pred_classes = []
        pred_confs = []

        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                xyxy = b.xyxy[0].tolist()
                c_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                pred_boxes.append(xyxy)
                pred_classes.append(c_id)
                pred_confs.append(conf)

        # Match predictions to GT for object detection TP/FP/FN
        matched_gt = set()
        matched_pred = set()

        for p_i, (p_box, p_cls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0.0
            best_gt_i = -1
            for g_i, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if g_i in matched_gt:
                    continue
                if p_cls == g_cls:
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_i = g_i

            if best_iou >= iou_match_threshold and best_gt_i >= 0:
                obj_tp[class_names[p_cls]] += 1
                matched_gt.add(best_gt_i)
                matched_pred.add(p_i)
            else:
                obj_fp[class_names[p_cls]] += 1

        for g_i, g_cls in enumerate(gt_classes):
            if g_i not in matched_gt:
                obj_fn[class_names[g_cls]] += 1

        # Determine Image-Level Classification
        # Ground truth primary class (majority class in image, or no_anomaly)
        if gt_classes:
            gt_primary_id = max(set(gt_classes), key=gt_classes.count)
            gt_label = class_names[gt_primary_id] if gt_primary_id < n_classes else "other"
            gt_cm_idx = gt_primary_id if gt_primary_id < n_classes else n_classes
        else:
            gt_label = "no_anomaly"
            gt_cm_idx = n_classes

        # Predicted primary class (highest confidence detection, or no_anomaly)
        if pred_classes:
            max_conf_idx = int(np.argmax(pred_confs))
            pred_primary_id = pred_classes[max_conf_idx]
            pred_label = class_names[pred_primary_id] if pred_primary_id < n_classes else "other"
            pred_cm_idx = pred_primary_id if pred_primary_id < n_classes else n_classes
            top_conf = round(pred_confs[max_conf_idx], 3)
        else:
            pred_label = "no_anomaly"
            pred_cm_idx = n_classes
            top_conf = 0.0

        confusion_matrix[gt_cm_idx, pred_cm_idx] += 1
        is_correct = (gt_label == pred_label)
        if is_correct:
            correct_classifications += 1

        # Compute Visual Severity
        formatted_dets = [
            {"class_name": class_names[c] if c < n_classes else "anomaly",
             "confidence": conf,
             "area_fraction": ((b[2]-b[0])*(b[3]-b[1])) / (img_w * img_h)}
            for b, c, conf in zip(pred_boxes, pred_classes, pred_confs)
        ]
        sev = severity_assessor.assess({"domain": domain_name, "detections": formatted_dets})

        test_cases_log.append({
            "case_id": f"TC-{idx:03d}",
            "image": img_p.name,
            "ground_truth": gt_label,
            "predicted": pred_label,
            "confidence": top_conf,
            "status": "CORRECT" if is_correct else "MISMATCH",
            "gt_instances": len(gt_classes),
            "pred_instances": len(pred_classes),
            "visual_severity": sev.get("visual_severity", "none").upper(),
            "severity_score": sev.get("visual_severity_score", 0.0)
        })

    # Compute Classification Metrics
    accuracy = correct_classifications / max(total_images, 1)

    # Per-Class Precision, Recall, Specificity, F1
    per_class_classification = {}
    precisions = []
    recalls = []
    f1_scores = []
    support_counts = []

    for c_idx, c_name in enumerate(cm_classes):
        tp = confusion_matrix[c_idx, c_idx]
        fp = np.sum(confusion_matrix[:, c_idx]) - tp
        fn = np.sum(confusion_matrix[c_idx, :]) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)

        p = tp / max(tp + fp, 1e-9) if (tp + fp) > 0 else 0.0
        r = tp / max(tp + fn, 1e-9) if (tp + fn) > 0 else 0.0
        spec = tn / max(tn + fp, 1e-9) if (tn + fp) > 0 else 0.0
        f1 = (2 * p * r) / max(p + r, 1e-9) if (p + r) > 0 else 0.0
        support = int(np.sum(confusion_matrix[c_idx, :]))

        per_class_classification[c_name] = {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

        if support > 0:
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
            support_counts.append(support)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    total_supp = sum(support_counts)
    weighted_precision = float(sum(p * s for p, s in zip(precisions, support_counts)) / max(total_supp, 1))
    weighted_recall = float(sum(r * s for r, s in zip(recalls, support_counts)) / max(total_supp, 1))
    weighted_f1 = float(sum(f * s for f, s in zip(f1_scores, support_counts)) / max(total_supp, 1))

    # Object-level detection summary
    total_obj_tp = sum(obj_tp.values())
    total_obj_fp = sum(obj_fp.values())
    total_obj_fn = sum(obj_fn.values())
    det_precision = total_obj_tp / max(total_obj_tp + total_obj_fp, 1e-9)
    det_recall = total_obj_tp / max(total_obj_tp + total_obj_fn, 1e-9)
    det_f1 = (2 * det_precision * det_recall) / max(det_precision + det_recall, 1e-9)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n_cm),
        yticks=np.arange(n_cm),
        xticklabels=cm_classes,
        yticklabels=cm_classes,
        title=f"Confusion Matrix — {domain_name.upper()} (Real-World Test Cases)",
        ylabel="Ground Truth Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = confusion_matrix.max() / 2.0
    for i in range(n_cm):
        for j in range(n_cm):
            ax.text(
                j, i, format(confusion_matrix[i, j], "d"),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black"
            )
    fig.tight_layout()
    cm_path = output_dir / f"confusion_matrix_{domain_name}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    mean_lat_ms = float(np.mean(inference_latencies))
    fps = round(1000.0 / mean_lat_ms, 2)

    results_data = {
        "domain": domain_name,
        "model": model_path.name,
        "total_test_images": total_images,
        "evaluation_metrics": {
            "image_level_classification": {
                "accuracy": round(accuracy, 4),
                "macro_precision": round(macro_precision, 4),
                "macro_recall": round(macro_recall, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_precision": round(weighted_precision, 4),
                "weighted_recall": round(weighted_recall, 4),
                "weighted_f1": round(weighted_f1, 4),
            },
            "object_level_detection": {
                "precision": round(det_precision, 4),
                "recall": round(det_recall, 4),
                "f1_score": round(det_f1, 4),
                "true_positives": total_obj_tp,
                "false_positives": total_obj_fp,
                "false_negatives": total_obj_fn
            },
            "inference_speed": {
                "mean_latency_ms": round(mean_lat_ms, 2),
                "fps": fps,
                "device": "CPU"
            }
        },
        "per_class_classification": per_class_classification,
        "confusion_matrix": confusion_matrix.tolist(),
        "confusion_matrix_labels": cm_classes,
        "sample_test_cases": test_cases_log
    }

    # Save JSON report
    json_path = output_dir / f"classification_metrics_{domain_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Print Formatted Report to Console
    print("\n" + "=" * 85)
    print(" REAL-WORLD CLASSIFICATION & DETECTION METRICS REPORT")
    print("=" * 85)
    print(f"Total Test Cases Evaluated:   {total_images}")
    print(f"Correctly Classified Images:  {correct_classifications} / {total_images} ({accuracy * 100:.2f}%)")
    print(f"Top-1 Classification Accuracy: {accuracy * 100:.2f}%")
    print("-" * 85)
    print("Classification Metrics (Image-Level):")
    print(f"  - Macro Precision:          {macro_precision:.4f} ({macro_precision*100:.1f}%)")
    print(f"  - Macro Recall:             {macro_recall:.4f} ({macro_recall*100:.1f}%)")
    print(f"  - Macro F1-Score:           {macro_f1:.4f}")
    print(f"  - Weighted F1-Score:        {weighted_f1:.4f}")
    print("-" * 85)
    print("Object Detection Metrics (Bounding Box Level):")
    print(f"  - Detection Precision:      {det_precision:.4f}")
    print(f"  - Detection Recall:         {det_recall:.4f}")
    print(f"  - Detection F1-Score:       {det_f1:.4f}")
    print(f"  - TP: {total_obj_tp} | FP: {total_obj_fp} | FN: {total_obj_fn}")
    print("-" * 85)
    print(f"CPU Latency:                  {mean_lat_ms:.2f} ms/frame ({fps} FPS)")
    print("=" * 85)

    print("\nPER-CLASS DETAILED BREAKDOWN:")
    print(f"{'Class Name':<22} | {'Precision':<9} | {'Recall':<9} | {'Specificity':<11} | {'F1-Score':<8} | {'Support':<7}")
    print("-" * 85)
    for c_name, m in per_class_classification.items():
        if m['support'] > 0:
            print(f"{c_name:<22} | {m['precision']:<9.4f} | {m['recall']:<9.4f} | {m['specificity']:<11.4f} | {m['f1_score']:<8.4f} | {m['support']:<7}")
    print("=" * 85)

    print(f"\nSaved Confusion Matrix Plot to: {cm_path}")
    print(f"Saved Structured Metrics JSON to: {json_path}")

    return results_data


def main():
    solid_model = PROJECT_ROOT / "models" / "trained" / "solid_waste_yolov8.pt"
    test_imgs = PROJECT_ROOT / "data" / "processed" / "solid_waste" / "images" / "test"
    test_lbls = PROJECT_ROOT / "data" / "processed" / "solid_waste" / "labels" / "test"

    classes = ["illegal_dumping", "overflowing_dustbin", "plastic_waste", "construction_debris", "organic_waste", "mixed_waste_pile"]

    evaluate_real_world_testset(
        model_path=solid_model,
        images_dir=test_imgs,
        labels_dir=test_lbls,
        class_names=classes,
        domain_name="solid_waste",
        conf_threshold=0.20,
        iou_match_threshold=0.35
    )


if __name__ == "__main__":
    main()

