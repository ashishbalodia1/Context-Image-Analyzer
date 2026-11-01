"""
Hyperparameter Tuning, Transfer Learning Optimization & Experiment Tracker for City Guardian.
Supports systematic comparison across model scales, optimizers, learning rates,
backbone freezing strategies, and post-training threshold tuning (Confidence & IoU).
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Manages controlled fine-tuning experiments and post-training optimization.
    """

    def __init__(self, experiment_log_path: str = "runs/experiments_log.json"):
        self.log_path = Path(experiment_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: List[Dict[str, Any]] = self._load_experiments()

    def _load_experiments(self) -> List[Dict[str, Any]]:
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_experiments(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)

    @staticmethod
    def configure_backbone_freezing(model: YOLO, freeze_layers: int = 10) -> int:
        """
        Freezes the first `freeze_layers` in the backbone.
        Layer 0-9 in YOLOv8 represent the convolutional feature extraction backbone
        (stem, C2f blocks, SPPF). Freezing preserves generic low-level edge features
        and stabilizes fine-tuning on smaller environmental datasets.

        Returns:
            Number of frozen parameters.
        """
        frozen_params = 0
        total_params = 0

        # YOLOv8 model layers are in model.model.model
        torch_model = model.model
        freeze_layer_names = [f"model.{x}." for x in range(freeze_layers)]

        for name, param in torch_model.named_parameters():
            total_params += param.numel()
            if any(fl in name for fl in freeze_layer_names):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.requires_grad = True

        logger.info(
            f"Backbone Freezing: Frozen {frozen_params:,} / {total_params:,} parameters "
            f"({frozen_params / total_params * 100:.1f}%) in first {freeze_layers} layers."
        )
        return frozen_params

    @staticmethod
    def find_optimal_confidence_threshold(
        precision_curve: np.ndarray,
        recall_curve: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[float, float]:
        """
        Finds the confidence threshold that maximizes the F1-score:
          F1 = 2 * (Precision * Recall) / (Precision + Recall + 1e-16)

        Returns:
            (optimal_conf_threshold, max_f1_score)
        """
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-16)
        best_idx = int(np.argmax(f1_scores))
        best_thresh = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        return round(best_thresh, 4), round(best_f1, 4)

    def run_controlled_experiment(
        self,
        exp_id: str,
        domain: str,
        data_yaml: str,
        model_size: str = "s",
        optimizer: str = "AdamW",
        lr0: float = 0.001,
        weight_decay: float = 0.0005,
        freeze_backbone: bool = False,
        freeze_layers: int = 10,
        epochs: int = 30,
        imgsz: int = 640,
        batch: int = 16,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes a controlled fine-tuning trial and records metrics into experiment log.
        """
        print(f"\n[EXPERIMENT: {exp_id}] Domain={domain} | Model=yolov8{model_size} | Opt={optimizer} | Freeze={freeze_backbone}")

        base_weights = f"yolov8{model_size}.pt"
        model = YOLO(base_weights)

        frozen_p = 0
        if freeze_backbone:
            frozen_p = self.configure_backbone_freezing(model, freeze_layers=freeze_layers)

        start_time = time.time()
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            optimizer=optimizer,
            lr0=lr0,
            weight_decay=weight_decay,
            freeze=freeze_layers if freeze_backbone else None,
            seed=seed,
            project="runs/tuning",
            name=exp_id,
            verbose=False,
            plots=True
        )
        elapsed = round(time.time() - start_time, 2)

        # Extract validation metrics
        val_metrics = results.results_dict if hasattr(results, "results_dict") else {}
        mAP50 = round(float(val_metrics.get("metrics/mAP50(B)", 0.0)), 4)
        mAP50_95 = round(float(val_metrics.get("metrics/mAP50-95(B)", 0.0)), 4)
        prec = round(float(val_metrics.get("metrics/precision(B)", 0.0)), 4)
        rec = round(float(val_metrics.get("metrics/recall(B)", 0.0)), 4)

        record = {
            "exp_id": exp_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "domain": domain,
            "hyperparameters": {
                "model_size": model_size,
                "optimizer": optimizer,
                "lr0": lr0,
                "weight_decay": weight_decay,
                "freeze_backbone": freeze_backbone,
                "freeze_layers": freeze_layers if freeze_backbone else 0,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
            },
            "metrics": {
                "mAP50": mAP50,
                "mAP50_95": mAP50_95,
                "precision": prec,
                "recall": rec,
                "training_duration_sec": elapsed,
            },
            "run_dir": f"runs/tuning/{exp_id}"
        }

        self.experiments.append(record)
        self._save_experiments()
        print(f"✅ Trial {exp_id} Complete | mAP50: {mAP50} | mAP50-95: {mAP50_95} | Time: {elapsed}s")
        return record

    def print_comparison_table(self):
        """Prints formatted comparison of all logged experiments."""
        if not self.experiments:
            print("No experiments logged yet.")
            return

        print("\n" + "=" * 100)
        print(f"{'Exp ID':<15} {'Domain':<15} {'Model':<8} {'Opt':<8} {'Freeze':<8} {'mAP50':<8} {'mAP50-95':<10} {'P':<8} {'R':<8} {'Time(s)':<8}")
        print("=" * 100)
        for exp in self.experiments:
            h = exp["hyperparameters"]
            m = exp["metrics"]
            freeze_str = f"L0-{h.get('freeze_layers', 0)}" if h.get("freeze_backbone") else "None"
            print(f"{exp['exp_id']:<15} {exp['domain']:<15} {h['model_size']:<8} {h['optimizer']:<8} {freeze_str:<8} {m['mAP50']:<8.4f} {m['mAP50_95']:<10.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['training_duration_sec']:<8.1f}")
        print("=" * 100)

