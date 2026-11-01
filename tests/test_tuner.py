"""
Unit tests for HyperparameterTuner.
Verifies layer freezing, F1 threshold optimization, and experiment persistence.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.tuner import HyperparameterTuner


def test_backbone_freezing():
    model = YOLO("yolov8n.yaml")
    frozen_p = HyperparameterTuner.configure_backbone_freezing(model, freeze_layers=10)
    assert frozen_p > 0, "Parameters in first 10 layers must be frozen"

    # Verify that requires_grad is False for frozen layers
    for name, param in model.model.named_parameters():
        if any(f"model.{i}." in name for i in range(10)):
            assert param.requires_grad is False
        elif "model.22." in name:  # Detection head
            assert param.requires_grad is True


def test_optimal_confidence_threshold_search():
    # Synthetic precision and recall curves across confidence thresholds [0.1 to 0.9]
    thresholds = np.linspace(0.1, 0.9, 9)
    # Synthetic: precision rises with threshold, recall drops
    precision = np.array([0.4, 0.5, 0.65, 0.78, 0.85, 0.90, 0.92, 0.95, 0.98])
    recall = np.array([0.95, 0.90, 0.85, 0.80, 0.72, 0.60, 0.45, 0.30, 0.15])

    best_thresh, best_f1 = HyperparameterTuner.find_optimal_confidence_threshold(
        precision, recall, thresholds
    )

    assert 0.3 <= best_thresh <= 0.6, f"Expected threshold around ~0.4-0.5, got {best_thresh}"
    assert best_f1 > 0.75, f"Expected strong peak F1, got {best_f1}"


def test_experiment_log_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "test_experiments.json"
        tuner = HyperparameterTuner(experiment_log_path=str(log_file))

        # Append dummy record
        tuner.experiments.append({
            "exp_id": "test_exp_01",
            "domain": "solid_waste",
            "hyperparameters": {"model_size": "s", "optimizer": "AdamW", "freeze_backbone": True, "freeze_layers": 10},
            "metrics": {"mAP50": 0.752, "mAP50_95": 0.514, "precision": 0.81, "recall": 0.72, "training_duration_sec": 45.2}
        })
        tuner._save_experiments()

        # Reload from disk
        tuner_reloaded = HyperparameterTuner(experiment_log_path=str(log_file))
        assert len(tuner_reloaded.experiments) == 1
        assert tuner_reloaded.experiments[0]["exp_id"] == "test_exp_01"


if __name__ == "__main__":
    test_backbone_freezing()
    test_optimal_confidence_threshold_search()
    test_experiment_log_persistence()
    print("HYPERPARAMETER TUNER UNIT TESTS PASSED (100% SUCCESS)!")

