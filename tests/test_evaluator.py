"""
Unit tests for ModelEvaluator.
Verifies evaluation metric extraction, latency profiling, and visual prediction gallery rendering.
"""
import sys
import tempfile
from pathlib import Path
from PIL import Image
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.evaluator import ModelEvaluator


def test_evaluator_metrics_and_gallery():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        dataset_dir = tmp / "test_dataset"
        runs_dir = tmp / "runs"
        gallery_dir = tmp / "gallery"

        # Create 2 test images and labels
        for split in ["train", "val", "test"]:
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)
            for i in range(2):
                img_p = dataset_dir / "images" / split / f"img_{i}.jpg"
                lbl_p = dataset_dir / "labels" / split / f"img_{i}.txt"
                im = Image.new("RGB", (128, 128), color=(80 + i * 30, 110, 140))
                im.save(img_p)
                lbl_p.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        yaml_content = {
            "path": str(dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": {0: "test_violation"}
        }
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        # Train a 1-epoch model to get valid .pt weights
        base_model = YOLO("yolov8n.yaml")
        results = base_model.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=64,
            batch=2,
            workers=0,
            device="cpu",
            project=str(runs_dir),
            name="eval_test_run",
            verbose=False,
            plots=False
        )
        weights_path = runs_dir / "eval_test_run" / "weights" / "last.pt"
        assert weights_path.exists()

        # Run evaluation on test split
        evaluator = ModelEvaluator(str(weights_path))
        summary = evaluator.evaluate(
            data_yaml=str(data_yaml),
            split="test",
            imgsz=64,
            device="cpu"
        )

        assert "global_metrics" in summary
        assert "mAP50" in summary["global_metrics"]
        assert "latency" in summary
        assert summary["latency"]["fps"] >= 0

        # Generate visual gallery
        gallery = evaluator.generate_visual_gallery(
            test_images_dir=dataset_dir / "images" / "test",
            output_dir=gallery_dir,
            conf_threshold=0.10
        )
        assert len(gallery) == 2
        for g in gallery:
            assert Path(g).exists()
            assert Path(g).stat().st_size > 0


if __name__ == "__main__":
    test_evaluator_metrics_and_gallery()
    print("MODEL EVALUATOR UNIT TESTS PASSED (100% SUCCESS)!")

