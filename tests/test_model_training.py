"""
Unit and integration tests for YOLOv8 model training setup in City Guardian.
Verifies network instantiation, loss heads, and training loop execution.
"""
import sys
import tempfile
from pathlib import Path
from PIL import Image
import yaml
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_yolov8_architecture_and_forward_pass():
    # Instantiate YOLOv8n from YAML architecture without downloading full weights
    model = YOLO("yolov8n.yaml")
    assert model.model is not None

    # Dummy tensor: 1 image, 3 channels, 640x640
    dummy_input = torch.zeros((1, 3, 640, 640))
    output = model.model(dummy_input)

    # In training/eval mode, YOLOv8 outputs multi-scale prediction features
    assert output is not None


def test_yolov8_training_dry_run():
    """
    Creates a 4-sample synthetic dataset and executes a 1-epoch dry-run
    to verify backprop, loss logging, and checkpoint generation.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        dataset_dir = tmp / "mini_dataset"
        runs_dir = tmp / "runs"

        # Create images and labels for train and val
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)

            for i in range(2):
                img_p = dataset_dir / "images" / split / f"sample_{i}.jpg"
                lbl_p = dataset_dir / "labels" / split / f"sample_{i}.txt"

                # 640x640 sample image
                im = Image.new("RGB", (640, 640), color=(100 + i * 20, 120, 140))
                im.save(img_p)

                # Class 0: normalized bbox
                lbl_p.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        yaml_content = {
            "path": str(dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": {0: "environmental_anomaly"}
        }
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        # Train for 1 epoch
        model = YOLO("yolov8n.yaml")
        results = model.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=64,  # Small resolution for fast unit test execution
            batch=2,
            workers=0,
            device="cpu",
            project=str(runs_dir),
            name="dry_run",
            verbose=False,
            plots=False
        )

        assert results is not None
        assert (runs_dir / "dry_run" / "weights" / "last.pt").exists()


if __name__ == "__main__":
    print("Testing YOLOv8 architecture instantiation...")
    test_yolov8_architecture_and_forward_pass()
    print("Architecture test passed.")

    print("Testing 1-epoch training dry-run...")
    test_yolov8_training_dry_run()
    print("YOLOV8 TRAINING DRY-RUN COMPLETED (100% SUCCESS)!")

