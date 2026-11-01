"""
Production YOLOv8 Fine-Tuning & Training Pipeline for City Guardian.
Supports transfer learning, optimizer selection (AdamW vs SGD), cosine LR scheduling,
early stopping, reproducible seeding, and automated metric tracking.

Usage:
  python scripts/train_model.py --domain solid_waste --model-size s --epochs 50 --batch 16 --optimizer AdamW
"""
import os
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
from typing import Optional, Dict, Any
import yaml
import torch
from ultralytics import YOLO

from src.preprocessing.image_enhancer import get_domain_augmentation_hyperparameters
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_environmental_model(
    domain: str,
    data_yaml: Optional[str] = None,
    model_size: str = "s",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    lrf: float = 0.01,
    patience: int = 20,
    cos_lr: bool = True,
    seed: int = 42,
    project_dir: str = "runs/train",
    workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes domain-specific YOLOv8 fine-tuning with full hyperparameter controls.
    """
    print("=" * 80)
    print(f"CITY GUARDIAN — YOLOv8 MODEL TRAINING: [{domain.upper()}]")
    print("=" * 80)

    # 1. Resolve dataset config path
    if data_yaml is None:
        processed_yaml = PROJECT_ROOT / "data" / "processed" / domain / "data.yaml"
        if processed_yaml.exists():
            data_yaml = str(processed_yaml)
        else:
            data_yaml = str(PROJECT_ROOT / "models" / "configs" / f"{domain}.yaml")

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_yaml}")

    # 2. Hardware and device auto-detection
    device = "0" if torch.cuda.is_available() else "cpu"
    num_cpus = os.cpu_count() or 4
    if workers is None:
        workers = min(4, num_cpus) if device == "cpu" else min(8, num_cpus)

    # 3. Base model selection (transfer learning from COCO)
    weights_dir = PROJECT_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    base_model_name = f"yolov8{model_size}.pt"
    base_model_path = weights_dir / base_model_name

    print(f"Hardware Profile:")
    print(f"  Target Device:    {device.upper()}")
    print(f"  CPU Worker Threads: {workers}")
    print(f"  Base Model:       {base_model_name}")
    print(f"  Dataset Config:   {data_yaml}")
    print(f"  Optimizer:        {optimizer} (lr0={lr0}, lrf={lrf})")
    print(f"  Cosine LR:        {cos_lr} | Early Stopping Patience: {patience}")
    print(f"  Random Seed:      {seed}")

    # Load base model (downloads to weights_dir if not cached)
    model = YOLO(str(base_model_path) if base_model_path.exists() else base_model_name)

    # 4. Ingest domain-specific augmentation hyperparameters
    aug_hyp = get_domain_augmentation_hyperparameters(domain)

    # 5. Launch fine-tuning
    exp_name = f"{domain}_yolov8{model_size}"
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        patience=patience,
        seed=seed,
        workers=workers,
        device=device,
        project=project_dir,
        name=exp_name,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        **aug_hyp  # Unpack domain-specific augmentation parameters
    )

    # 6. Copy best weights to models/trained/
    trained_dir = PROJECT_ROOT / "models" / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)
    save_run_dir = Path(project_dir) / exp_name
    best_pt = save_run_dir / "weights" / "best.pt"

    dest_best_pt = trained_dir / f"{domain}_yolov8.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(str(best_pt), str(dest_best_pt))
        print(f"\n✅ Production checkpoint saved: {dest_best_pt}")

    return {
        "domain": domain,
        "base_model": base_model_name,
        "run_directory": str(save_run_dir),
        "best_weights": str(dest_best_pt) if dest_best_pt.exists() else None,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Fine-Tuning Pipeline for City Guardian")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["air_pollution", "water_quality", "sewage_drainage", "mining_dust", "solid_waste"])
    parser.add_argument("--data-yaml", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--model-size", type=str, default="s", choices=["n", "s", "m", "l", "x"], help="Model size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD", "Adam"])
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    train_environmental_model(
        domain=args.domain,
        data_yaml=args.data_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
