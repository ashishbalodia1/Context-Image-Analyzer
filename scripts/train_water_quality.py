"""
Fine-tuning pipeline for Water Quality (Floating Ocean/River Waste).
Trains YOLOv8 nano on authentic real-world aquatic waste imagery.
"""
import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def prepare_water_data():
    source_root = Path(os.path.expanduser("~/.cache/kagglehub/datasets/alexeydzhevello/ocean-waste-test-subset/versions/1/test"))
    dest_root = PROJECT_ROOT / "data" / "processed" / "water_quality"
    
    # Clean previous directory if exists
    if dest_root.exists():
        shutil.rmtree(dest_root)
        
    for split in ["train", "val", "test"]:
        (dest_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    imgs = sorted(list((source_root / "images").glob("*.JPG")) + list((source_root / "images").glob("*.jpg")))
    print(f"Total available real water images: {len(imgs)}")
    
    # 65 train, 16 val, 10 test
    train_imgs = imgs[:65]
    val_imgs = imgs[65:81]
    test_imgs = imgs[81:]
    
    splits = [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]
    
    for split_name, split_list in splits:
        for p in split_list:
            dest_img = dest_root / "images" / split_name / p.name
            shutil.copy2(p, dest_img)
            
            # Map label: class 0 in ocean dataset -> class 1 (floating_waste)
            src_lbl = source_root / "labels" / f"{p.stem}.txt"
            dest_lbl = dest_root / "labels" / split_name / f"{p.stem}.txt"
            
            if src_lbl.exists():
                with open(src_lbl, "r") as r_fp, open(dest_lbl, "w") as w_fp:
                    for line in r_fp:
                        parts = line.strip().split()
                        if parts:
                            # Map 0 to 1 (floating_waste)
                            w_fp.write(f"1 {' '.join(parts[1:])}\n")
            else:
                dest_lbl.touch()

    # Create data.yaml
    yaml_content = f"""path: {dest_root.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: water_discoloration
  1: floating_waste
  2: oil_film
  3: foam_accumulation
  4: illegal_discharge_pipe
"""
    yaml_path = dest_root / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Prepared water quality dataset with data.yaml at: {yaml_path}")
    return yaml_path

def train_model(yaml_path):
    print("Initiating YOLOv8n fine-tuning on real water quality dataset...")
    base_model = PROJECT_ROOT / "models" / "weights" / "yolov8n.pt"
    model = YOLO(str(base_model))
    
    results = model.train(
        data=str(yaml_path),
        epochs=12,
        imgsz=640,
        batch=16,
        device="cpu",
        workers=2,
        project=str(PROJECT_ROOT / "runs" / "detect" / "train_water"),
        name="water_quality_yolov8n",
        exist_ok=True,
        verbose=True
    )
    
    # Copy best weights to models/trained/water_quality_yolov8.pt
    best_weights = PROJECT_ROOT / "runs" / "detect" / "train_water" / "water_quality_yolov8n" / "weights" / "best.pt"
    target_weights = PROJECT_ROOT / "models" / "trained" / "water_quality_yolov8.pt"
    
    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"Successfully copied trained weights to {target_weights} ({target_weights.stat().st_size / 1e6:.2f} MB)")
    else:
        print("Training finished but best.pt not found!")

if __name__ == "__main__":
    y_path = prepare_water_data()
    train_model(y_path)
