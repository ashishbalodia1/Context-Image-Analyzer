"""
Real Dataset Ingestion & Converter for City Guardian.
Processes:
  1. TACO Trash Dataset (kneroma/tacotrashdataset) -> Converts COCO to YOLO, maps 60 categories to 4 solid_waste classes.
  2. Floating Ocean/River Waste (alexeydzhevello/ocean-waste-test-subset) -> Ingests into water_quality.
  3. Daily Solid Waste Municipal Dataset (ivantha/daily-solid-waste-dataset) -> Ingests municipal time-series into data/raw/solid_waste/municipal_timeseries/.
"""
import os
import sys
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Production class mapping for Solid Waste
# 0: plastic_waste, 1: organic_waste, 2: illegal_dumping, 3: overflowing_dustbin
TACO_CATEGORY_MAP = {
    # Plastic waste (0)
    'Other plastic bottle': 0, 'Clear plastic bottle': 0, 'Plastic bottle cap': 0,
    'Disposable plastic cup': 0, 'Other plastic cup': 0, 'Plastic lid': 0,
    'Other plastic': 0, 'Plastified paper bag': 0, 'Plastic film': 0,
    'Six pack rings': 0, 'Garbage bag': 0, 'Other plastic wrapper': 0,
    'Single-use carrier bag': 0, 'Polypropylene bag': 0, 'Crisp packet': 0,
    'Spread tub': 0, 'Tupperware': 0, 'Disposable food container': 0,
    'Foam food container': 0, 'Other plastic container': 0, 'Plastic glooves': 0,
    'Plastic utensils': 0, 'Plastic straw': 0, 'Styrofoam piece': 0,

    # Organic / Paper waste (1)
    'Food waste': 1, 'Other carton': 1, 'Egg carton': 1, 'Drink carton': 1,
    'Corrugated carton': 1, 'Meal carton': 1, 'Pizza box': 1, 'Paper cup': 1,
    'Magazine paper': 1, 'Tissues': 1, 'Wrapping paper': 1, 'Normal paper': 1,
    'Paper bag': 1, 'Toilet tube': 1, 'Paper straw': 1,

    # Illegal dumping / Mixed debris (2)
    'Broken glass': 2, 'Glass bottle': 2, 'Glass jar': 2, 'Glass cup': 2,
    'Aluminium foil': 2, 'Battery': 2, 'Aluminium blister pack': 2,
    'Carded blister pack': 2, 'Food Can': 2, 'Aerosol': 2, 'Drink can': 2,
    'Metal bottle cap': 2, 'Metal lid': 2, 'Pop tab': 2, 'Rope & strings': 2,
    'Scrap metal': 2, 'Shoe': 2, 'Squeezable tube': 2, 'Unlabeled litter': 2,
    'Cigarette': 2
}


def ingest_taco(max_images: int = 400):
    """
    Ingests real images from TACO dataset, converting COCO format to YOLOv8.
    """
    taco_root = Path(r"C:\Users\ashis\.cache\kagglehub\datasets\kneroma\tacotrashdataset\versions\3\data")
    if not taco_root.exists():
        logger.error(f"TACO dataset not found at {taco_root}")
        return

    dest_images = PROJECT_ROOT / "data" / "raw" / "solid_waste" / "images"
    dest_labels = PROJECT_ROOT / "data" / "raw" / "solid_waste" / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    # Clean existing placeholder files if any
    for f in dest_images.glob("ward_*_sample_*.jpg"):
        f.unlink()
    for f in dest_labels.glob("ward_*_sample_*.txt"):
        f.unlink()

    ann_file = taco_root / "annotations.json"
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image id lookup
    images_dict = {img["id"]: img for img in coco["images"]}

    # Build category id lookup
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image_id
    ann_by_img = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        ann_by_img.setdefault(img_id, []).append(ann)

    print(f"\n[INGEST] Processing TACO real trash dataset...")
    count = 0
    total_boxes = 0

    for img_id, img_info in images_dict.items():
        if count >= max_images:
            break

        file_name = img_info["file_name"]
        src_img_path = taco_root / file_name

        if not src_img_path.exists():
            continue

        width = img_info.get("width")
        height = img_info.get("height")
        if not width or not height or width <= 0 or height <= 0:
            continue

        anns = ann_by_img.get(img_id, [])
        yolo_lines = []

        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"])
            target_cls = TACO_CATEGORY_MAP.get(cat_name, 2)  # default to illegal_dumping

            bbox = ann.get("bbox")  # [x, y, w, h] in absolute pixels
            if not bbox or len(bbox) != 4:
                continue

            bx, by, bw, bh = bbox
            if bw <= 0 or bh <= 0:
                continue

            # Convert to normalized YOLO coordinates
            x_center = (bx + bw / 2.0) / width
            y_center = (by + bh / 2.0) / height
            norm_w = bw / width
            norm_h = bh / height

            # Bound strictly to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            yolo_lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if not yolo_lines:
            continue

        # Save to data/raw/solid_waste
        clean_stem = f"taco_{img_id:05d}"
        target_img_p = dest_images / f"{clean_stem}.jpg"
        target_lbl_p = dest_labels / f"{clean_stem}.txt"

        shutil.copy2(src_img_path, target_img_p)
        target_lbl_p.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        count += 1
        total_boxes += len(yolo_lines)

    print(f"  -> Ingested {count} real photographic images and {total_boxes} YOLO annotations into data/raw/solid_waste/")


def ingest_ocean_waste():
    """
    Ingests real floating waste images and annotations into water_quality.
    """
    ocean_root = Path(r"C:\Users\ashis\.cache\kagglehub\datasets\alexeydzhevello\ocean-waste-test-subset\versions\1\test")
    if not ocean_root.exists():
        logger.error(f"Ocean waste dataset not found at {ocean_root}")
        return

    dest_images = PROJECT_ROOT / "data" / "raw" / "water_quality" / "images"
    dest_labels = PROJECT_ROOT / "data" / "raw" / "water_quality" / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    # Clean existing placeholders
    for f in dest_images.glob("ward_*_sample_*.jpg"):
        f.unlink()
    for f in dest_labels.glob("ward_*_sample_*.txt"):
        f.unlink()

    src_imgs = list((ocean_root / "images").glob("*"))
    print(f"\n[INGEST] Processing Ocean & River Floating Waste dataset ({len(src_imgs)} files)...")

    count = 0
    for img_p in src_imgs:
        lbl_p = ocean_root / "labels" / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        target_img = dest_images / f"ocean_{img_p.name}"
        target_lbl = dest_labels / f"ocean_{img_p.stem}.txt"

        shutil.copy2(img_p, target_img)
        # Class 1 is floating_waste in water_quality domain
        raw_lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        remapped_lines = []
        for line in raw_lines:
            parts = line.split()
            if len(parts) >= 5:
                # Set class to 1 (floating_waste)
                remapped_lines.append(f"1 {' '.join(parts[1:])}")

        target_lbl.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
        count += 1

    print(f"  -> Ingested {count} real floating waste images and annotations into data/raw/water_quality/")


def ingest_daily_solid_waste():
    """
    Ingests municipal waste time-series CSVs into data/raw/solid_waste/municipal_timeseries/
    """
    csv_root = Path(r"C:\Users\ashis\.cache\kagglehub\datasets\ivantha\daily-solid-waste-dataset\versions\1")
    if not csv_root.exists():
        logger.error(f"Daily solid waste dataset not found at {csv_root}")
        return

    dest_dir = PROJECT_ROOT / "data" / "raw" / "solid_waste" / "municipal_timeseries"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(csv_root.glob("*.csv"))
    print(f"\n[INGEST] Copying {len(csv_files)} Daily Solid Waste municipal time-series CSVs...")

    for f in csv_files:
        dest_p = dest_dir / f.name
        shutil.copy2(f, dest_p)
        print(f"   + Ingested: {f.name} ({round(f.stat().st_size / 1024, 1)} KB)")


def main():
    print("=" * 70)
    print(" CITY GUARDIAN — REAL KAGGLE DATASET INGESTION PIPELINE")
    print("=" * 70)

    # 1. Ingest TACO
    ingest_taco(max_images=500)

    # 2. Ingest Ocean Waste
    ingest_ocean_waste()

    # 3. Ingest Municipal Time Series
    ingest_daily_solid_waste()

    print("\n" + "=" * 70)
    print(" ALL REAL DATASETS INGESTED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

