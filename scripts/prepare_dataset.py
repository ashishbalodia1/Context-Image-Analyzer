"""
Dataset preparation script for City Guardian.
Cleans raw inputs, validates annotations, isolates invalid labels/images,
and partitions datasets using group-aware splitting into standard YOLOv8 layout:

  data/processed/<domain>/
    ├── images/{train, val, test}
    ├── labels/{train, val, test}
    └── data.yaml

Usage:
  python scripts/prepare_dataset.py --domain solid_waste --raw-images data/raw/solid_waste/images --raw-labels data/raw/solid_waste/labels
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
import yaml
from src.preprocessing.dataset_cleaner import DatasetCleaner
from src.preprocessing.dataset_validator import DatasetValidator, print_validation_summary

DOMAINS_CONFIG = PROJECT_ROOT / "config" / "domains.yaml"

def prepare_domain_dataset(
    domain: str,
    raw_images_dir: Path,
    raw_labels_dir: Path,
    output_dir: Path,
    group_delimiter: str = "_",
    quarantine: bool = True
):
    print(f"\n=======================================================")
    print(f"Preparing Dataset for Domain: [{domain.upper()}]")
    print(f"Raw Images:  {raw_images_dir}")
    print(f"Raw Labels:  {raw_labels_dir}")
    print(f"Destination: {output_dir}")
    print(f"=======================================================\n")

    # Load domain config
    with open(DOMAINS_CONFIG, "r", encoding="utf-8") as f:
        domains_data = yaml.safe_load(f)["domains"]

    if domain not in domains_data:
        raise ValueError(f"Domain '{domain}' not found in domains.yaml. Available: {list(domains_data.keys())}")

    classes = [c["name"] for c in domains_data[domain]["classes"]]
    cleaner = DatasetCleaner()

    # Step 1: Quarantine invalid files if requested
    if quarantine:
        quarantine_dir = output_dir.parent / "quarantine" / domain
        print(f"Checking data integrity & isolating corrupt items to {quarantine_dir}...")
        quarantined = cleaner.clean_and_quarantine(
            raw_images_dir,
            raw_labels_dir,
            quarantine_dir
        )
        print(f"Quarantined: {quarantined}")

    # Step 2: Discover valid images
    validator = DatasetValidator()
    image_files = [
        f for f in raw_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in validator.SUPPORTED_IMG_EXTS
    ]

    if not image_files:
        print(f"⚠️  No images found in {raw_images_dir}. Place raw dataset files here first.")
        return

    # Step 3: Group-aware split (70 / 20 / 10)
    print(f"Partitioning {len(image_files)} images using group-aware clustering (70% train / 20% val / 10% test)...")
    train_imgs, val_imgs, test_imgs = cleaner.group_aware_split(
        image_files,
        group_delimiter=group_delimiter,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10
    )
    print(f"Split results: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    # Step 4: Export formatted YOLO directory
    yaml_path = cleaner.export_yolo_dataset(
        output_dir=output_dir,
        train_images=train_imgs,
        val_images=val_imgs,
        test_images=test_imgs,
        source_labels_dir=raw_labels_dir,
        class_names=classes,
        domain_name=domain
    )
    print(f"✅ Generated YOLO data.yaml: {yaml_path}")

    # Step 5: Final Validation Audit on exported dataset
    print("\nRunning post-export validation audit...")
    report = validator.generate_comprehensive_report(output_dir)
    print_validation_summary(report)

def main():
    parser = argparse.ArgumentParser(description="Prepare and split YOLOv8 domain dataset")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["air_pollution", "water_quality", "sewage_drainage", "mining_dust", "solid_waste"])
    parser.add_argument("--raw-images", type=str, default=None, help="Path to raw images folder (defaults to data/raw/<domain>/images)")
    parser.add_argument("--raw-labels", type=str, default=None, help="Path to raw labels folder (defaults to data/raw/<domain>/labels)")
    parser.add_argument("--out-dir", type=str, default=None, help="Destination directory (defaults to data/processed/<domain>)")
    parser.add_argument("--delimiter", type=str, default="_", help="Prefix delimiter for group-aware clustering")
    args = parser.parse_args()

    raw_images = Path(args.raw_images) if args.raw_images else PROJECT_ROOT / "data" / "raw" / args.domain / "images"
    raw_labels = Path(args.raw_labels) if args.raw_labels else PROJECT_ROOT / "data" / "raw" / args.domain / "labels"
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "data" / "processed" / args.domain
    prepare_domain_dataset(
        domain=args.domain,
        raw_images_dir=raw_images,
        raw_labels_dir=raw_labels,
        output_dir=out_dir,
        group_delimiter=args.delimiter
    )

if __name__ == "__main__":
    main()
