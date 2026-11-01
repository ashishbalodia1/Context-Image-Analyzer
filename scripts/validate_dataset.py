"""
CLI utility to audit and validate any YOLOv8 dataset directory.
Usage:
  python scripts/validate_dataset.py --data-dir data/processed/solid_waste --yaml models/configs/solid_waste.yaml
"""
import sys
from pathlib import Path

# Ensure project root in sys.path
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
from src.preprocessing.dataset_validator import DatasetValidator, print_validation_summary

def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 Dataset for City Guardian")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder containing images/ and labels/")
    parser.add_argument("--yaml", type=str, default=None, help="Path to domain data.yaml or models/configs/<domain>.yaml")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}")
        sys.exit(1)

    print(f"Auditing YOLO dataset at: {data_dir}...")
    validator = DatasetValidator(args.yaml)
    report = validator.generate_comprehensive_report(data_dir)
    print_validation_summary(report)

if __name__ == "__main__":
    main()

