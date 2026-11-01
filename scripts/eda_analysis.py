"""
CLI Script to execute Exploratory Data Analysis (EDA) on YOLOv8 datasets.
Generates statistical metrics and visual diagnostic plots.

Usage:
  python scripts/eda_analysis.py --labels-dir data/processed/solid_waste/labels/train --yaml models/configs/solid_waste.yaml --domain solid_waste
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
from src.preprocessing.eda_analyzer import DatasetEDAAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis (EDA) on YOLO dataset")
    parser.add_argument("--domain", type=str, default="solid_waste", help="Name of environmental domain")
    parser.add_argument("--labels-dir", type=str, default=None, help="Path to folder containing YOLO .txt labels")
    parser.add_argument("--yaml", type=str, default=None, help="Path to domain data.yaml or config")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to save generated plots")
    args = parser.parse_args()

    domain = args.domain
    labels_dir = Path(args.labels_dir) if args.labels_dir else PROJECT_ROOT / "data" / "processed" / domain / "labels" / "train"
    yaml_path = Path(args.yaml) if args.yaml else PROJECT_ROOT / "data" / "processed" / domain / "data.yaml"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "reports" / "eda" / domain

    analyzer = DatasetEDAAnalyzer(str(yaml_path) if yaml_path.exists() else None)
    print(f"Running EDA on labels in: {labels_dir}...")
    metrics = analyzer.analyze_dataset(labels_dir)

    print("\n" + "=" * 60)
    print(f"EDA REPORT: {args.domain.upper()}")
    print("=" * 60)
    print(f"Total Images:           {metrics.get('total_images', 0)}")
    print(f"Total Bounding Boxes:   {metrics.get('total_boxes', 0)}")
    print(f"Empty/Background Images: {metrics.get('empty_images', 0)}")
    print(f"Avg Objects per Image:  {metrics.get('avg_boxes_per_image', 0)}")

    print("\nClass Distribution:")
    for cls_name, count in metrics.get("class_distribution", {}).items():
        pct = (count / metrics['total_boxes'] * 100) if metrics.get('total_boxes') else 0
        print(f"  - {cls_name:25s}: {count:5d} ({pct:5.1f}%)")

    print("\nBounding Box Scales:")
    for scale, frac in metrics.get("box_scales", {}).items():
        print(f"  - {scale:25s}: {frac * 100:5.1f}%")

    if metrics.get("total_boxes", 0) > 0:
        plots = analyzer.generate_eda_plots(metrics, output_dir, domain_name=domain)
        print(f"\nGenerated {len(plots)} Diagnostic Plots in: {output_dir}")
        for p in plots:
            print(f"  📈 {Path(p).name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

