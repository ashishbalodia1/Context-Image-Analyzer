"""
Dataset Download & Ingestion Helper Script for City Guardian.
Provides verified dataset registries and download commands for authentic environmental datasets:
  - TACO Trash (kneroma/tacotrashdataset)
  - Daily Solid Waste Municipal Tonnage (ivantha/daily-solid-waste-dataset)
  - Floating Ocean & River Waste (alexeydzhevello/ocean-waste-test-subset)
  - D-Fire Fire & Smoke (DFireDataset)
  - FASDD Fire & Smoke (seawsurf/fire_smoke_dataset_fasdd_cv)
  - RDD2022 Road & Surface Damage (RDD2022)

Usage:
  python scripts/download_datasets.py --list
  python scripts/download_datasets.py --download-kaggle
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASETS = {
    "TACO (Trash Annotations in Context)": {
        "domain": "solid_waste",
        "description": "1,500 real-world photographic images with 4,784 bounding box annotations across 60 litter classes.",
        "verified_link": "https://www.kaggle.com/datasets/kneroma/tacotrashdataset",
        "kagglehub_handle": "kneroma/tacotrashdataset",
        "format": "COCO annotations & real JPG photos"
    },
    "Daily Solid Waste Municipal Dataset": {
        "domain": "solid_waste (Context)",
        "description": "Historical time-series of daily municipal solid waste generation tonnage across 6 municipal councils.",
        "verified_link": "https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset",
        "kagglehub_handle": "ivantha/daily-solid-waste-dataset",
        "format": "Tabular CSV records"
    },
    "Floating Ocean & River Waste": {
        "domain": "water_quality",
        "description": "Real-world water surface floating debris, plastics, and surface contaminants.",
        "verified_link": "https://www.kaggle.com/datasets/alexeydzhevello/ocean-waste-test-subset",
        "kagglehub_handle": "alexeydzhevello/ocean-waste-test-subset",
        "format": "YOLO format (images & labels)"
    },
    "D-Fire (Fire & Smoke Detection)": {
        "domain": "air_pollution",
        "description": "Real fire and industrial smoke detection dataset containing outdoor and industrial images.",
        "verified_link": "https://github.com/gaiasd/DFireDataset",
        "instructions": "git clone https://github.com/gaiasd/DFireDataset.git data/raw/air_pollution/D-Fire",
        "format": "YOLO format (images & labels)"
    },
    "FASDD (Fire and Smoke Dataset CV)": {
        "domain": "air_pollution",
        "description": "Comprehensive aerial and ground fire/smoke dataset on Hugging Face S3 Bucket.",
        "verified_link": "https://s3.hf.co/Ashishbalodia",
        "instructions": "python scripts/ingest_custom_datasets.py --fasdd",
        "format": "YOLO / Images & Labels"
    },
    "RDD2022 (Road Damage Dataset 2022)": {
        "domain": "mining_dust & road condition",
        "description": "Road damage & surface deterioration dataset from Roboflow Universe.",
        "verified_link": "https://universe.roboflow.com/rdd2022/road-damage-iztzn",
        "instructions": "python scripts/ingest_custom_datasets.py --rdd",
        "format": "YOLOv8 txt"
    }
}


def print_guide():
    print("=" * 80)
    print("CITY GUARDIAN — REAL ENVIRONMENTAL DATASET REGISTRY")
    print("=" * 80)
    for name, info in DATASETS.items():
        print(f"\n📂 Dataset: {name}")
        print(f"   Domain:       {info['domain']}")
        print(f"   Description:  {info['description']}")
        print(f"   Primary Link: {info['verified_link']}")
        if "kagglehub_handle" in info:
            print(f"   KaggleHub:    kagglehub.dataset_download('{info['kagglehub_handle']}')")
        if "instructions" in info:
            print(f"   Download Cmd: {info['instructions']}")
        print("-" * 80)


def download_kaggle_datasets():
    """Triggers automated download and ingestion of real Kaggle datasets."""
    from scripts.ingest_kaggle_datasets import main as ingest_main
    print("[DOWNLOAD] Launching automated Kaggle dataset ingestion pipeline...")
    ingest_main()


def main():
    parser = argparse.ArgumentParser(description="City Guardian Real Dataset Registry & Ingestion Tool")
    parser.add_argument("--list", action="store_true", help="List all verified real datasets")
    parser.add_argument("--download-kaggle", action="store_true", help="Download and ingest real Kaggle datasets")

    args = parser.parse_args()

    if args.download_kaggle:
        download_kaggle_datasets()
    else:
        print_guide()


if __name__ == "__main__":
    main()
