"""
Automated Ingestion Pipeline for:
1. Fire & Smoke Dataset (FASDD) via Hugging Face S3 Bucket (boto3)
2. Road Damage Dataset (RDD2022) via Roboflow Universe (roboflow SDK)

Saves and structures into City Guardian's data pipeline:
- data/raw/air_pollution/ & data/processed/air_pollution/
- data/raw/mining_dust/   & data/processed/mining_dust/
"""
import os
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try loading from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def download_fasdd_s3(
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bucket_name: str = "fire_smoke_dataset_fasdd_cv-bucket",
    endpoint_url: str = "https://s3.hf.co/Ashishbalodia",
    output_raw: Path = PROJECT_ROOT / "data" / "raw" / "air_pollution",
    max_files: int = 500
) -> bool:
    """
    Downloads fire & smoke dataset from user's Hugging Face S3 bucket.
    """
    import boto3
    from botocore.config import Config

    ak = access_key or os.getenv("AWS_ACCESS_KEY_ID")
    sk = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")

    if not ak or not sk:
        logger.error(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to access "
            f"HF S3 endpoint {endpoint_url}. Please set them in your environment or .env file."
        )
        return False

    logger.info(f"Connecting to S3 endpoint: {endpoint_url} (Bucket: {bucket_name})...")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=Config(
            region_name="us-east-1",
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )

    output_raw.mkdir(parents=True, exist_ok=True)
    images_dir = output_raw / "images"
    labels_dir = output_raw / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Listing objects in bucket '{bucket_name}'...")
        paginator = s3.get_paginator('list_objects_v2')
        count = 0

        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Determine destination
                if key.endswith(('.jpg', '.jpeg', '.png')):
                    dest = images_dir / Path(key).name
                elif key.endswith('.txt'):
                    dest = labels_dir / Path(key).name
                elif key.endswith('.zip'):
                    dest = output_raw / Path(key).name
                else:
                    dest = output_raw / key

                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading [{count + 1}] {key} -> {dest.name}")
                s3.download_file(bucket_name, key, str(dest))
                count += 1

                # If zip file, extract it
                if dest.suffix == '.zip':
                    logger.info(f"Extracting {dest.name}...")
                    with zipfile.ZipFile(dest, 'r') as zip_ref:
                        zip_ref.extractall(output_raw)
                    dest.unlink()

                if count >= max_files:
                    logger.info(f"Reached max limit of {max_files} files.")
                    break
            if count >= max_files:
                break

        logger.info(f"Successfully downloaded {count} items for air_pollution (FASDD).")
        return count > 0

    except Exception as e:
        logger.error(f"Failed to download from S3 bucket '{bucket_name}': {e}")
        return False


def download_road_damage_roboflow(
    api_key: Optional[str] = None,
    workspace: str = "rdd2022",
    project_slug: str = "road-damage-iztzn",
    version: int = 1,
    output_raw: Path = PROJECT_ROOT / "data" / "raw" / "mining_dust"
) -> bool:
    """
    Downloads Road Damage dataset from Roboflow Universe using official roboflow package.
    """
    rf_key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not rf_key:
        logger.error(
            "ROBOFLOW_API_KEY is required to download dataset from Roboflow Universe. "
            "Please set ROBOFLOW_API_KEY in your environment or .env file."
        )
        return False

    try:
        from roboflow import Roboflow
        logger.info(f"Authenticating with Roboflow Universe (Project: {workspace}/{project_slug} v{version})...")
        rf = Roboflow(api_key=rf_key)
        proj = rf.workspace(workspace).project(project_slug)
        v = proj.version(version)

        temp_dir = PROJECT_ROOT / "data" / "temp_roboflow_rdd"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        logger.info("Downloading dataset in YOLOv8 format from Roboflow...")
        dataset = v.download("yolov8", location=str(temp_dir))

        output_raw.mkdir(parents=True, exist_ok=True)
        # Move files to data/raw/mining_dust
        for item in temp_dir.iterdir():
            dest = output_raw / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(output_raw))

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        logger.info(f"Successfully ingested Road Damage dataset into {output_raw}.")
        return True

    except Exception as e:
        logger.error(f"Failed to download from Roboflow Universe: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest FASDD Fire/Smoke and RDD2022 Road Damage datasets")
    parser.add_argument("--fasdd", action="store_true", help="Download FASDD fire/smoke dataset from HF S3")
    parser.add_argument("--rdd", action="store_true", help="Download Road Damage dataset from Roboflow")
    parser.add_argument("--all", action="store_true", help="Download both datasets")
    parser.add_argument("--aws-key", type=str, default=None, help="AWS Access Key ID for HF S3")
    parser.add_argument("--aws-secret", type=str, default=None, help="AWS Secret Access Key for HF S3")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key")

    args = parser.parse_args()

    if not (args.fasdd or args.rdd or args.all):
        parser.print_help()
        print("\nNote: Please provide credentials via flags or environment variables (.env file).")
        return

    if args.fasdd or args.all:
        logger.info("Starting FASDD Fire & Smoke Ingestion...")
        download_fasdd_s3(access_key=args.aws_key, secret_key=args.aws_secret)

    if args.rdd or args.all:
        logger.info("Starting RDD2022 Road Damage Ingestion...")
        download_road_damage_roboflow(api_key=args.roboflow_key)


if __name__ == "__main__":
    main()

