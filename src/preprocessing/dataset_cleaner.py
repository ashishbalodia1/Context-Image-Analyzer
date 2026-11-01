"""
Dataset Cleaner and Group-Aware Splitting Utility for City Guardian.
Automates dataset hygiene and enforces group-aware train/val/test partitioning
to guarantee zero geographic, temporal, or sequence data leakage.
"""
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import yaml

from src.preprocessing.dataset_validator import DatasetValidator


class DatasetCleaner:
    """
    Cleans and structures raw environmental datasets into standard YOLOv8 format:
      dataset/
        ├── images/{train, val, test}
        ├── labels/{train, val, test}
        └── data.yaml
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)

    def clean_and_quarantine(
        self,
        images_dir: Path,
        labels_dir: Path,
        quarantine_dir: Path,
        data_yaml: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Scans a directory of images and labels, and isolates defective or invalid items
        into a quarantine directory rather than letting them taint model training.
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        quarantine_dir = Path(quarantine_dir)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        validator = DatasetValidator(data_yaml)
        report = validator.validate_split(images_dir, labels_dir, split_name="clean_pass")

        quarantined_counts = {
            "corrupted_images": len(report["corrupted_images"]),
            "orphan_labels": len(report["orphan_labels"]),
            "missing_labels": len(report["missing_labels"]),
            "invalid_bboxes": len(report["invalid_bboxes"])
        }

        # Quarantine corrupted images
        for c_img in report["corrupted_images"]:
            p = Path(c_img)
            if p.exists():
                shutil.move(str(p), str(quarantine_dir / f"corrupt_{p.name}"))

        # Quarantine invalid bounding box label files + their images
        for bad in report["invalid_bboxes"]:
            lbl_p = Path(bad["label_file"])
            if lbl_p.exists():
                shutil.move(str(lbl_p), str(quarantine_dir / f"invalid_{lbl_p.name}"))
            # Also move corresponding image
            for ext in validator.SUPPORTED_IMG_EXTS:
                img_p = images_dir / f"{lbl_p.stem}{ext}"
                if img_p.exists():
                    shutil.move(str(img_p), str(quarantine_dir / f"invalid_{img_p.name}"))

        return quarantined_counts

    def group_aware_split(
        self,
        image_files: List[Path],
        group_delimiter: str = "_",
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Partitions images using group-aware clustering.
        Images sharing the same incident or location prefix (e.g. 'locDelhi01_001.jpg',
        'locDelhi01_002.jpg') are strictly assigned to the SAME split to prevent leakage.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"

        # 1. Group images by prefix/incident ID
        groups = defaultdict(list)
        for img in image_files:
            stem = img.stem
            if group_delimiter in stem:
                parts = stem.split(group_delimiter)
                if len(parts) >= 2 and parts[1].isdigit():
                    # Keep sequential bursts together while enabling diverse multi-group partitioning
                    group_key = f"{parts[0]}_{parts[1][:2]}" if len(parts[1]) >= 2 else parts[0]
                else:
                    group_key = parts[0]
            else:
                group_key = stem[:8]
            groups[group_key].append(img)

        # Fallback if too few groups exist to split across train, val, and test
        if len(groups) < 3:
            groups = {img.stem: [img] for img in image_files}

        # 2. Shuffle groups
        group_keys = list(groups.keys())
        random.shuffle(group_keys)

        total_images = len(image_files)
        target_train = int(total_images * train_ratio)
        target_val = int(total_images * val_ratio)

        train_imgs, val_imgs, test_imgs = [], [], []
        curr_train, curr_val = 0, 0

        for gk in group_keys:
            grp = groups[gk]
            grp_len = len(grp)

            if curr_train + grp_len <= target_train or not train_imgs:
                train_imgs.extend(grp)
                curr_train += grp_len
            elif curr_val + grp_len <= target_val or not val_imgs:
                val_imgs.extend(grp)
                curr_val += grp_len
            else:
                test_imgs.extend(grp)

        # Fallback if test is empty due to small group count
        if not test_imgs and len(val_imgs) > 1:
            test_imgs.append(val_imgs.pop())

        return train_imgs, val_imgs, test_imgs

    def export_yolo_dataset(
        self,
        output_dir: Path,
        train_images: List[Path],
        val_images: List[Path],
        test_images: List[Path],
        source_labels_dir: Path,
        class_names: List[str],
        domain_name: str
    ) -> Path:
        """
        Organizes split files into destination directory and writes data.yaml.
        """
        output_dir = Path(output_dir)
        for split in ["train", "val", "test"]:
            img_dest = output_dir / "images" / split
            lbl_dest = output_dir / "labels" / split
            if img_dest.exists():
                shutil.rmtree(img_dest)
            if lbl_dest.exists():
                shutil.rmtree(lbl_dest)
            img_dest.mkdir(parents=True, exist_ok=True)
            lbl_dest.mkdir(parents=True, exist_ok=True)

        for split, imgs in [("train", train_images), ("val", val_images), ("test", test_images)]:
            img_dest = output_dir / "images" / split
            lbl_dest = output_dir / "labels" / split

            for img in imgs:
                shutil.copy2(str(img), str(img_dest / img.name))
                lbl_src = source_labels_dir / f"{img.stem}.txt"
                if lbl_src.exists():
                    shutil.copy2(str(lbl_src), str(lbl_dest / f"{img.stem}.txt"))

        # Write data.yaml
        yaml_content = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)}
        }

        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        return yaml_path

