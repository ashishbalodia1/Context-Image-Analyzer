"""
Dataset Validation and Cleaning Engine for City Guardian.
Verifies YOLO-format annotations, image integrity, bounding box geometry,
class consistency, duplicate detection, and train/val/test leakage prevention.
"""
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import yaml
from PIL import Image

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class DatasetValidator:
    """
    Production-grade dataset validator for YOLOv8 environmental datasets.

    Checks:
    1. Image-label pair correspondence (missing pairs, orphan labels).
    2. Image integrity (corruption, dimensions, readable headers).
    3. Duplicate images (via MD5 image hashing).
    4. Bounding box validity:
       - 5-element format: <class_id> <x_center> <y_center> <width> <height>
       - Normalization in [0.0, 1.0]
       - Positive widths and heights (w > 0, h > 0)
       - Boundary containment (x ± w/2, y ± h/2 within [0, 1])
    5. Class ID validity against data.yaml class list.
    6. Object size distribution (small, medium, large area fractions).
    7. Class imbalance metrics (counts, percentages, max/min imbalance ratio).
    8. Cross-split data leakage detection (train vs val vs test).
    """

    SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    BBOX_TOLERANCE = 1e-3  # Numerical tolerance for boundary checks

    def __init__(self, data_yaml_path: Optional[str] = None):
        """
        Args:
            data_yaml_path: Path to YOLO data.yaml configuration containing nc and names.
        """
        self.data_yaml_path = data_yaml_path
        self.nc: Optional[int] = None
        self.class_names: Dict[int, str] = {}
        if data_yaml_path and Path(data_yaml_path).exists():
            self._load_yaml_config(data_yaml_path)

    def _load_yaml_config(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            self.nc = cfg.get("nc")
            names = cfg.get("names", {})
            if isinstance(names, list):
                self.class_names = {i: name for i, name in enumerate(names)}
            elif isinstance(names, dict):
                self.class_names = {int(k): str(v) for k, v in names.items()}
            if self.nc is None and self.class_names:
                self.nc = len(self.class_names)

    def validate_split(
        self,
        images_dir: Path,
        labels_dir: Path,
        split_name: str = "train"
    ) -> Dict[str, Any]:
        """
        Validates all image-label pairs inside a specific split folder.

        Returns:
            Dict containing errors, warnings, class distributions, and bounding box metrics.
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        report = {
            "split": split_name,
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "total_images": 0,
            "total_labels": 0,
            "valid_pairs": 0,
            "missing_labels": [],
            "orphan_labels": [],
            "corrupted_images": [],
            "duplicate_images": [],
            "invalid_bboxes": [],
            "class_distribution": defaultdict(int),
            "bbox_size_distribution": {
                "small (<0.05)": 0,
                "medium (0.05-0.25)": 0,
                "large (>0.25)": 0
            },
            "hashes": {},  # hash -> image_path
            "is_valid": True,
        }

        if not images_dir.exists():
            report["errors"] = [f"Images directory does not exist: {images_dir}"]
            report["is_valid"] = False
            return report

        # 1. Discover all image files
        image_files = [
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMG_EXTS
        ]
        report["total_images"] = len(image_files)

        # 2. Discover all label files
        label_files = (
            [f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
            if labels_dir.exists() else []
        )
        report["total_labels"] = len(label_files)
        label_stem_map = {f.stem: f for f in label_files}
        image_stem_set = {f.stem for f in image_files}

        # 3. Check orphan labels (labels without images)
        for stem, lbl_file in label_stem_map.items():
            if stem not in image_stem_set:
                report["orphan_labels"].append(str(lbl_file))

        # 4. Validate images and their corresponding labels
        for img_path in image_files:
            stem = img_path.stem

            # Check image integrity
            img_valid, img_dims, img_hash = self._verify_image(img_path)
            if not img_valid:
                report["corrupted_images"].append(str(img_path))
                continue

            # Check duplicates
            if img_hash in report["hashes"]:
                report["duplicate_images"].append({
                    "original": report["hashes"][img_hash],
                    "duplicate": str(img_path)
                })
            else:
                report["hashes"][img_hash] = str(img_path)

            # Check matching label
            if stem not in label_stem_map:
                report["missing_labels"].append(str(img_path))
                continue

            # Validate label content
            lbl_path = label_stem_map[stem]
            bbox_errors, bboxes = self._validate_label_file(lbl_path)

            if bbox_errors:
                report["invalid_bboxes"].append({
                    "label_file": str(lbl_path),
                    "errors": bbox_errors
                })
            else:
                report["valid_pairs"] += 1

            # Accumulate class counts and box size metrics
            for cls_id, xc, yc, w, h in bboxes:
                report["class_distribution"][cls_id] += 1
                area = w * h
                if area < 0.05:
                    report["bbox_size_distribution"]["small (<0.05)"] += 1
                elif area <= 0.25:
                    report["bbox_size_distribution"]["medium (0.05-0.25)"] += 1
                else:
                    report["bbox_size_distribution"]["large (>0.25)"] += 1

        # Summary validity check
        has_critical_issues = bool(
            report["corrupted_images"] or
            report["missing_labels"] or
            report["invalid_bboxes"]
        )
        report["is_valid"] = not has_critical_issues
        return report

    def _verify_image(self, img_path: Path) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
        """Verify image can be opened, is uncorrupted, and compute its MD5 hash."""
        try:
            with open(img_path, "rb") as f:
                data = f.read()
                img_hash = hashlib.md5(data).hexdigest()

            with Image.open(img_path) as img:
                img.verify()

            # Reopen to read dimensions (verify() closes/invalidates the image object)
            with Image.open(img_path) as img:
                dims = img.size
                if dims[0] < 10 or dims[1] < 10:
                    return False, None, None

            return True, dims, img_hash
        except Exception:
            return False, None, None

    def _validate_label_file(self, lbl_path: Path) -> Tuple[List[str], List[Tuple[int, float, float, float, float]]]:
        """
        Validates individual YOLO format label file.
        Each line: <class_id> <x_center> <y_center> <width> <height>
        """
        errors = []
        valid_bboxes = []

        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                # Empty label file = background image without objects (valid in YOLO)
                return errors, valid_bboxes

            for line_idx, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Line {line_idx}: Expected 5 elements, got {len(parts)} ('{line}')")
                    continue

                # 1. Class ID check
                try:
                    cls_id = int(parts[0])
                    if cls_id < 0:
                        errors.append(f"Line {line_idx}: Negative class_id '{cls_id}'")
                        continue
                    if self.nc is not None and cls_id >= self.nc:
                        errors.append(f"Line {line_idx}: class_id '{cls_id}' >= nc '{self.nc}'")
                        continue
                except ValueError:
                    errors.append(f"Line {line_idx}: Non-integer class_id '{parts[0]}'")
                    continue

                # 2. Coordinates checks
                try:
                    xc, yc, w, h = [float(p) for p in parts[1:5]]
                except ValueError:
                    errors.append(f"Line {line_idx}: Non-float coordinates in '{line}'")
                    continue

                # Check width & height positivity
                if w <= 0.0 or h <= 0.0:
                    errors.append(f"Line {line_idx}: Non-positive dimension w={w}, h={h}")
                    continue

                # Check normalization in [0.0, 1.0]
                tol = self.BBOX_TOLERANCE
                if not (0.0 - tol <= xc <= 1.0 + tol and 0.0 - tol <= yc <= 1.0 + tol):
                    errors.append(f"Line {line_idx}: Center coords out of [0, 1] range: xc={xc}, yc={yc}")
                    continue

                if not (0.0 < w <= 1.0 + tol and 0.0 < h <= 1.0 + tol):
                    errors.append(f"Line {line_idx}: Dimensions out of (0, 1] range: w={w}, h={h}")
                    continue

                # Check boundaries: x1, y1 >= 0, x2, y2 <= 1
                x1 = xc - w / 2.0
                y1 = yc - h / 2.0
                x2 = xc + w / 2.0
                y2 = yc + h / 2.0

                if x1 < 0.0 - tol or y1 < 0.0 - tol or x2 > 1.0 + tol or y2 > 1.0 + tol:
                    errors.append(f"Line {line_idx}: BBox overflows image bounds: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
                    continue

                valid_bboxes.append((cls_id, xc, yc, w, h))

        except Exception as e:
            errors.append(f"File read error: {str(e)}")

        return errors, valid_bboxes

    def check_dataset_leakage(
        self,
        train_report: Dict[str, Any],
        val_report: Dict[str, Any],
        test_report: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Detects data leakage between train, val, and test splits using image hashes.
        """
        leakages = []
        train_hashes = train_report.get("hashes", {})
        val_hashes = val_report.get("hashes", {})
        test_hashes = test_report.get("hashes", {}) if test_report else {}

        # Train vs Val
        for h, train_img in train_hashes.items():
            if h in val_hashes:
                leakages.append({
                    "split_1": "train",
                    "file_1": train_img,
                    "split_2": "val",
                    "file_2": val_hashes[h],
                    "hash": h
                })

        # Train vs Test
        for h, train_img in train_hashes.items():
            if h in test_hashes:
                leakages.append({
                    "split_1": "train",
                    "file_1": train_img,
                    "split_2": "test",
                    "file_2": test_hashes[h],
                    "hash": h
                })

        # Val vs Test
        for h, val_img in val_hashes.items():
            if h in test_hashes:
                leakages.append({
                    "split_1": "val",
                    "file_1": val_img,
                    "split_2": "test",
                    "file_2": test_hashes[h],
                    "hash": h
                })

        return leakages

    def generate_comprehensive_report(self, dataset_root: Path) -> Dict[str, Any]:
        """
        Executes complete audit across train, val, and test splits of a YOLO dataset.
        """
        dataset_root = Path(dataset_root)
        results = {
            "dataset_root": str(dataset_root),
            "splits": {},
            "leakages": [],
            "overall_is_valid": True,
            "total_objects": 0,
            "overall_class_distribution": defaultdict(int),
            "imbalance_ratio": 1.0,
        }

        for split in ["train", "val", "test"]:
            img_dir = dataset_root / "images" / split
            lbl_dir = dataset_root / "labels" / split
            if img_dir.exists():
                split_res = self.validate_split(img_dir, lbl_dir, split_name=split)
                results["splits"][split] = split_res
                if not split_res["is_valid"]:
                    results["overall_is_valid"] = False

                for cls_id, cnt in split_res["class_distribution"].items():
                    results["overall_class_distribution"][cls_id] += cnt
                    results["total_objects"] += cnt

        # Check leakage across available splits
        if "train" in results["splits"] and "val" in results["splits"]:
            leakages = self.check_dataset_leakage(
                results["splits"]["train"],
                results["splits"]["val"],
                results["splits"].get("test")
            )
            results["leakages"] = leakages
            if leakages:
                results["overall_is_valid"] = False

        # Compute class imbalance ratio
        counts = list(results["overall_class_distribution"].values())
        if counts:
            max_c = max(counts)
            min_c = min(counts)
            results["imbalance_ratio"] = round(max_c / min_c, 2) if min_c > 0 else float("inf")

        return results


def print_validation_summary(report: Dict[str, Any]):
    """CLI formatter for dataset audit report."""
    print("=" * 80)
    print(f"DATASET VALIDATION AUDIT: {report['dataset_root']}")
    print("=" * 80)
    print(f"Overall Valid: {'✅ PASS' if report['overall_is_valid'] else '❌ FAIL'}")
    print(f"Total Objects: {report['total_objects']}")
    print(f"Imbalance Ratio (Max/Min): {report['imbalance_ratio']}")

    for split, data in report.get("splits", {}).items():
        print(f"\n[{split.upper()} SPLIT]")
        print(f"  Images: {data['total_images']} | Labels: {data['total_labels']} | Valid Pairs: {data['valid_pairs']}")
        if data["missing_labels"]:
            print(f"  ⚠️  Missing Labels: {len(data['missing_labels'])}")
        if data["orphan_labels"]:
            print(f"  ⚠️  Orphan Labels: {len(data['orphan_labels'])}")
        if data["corrupted_images"]:
            print(f"  ❌ Corrupted Images: {len(data['corrupted_images'])}")
        if data["duplicate_images"]:
            print(f"  ⚠️  Duplicates Found: {len(data['duplicate_images'])}")
        if data["invalid_bboxes"]:
            print(f"  ❌ Files with Invalid BBoxes: {len(data['invalid_bboxes'])}")

        print(f"  Box Sizes: {dict(data['bbox_size_distribution'])}")

    if report.get("leakages"):
        print(f"\n❌ CRITICAL: Data Leakage Detected ({len(report['leakages'])} duplicate images across splits):")
        for leak in report["leakages"][:5]:
            print(f"  - {leak['split_1']} vs {leak['split_2']}: {Path(leak['file_1']).name}")
    else:
        print("\n✅ Zero Data Leakage: Splits are completely disjoint.")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Quality Validator")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--yaml", type=str, default=None, help="Path to data.yaml")
    args = parser.parse_args()

    validator = DatasetValidator(args.yaml)
    res = validator.generate_comprehensive_report(Path(args.data_dir))
    print_validation_summary(res)

