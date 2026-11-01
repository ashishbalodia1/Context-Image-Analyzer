"""
Exploratory Data Analysis (EDA) Engine for YOLOv8 Environmental Datasets.
Analyzes class frequency, bounding-box geometry, aspect ratios, scale distributions,
and spatial centroid densities to identify annotation bias and class imbalance.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import yaml

# Set MPLCONFIGDIR before importing matplotlib to avoid Windows permission warnings
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


class DatasetEDAAnalyzer:
    """
    Automated EDA analyzer for YOLO format datasets.
    """

    def __init__(self, data_yaml_path: Optional[str] = None):
        self.class_names: Dict[int, str] = {}
        if data_yaml_path and Path(data_yaml_path).exists():
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                names = cfg.get("names", {})
                if isinstance(names, list):
                    self.class_names = {i: name for i, name in enumerate(names)}
                elif isinstance(names, dict):
                    self.class_names = {int(k): str(v) for k, v in names.items()}

    def analyze_dataset(self, labels_dir: Path) -> Dict[str, Any]:
        """
        Parses all YOLO .txt label files in labels_dir and computes statistical metrics.
        """
        labels_dir = Path(labels_dir)
        files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        class_counts = defaultdict(int)
        widths, heights, areas, aspect_ratios = [], [], [], []
        x_centers, y_centers = [], []
        objects_per_image = []

        total_boxes = 0
        empty_images = 0

        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                lines = [line.strip() for line in fp if line.strip()]

            if not lines:
                empty_images += 1
                objects_per_image.append(0)
                continue

            objects_per_image.append(len(lines))

            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if w > 0 and h > 0:
                            class_counts[cls_id] += 1
                            widths.append(w)
                            heights.append(h)
                            areas.append(w * h)
                            aspect_ratios.append(w / h)
                            x_centers.append(xc)
                            y_centers.append(yc)
                            total_boxes += 1
                    except ValueError:
                        continue

        if total_boxes == 0:
            return {
                "total_images": len(files),
                "total_boxes": 0,
                "empty_images": empty_images,
                "class_distribution": {},
                "metrics": {}
            }

        w_arr = np.array(widths)
        h_arr = np.array(heights)
        area_arr = np.array(areas)
        ar_arr = np.array(aspect_ratios)

        metrics = {
            "total_images": len(files),
            "total_boxes": total_boxes,
            "empty_images": empty_images,
            "avg_boxes_per_image": round(float(np.mean(objects_per_image)), 2),
            "class_distribution": {
                self.class_names.get(cls_id, f"class_{cls_id}"): count
                for cls_id, count in sorted(class_counts.items())
            },
            "box_scales": {
                "small_fraction (<0.05)": round(float(np.mean(area_arr < 0.05)), 4),
                "medium_fraction (0.05-0.25)": round(float(np.mean((area_arr >= 0.05) & (area_arr <= 0.25))), 4),
                "large_fraction (>0.25)": round(float(np.mean(area_arr > 0.25)), 4),
            },
            "dimensions_percentiles": {
                "width": {
                    "p25": round(float(np.percentile(w_arr, 25)), 4),
                    "median": round(float(np.median(w_arr)), 4),
                    "p75": round(float(np.percentile(w_arr, 75)), 4),
                },
                "height": {
                    "p25": round(float(np.percentile(h_arr, 25)), 4),
                    "median": round(float(np.median(h_arr)), 4),
                    "p75": round(float(np.percentile(h_arr, 75)), 4),
                },
                "aspect_ratio": {
                    "p25": round(float(np.percentile(ar_arr, 25)), 4),
                    "median": round(float(np.median(ar_arr)), 4),
                    "p75": round(float(np.percentile(ar_arr, 75)), 4),
                }
            },
            "raw_arrays": {
                "widths": widths,
                "heights": heights,
                "aspect_ratios": aspect_ratios,
                "x_centers": x_centers,
                "y_centers": y_centers
            }
        }
        return metrics

    def generate_eda_plots(self, metrics: Dict[str, Any], output_dir: Path, domain_name: str = "Environmental") -> List[str]:
        """
        Renders statistical diagnostic visual plots and saves them as high-res PNGs.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = []

        if metrics.get("total_boxes", 0) == 0:
            return saved_plots

        raw = metrics["raw_arrays"]

        # 1. Class Distribution Bar Chart
        plt.figure(figsize=(10, 5))
        cls_dist = metrics["class_distribution"]
        classes = list(cls_dist.keys())
        counts = list(cls_dist.values())
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
        plt.bar(classes, counts, color=colors, edgecolor="black", alpha=0.85)
        plt.title(f"Class Distribution: {domain_name.upper()}", fontsize=14, fontweight="bold")
        plt.xlabel("Target Environmental Classes", fontsize=11)
        plt.ylabel("Instance Count (BBoxes)", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        p1 = output_dir / f"{domain_name}_class_distribution.png"
        plt.savefig(p1, dpi=200)
        plt.close()
        saved_plots.append(str(p1))

        # 2. Bounding Box Width vs Height Scatter & Aspect Ratio Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.scatter(raw["widths"], raw["heights"], c="#2b5c8f", alpha=0.4, edgecolors="none", s=25)
        ax1.plot([0, 1], [0, 1], color="red", linestyle="--", label="Square (1:1)")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title("Normalized Bounding Box Dimensions (Width vs Height)", fontsize=12)
        ax1.set_xlabel("Normalized Width", fontsize=10)
        ax1.set_ylabel("Normalized Height", fontsize=10)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Aspect ratios histogram
        ar_clipped = np.clip(raw["aspect_ratios"], 0.1, 5.0)
        ax2.hist(ar_clipped, bins=30, color="#4a90e2", edgecolor="black", alpha=0.8)
        ax2.axvline(1.0, color="red", linestyle="--", label="1.0 (Square)")
        ax2.set_title("Aspect Ratio (Width / Height) Distribution", fontsize=12)
        ax2.set_xlabel("Aspect Ratio", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p2 = output_dir / f"{domain_name}_bbox_geometry.png"
        plt.savefig(p2, dpi=200)
        plt.close()
        saved_plots.append(str(p2))

        # 3. Spatial Centroid Heatmap (Spatial Bias Inspection)
        plt.figure(figsize=(7, 6))
        plt.hist2d(raw["x_centers"], raw["y_centers"], bins=25, range=[[0, 1], [0, 1]], cmap="YlOrRd")
        plt.colorbar(label="Centroid Density")
        plt.gca().invert_yaxis()  # Image coordinates (0,0 is top-left)
        plt.title(f"Spatial Centroid Heatmap (Location Bias): {domain_name}", fontsize=12, fontweight="bold")
        plt.xlabel("Normalized X-Center", fontsize=10)
        plt.ylabel("Normalized Y-Center", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        p3 = output_dir / f"{domain_name}_spatial_density.png"
        plt.savefig(p3, dpi=200)
        plt.close()
        saved_plots.append(str(p3))

        return saved_plots

