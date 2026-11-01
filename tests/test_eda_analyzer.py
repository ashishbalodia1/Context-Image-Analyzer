"""
Unit tests for DatasetEDAAnalyzer.
Verifies class distribution counting, bounding box geometry statistics,
and automated plot file generation.
"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.eda_analyzer import DatasetEDAAnalyzer


def test_eda_metrics_and_plots():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        labels_dir = tmp / "labels"
        labels_dir.mkdir()
        plots_dir = tmp / "plots"

        # Create synthetic label files:
        # File 1: class 0 (small square), class 1 (tall box)
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.4 0.6 0.1 0.4\n", encoding="utf-8")
        # File 2: class 0 (large box)
        (labels_dir / "img2.txt").write_text("0 0.6 0.6 0.6 0.5\n", encoding="utf-8")
        # File 3: empty background image
        (labels_dir / "img3.txt").write_text("", encoding="utf-8")

        analyzer = DatasetEDAAnalyzer()
        metrics = analyzer.analyze_dataset(labels_dir)

        assert metrics["total_images"] == 3
        assert metrics["total_boxes"] == 3
        assert metrics["empty_images"] == 1
        assert metrics["class_distribution"]["class_0"] == 2
        assert metrics["class_distribution"]["class_1"] == 1

        # Check scale fractions sum to ~1.0
        scales = metrics["box_scales"]
        total_frac = sum(scales.values())
        assert abs(total_frac - 1.0) < 1e-3

        # Test plot generation
        plots = analyzer.generate_eda_plots(metrics, plots_dir, domain_name="test_env")
        assert len(plots) == 3
        for p in plots:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0


if __name__ == "__main__":
    test_eda_metrics_and_plots()
    print("EDA ANALYZER UNIT TESTS PASSED (100% SUCCESS)!")

