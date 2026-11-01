"""
Unit tests for DatasetValidator.
Verifies all validation criteria: bounding boxes, corrupted images, duplicate detection,
missing pairs, class IDs, and train/val/test data leakage.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

import shutil
from pathlib import Path
from PIL import Image
from src.preprocessing.dataset_validator import DatasetValidator


if pytest:
    fixture_decorator = pytest.fixture
else:
    def fixture_decorator(func):
        return func

@fixture_decorator
def mock_dataset_dir(tmp_path):
    """Generates a temporary synthetic dataset structure for testing."""
    root = tmp_path / "mock_dataset"
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    # 1. Create a valid image and label in train
    img_valid = root / "images" / "train" / "sample_valid.jpg"
    im = Image.new("RGB", (100, 100), color=(73, 109, 137))
    im.save(img_valid)

    lbl_valid = root / "labels" / "train" / "sample_valid.txt"
    # class 0, xc=0.5, yc=0.5, w=0.2, h=0.2 (valid)
    lbl_valid.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # 2. Create an image with missing label in train
    img_no_lbl = root / "images" / "train" / "no_label.jpg"
    im.save(img_no_lbl)

    # 3. Create an orphan label in train
    orphan_lbl = root / "labels" / "train" / "orphan_sample.txt"
    orphan_lbl.write_text("1 0.3 0.3 0.1 0.1\n", encoding="utf-8")

    # 4. Create an invalid bbox file (out of bounds)
    img_bad_bbox = root / "images" / "train" / "bad_bbox.jpg"
    im.save(img_bad_bbox)
    lbl_bad_bbox = root / "labels" / "train" / "bad_bbox.txt"
    # xc=1.2 (out of bounds)
    lbl_bad_bbox.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")

    # 5. Create a corrupted image file
    corrupt_img = root / "images" / "train" / "corrupted.jpg"
    corrupt_img.write_bytes(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")

    # 6. Create a duplicate image in val to test data leakage
    img_leakage = root / "images" / "val" / "sample_valid_copy.jpg"
    im.save(img_leakage)  # Identical content to sample_valid.jpg
    lbl_leakage = root / "labels" / "val" / "sample_valid_copy.txt"
    lbl_leakage.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    return root


def test_validator_detects_all_anomalies(mock_dataset_dir):
    validator = DatasetValidator()
    report = validator.generate_comprehensive_report(mock_dataset_dir)

    train_data = report["splits"]["train"]

    # Valid pairs check
    assert train_data["valid_pairs"] == 1

    # Missing label detection
    assert len(train_data["missing_labels"]) == 1
    assert "no_label.jpg" in train_data["missing_labels"][0]

    # Orphan label detection
    assert len(train_data["orphan_labels"]) == 1
    assert "orphan_sample.txt" in train_data["orphan_labels"][0]

    # Corrupted image detection
    assert len(train_data["corrupted_images"]) == 1
    assert "corrupted.jpg" in train_data["corrupted_images"][0]

    # Invalid bounding box detection
    assert len(train_data["invalid_bboxes"]) == 1
    assert "bad_bbox.txt" in train_data["invalid_bboxes"][0]["label_file"]

    # Cross-split leakage detection
    assert len(report["leakages"]) > 0
    assert report["leakages"][0]["split_1"] == "train"
    assert report["leakages"][0]["split_2"] == "val"

    # Overall dataset validity flag must be FALSE
    assert report["overall_is_valid"] is False


def test_validator_class_boundary_enforcement(tmp_path):
    lbl_file = tmp_path / "test_label.txt"
    validator = DatasetValidator()
    validator.nc = 3  # Only classes 0, 1, 2 allowed

    # Negative width
    lbl_file.write_text("0 0.5 0.5 -0.1 0.2\n", encoding="utf-8")
    errs, _ = validator._validate_label_file(lbl_file)
    assert any("Non-positive dimension" in e for e in errs)

    # Class ID >= nc
    lbl_file.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    errs, _ = validator._validate_label_file(lbl_file)
    assert any("class_id '5' >= nc '3'" in e for e in errs)

    # Box overflow (xc=0.9, w=0.4 -> x2 = 1.1)
    lbl_file.write_text("1 0.9 0.5 0.4 0.2\n", encoding="utf-8")
    errs, _ = validator._validate_label_file(lbl_file)
    assert any("overflows image bounds" in e for e in errs)

    # Valid bbox
    lbl_file.write_text("2 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    errs, bboxes = validator._validate_label_file(lbl_file)
    assert len(errs) == 0
    assert len(bboxes) == 1


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tpath = Path(tmp_dir)
        mock_dir = mock_dataset_dir(tpath)
        test_validator_detects_all_anomalies(mock_dir)
        test_validator_class_boundary_enforcement(tpath)
        print("ALL DATASET VALIDATOR UNIT TESTS PASSED (100% SUCCESS)!")

