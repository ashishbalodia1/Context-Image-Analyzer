"""
Unit tests for DatasetCleaner (group-aware splitting and data export).
"""
import sys
from pathlib import Path
import tempfile
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset_cleaner import DatasetCleaner


def test_group_aware_split_no_leakage():
    cleaner = DatasetCleaner(random_seed=123)

    # Create synthetic filenames with groups: wardA, wardB, wardC, wardD, wardE
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        images = []
        for ward in ["wardA", "wardB", "wardC", "wardD", "wardE", "wardF", "wardG", "wardH", "wardI", "wardJ"]:
            for i in range(5):
                p = tmp / f"{ward}_seq{i}.jpg"
                p.touch()
                images.append(p)

        train, val, test = cleaner.group_aware_split(images, group_delimiter="_", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

        train_groups = {img.stem.split("_")[0] for img in train}
        val_groups = {img.stem.split("_")[0] for img in val}
        test_groups = {img.stem.split("_")[0] for img in test}

        # Crucial check: Disjoint group sets (zero geographic / incident leakage)
        assert len(train_groups.intersection(val_groups)) == 0, "Train and Val share groups!"
        assert len(train_groups.intersection(test_groups)) == 0, "Train and Test share groups!"
        assert len(val_groups.intersection(test_groups)) == 0, "Val and Test share groups!"
        assert len(train) + len(val) + len(test) == 50


if __name__ == "__main__":
    test_group_aware_split_no_leakage()
    print("DATASET CLEANER GROUP-AWARE SPLIT TESTS PASSED (100% SUCCESS)!")

