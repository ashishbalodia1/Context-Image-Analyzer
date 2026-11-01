"""Unit tests for preprocessing modules."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.image_validator import ImageValidator
from src.preprocessing.geo_extractor import resolve_location, build_geo_context


def test_image_validator_missing_file():
    validator = ImageValidator()
    is_valid, result = validator.validate("non_existent_file.jpg")
    assert not is_valid
    assert len(result["errors"]) > 0


def test_geo_extractor_no_coords():
    context = build_geo_context(lat=None, lon=None)
    assert not context.get("location_available", True)


if __name__ == "__main__":
    test_image_validator_missing_file()
    test_geo_extractor_no_coords()
    print("PREPROCESSING TESTS PASSED!")
