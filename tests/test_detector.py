"""Unit tests for YOLO Detector and Domain Router."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import YOLODetector
from src.detection.domain_router import DomainRouter


def test_yolo_detector_instantiation():
    detector = YOLODetector()
    assert detector.conf_threshold > 0.0
    assert detector.iou_threshold > 0.0


def test_domain_router_domains():
    router = DomainRouter()
    expected_domains = ["air_pollution", "water_quality", "sewage_drainage", "mining_dust", "solid_waste"]
    for d in expected_domains:
        assert d in router.VALID_DOMAINS


if __name__ == "__main__":
    test_yolo_detector_instantiation()
    test_domain_router_domains()
    print("DETECTOR AND DOMAIN ROUTER TESTS PASSED!")
