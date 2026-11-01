"""
Unit tests for the Environmental Context Engine and Visual Severity Assessor.
Verifies compliance with Section 9 & 10 of Master Prompt.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_engine.severity_assessor import SeverityAssessor
from src.context_engine.report_generator import ReportGenerator


def test_visual_severity_assessment():
    assessor = SeverityAssessor()

    # Test 1: Empty detections
    res_empty = assessor.assess({"domain": "solid_waste", "detections": []})
    assert res_empty["visual_severity"] == "none"
    assert res_empty["visual_severity_score"] == 0.0

    # Test 2: High severity detection (large illegal dumping mound)
    mock_detection = {
        "domain": "solid_waste",
        "detections": [
            {
                "class_name": "illegal_dumping",
                "confidence": 0.92,
                "area_fraction": 0.35,
                "bbox": [50, 100, 450, 500]
            }
        ]
    }
    res_high = assessor.assess(mock_detection)
    assert res_high["visual_severity"] in ["high", "critical"]
    assert res_high["visual_severity_score"] >= 0.65
    assert res_high["affected_area_fraction"] == 0.35


def test_report_generator_schema_and_statutory_acts():
    reporter = ReportGenerator(use_llm=False)
    assessor = SeverityAssessor()

    mock_detection = {
        "domain": "sewage_drainage",
        "domain_name": "Sewage & Drainage",
        "image_path": "uploads/drain_overflow.jpg",
        "annotated_image_path": "uploads/annotated/drain_overflow.jpg",
        "detections": [
            {
                "class_name": "sewage_overflow",
                "confidence": 0.94,
                "area_fraction": 0.25,
                "bbox": [120, 80, 420, 350]
            }
        ]
    }
    severity_res = assessor.assess(mock_detection)
    geo_context = {
        "latitude": 28.7041,
        "longitude": 77.1025,
        "display_address": "Sector 14, Rohini, North Delhi",
        "city_district": "Rohini",
        "city": "Delhi",
        "location_available": True
    }

    report = reporter.generate(
        detection_result=mock_detection,
        severity_result=severity_res,
        geo_context=geo_context,
        sensor_data={"water_level_alert": True},
        image_metadata={"timestamp": "2026-11-01T10:30:00Z"},
        citizen_description="Sewage overflowing onto pedestrian pavement since morning"
    )

    # Verify all fields required by Section 9
    assert report["incident_id"].startswith("CG-")
    assert report["domain"] == "sewage_drainage"
    assert report["detected_issue"] == "sewage_overflow"
    assert len(report["detections"]) == 1
    assert report["detections"][0]["class"] == "sewage_overflow"
    assert report["detections"][0]["confidence"] == 0.94
    assert report["detections"][0]["bbox"] == [120, 80, 420, 350]
    assert report["location"]["latitude"] == 28.7041
    assert report["location"]["longitude"] == 77.1025
    assert report["visual_severity"] in ["high", "critical"]
    assert "Drainage" in report["department"]
    assert report["verification_required"] is True
    assert "Public Health" in report["statutory_act"]
    assert "pedestrian pavement" in report["description"]


if __name__ == "__main__":
    test_visual_severity_assessment()
    test_report_generator_schema_and_statutory_acts()
    print("CONTEXT ENGINE & REPORT GENERATOR TESTS PASSED (100% SUCCESS)!")
