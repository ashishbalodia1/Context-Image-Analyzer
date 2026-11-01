"""
Comprehensive Integration Tests for City Guardian REST API
Verifies:
  - System health (/api/health and /api/v1/health)
  - Image analysis pipeline (/api/analyze-image)
  - Incident retrieval (/api/detection/<id>)
  - Incident filtering & listing (/api/environmental-incidents)
  - Hotspot clustering (/api/environmental-hotspots)
  - Model performance benchmarking (/api/model-performance)
  - Status updates (/api/detection/<id>/status)
  - Error handling (invalid coordinates, missing files, non-existent incidents)
"""
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from PIL import Image
from backend.app import create_app


def _create_test_image_bytes(color=(120, 180, 90), size=(640, 640)) -> io.BytesIO:
    """Creates a valid JPEG image buffer for multipart upload testing."""
    img_array = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    # Add some high-contrast shape to simulate visual features
    cv2.rectangle(img_array, (100, 100), (300, 300), (30, 30, 200), -1)
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_endpoints(client):
    """Test health check on both /api/health and /api/v1/health."""
    for path in ["/api/health", "/api/v1/health"]:
        res = client.get(path)
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "healthy"
        assert "City Guardian" in data["service"]


def test_model_performance_endpoint(client):
    """Test GET /api/model-performance."""
    res = client.get("/api/model-performance")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ready"
    assert "performance_benchmarks" in data
    assert data["performance_benchmarks"]["mean_inference_latency_ms"] > 0
    assert len(data["domains_supported"]) == 5


def test_analyze_image_success(client):
    """Test POST /api/analyze-image with image and valid geo coordinates."""
    img_bytes = _create_test_image_bytes()
    data = {
        "image": (img_bytes, "test_environmental_photo.jpg"),
        "latitude": "28.6139",
        "longitude": "77.2090",
        "domain": "solid_waste",
        "ward": "Ward-12"
    }

    res = client.post(
        "/api/analyze-image",
        data=data,
        content_type="multipart/form-data"
    )
    assert res.status_code in (200, 201)
    resp_data = res.get_json()

    assert "incident_id" in resp_data
    assert "visual_severity" in resp_data
    assert resp_data["domain"] == "solid_waste"
    assert resp_data["location"]["latitude"] == 28.6139

    incident_id = resp_data["incident_id"]

    # Now verify retrieving this specific incident via GET /api/detection/<id>
    get_res = client.get(f"/api/detection/{incident_id}")
    assert get_res.status_code == 200
    get_data = get_res.get_json()
    assert get_data["incident_id"] == incident_id
    assert get_data["data"]["incident_id"] == incident_id


def test_analyze_image_validation_errors(client):
    """Test error handling on invalid image submissions."""
    # 1. No image provided
    res = client.post("/api/analyze-image", data={"latitude": "28.61"})
    assert res.status_code == 400

    # 2. Invalid GPS latitude
    img_bytes = _create_test_image_bytes()
    res = client.post(
        "/api/analyze-image",
        data={
            "image": (img_bytes, "photo.jpg"),
            "latitude": "199.99",  # Invalid latitude > 90
            "longitude": "77.20"
        },
        content_type="multipart/form-data"
    )
    assert res.status_code == 400
    assert "Invalid latitude" in res.get_json()["error"]


def test_incidents_listing_and_filtering(client):
    """Test GET /api/environmental-incidents with filters and pagination."""
    # Post a second incident in air_pollution domain
    img_bytes = _create_test_image_bytes(color=(200, 200, 200))
    client.post(
        "/api/analyze-image",
        data={
            "image": (img_bytes, "smoke.jpg"),
            "latitude": "28.6200",
            "longitude": "77.2100",
            "domain": "air_pollution",
            "ward": "Ward-04"
        },
        content_type="multipart/form-data"
    )

    # List all incidents
    res = client.get("/api/environmental-incidents")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] >= 2
    assert len(data["incidents"]) >= 2

    # Filter by domain
    res_filtered = client.get("/api/environmental-incidents?domain=air_pollution")
    assert res_filtered.status_code == 200
    filtered_data = res_filtered.get_json()
    for inc in filtered_data["incidents"]:
        assert inc["domain"] == "air_pollution"


def test_hotspots_endpoint(client):
    """Test GET /api/environmental-hotspots."""
    res = client.get("/api/environmental-hotspots?min_incidents=1")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "hotspots" in data
    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"


def test_incident_status_update(client):
    """Test PATCH /api/detection/<id>/status for resolution lifecycle."""
    # Create incident
    img_bytes = _create_test_image_bytes()
    res = client.post(
        "/api/analyze-image",
        data={
            "image": (img_bytes, "drain.jpg"),
            "latitude": "28.6500",
            "longitude": "77.2500",
            "domain": "sewage_drainage"
        },
        content_type="multipart/form-data"
    )
    inc_id = res.get_json()["incident_id"]

    # Patch status to IN_PROGRESS
    res_patch = client.patch(
        f"/api/detection/{inc_id}/status",
        json={"status": "IN_PROGRESS"}
    )
    assert res_patch.status_code == 200
    assert res_patch.get_json()["status_code"] == "IN_PROGRESS"

    # Verify retrieval reflects new status
    get_res = client.get(f"/api/detection/{inc_id}")
    assert get_res.get_json()["data"]["status"] == "IN_PROGRESS"

    # Test invalid status
    bad_res = client.patch(
        f"/api/detection/{inc_id}/status",
        json={"status": "FLYING_TO_SPACE"}
    )
    assert bad_res.status_code == 400


if __name__ == "__main__":
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        print("[TEST] Running health endpoints test...")
        test_health_endpoints(c)
        print("  -> Passed.")

        print("[TEST] Running model performance endpoint test...")
        test_model_performance_endpoint(c)
        print("  -> Passed.")

        print("[TEST] Running image analysis pipeline test...")
        test_analyze_image_success(c)
        print("  -> Passed.")

        print("[TEST] Running validation error handling test...")
        test_analyze_image_validation_errors(c)
        print("  -> Passed.")

        print("[TEST] Running incident filtering and listing test...")
        test_incidents_listing_and_filtering(c)
        print("  -> Passed.")

        print("[TEST] Running hotspot clustering endpoint test...")
        test_hotspots_endpoint(c)
        print("  -> Passed.")

        print("[TEST] Running status update lifecycle test...")
        test_incident_status_update(c)
        print("  -> Passed.")

    print("\n=======================================================")
    print("ALL API INTEGRATION TESTS PASSED (100% SUCCESS)!")
    print("=======================================================")

