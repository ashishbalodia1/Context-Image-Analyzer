"""
Unit tests for API Key Authentication in City Guardian backend.
"""
import io
import pytest
from backend.app import create_app
from backend.middleware.auth import DEFAULT_DEV_KEY


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_verify_key_without_header(client):
    """Should return 401 Unauthorized when no API key is provided."""
    res = client.get("/api/verify-key")
    assert res.status_code == 401
    data = res.get_json()
    assert data["error"] == "Unauthorized"


def test_verify_key_with_invalid_key(client):
    """Should return 403 Forbidden when an invalid API key is provided."""
    res = client.get("/api/verify-key", headers={"X-API-Key": "invalid_random_key_999"})
    assert res.status_code == 403
    data = res.get_json()
    assert data["error"] == "Forbidden"


def test_verify_key_with_valid_key(client):
    """Should return 200 OK when valid default API key is provided."""
    res = client.get("/api/verify-key", headers={"X-API-Key": DEFAULT_DEV_KEY})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "authenticated"


def test_verify_key_with_bearer_token(client):
    """Should support Authorization: Bearer <key> format."""
    res = client.get("/api/verify-key", headers={"Authorization": f"Bearer {DEFAULT_DEV_KEY}"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "authenticated"


def test_analyze_image_blocked_without_key(client):
    """Should block POST /api/analyze-image if API key is missing."""
    data = {
        "domain": "solid_waste",
        "file": (io.BytesIO(b"fake_image_bytes"), "test.jpg")
    }
    res = client.post("/api/analyze-image", data=data, content_type="multipart/form-data")
    assert res.status_code == 401
    assert res.get_json()["error"] == "Unauthorized"

