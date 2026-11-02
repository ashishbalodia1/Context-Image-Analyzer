"""
Image Analysis and Upload Route for City Guardian.
Exposes POST /api/analyze-image and executes the complete pipeline:
Validation -> EXIF/Manual GeoTagging -> CLAHE Enhancement -> YOLOv8 Detection ->
Severity Assessment -> Context Synthesis -> Storage.
"""
import os
import uuid
import base64
from pathlib import Path
from flask import Blueprint, request, jsonify

from config.settings import UPLOAD_DIR, ALLOWED_EXTENSIONS
from src.preprocessing.pipeline import PreprocessingPipeline
from src.detection.domain_router import DomainRouter
from src.context_engine.severity_assessor import SeverityAssessor
from src.context_engine.report_generator import ReportGenerator
from src.context_engine.external_apis import fetch_aqi_data
from src.utils.logger import get_logger
from backend.middleware.auth import require_api_key

logger = get_logger(__name__)
upload_bp = Blueprint("upload", __name__)

# Singletons (loaded once at API startup to eliminate per-request overhead)
_pipeline = PreprocessingPipeline()
_router = DomainRouter()
_assessor = SeverityAssessor()
_report_gen = ReportGenerator(use_llm=False)

# Shared in-memory incident database (synchronized with reports and map blueprints)
from backend.routes.reports import _reports_store


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@upload_bp.route("/verify-key", methods=["GET"])
@require_api_key
def verify_key():
    """GET /api/verify-key - Checks if the client's API Key is valid."""
    return jsonify({
        "status": "authenticated",
        "message": "API Key is valid. Access granted to City Guardian Environmental Intelligence Pipeline."
    }), 200


@upload_bp.route("/analyze-image", methods=["POST"])
@upload_bp.route("/upload", methods=["POST"])
@upload_bp.route("/capture-camera", methods=["POST"])
@require_api_key
def analyze_image():
    """
    POST /api/analyze-image (or /api/capture-camera)
    Supports both Multipart Form-Data (file upload or camera snapshot)
    and JSON body (with base64 encoded camera frame).
    """
    save_path = None
    source_type = "upload"

    # A. Check for JSON Payload (Direct Base64 from Camera/Webcam Canvas)
    if request.is_json:
        json_data = request.get_json() or {}
        b64_data = json_data.get("image_base64") or json_data.get("camera_base64") or json_data.get("image")
        if not b64_data:
            return jsonify({"error": "No image data provided. Pass 'image_base64' in JSON body or 'file' in multipart form."}), 400

        source_type = json_data.get("source", "camera")
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]

        unique_name = f"camera_{uuid.uuid4().hex}.jpg"
        os.makedirs(str(UPLOAD_DIR), exist_ok=True)
        save_path = str(UPLOAD_DIR / unique_name)
        try:
            with open(save_path, "wb") as fh:
                fh.write(base64.b64decode(b64_data))
            logger.info(f"Live camera frame decoded and saved securely: {save_path}")
        except Exception as e:
            return jsonify({"error": f"Invalid base64 camera image: {str(e)}"}), 400

        domain = json_data.get("domain")
        citizen_desc = json_data.get("description")
        lat_raw = json_data.get("latitude") or json_data.get("lat")
        lon_raw = json_data.get("longitude") or json_data.get("lon")
        req_timestamp = json_data.get("timestamp")

    # B. Multipart Form-Data (File Upload or Camera Snapshot Blob)
    else:
        file = request.files.get("file") or request.files.get("image") or request.files.get("camera")
        if not file:
            return jsonify({"error": "No file part in request. Provide an image under key 'image', 'file', or 'camera'."}), 400

        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not _allowed_file(file.filename):
            return jsonify({"error": f"Unsupported file type. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}"}), 400

        source_type = request.form.get("source", "upload")
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{'camera_' if source_type == 'camera' else ''}{uuid.uuid4().hex}.{ext}"
        os.makedirs(str(UPLOAD_DIR), exist_ok=True)
        save_path = str(UPLOAD_DIR / unique_name)
        file.save(save_path)
        logger.info(f"File/Camera image saved securely: {save_path}")

        domain = request.form.get("domain")
        citizen_desc = request.form.get("description")
        lat_raw = request.form.get("latitude") or request.form.get("lat")
        lon_raw = request.form.get("longitude") or request.form.get("lon")
        req_timestamp = request.form.get("timestamp")
    if lat_raw and lon_raw:
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                return jsonify({"error": f"Invalid latitude ({lat}) or longitude ({lon}) bounds."}), 400
            manual_location = {
                "latitude": lat,
                "longitude": lon
            }
        except ValueError:
            return jsonify({"error": f"Coordinates must be numeric. Received lat={lat_raw}, lon={lon_raw}"}), 400

    # 3. Preprocessing (Validation, EXIF extraction, CLAHE contrast enhancement, resize)
    preproc_result = _pipeline.run(
        image_path=save_path,
        manual_location=manual_location,
        enhance=True
    )

    if not preproc_result.get("success"):
        return jsonify({
            "error": "Image preprocessing or validation failed",
            "details": preproc_result.get("errors", [])
        }), 422

    processed_path = preproc_result["processed_path"]
    geo_context = preproc_result["geo_context"]
    image_metadata = preproc_result["image_metadata"]
    if request.form.get("timestamp"):
        image_metadata["timestamp"] = request.form.get("timestamp")

    # 4. YOLOv8 Domain Routing & Detection
    annotated_dir = str(UPLOAD_DIR / "annotated")
    detection_result = _router.route_and_detect(
        image_path=processed_path,
        domain=domain,
        save_annotated=True,
        output_dir=annotated_dir
    )

    # 5. Visual Severity Assessment
    severity_result = _assessor.assess(detection_result)

    # 6. Sensor Telemetry Fusion (AQI for Air Pollution)
    sensor_data = {}
    if detection_result.get("domain") == "air_pollution" and geo_context.get("latitude"):
        aqi_info = fetch_aqi_data(geo_context["latitude"], geo_context["longitude"])
        if aqi_info:
            sensor_data = aqi_info

    # 7. Context Synthesis & Structured Report Generation
    report = _report_gen.generate(
        detection_result=detection_result,
        severity_result=severity_result,
        geo_context=geo_context,
        sensor_data=sensor_data,
        image_metadata=image_metadata,
        citizen_description=citizen_desc
    )
    report["capture_source"] = "Live Camera Snapshot" if source_type == "camera" else "Inspection File Upload"

    # 7b. Attach selected domain's dataset evaluation & benchmark metrics
    domain_key = detection_result.get("domain", "solid_waste")
    project_root = Path(__file__).resolve().parent.parent.parent
    metrics_file = project_root / "reports" / "evaluation" / f"classification_metrics_{domain_key}.json"
    if metrics_file.exists():
        try:
            import json
            with open(metrics_file, "r", encoding="utf-8") as mf:
                report["domain_evaluation_metrics"] = json.load(mf)
        except Exception as e:
            logger.warning(f"Could not load domain metrics for {domain_key}: {e}")

    # 8. Persist incident into storage for dashboard & spatial mapping
    _reports_store.append(report)

    return jsonify(report), 201


@upload_bp.route("/domain-metrics/<domain>", methods=["GET"])
def get_domain_metrics(domain: str):
    """Returns the individual dataset classification & evaluation metrics for a specific domain."""
    project_root = Path(__file__).resolve().parent.parent.parent
    metrics_file = project_root / "reports" / "evaluation" / f"classification_metrics_{domain}.json"
    if metrics_file.exists():
        import json
        with open(metrics_file, "r", encoding="utf-8") as mf:
            return jsonify(json.load(mf)), 200
    return jsonify({"error": f"No evaluation metrics found for domain: {domain}"}), 404
