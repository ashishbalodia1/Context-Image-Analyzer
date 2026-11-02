"""
Flask application factory for City Guardian REST API.
Geo-Aware AI Environmental Intelligence & Compliance Monitoring System
"""
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import FLASK_CONFIG, DATABASE_URL, UPLOAD_DIR

# Safe DB initialization
try:
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()
    HAS_SQLALCHEMY = True
except ImportError:
    db = None
    HAS_SQLALCHEMY = False


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))

    # Config
    app.config["SECRET_KEY"] = FLASK_CONFIG.get("SECRET_KEY", "dev-secret")
    app.config["DEBUG"] = FLASK_CONFIG.get("DEBUG", False)
    app.config["MAX_CONTENT_LENGTH"] = FLASK_CONFIG.get("MAX_CONTENT_LENGTH", 20 * 1024 * 1024)
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # CORS enabled for mobile and React dashboard
    CORS(app)

    if HAS_SQLALCHEMY and db:
        try:
            db.init_app(app)
        except Exception:
            pass

    # Register blueprints (both /api and /api/v1 supported)
    from backend.routes.upload import upload_bp
    from backend.routes.reports import reports_bp
    from backend.routes.map import map_bp
    from backend.routes.health import health_bp

    app.register_blueprint(upload_bp, url_prefix="/api")
    app.register_blueprint(reports_bp, url_prefix="/api")
    app.register_blueprint(map_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")

    app.register_blueprint(upload_bp, url_prefix="/api/v1", name="upload_v1")
    app.register_blueprint(reports_bp, url_prefix="/api/v1", name="reports_v1")
    app.register_blueprint(map_bp, url_prefix="/api/v1", name="map_v1")
    app.register_blueprint(health_bp, url_prefix="/api/v1", name="health_v1")

    # Interactive Dashboard & Media Routes
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/uploads/<path:filename>")
    def serve_upload(filename):
        return send_from_directory(str(UPLOAD_DIR), filename)

    @app.route("/sample-image")
    def sample_image():
        sample_path = PROJECT_ROOT / "data" / "processed" / "solid_waste" / "images" / "test"
        if sample_path.exists():
            imgs = list(sample_path.glob("*.jpg"))
            if imgs:
                return send_from_directory(str(sample_path), imgs[0].name)
        return jsonify({"error": "No sample image found"}), 404

    # Global error handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": "File size exceeds limit (maximum 20 MB allowed)"
        }), 413

    @app.errorhandler(404)
    def resource_not_found(error):
        return jsonify({
            "error": "Endpoint or resource not found"
        }), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "error": "Internal server error occurred in pipeline"
        }), 500

    # Pre-warm all 5 domain detectors at startup so all domains respond instantly
    try:
        from backend.routes.upload import _router
        for dom in ["solid_waste", "water_quality", "air_pollution", "sewage_drainage", "mining_dust"]:
            _router._get_detector(dom)
    except Exception as e:
        app.logger.warning(f"Detector pre-warm warning: {e}")

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
