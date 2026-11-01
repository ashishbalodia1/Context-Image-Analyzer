"""
Health check route.
"""
import platform
import sys
from datetime import datetime
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """GET /api/health — system health check."""
    return jsonify({
        "status": "healthy",
        "service": "City Guardian Environmental Intelligence API",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.system(),
        "version": "1.0.0",
    }), 200
