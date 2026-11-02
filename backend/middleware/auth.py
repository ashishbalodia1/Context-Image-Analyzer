"""
API Authentication Middleware for City Guardian.
Validates client requests via 'X-API-Key', 'Authorization: Bearer <key>', or query parameter.
"""
import os
from functools import wraps
from flask import request, jsonify

# Load configured keys from environment or fallback default
DEFAULT_DEV_KEY = "cg_live_cityguardian_2026"
_RAW_KEYS = os.getenv("CITY_GUARDIAN_API_KEY", DEFAULT_DEV_KEY)
VALID_API_KEYS = {k.strip() for k in _RAW_KEYS.split(",") if k.strip()}


def get_request_api_key() -> str | None:
    """Extracts API key from headers or query parameters."""
    # 1. Custom Header: X-API-Key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip()

    # 2. Standard Header: Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        parts = auth_header.split(" ", 1)
        if len(parts) == 2:
            return parts[1].strip()

    # 3. Query Parameter: ?api_key=... (convenient for testing & webhooks)
    query_key = request.args.get("api_key")
    if query_key:
        return query_key.strip()

    return None


def require_api_key(f):
    """
    Decorator to protect API endpoints with API Key verification.
    Returns 401 if key is missing, 403 if key is invalid.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow OPTIONS requests for CORS pre-flight
        if request.method == "OPTIONS":
            return f(*args, **kwargs)

        api_key = get_request_api_key()

        if not api_key:
            return jsonify({
                "status": "error",
                "error": "Unauthorized",
                "message": "Missing API Key. Pass your key in 'X-API-Key' header or 'Authorization: Bearer <key>'."
            }), 401

        if api_key not in VALID_API_KEYS:
            return jsonify({
                "status": "error",
                "error": "Forbidden",
                "message": "Invalid API Key. Access denied to City Guardian ML Pipeline."
            }), 403

        return f(*args, **kwargs)

    return decorated_function
