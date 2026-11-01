"""
Utility helper functions for City Guardian.
"""
import uuid
import os
from pathlib import Path
from typing import Optional
from src.utils.constants import ALLOWED_IMAGE_EXTENSIONS


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def generate_unique_filename(original_filename: str) -> str:
    """Generate a UUID-based unique filename preserving the extension."""
    ext = Path(original_filename).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
    return path


def format_severity_badge(severity_level: str) -> str:
    """Return an emoji badge for a severity level."""
    badges = {
        "CRITICAL": "🔴 CRITICAL",
        "HIGH": "🟠 HIGH",
        "MEDIUM": "🟡 MEDIUM",
        "LOW": "🟢 LOW",
        "NONE": "✅ NONE",
    }
    return badges.get(severity_level, "❓ UNKNOWN")


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float value between min and max."""
    return max(min_val, min(max_val, value))
