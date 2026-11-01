"""
Shared constants for City Guardian system.
"""

SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]

DOMAIN_ICONS = {
    "air_pollution": "🏭",
    "water_quality": "💧",
    "sewage_drainage": "🚰",
    "mining_dust": "⛏️",
    "solid_waste": "🗑️",
}

VALID_DOMAINS = list(DOMAIN_ICONS.keys())

SEVERITY_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH": "#FF8C00",
    "MEDIUM": "#FFD700",
    "LOW": "#32CD32",
    "NONE": "#808080",
}

SLA_HOURS = {
    "CRITICAL": 2,
    "HIGH": 6,
    "MEDIUM": 24,
    "LOW": 72,
    "NONE": 0,
}

MAX_IMAGE_SIZE_MB = 20
TARGET_IMAGE_SIZE = (640, 640)
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
