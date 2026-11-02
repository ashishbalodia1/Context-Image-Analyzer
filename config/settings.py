"""
Central configuration for City Guardian - Environmental Intelligence System
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Base paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = DATA_DIR / "uploads"

# ─── Security & Authentication ───────────────────────────────────────────────
CITY_GUARDIAN_API_KEY = os.getenv("CITY_GUARDIAN_API_KEY", "cg_live_cityguardian_2026")

# ─── Flask ───────────────────────────────────────────────────────────────────
FLASK_CONFIG = {
    "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key"),
    "DEBUG": os.getenv("DEBUG", "True").lower() == "true",
    "MAX_CONTENT_LENGTH": int(os.getenv("MAX_UPLOAD_SIZE_MB", 20)) * 1024 * 1024,
}

# ─── Database ────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://user:password@localhost:5432/city_guardian_db"
)

# ─── Model paths ─────────────────────────────────────────────────────────────
YOLO_MODELS = {
    "air_pollution":   os.getenv("AIR_POLLUTION_MODEL",   "models/trained/air_pollution_yolov8.pt"),
    "water_quality":   os.getenv("WATER_QUALITY_MODEL",   "models/trained/water_quality_yolov8.pt"),
    "sewage_drainage": os.getenv("SEWAGE_DRAINAGE_MODEL", "models/trained/sewage_drainage_yolov8.pt"),
    "mining_dust":     os.getenv("MINING_DUST_MODEL",     "models/trained/mining_dust_yolov8.pt"),
    "solid_waste":     os.getenv("SOLID_WASTE_MODEL",     "models/trained/solid_waste_yolov8.pt"),
}

BASE_YOLO_MODEL = os.getenv("BASE_YOLO_MODEL", "yolov8m.pt")

# ─── Detection thresholds ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.20))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))

# ─── External APIs ───────────────────────────────────────────────────────────
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY", "")

AQI_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# ─── Geospatial defaults ──────────────────────────────────────────────────────
DEFAULT_LOCATION = {
    "lat": float(os.getenv("DEFAULT_LAT", 28.6139)),
    "lon": float(os.getenv("DEFAULT_LON", 77.2090)),
    "city": os.getenv("DEFAULT_CITY", "Delhi"),
}

# ─── Upload settings ────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = set(
    os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp").split(",")
)
