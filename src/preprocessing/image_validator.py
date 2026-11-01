"""
Image validation: checks format, size, corruption, and metadata presence.
"""
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
from PIL import Image
try:
    import exifread
    HAS_EXIFREAD = True
except ImportError:
    exifread = None
    HAS_EXIFREAD = False
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageValidator:
    """
    Validates uploaded images before processing.
    Checks: format, resolution, file size, EXIF metadata.
    """

    ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
    MIN_RESOLUTION = (224, 224)
    MAX_RESOLUTION = (7680, 4320)  # 8K max
    MAX_FILE_SIZE_MB = 20

    def __init__(self):
        self.errors: list = []
        self.warnings: list = []

    def validate(self, image_path: str) -> Tuple[bool, Dict]:
        """
        Full validation pipeline.

        Returns:
            (is_valid, result_dict)
        """
        self.errors = []
        self.warnings = []
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "metadata": {},
        }

        path = Path(image_path)

        # 1. File existence
        if not path.exists():
            self.errors.append(f"File not found: {image_path}")
            result["errors"] = self.errors
            return False, result

        # 2. File size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            self.errors.append(
                f"File too large: {size_mb:.1f}MB (max {self.MAX_FILE_SIZE_MB}MB)"
            )

        # 3. Format check
        fmt = "UNKNOWN"
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)  # Re-open after verify
            fmt = img.format
            if fmt not in self.ALLOWED_FORMATS:
                self.errors.append(
                    f"Unsupported format: {fmt}. Allowed: {self.ALLOWED_FORMATS}"
                )
        except Exception as e:
            self.errors.append(f"Corrupt or unreadable image: {e}")
            result["errors"] = self.errors
            return False, result

        # 4. Resolution check
        width, height = img.size
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            self.errors.append(
                f"Resolution too low: {width}x{height}. Minimum: {self.MIN_RESOLUTION}"
            )
        if width > self.MAX_RESOLUTION[0] or height > self.MAX_RESOLUTION[1]:
            self.warnings.append(
                f"Very high resolution ({width}x{height}) — will be resized."
            )

        # 5. EXIF metadata (GPS)
        gps_data = self._extract_gps_exif(image_path)
        if not gps_data:
            self.warnings.append(
                "No GPS metadata found in image. Location must be provided manually."
            )

        is_valid = len(self.errors) == 0
        result.update({
            "valid": is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": {
                "format": fmt,
                "resolution": f"{width}x{height}",
                "size_mb": round(size_mb, 2),
                "gps": gps_data,
            },
        })
        logger.info(f"Validation {'passed' if is_valid else 'failed'} for {path.name}")
        return is_valid, result

    def _extract_gps_exif(self, image_path: str) -> Optional[Dict]:
        """Extract GPS coordinates from EXIF data if present."""
        if not HAS_EXIFREAD:
            # Fallback to PIL Image EXIF extraction if exifread not installed
            try:
                with Image.open(image_path) as img:
                    exif = img._getexif()
                    if not exif:
                        return None
                    # Tag 34853 is GPSInfo
                    gps_info = exif.get(34853)
                    if gps_info and 2 in gps_info and 4 in gps_info:
                        lat_raw = gps_info[2]
                        lon_raw = gps_info[4]
                        lat_ref = gps_info.get(1, "N")
                        lon_ref = gps_info.get(3, "E")
                        lat = float(lat_raw[0]) + float(lat_raw[1])/60.0 + float(lat_raw[2])/3600.0
                        if lat_ref == "S":
                            lat = -lat
                        lon = float(lon_raw[0]) + float(lon_raw[1])/60.0 + float(lon_raw[2])/3600.0
                        if lon_ref == "W":
                            lon = -lon
                        return {"latitude": round(lat, 7), "longitude": round(lon, 7), "timestamp": None}
            except Exception:
                pass
            return None

        try:
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            lat = tags.get("GPS GPSLatitude")
            lon = tags.get("GPS GPSLongitude")
            lat_ref = tags.get("GPS GPSLatitudeRef")
            lon_ref = tags.get("GPS GPSLongitudeRef")
            timestamp = tags.get("EXIF DateTimeOriginal")

            if lat and lon:
                lat_val = self._convert_dms_to_dd(lat.values, str(lat_ref))
                lon_val = self._convert_dms_to_dd(lon.values, str(lon_ref))
                return {
                    "latitude": lat_val,
                    "longitude": lon_val,
                    "timestamp": str(timestamp) if timestamp else None,
                }
        except Exception as e:
            logger.debug(f"EXIF extraction error: {e}")
        return None

    @staticmethod
    def _convert_dms_to_dd(dms_values, ref: str) -> float:
        """Convert DMS (degrees, minutes, seconds) to decimal degrees."""
        d, m, s = [float(x.num) / float(x.den) for x in dms_values]
        dd = d + m / 60 + s / 3600
        if ref in ["S", "W"]:
            dd = -dd
        return round(dd, 7)
