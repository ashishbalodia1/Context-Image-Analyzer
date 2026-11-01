"""
Geo-tagging extractor: pulls GPS from EXIF or accepts manual input.
Resolves coordinates to ward/district info via reverse geocoding.
"""
from typing import Optional, Dict
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    HAS_GEOPY = True
    geolocator = Nominatim(user_agent="city-guardian-env-system")
except ImportError:
    HAS_GEOPY = False
    GeocoderTimedOut = Exception
    geolocator = None
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_location(lat: float, lon: float, retries: int = 3) -> Dict:
    """
    Reverse geocode coordinates to get address, ward, district, city.

    Args:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        retries: Number of retry attempts on timeout

    Returns:
        dict with address components
    """
    if not HAS_GEOPY or geolocator is None:
        return {
            "latitude": lat,
            "longitude": lon,
            "display_address": f"Lat: {lat}, Lon: {lon}",
            "suburb": "Local Ward",
            "city_district": "Municipal Area",
            "city": "Metropolitan City",
            "state": "National Capital Region",
            "postcode": "110001",
            "country": "India",
            "raw": {},
        }

    for attempt in range(retries):
        try:
            location = geolocator.reverse(f"{lat}, {lon}", language="en", timeout=10)
            if location:
                addr = location.raw.get("address", {})
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "display_address": location.address,
                    "suburb": addr.get("suburb", ""),
                    "city_district": addr.get("city_district", ""),
                    "city": addr.get("city", addr.get("town", "")),
                    "state": addr.get("state", ""),
                    "postcode": addr.get("postcode", ""),
                    "country": addr.get("country", ""),
                    "raw": addr,
                }
        except GeocoderTimedOut:
            logger.warning(f"Geocoder timeout (attempt {attempt + 1}/{retries})")
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            break
    return {
        "latitude": lat,
        "longitude": lon,
        "display_address": f"Lat: {lat}, Lon: {lon}",
        "error": "Could not resolve address",
    }


def build_geo_context(
    lat: Optional[float],
    lon: Optional[float],
    manual_override: Optional[Dict] = None,
) -> Dict:
    """
    Build full geo context. Uses EXIF-extracted coords or manual input.

    Args:
        lat: Latitude (may be None if not extracted from EXIF)
        lon: Longitude
        manual_override: User-provided location data from mobile app

    Returns:
        Geo context dict
    """
    if manual_override:
        lat = manual_override.get("latitude", lat)
        lon = manual_override.get("longitude", lon)

    if lat is None or lon is None:
        logger.warning("No GPS data available — location context incomplete")
        return {"error": "No GPS coordinates available", "location_available": False}

    geo_info = resolve_location(lat, lon)
    geo_info["location_available"] = True
    return geo_info
