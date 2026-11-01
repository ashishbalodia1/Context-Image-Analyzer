"""
Ward Locator: maps GPS coordinates to administrative ward/zone boundaries.
Currently uses reverse geocoding fallback; can be upgraded to PostGIS spatial queries.
"""
from typing import Dict, Optional
from src.preprocessing.geo_extractor import resolve_location
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WardLocator:
    """
    Resolves a GPS coordinate to the corresponding ward/zone/district.
    In production, this should query a PostGIS database with ward boundary polygons.
    """

    def locate(self, lat: float, lon: float) -> Dict:
        """
        Locate the administrative ward for the given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Ward info dict
        """
        geo = resolve_location(lat, lon)
        return {
            "latitude": lat,
            "longitude": lon,
            "ward": geo.get("city_district") or geo.get("suburb", "Unknown Ward"),
            "district": geo.get("city_district", ""),
            "city": geo.get("city", ""),
            "state": geo.get("state", ""),
            "postcode": geo.get("postcode", ""),
            "display_address": geo.get("display_address", ""),
        }
