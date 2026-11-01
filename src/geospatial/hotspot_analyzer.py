"""
Hotspot Analyzer: identifies geographic clusters of environmental violations.
Uses simple grid-based density analysis.
"""
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HotspotAnalyzer:
    """
    Analyzes geographic distribution of incidents to find hotspots.
    Uses a grid-cell density approach with configurable resolution.
    """

    def __init__(self, grid_resolution_km: float = 1.0):
        """
        Args:
            grid_resolution_km: Size of each grid cell in kilometers
        """
        self.grid_resolution_km = grid_resolution_km
        # Approx degrees per km at mid-latitudes
        self.lat_per_km = 1 / 111.0
        self.lon_per_km = 1 / 111.0  # Will vary with latitude

    def _to_grid_key(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to a grid cell key."""
        grid_lat = int(lat / (self.lat_per_km * self.grid_resolution_km))
        grid_lon = int(lon / (self.lon_per_km * self.grid_resolution_km))
        return (grid_lat, grid_lon)

    def _grid_to_centroid(self, grid_key: Tuple[int, int]) -> Tuple[float, float]:
        """Convert a grid key back to approximate center coordinates."""
        lat = grid_key[0] * self.lat_per_km * self.grid_resolution_km
        lon = grid_key[1] * self.lon_per_km * self.grid_resolution_km
        return (round(lat, 6), round(lon, 6))

    def find_hotspots(
        self,
        reports: List[Dict[str, Any]],
        min_incidents: int = 3,
        domain_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Find geographic hotspots from a list of incident reports.

        Args:
            reports: List of report dicts
            min_incidents: Minimum number of incidents to qualify as a hotspot
            domain_filter: Optional domain to filter (e.g., 'solid_waste')

        Returns:
            List of hotspot dicts sorted by incident count descending
        """
        grid: Dict[Tuple, List] = defaultdict(list)

        for report in reports:
            if domain_filter and report.get("domain") != domain_filter:
                continue

            location = report.get("location", {})
            lat = location.get("latitude")
            lon = location.get("longitude")
            if lat is None or lon is None:
                continue

            key = self._to_grid_key(lat, lon)
            grid[key].append(report)

        hotspots = []
        for key, incidents in grid.items():
            if len(incidents) < min_incidents:
                continue
            centroid = self._grid_to_centroid(key)
            severity_counts = defaultdict(int)
            domains = defaultdict(int)
            for inc in incidents:
                level = inc.get("severity", {}).get("level", "LOW")
                severity_counts[level] += 1
                domains[inc.get("domain", "unknown")] += 1

            hotspots.append({
                "centroid_lat": centroid[0],
                "centroid_lon": centroid[1],
                "incident_count": len(incidents),
                "severity_breakdown": dict(severity_counts),
                "dominant_domain": max(domains, key=domains.get),
                "domain_breakdown": dict(domains),
                "grid_cell": key,
            })

        return sorted(hotspots, key=lambda h: h["incident_count"], reverse=True)
