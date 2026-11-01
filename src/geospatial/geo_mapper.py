"""
Geo Mapper: generates GeoJSON from incident reports for map visualization.
"""
import json
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH": "#FF8C00",
    "MEDIUM": "#FFD700",
    "LOW": "#32CD32",
    "NONE": "#808080",
}


class GeoMapper:
    """
    Converts incident reports to GeoJSON for Leaflet/Mapbox rendering.
    """

    def reports_to_geojson(self, reports: List[Dict[str, Any]]) -> Dict:
        """
        Convert a list of incident reports to a GeoJSON FeatureCollection.

        Args:
            reports: List of report dicts from ReportGenerator

        Returns:
            GeoJSON FeatureCollection dict
        """
        features = []
        for report in reports:
            location = report.get("location", {})
            lat = location.get("latitude")
            lon = location.get("longitude")

            if lat is None or lon is None:
                logger.debug(f"Skipping report {report.get('report_id')} — no coordinates")
                continue

            severity_level = report.get("severity", {}).get("level", "LOW")
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "report_id": report.get("report_id"),
                    "domain": report.get("domain"),
                    "domain_name": report.get("domain_name"),
                    "severity_level": severity_level,
                    "severity_score": report.get("severity", {}).get("score"),
                    "priority_label": report.get("severity", {}).get("priority_label"),
                    "address": location.get("address"),
                    "ward": location.get("ward"),
                    "city": location.get("city"),
                    "timestamp": report.get("timestamp"),
                    "status": report.get("status"),
                    "color": SEVERITY_COLORS.get(severity_level, "#808080"),
                    "detected_objects": report.get("detection_summary", {}).get(
                        "detected_objects", []
                    ),
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "total_incidents": len(features),
        }

    def export_geojson_file(
        self, reports: List[Dict], output_path: str = "data/map_incidents.geojson"
    ) -> str:
        """Export GeoJSON to a file."""
        geojson = self.reports_to_geojson(reports)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        logger.info(f"GeoJSON exported: {output_path} ({len(geojson['features'])} features)")
        return output_path
