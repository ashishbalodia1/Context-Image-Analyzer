"""
Map route: returns GeoJSON for incident map visualization and hotspot analysis.
Endpoints:
  GET /api/map/incidents
  GET /api/map/hotspots, GET /api/environmental-hotspots
"""
from flask import Blueprint, jsonify, request
from src.geospatial.geo_mapper import GeoMapper
from src.geospatial.hotspot_analyzer import HotspotAnalyzer
from src.utils.logger import get_logger
from backend.routes.reports import _reports_store

logger = get_logger(__name__)
map_bp = Blueprint("map", __name__)

_geo_mapper = GeoMapper()
_hotspot_analyzer = HotspotAnalyzer()


@map_bp.route("/map/incidents", methods=["GET"])
def get_incident_geojson():
    """
    GET /api/map/incidents
    Returns GeoJSON FeatureCollection of all recorded incidents with geographic markers.
    """
    domain_filter = request.args.get("domain")
    severity_filter = request.args.get("visual_severity") or request.args.get("severity")

    reports = _reports_store
    if domain_filter:
        reports = [r for r in reports if r.get("domain") == domain_filter]
    if severity_filter:
        sev_upper = severity_filter.upper()
        reports = [
            r for r in reports
            if str(r.get("visual_severity", "")).upper() == sev_upper
            or str(r.get("severity", {}).get("visual_severity", "")).upper() == sev_upper
        ]

    geojson = _geo_mapper.reports_to_geojson(reports)
    return jsonify(geojson), 200


@map_bp.route("/map/hotspots", methods=["GET"])
@map_bp.route("/environmental-hotspots", methods=["GET"])
def get_hotspots():
    """
    GET /api/environmental-hotspots (or /api/map/hotspots)
    Identifies geographic clusters and density hotspots across municipal wards.
    """
    domain_filter = request.args.get("domain")
    try:
        min_incidents = int(request.args.get("min_incidents", 2))
    except ValueError:
        min_incidents = 2

    hotspots = _hotspot_analyzer.find_hotspots(
        _reports_store,
        min_incidents=min_incidents,
        domain_filter=domain_filter,
    )

    # Convert hotspots to GeoJSON representation as well
    features = []
    for h in hotspots:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [h.get("center_lon"), h.get("center_lat")]
            },
            "properties": {
                "hotspot_id": h.get("hotspot_id"),
                "incident_count": h.get("incident_count"),
                "dominant_domain": h.get("dominant_domain"),
                "hotspot_score": h.get("hotspot_score"),
                "ward": h.get("ward"),
                "priority": h.get("priority", "HIGH")
            }
        })

    geojson_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    return jsonify({
        "status": "success",
        "total_hotspots": len(hotspots),
        "hotspots": hotspots,
        "geojson": geojson_collection
    }), 200
