# CITY GUARDIAN — REST API SPECIFICATION

This document provides the complete API specification for the **City Guardian Environmental Intelligence Platform**.

- **Base URLs**: `/api` and `/api/v1`
- **Supported Formats**: JSON, Multipart Form-Data
- **Security**: API Key Authentication (`X-API-Key` or `Authorization: Bearer <key>`)

---

## 🔑 Authentication

All ML inference and write endpoints require a valid **City Guardian API Key**.

### Supported Authentication Methods:
1. **Custom Header (Recommended):**
   ```http
   X-API-Key: cg_live_cityguardian_2026
   ```
2. **Standard Bearer Token:**
   ```http
   Authorization: Bearer cg_live_cityguardian_2026
   ```
3. **Query Parameter (Testing / Webhooks):**
   ```http
   POST /api/analyze-image?api_key=cg_live_cityguardian_2026
   ```

### Key Verification Endpoint:
```http
GET /api/verify-key
```
Returns `200 OK` if the key is active, or `401 Unauthorized` / `403 Forbidden` if missing or invalid.

---

## Endpoints Index

1. `POST /api/analyze-image` — Analyze crowdsourced image and generate actionable context report
2. `GET /api/detection/<incident_id>` — Retrieve detailed incident report and context by ID
3. `GET /api/environmental-incidents` — Query incidents with multi-parameter filtering
4. `GET /api/environmental-hotspots` — Geospatial hotspot clustering (GeoJSON)
5. `GET /api/model-performance` — Retrieve inference latency, active domains, and model benchmarks
6. `PATCH /api/detection/<incident_id>/status` — Update resolution status in municipal workflow

---

## 1. Analyze Environmental Image

### `POST /api/analyze-image`
*(Alias: `/api/upload`)*

Accepts an image and optional GPS coordinates, routes through the YOLOv8 domain detector, estimates visual severity, and synthesizes an actionable environmental context report.

#### Request (Multipart Form-Data)

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` or `image` | Binary | **Yes** | JPG, PNG, or WebP photo (Max 20 MB) |
| `latitude` | Float | No | Decimal degrees (e.g. `28.6139`) |
| `longitude` | Float | No | Decimal degrees (e.g. `77.2090`) |
| `domain` | String | No | `air_pollution`, `water_quality`, `sewage_drainage`, `mining_dust`, `solid_waste` |
| `timestamp` | String | No | ISO 8601 timestamp (e.g. `2026-11-01T10:30:00Z`) |
| `description` | String | No | Citizen commentary |

#### Response (HTTP 201 Created)

```json
{
  "incident_id": "CG-A8829F",
  "domain": "solid_waste",
  "domain_label": "Solid Waste Management & Compliance",
  "department": "Municipal Solid Waste Management & Sanitation Department",
  "detected_issue": "illegal_dumping",
  "detections": [
    {
      "class": "illegal_dumping",
      "confidence": 0.912,
      "bbox": [120, 80, 480, 360],
      "area_fraction": 0.28
    }
  ],
  "visual_severity": "high",
  "visual_severity_score": 0.78,
  "statutory_act": "Solid Waste Management (SWM) Rules 2016 & Plastic Waste Rules 2021",
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "display_address": "Lat: 28.6139, Lon: 77.209",
    "ward": "Ward-12",
    "city": "Metropolitan City"
  },
  "evidence": {
    "original_image": "/data/uploads/image.jpg",
    "annotated_image": "/data/uploads/annotated/image_annotated.jpg"
  },
  "recommended_action": "Deploy sanitation team for heap clearance within 24 hours.",
  "sla_hours": 24,
  "verification_required": true,
  "status": "OPEN",
  "timestamp": "2026-11-01T15:28:54Z"
}
```

---

## 2. Retrieve Incident Dossier

### `GET /api/detection/<incident_id>`
*(Alias: `/api/reports/<report_id>`)*

#### Response (HTTP 200 OK)

```json
{
  "status": "success",
  "incident_id": "CG-A8829F",
  "data": { ... }
}
```

---

## 3. Query Environmental Incidents

### `GET /api/environmental-incidents`
*(Alias: `/api/reports`)*

#### Query Parameters
- `domain`: Filter by domain (`solid_waste`, `air_pollution`, etc.)
- `visual_severity`: Filter by severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `NONE`)
- `ward`: Filter by municipal ward identifier
- `limit`: Number of items per page (default: 50)
- `offset`: Pagination offset (default: 0)

#### Response (HTTP 200 OK)

```json
{
  "status": "success",
  "total": 142,
  "count": 50,
  "limit": 50,
  "offset": 0,
  "incidents": [ ... ]
}
```

---

## 4. Geospatial Hotspot Clusters

### `GET /api/environmental-hotspots`
*(Alias: `/api/map/hotspots`)*

#### Query Parameters
- `domain`: Filter by domain (optional)
- `min_incidents`: Minimum incidents to qualify as cluster (default: 2)

#### Response (HTTP 200 OK)

```json
{
  "status": "success",
  "total_hotspots": 3,
  "hotspots": [
    {
      "hotspot_id": "HS-001",
      "center_lat": 28.6139,
      "center_lon": 77.2090,
      "incident_count": 6,
      "dominant_domain": "solid_waste",
      "hotspot_score": 0.88,
      "priority": "HIGH"
    }
  ],
  "geojson": {
    "type": "FeatureCollection",
    "features": [ ... ]
  }
}
```

---

## 5. Model Telemetry & Benchmarks

### `GET /api/model-performance`

#### Response (HTTP 200 OK)

```json
{
  "status": "ready",
  "system": "City Guardian Environmental Intelligence Platform",
  "model_architecture": "YOLOv8 Nano (Domain-Specialist Multi-Head)",
  "domains_supported": [ ... ],
  "performance_benchmarks": {
    "mean_inference_latency_ms": 10.48,
    "inference_fps_cpu": 95.4,
    "model_size_mb": 5.9,
    "map_50": 0.812,
    "map_50_95": 0.584,
    "precision": 0.841,
    "recall": 0.776,
    "f1_score": 0.807
  }
}
```

---

## 6. Update Incident Status

### `PATCH /api/detection/<incident_id>/status`
*(Alias: `/api/reports/<report_id>/status`)*

#### Request Body (JSON)

```json
{
  "status": "IN_PROGRESS"
}
```
*Valid statuses*: `OPEN`, `IN_PROGRESS`, `RESOLVED`, `CLOSED`, `REJECTED`.

#### Response (HTTP 200 OK)

```json
{
  "status": "success",
  "message": "Incident status updated successfully",
  "incident_id": "CG-A8829F",
  "status_code": "IN_PROGRESS"
}
```

