"""
Environmental Incident Report & Context Generation Engine for City Guardian.
Combines YOLOv8 detections, visual severity, geolocation metadata, statutory compliance acts,
and external sensor data into legally-grounded, actionable municipal incident reports.
"""
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Environmental Incident Report Generator.
    Combines computer vision evidence with geospatial metadata,
    statutory compliance rules, and external sensor data.
    """

    STATUTORY_ACTS = {
        "air_pollution": "Air (Prevention and Control of Pollution) Act 1981 & DPCC Norms",
        "water_quality": "Water (Prevention and Control of Pollution) Act 1974 & CPCB Standards",
        "sewage_drainage": "Municipal Corporation Public Health & Sanitation Bye-laws 2007",
        "mining_dust": "Mines and Minerals (Development and Regulation) Act 1957 (MMDR Act)",
        "solid_waste": "Solid Waste Management (SWM) Rules 2016 & Plastic Waste Rules 2021"
    }

    DEPARTMENT_MAP = {
        "air_pollution": "Pollution Control Committee (Air Surveillance Cell)",
        "water_quality": "State Jal Board / CPCB Water Quality Division",
        "sewage_drainage": "Municipal Drainage & Sewerage Maintenance Department",
        "mining_dust": "State Mining Directorate & District Mineral Foundation",
        "solid_waste": "Municipal Solid Waste Management & Sanitation Department"
    }

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def generate(
        self,
        detection_result: Dict[str, Any],
        severity_result: Dict[str, Any],
        geo_context: Dict[str, Any],
        sensor_data: Optional[Dict[str, Any]] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
        citizen_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates full structured environmental incident report matching production specifications.
        """
        incident_id = f"CG-{uuid.uuid4().hex[:6].upper()}"
        timestamp = image_metadata.get("timestamp") or datetime.utcnow().isoformat() + "Z"

        domain = detection_result.get("domain", "solid_waste")
        domain_name = detection_result.get("domain_name", domain.replace("_", " ").title())
        detections = detection_result.get("detections", [])
        visual_severity = severity_result.get("visual_severity", "medium")
        severity_score = severity_result.get("visual_severity_score", 0.0)

        # Primary detected issue
        primary_issue = detections[0]["class_name"] if detections else "no_visible_anomaly"

        # Department & Act
        department = self.DEPARTMENT_MAP.get(domain, "Municipal Grievance Cell")
        statutory_act = self.STATUTORY_ACTS.get(domain, "Environment (Protection) Act 1986")

        # Format detection list matching standard schema
        formatted_detections = []
        for det in detections:
            bbox = det.get("bbox", {})
            if isinstance(bbox, dict):
                bbox_coords = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
            else:
                bbox_coords = list(bbox)

            formatted_detections.append({
                "class": det.get("class_name"),
                "confidence": det.get("confidence"),
                "bbox": bbox_coords,
                "area_fraction": det.get("area_fraction")
            })

        # Generate narrative description (rule-based or LLM)
        description = self._generate_description(
            domain=domain,
            primary_issue=primary_issue,
            detections=formatted_detections,
            geo_context=geo_context,
            visual_severity=visual_severity,
            sensor_data=sensor_data,
            citizen_notes=citizen_description
        )

        recommended_action = self._get_recommended_action(domain, visual_severity)

        # Verification requirement flag
        verification_required = visual_severity in ["high", "critical"] or not geo_context.get("location_available", True)

        # Build official report payload
        report = {
            "incident_id": incident_id,
            "status": "OPEN",
            "timestamp": timestamp,
            "domain": domain,
            "domain_label": domain_name,
            "detected_issue": primary_issue,
            "detections": formatted_detections,
            "total_objects": len(formatted_detections),
            "location": {
                "latitude": geo_context.get("latitude"),
                "longitude": geo_context.get("longitude"),
                "display_address": geo_context.get("display_address", "Location not geotagged"),
                "ward": geo_context.get("city_district") or geo_context.get("suburb", "Unspecified Ward"),
                "city": geo_context.get("city", "Delhi NCR")
            },
            "visual_severity": visual_severity,
            "visual_severity_score": severity_score,
            "description": description,
            "statutory_act": statutory_act,
            "recommended_action": recommended_action,
            "department": department,
            "sla_hours": self._get_sla(visual_severity),
            "verification_required": verification_required,
            "external_sensor_data": sensor_data or {},
            "evidence": {
                "original_image": detection_result.get("image_path"),
                "annotated_image": detection_result.get("annotated_image_path")
            }
        }

        logger.info(f"Report Generated: [{incident_id}] Domain={domain} | Severity={visual_severity.upper()}")
        return report

    def _generate_description(
        self,
        domain: str,
        primary_issue: str,
        detections: List[Dict],
        geo_context: Dict,
        visual_severity: str,
        sensor_data: Optional[Dict],
        citizen_notes: Optional[str]
    ) -> str:
        """
        Generates factual, scientifically-grounded narrative without hallucinating facts.
        """
        location_str = geo_context.get("display_address") or "the uploaded site"
        object_names = list(set([d["class"].replace("_", " ") for d in detections]))
        obj_text = ", ".join(object_names) if object_names else "anomalies"

        templates = {
            "air_pollution": (
                f"Visible {obj_text} detected in photograph at {location_str}. "
                f"Visual severity is assessed as {visual_severity.upper()}. "
                f"This condition indicates potential atmospheric particulate non-compliance under {self.STATUTORY_ACTS['air_pollution']}. "
            ),
            "water_quality": (
                f"Water body inspection at {location_str} reveals visible {obj_text}. "
                f"Visual severity is assessed as {visual_severity.upper()}. "
                f"Indicates surface contamination under {self.STATUTORY_ACTS['water_quality']}. "
                f"Physical water sampling (pH, TDS, turbidity) is required for legal confirmation."
            ),
            "sewage_drainage": (
                f"Visible {obj_text} detected at {location_str}. "
                f"Visual severity is assessed as {visual_severity.upper()}. "
                f"This condition suggests acute drainage channel choking or pipeline leakage. "
                f"Municipal desilting and repair team dispatch is advised."
            ),
            "mining_dust": (
                f"Industrial/quarry surveillance at {location_str} shows visible {obj_text}. "
                f"Visual severity is assessed as {visual_severity.upper()}. "
                f"Potential non-compliance with dust suppression protocols under {self.STATUTORY_ACTS['mining_dust']}."
            ),
            "solid_waste": (
                f"Unauthorized accumulation of {obj_text} detected at {location_str}. "
                f"Visual severity is assessed as {visual_severity.upper()}. "
                f"Constitutes a municipal solid waste management violation under {self.STATUTORY_ACTS['solid_waste']}."
            )
        }

        base_desc = templates.get(
            domain,
            f"Environmental anomaly ({obj_text}) detected at {location_str}. Visual severity: {visual_severity.upper()}."
        )

        # Append sensor correlation if available
        if sensor_data:
            if "pm2_5" in sensor_data and sensor_data["pm2_5"] is not None:
                base_desc += f" Concurrent ambient AQI sensor reads PM2.5: {sensor_data['pm2_5']} µg/m³ ({sensor_data.get('aqi_label', 'Unknown')})."
            if "pH" in sensor_data and sensor_data["pH"] is not None:
                base_desc += f" Water sensor reading indicates pH: {sensor_data['pH']}."

        if citizen_notes:
            base_desc += f" Citizen remarks: '{citizen_notes.strip()}'."

        return base_desc

    @staticmethod
    def _get_recommended_action(domain: str, severity: str) -> str:
        actions = {
            "air_pollution": {
                "critical": "Emergency emission shutdown notice + immediate DPCC field inspection",
                "high": "Dispatch mobile air monitoring van + issue formal emission show-cause notice",
                "medium": "Schedule site inspection within 24 hours",
                "low": "Log incident for routine emissions audit",
                "none": "No immediate enforcement needed"
            },
            "water_quality": {
                "critical": "Immediate industrial outfall pipe sealing + CPCB emergency water sampling",
                "high": "Deploy floating waste barrier + collect grab samples within 6 hours",
                "medium": "Schedule surface skimming and water quality test within 24 hours",
                "low": "Log for routine water body maintenance",
                "none": "Water surface visually clear"
            },
            "sewage_drainage": {
                "critical": "Emergency super-sucker tanker dispatch + bio-sanitization of overflow zone",
                "high": "Drainage desilting squad dispatch within 6 hours",
                "medium": "Schedule maintenance and sewer line rodding within 24 hours",
                "low": "Add to scheduled weekly drain clearance rota",
                "none": "Drainage channel unobstructed"
            },
            "mining_dust": {
                "critical": "Temporary work suspension order + mandatory water-sprinkler audit",
                "high": "Enforce mandatory vehicle tarpaulin covering + dust barrier inspection",
                "medium": "Issue advisory notice to haul road maintenance operator",
                "low": "Log for monthly environmental compliance audit",
                "none": "Dust levels within visual baseline"
            },
            "solid_waste": {
                "critical": "Immediate waste clearance order + issue challan under SWM Rules 2016",
                "high": "Deploy municipal waste compactors within 6 hours + zone officer alert",
                "medium": "Schedule waste collection within 24 hours",
                "low": "Add to daily municipal sweeping schedule",
                "none": "Public area free of visible litter"
            }
        }
        return actions.get(domain, {}).get(severity, "Municipal inspection recommended")

    @staticmethod
    def _get_sla(severity: str) -> int:
        return {"critical": 2, "high": 6, "medium": 24, "low": 72, "none": 0}.get(severity, 24)
