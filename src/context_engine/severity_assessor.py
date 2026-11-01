"""
Visual Severity Assessment Engine for City Guardian.
Evaluates VISUAL SEVERITY from YOLOv8 detections, bounding-box area coverage,
object counts, and domain severity weights.

IMPORTANT SCIENTIFIC DISTINCTION:
This computes VISUAL SEVERITY from photographic evidence alone.
It does NOT directly measure chemical concentration, ppm, or legally confirmed violation risk.
"""
from typing import Dict, Any, List
import yaml
from pathlib import Path

DOMAIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "domains.yaml"


class SeverityAssessor:
    """
    Computes visual severity levels:
        CRITICAL  → score ≥ 0.85
        HIGH      → score ≥ 0.65
        MEDIUM    → score ≥ 0.40
        LOW       → score < 0.40
        NONE      → no detections
    """

    THRESHOLDS = {
        "CRITICAL": 0.85,
        "HIGH": 0.65,
        "MEDIUM": 0.40,
        "LOW": 0.0,
    }

    def __init__(self):
        if DOMAIN_CONFIG_PATH.exists():
            with open(DOMAIN_CONFIG_PATH, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = {"domains": {}}

    def assess(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate visual severity from YOLO detection output.

        Args:
            detection_result: Dict containing 'domain', 'detections' list with bbox, confidence, class_name.

        Returns:
            Dict with visual_severity_score, visual_severity, priority_label, and breakdown.
        """
        domain = detection_result.get("domain", "solid_waste")
        detections: List[Dict] = detection_result.get("detections", [])
        domain_classes = self._get_domain_class_weights(domain)

        if not detections:
            return {
                "visual_severity_score": 0.0,
                "visual_severity": "none",
                "severity_level": "NONE",
                "priority": "No Visible Anomaly",
                "affected_area_fraction": 0.0,
                "detection_count": 0,
                "breakdown": [],
            }

        breakdown = []
        weighted_scores = []
        total_area_fraction = 0.0

        for det in detections:
            class_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 0.5)
            area_fraction = det.get("area_fraction", 0.05)
            total_area_fraction += area_fraction

            # Default class weight if unlisted
            domain_weight = domain_classes.get(class_name, 0.5)

            # Mathematical score: confidence * domain_weight * (1 + min(2 * area, 0.5))
            area_bonus = min(area_fraction * 2.0, 0.5)
            score = confidence * domain_weight * (1.0 + area_bonus)
            weighted_scores.append(score)

            breakdown.append({
                "class": class_name,
                "confidence": round(confidence, 4),
                "domain_weight": round(domain_weight, 2),
                "area_fraction": round(area_fraction, 4),
                "contribution_score": round(score, 4),
            })

        # Final score = average of detection scores + count scaling
        avg_score = sum(weighted_scores) / len(weighted_scores)
        count_bonus = min(len(detections) * 0.05, 0.20)
        final_score = min(avg_score + count_bonus, 1.0)

        severity_level = self._get_level(final_score)
        priority = self._get_priority_label(severity_level)

        return {
            "visual_severity_score": round(final_score, 4),
            "visual_severity": severity_level.lower(),
            "severity_level": severity_level,
            "priority": priority,
            "affected_area_fraction": round(min(total_area_fraction, 1.0), 4),
            "detection_count": len(detections),
            "breakdown": breakdown,
        }

    def _get_domain_class_weights(self, domain: str) -> Dict[str, float]:
        """Fetch class severity weights from domains.yaml."""
        classes = self._config.get("domains", {}).get(domain, {}).get("classes", [])
        return {cls["name"]: cls["severity_weight"] for cls in classes}

    def _get_level(self, score: float) -> str:
        for level, threshold in self.THRESHOLDS.items():
            if score >= threshold:
                return level
        return "LOW"

    @staticmethod
    def _get_priority_label(level: str) -> str:
        return {
            "CRITICAL": "🔴 Critical — Immediate Action Required",
            "HIGH": "🟠 High Priority",
            "MEDIUM": "🟡 Medium Priority",
            "LOW": "🟢 Low Priority",
            "NONE": "✅ No Violation Detected",
        }.get(level, "Unknown")
