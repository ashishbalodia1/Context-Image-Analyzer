"""
Rule Engine: applies compliance rules from domains.yaml to detection results.
Produces actionable violation flags based on detected classes and thresholds.
"""
from typing import Dict, Any, List
import yaml
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

DOMAIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "domains.yaml"


class RuleEngine:
    """
    Evaluates compliance rules for each environmental domain.
    Flags violations and determines enforcement actions.
    """

    def __init__(self):
        with open(DOMAIN_CONFIG_PATH, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def evaluate(
        self, domain: str, detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate compliance rules for the given domain and detections.

        Args:
            domain: Environmental domain key
            detection_result: Output from YOLODetector.detect()

        Returns:
            Dict with violations list and enforcement actions
        """
        detections = detection_result.get("detections", [])
        rules = self._config["domains"].get(domain, {}).get("compliance_rules", [])
        detected_classes = {d["class_name"] for d in detections}

        violations = []
        for rule in rules:
            threshold = rule.get("threshold", 1)
            # Simple rule: if any detection exists, rule is triggered
            count = len(detections)
            if count >= threshold:
                violations.append({
                    "rule": rule["rule"],
                    "action": rule["action"],
                    "detections_triggering": count,
                    "triggered": True,
                })

        return {
            "domain": domain,
            "violations_found": len(violations) > 0,
            "violation_count": len(violations),
            "violations": violations,
            "enforcement_required": any(v["triggered"] for v in violations),
        }
