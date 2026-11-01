# src/detection/__init__.py
from .yolo_detector import YOLODetector
from .domain_router import DomainRouter

__all__ = ["YOLODetector", "DomainRouter"]
