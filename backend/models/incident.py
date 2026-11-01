"""
SQLAlchemy Incident model (detected objects within a report).
"""
from backend.app import db


class Incident(db.Model):
    __tablename__ = "incidents"

    id = db.Column(db.Integer, primary_key=True)
    report_db_id = db.Column(db.Integer, db.ForeignKey("reports.id"), nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bbox_x1 = db.Column(db.Float)
    bbox_y1 = db.Column(db.Float)
    bbox_x2 = db.Column(db.Float)
    bbox_y2 = db.Column(db.Float)
    area_fraction = db.Column(db.Float)

    def to_dict(self):
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
                "x2": self.bbox_x2,
                "y2": self.bbox_y2,
            },
            "area_fraction": self.area_fraction,
        }
