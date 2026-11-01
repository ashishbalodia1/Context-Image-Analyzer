"""
SQLAlchemy Report model.
"""
from datetime import datetime
from backend.app import db


class Report(db.Model):
    __tablename__ = "reports"

    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.String(20), unique=True, nullable=False)
    domain = db.Column(db.String(50), nullable=False)
    severity_level = db.Column(db.String(20), nullable=False)
    severity_score = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default="OPEN")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.Text)
    annotated_path = db.Column(db.Text)
    ai_context = db.Column(db.Text)
    responding_department = db.Column(db.String(200))
    recommended_action = db.Column(db.Text)
    total_detections = db.Column(db.Integer, default=0)

    # Relationships
    incidents = db.relationship("Incident", backref="report", lazy=True)
    location = db.relationship("Location", backref="report", uselist=False, lazy=True)

    def to_dict(self):
        return {
            "report_id": self.report_id,
            "domain": self.domain,
            "severity_level": self.severity_level,
            "severity_score": self.severity_score,
            "status": self.status,
            "timestamp": self.timestamp.isoformat() + "Z",
            "total_detections": self.total_detections,
            "ai_context": self.ai_context,
        }
