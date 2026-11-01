"""
SQLAlchemy Location model.
"""
from backend.app import db


class Location(db.Model):
    __tablename__ = "locations"

    id = db.Column(db.Integer, primary_key=True)
    report_db_id = db.Column(db.Integer, db.ForeignKey("reports.id"), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    display_address = db.Column(db.Text)
    ward = db.Column(db.String(200))
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    postcode = db.Column(db.String(20))
    country = db.Column(db.String(100))

    def to_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "address": self.display_address,
            "ward": self.ward,
            "city": self.city,
            "state": self.state,
            "postcode": self.postcode,
            "country": self.country,
        }
