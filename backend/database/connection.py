"""
Database connection utilities.
"""
from backend.app import db
from src.utils.logger import get_logger

logger = get_logger(__name__)


def init_db(app):
    """Initialize database tables."""
    with app.app_context():
        db.create_all()
        logger.info("Database tables initialized.")


def get_db():
    """Get the SQLAlchemy db instance."""
    return db
