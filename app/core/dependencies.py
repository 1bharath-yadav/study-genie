from app.db.models import DatabaseManager
from app.core import db_manager

def get_db() -> DatabaseManager:
    return db_manager