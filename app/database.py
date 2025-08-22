"""Compatibility shim: re-export database manager and manager classes.

Some modules import `app.database` (e.g. `app.services`) while the real
implementations live in `app.db.models`. This file re-exports the needed
symbols so imports succeed in tests and at runtime.
"""
from .db.models import (
    DatabaseManager,
    StudentManager,
    SubjectManager,
    ProgressTracker,
    RecommendationEngine,
    AnalyticsManager,
    StudySessionManager,
    initialize_database,
)

__all__ = [
    "DatabaseManager",
    "StudentManager",
    "SubjectManager",
    "ProgressTracker",
    "RecommendationEngine",
    "AnalyticsManager",
    "StudySessionManager",
    "initialize_database",
]
