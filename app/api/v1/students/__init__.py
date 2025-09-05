"""
Students API module with modular structure
"""
from fastapi import APIRouter

from . import crud, progress, recommendations, api_keys

# Create a combined router for all student endpoints
router = APIRouter()

# Include all student-related routers
router.include_router(crud.router, tags=["students"])
router.include_router(progress.router, tags=["student-progress"])
router.include_router(recommendations.router, tags=["student-recommendations"])
router.include_router(api_keys.router, tags=["student-api-keys"])
