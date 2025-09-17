"""
Main functional API router combining all functional routes for API v1.
"""
from fastapi import APIRouter
from . import (
    auth,
    student_routes,
    api_key_routes,
    provider_routes,
    llm_routes,
    session_routes,
    model_preferences,
    models,
    content_routes,
    cache_routes,
    export_routes,
)

router = APIRouter()

# Include all functional route modules
router.include_router(auth.router)
router.include_router(student_routes.router)
router.include_router(api_key_routes.router)
router.include_router(provider_routes.router)
router.include_router(llm_routes.router)
router.include_router(content_routes.router)
router.include_router(cache_routes.router)
router.include_router(session_routes.router)
router.include_router(export_routes.router)
router.include_router(model_preferences.router)
router.include_router(models.router)


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}
