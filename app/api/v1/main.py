"""
Main functional API router combining all functional routes
"""
from fastapi import APIRouter
from . import (
    auth,
    student_routes, 
    api_key_routes, 
    provider_routes, 
    llm_routes,
    analytics_routes
)

router = APIRouter()

# Include all functional route modules
router.include_router(auth.router)  # Authentication routes
router.include_router(student_routes.router)
router.include_router(api_key_routes.router) 
router.include_router(provider_routes.router)
router.include_router(llm_routes.router)
router.include_router(analytics_routes.router)  # Analytics routes

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}
