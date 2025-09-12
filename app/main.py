"""
Main FastAPI application with Supabase integration - Pure Functional Version
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import functional routes
from app.api.v1.main import router as v1_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    # Startup
    logger.info("Starting StudyGenie API...")
    try:
        # Test Supabase connection
        logger.info("Testing Supabase connection...")
        from app.services.db import test_connection
        
        if await test_connection():
            logger.info("✅ Supabase connection successful")
        else:
            logger.warning("⚠️  Supabase connection failed")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Don't fail the startup, just warn
        logger.info("💡 Application will continue, but some features may not work")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down StudyGenie API...")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="StudyGenie API",
    description="AI-powered personalized learning platform - Pure Functional",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(v1_router, prefix="/api/v1")


# Health endpoint
@app.get("/health")
async def health_check():
    """Health check with Supabase connectivity test"""
    health_status = {
        "status": "healthy",
        "message": "StudyGenie API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0"
    }

    try:
        from app.services.db import test_connection
        if await test_connection():
            health_status["database"] = "connected"
        else:
            health_status["database"] = "disconnected"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["database"] = "error"
        health_status["db_error"] = str(e)
        health_status["status"] = "degraded"

    return health_status


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to StudyGenie API v3.0 - Pure Functional",
        "health": "/health",
        "docs": "/docs",
        "api": "/api/v1"
    }
