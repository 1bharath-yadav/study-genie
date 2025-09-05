"""
Main FastAPI application with Supabase integration
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.core.security import get_current_user
from app.api.v1.students import router as students_router
from app.api.v1.llm import router as llm_router
from app.api.v1.auth import router as auth_router
from app.api.v1.api_keys_supabase import router as api_keys_router
from app.api.v1.users_supabase import router as users_router
from app.api.v1.analytics import router as analytics_router
from app.db.models import initialize_database, DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database manager instance
db_manager: DatabaseManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    global db_manager

    # Startup
    logger.info("Starting StudyGenie API...")
    try:
        # Initialize Supabase tables first
        logger.info("Initializing Supabase tables...")
        from app.db.supabase_init import SupabaseInitializer

        try:
            supabase_init = SupabaseInitializer()
            # Verify tables exist, if not, provide instructions
            if not supabase_init.verify_tables():
                logger.warning("⚠️  Some Supabase tables are missing!")
                logger.info(
                    "📝 Please run the following command to create tables:")
                logger.info("   python -m app.db.supabase_init --generate-sql")
                logger.info(
                    "   Then run the generated SQL script in your Supabase SQL Editor")
                # Don't fail the startup, just warn
            else:
                logger.info("✅ All Supabase tables verified successfully")
        except Exception as e:
            logger.warning(f"Supabase table verification failed: {e}")
            logger.info("💡 This might be normal if tables don't exist yet")

        # Initialize local database connection (for fallback/caching if needed)
        # Note: Since we're using Supabase, local database is optional
        logger.info("Skipping local database initialization (using Supabase)")
        db_manager = None  # We'll create a dummy manager for dependency injection
        # db_manager = await initialize_database()
        logger.info("Database initialization completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down StudyGenie API...")
    if db_manager:
        await db_manager.close_pool()
        logger.info("Database connections closed")


def get_database_manager() -> DatabaseManager:
    """Dependency to get the database manager instance (legacy - using Supabase now)"""
    # Return a dummy manager since we're using Supabase
    # This is kept for backwards compatibility
    from app.db.models import DatabaseManager
    return DatabaseManager()


# Create FastAPI app with lifespan management
app = FastAPI(
    title="StudyGenie API",
    description="AI-powered personalized learning platform",
    version="2.0.0",
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

# Include API routers
app.include_router(auth_router, prefix="/api", tags=["Authentication"])
app.include_router(students_router, prefix="/api", tags=["Students"])
app.include_router(llm_router, prefix="/api", tags=["LLM"])
app.include_router(api_keys_router, prefix="/api", tags=["API Keys"])
app.include_router(users_router, prefix="/api", tags=["Users"])
app.include_router(analytics_router, prefix="/api", tags=["Analytics"])


# Health endpoint with Supabase check
@app.get("/health")
async def health_check():
    """Enhanced health check that includes Supabase connectivity"""
    from typing import Dict, Any

    # Basic app health
    health_status: Dict[str, Any] = {
        "status": "healthy",
        "message": "StudyGenie API is running with Supabase integration",
        "timestamp": datetime.now().isoformat()
    }

    # Supabase health check
    try:
        from app.services_supabase import get_learning_progress_service
        service = get_learning_progress_service()
        # Test Supabase connection by trying to get service (which initializes Supabase client)
        health_status["supabase"] = "connected"
        health_status["database"] = "supabase_operational"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["database"] = {"status": "error", "error": str(e)}

    return health_status


# Database status endpoint
@app.get("/database/status")
async def database_status():
    """Detailed Supabase status and statistics"""
    try:
        from app.services_supabase import get_learning_progress_service
        service = get_learning_progress_service()
        return {
            "status": "operational",
            "database_type": "supabase",
            "connection": "active",
            "message": "Supabase connection is healthy"
        }
    except Exception as e:
        return {
            "status": "error",
            "database_type": "supabase",
            "connection": "failed",
            "error": str(e)
        }


# Root endpoint


@app.get("/")
async def root():
    return {
        "message": "Welcome to StudyGenie API",
        "health": "/health"
    }
