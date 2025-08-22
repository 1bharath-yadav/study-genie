import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .db.db import DatabaseManager, initialize_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("progress_tracker_api")

# Global database manager used by the app
db_manager: Optional[DatabaseManager] = None


app = FastAPI(
    title="Study-genie",
    description="Interactive RAG-powered AI tutor",
    version="1.0.0",
)


@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    global db_manager
    try:
        db_manager = await initialize_database()
        logger.info("Database initialized successfully")
        yield
    finally:
        if db_manager:
            await db_manager.close_pool()
            logger.info("Database connections closed")


app.router.lifespan_context = lifespan  # simple lifespan binding for tests

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_db_manager() -> DatabaseManager:
    if db_manager is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db_manager
