# db.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Database URL from .env file
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in your .env file")

# ---------------------------------------------------------------------
# 1. Create Async Engine
# ---------------------------------------------------------------------
# Using NullPool so each session has isolated connections (better for async + FastAPI)
# For heavy concurrency, we can switch to asyncpg connection pooling.
engine = create_async_engine(
    DATABASE_URL,
    echo=False,            # Set True for SQL debug logging
    poolclass=NullPool,    # Each session = independent connection
    future=True
)

# ---------------------------------------------------------------------
# 2. Create Session Maker
# ---------------------------------------------------------------------
# Expire on commit = False → objects remain usable after commit.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------
# 3. Declarative Base
# ---------------------------------------------------------------------
Base = declarative_base()

# ---------------------------------------------------------------------
# 4. Dependency Injection Helper (for FastAPI routes)
# ---------------------------------------------------------------------


async def get_db():
    """Yield an async database session for dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# ---------------------------------------------------------------------
# 5. Initialize Database (Create Tables)
# ---------------------------------------------------------------------


async def init_db():
    # from models import Base  # Import here to avoid circular imports
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database initialized successfully.")

# ---------------------------------------------------------------------
# 6. Health Check Utility
# ---------------------------------------------------------------------


async def test_connection():
    """
    Simple async DB connection test.
    """
    try:
        async with engine.connect() as conn:
            from sqlalchemy import text
            result = await conn.execute(text("SELECT 1"))
            print("✅ DB Connection OK:", result.scalar())
    except Exception as e:
        print("❌ DB Connection Failed:", e)


# ---------------------------------------------------------------------
# Re-exports for compatibility with the rest of the application
# Main app expects `DatabaseManager` and `initialize_database` to be
# available from `app.db.db` (see app/main.py). The real implementations
# live in `app.db.models`, so import and re-export them here.
# ---------------------------------------------------------------------
from .models import DatabaseManager, initialize_database  # noqa: E402,F401
