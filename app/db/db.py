# db.py
import os
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import InvalidRequestError
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Database URL from .env file
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in your .env file")


def _ensure_async_driver(database_url: str) -> str:
    """Ensure the URL requests an async driver supported by SQLAlchemy asyncio.

    For PostgreSQL we prefer `asyncpg` and will coerce common forms like:
      - `postgres://...` -> `postgresql+asyncpg://...`
      - `postgresql://...` -> `postgresql+asyncpg://...`

    If the URL already explicitly contains a +driver (e.g. `+psycopg2`) we won't
    silently overwrite it; instead we only add `+asyncpg` when no driver is
    specified. This keeps behavior predictable while helping devs avoid the
    common psycopg2 sync/async mismatch.
    """
    lower = database_url.lower()
    # If it's postgres and no explicit +driver present, add +asyncpg
    if (lower.startswith("postgres://") or lower.startswith("postgresql://")) and "+" not in database_url.split("://", 1)[0]:
        # prefer the full scheme name
        if lower.startswith("postgres://"):
            tail = database_url.split("://", 1)[1]
            return f"postgresql+asyncpg://{tail}"
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


# ---------------------------------------------------------------------
# 1. Create Async Engine
# ---------------------------------------------------------------------
# Using NullPool so each session has isolated connections (better for async + FastAPI)
# For heavy concurrency, we can switch to asyncpg connection pooling.
try:
    engine = create_async_engine(
        _ensure_async_driver(DATABASE_URL),
        echo=False,            # Set True for SQL debug logging
        poolclass=NullPool,    # Each session = independent connection
        future=True,
    )
except InvalidRequestError as exc:
    # This most commonly happens when a sync DB driver (eg. psycopg2) is
    # provided with SQLAlchemy's asyncio helpers. Provide a clearer, actionable
    # error message that points the developer to the fix.
    raise RuntimeError(
        "The asyncio extension requires an async DB driver (for Postgres use 'asyncpg'). "
        "Check your DATABASE_URL. Example: 'postgresql+asyncpg://user:pass@host:5432/dbname'. "
        "If you intended to use psycopg2 (sync) switch to a sync engine, or install asyncpg: 'pip install asyncpg'"
    ) from exc

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
