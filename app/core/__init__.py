from fastapi import FastAPI
from app.db.models import DatabaseManager
from contextlib import asynccontextmanager

app = FastAPI()

# Global database manager instance
db_manager: DatabaseManager = None

def get_db_manager() -> DatabaseManager:
    return DatabaseManager()

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    global db_manager
    db_manager = get_db_manager()
    await db_manager.initialize_pool()
    await db_manager.create_tables()
    print("Database initialized and tables created.")
    yield
    await db_manager.close_pool()
    print("Database pool closed.")

app.router.lifespan_context = lifespan