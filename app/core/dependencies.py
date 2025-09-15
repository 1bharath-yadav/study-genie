"""
Functional Dependencies Module
Pure functions and dependency injection helpers
"""
from typing import Optional

from app.db.db_client import get_supabase_client
from app.services.db import create_supabase_client
from app.core.auth_service import get_auth_service_instance
from supabase import Client


def get_db() -> Client:
    """Get Supabase client for dependency injection"""
    return get_supabase_client()


def create_db() -> Optional[Client]:
    """Get functional Supabase client with error handling"""
    return create_supabase_client()


async def get_auth_service():
    """Get functional auth service instance"""
    return await get_auth_service_instance()


# Utility dependencies for common operations
async def get_db_with_auth_service():
    """Get both database client and auth service"""
    return {
        "db": get_db(),
        "auth": await get_auth_service()
    }


# Export commonly used dependencies
__all__ = [
    "get_db",
    "get_db", 
    "get_auth_service",
    "get_db_with_auth_service"
]