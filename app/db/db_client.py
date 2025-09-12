"""
Pure Functional Supabase Client
"""
from supabase import create_client, Client
from app.config import settings


def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_API_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_API_KEY must be set")
    
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_API_KEY)


async def get_supabase_async_client():
    """Get async Supabase client - currently returns sync client as supabase-py doesn't have true async client"""
    return get_supabase_client()
