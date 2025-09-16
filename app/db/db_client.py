"""
Pure Functional Supabase Client
"""
from supabase import create_client, Client
from app.config import settings


def get_supabase_client() -> Client:
    """Get Supabase client instance (synchronous)."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_API_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_API_KEY must be set")
    
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_API_KEY)


def get_supabase_async_client() -> Client:
    """
    Placeholder async client.
    supabase-py v2 doesn’t yet support true async I/O, so this returns
    the same sync Client for compatibility.
    """
    return get_supabase_client()
