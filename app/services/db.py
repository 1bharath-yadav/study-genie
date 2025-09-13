"""
Pure functional database operations
"""
# app/services/db.py
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)


def create_supabase_client() -> Optional[Client]:
    """Create and return Supabase client."""
    try:
        if not settings.SUPABASE_URL or not settings.SUPABASE_API_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_API_KEY must be set")
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None


async def test_connection() -> bool:
    """Test Supabase connection."""
    try:
        client = create_supabase_client()
        if client:
            # Try a simple query to test connection
            client.table("students").select("*").limit(1).execute()
            return True
        return False
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


def safe_extract_data(response) -> List[Dict[str, Any]]:
    """Safely extract data from Supabase response."""
    if hasattr(response, 'data') and response.data is not None:
        return response.data if isinstance(response.data, list) else [response.data]
    return []


def safe_extract_single(response) -> Optional[Dict[str, Any]]:
    """Safely extract single item from Supabase response."""
    data = safe_extract_data(response)
    return data[0] if data else None

# Student operations (remove student_id references)
def create_student(student_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("students").insert(student_data).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to create student: {str(e)}")
        return None

def get_student_by_id(student_id: str) -> Optional[Dict[str, Any]]:
    try:
        client = create_supabase_client()
        if not client:
            return None
        # students table uses `student_id` as primary key
        response = client.table("students").select("*").eq("student_id", student_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get student: {str(e)}")
        return None

def update_student(student_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        client = create_supabase_client()
        if not client:
            return None
        update_data['updated_at'] = datetime.now().isoformat()
        response = client.table("students").update(update_data).eq("student_id", student_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to update student: {str(e)}")
        return None

def delete_student(student_id: str) -> bool:
    try:
        client = create_supabase_client()
        if not client:
            return False
        response = client.table("students").delete().eq("student_id", student_id).execute()
        return len(safe_extract_data(response)) > 0
    except Exception as e:
        logger.error(f"Failed to delete student: {str(e)}")
        return False

