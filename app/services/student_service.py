# app/services/student_service.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from app.models import StudentData, StudentUpdate
from app.services.db import (
    create_supabase_client,
    get_student_by_id as db_get_student_by_id,
    safe_extract_data,
    safe_extract_single,
    update_student as db_update_student,
    delete_student as db_delete_student,
)
import logging

logger = logging.getLogger(__name__)

def _normalize_learning_preferences(raw: Any) -> list:
    """Normalize the stored learning_preferences JSONB to a list for the frontend.

    Accepts lists, dicts, or null and returns a list of strings (or empty list).
    """
    try:
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            # Prefer explicit 'preferences' or values; otherwise use keys
            if "preferences" in raw and isinstance(raw["preferences"], list):
                return raw["preferences"]
            # Convert dict values to list of strings
            return [str(v) for v in raw.values()]
        # Fallback: try parsing if it's a JSON string
        import json

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return _normalize_learning_preferences(parsed)
            except Exception:
                return [raw]
        return []
    except Exception:
        return []


def get_student_by_id(student_id: str) -> Optional[StudentData]:
    try:
        result = db_get_student_by_id(student_id)
        if not result:
            return None

        # Map DB columns to API model fields expected by frontend
        mapped = {
            "student_id": result.get("student_id") or result.get("id"),
            "username": result.get("username"),
            "email": result.get("email"),
            "full_name": result.get("full_name") or result.get("username"),
            "grade_level": result.get("grade_level"),
            "bio": result.get("bio"),
            "learning_preferences": result.get("learning_preferences") or {},
            "created_at": result.get("created_at"),
            "updated_at": result.get("updated_at"),
        }
        return StudentData(**mapped)
    except Exception as e:
        logger.error(f"Failed to get student {student_id}: {str(e)}")
        return None

def update_student_data(student_id: str, student_data: StudentUpdate) -> Optional[StudentData]:
    try:
        update_data: Dict[str, Any] = {}
        # Accept both frontend and backend keys for compatibility
        # frontend may send 'name' or 'full_name'
        if getattr(student_data, "full_name", None) is not None:
            update_data["full_name"] = student_data.full_name
        if getattr(student_data, "grade_level", None) is not None:
            update_data["grade_level"] = student_data.grade_level
        if getattr(student_data, "bio", None) is not None:
            update_data["bio"] = student_data.bio
        if getattr(student_data, "learning_preferences", None) is not None:
            update_data["learning_preferences"] = student_data.learning_preferences

        if not update_data:
            return get_student_by_id(student_id)

        result = db_update_student(student_id, update_data)
        if not result:
            return None

        # Map result back to API model
        mapped = {
            "student_id": result.get("student_id") or result.get("id"),
            "username": result.get("username"),
            "email": result.get("email"),
            "full_name": result.get("full_name") or result.get("username"),
            "grade_level": result.get("grade_level"),
            "bio": result.get("bio"),
            "learning_preferences": result.get("learning_preferences") or {},
            "created_at": result.get("created_at"),
            "updated_at": result.get("updated_at"),
        }
        return StudentData(**mapped)
    except Exception as e:
        logger.error(f"Failed to update student {student_id}: {str(e)}")
        return None

def delete_student_by_id(student_id: str) -> bool:
    try:
        return db_delete_student(student_id)
    except Exception as e:
        logger.error(f"Failed to delete student {student_id}: {str(e)}")
        return False


###  api key management 


# API Key operations
def create_api_key(api_key_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new API key."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        api_key_data['created_at'] = datetime.now().isoformat()
        api_key_data['updated_at'] = datetime.now().isoformat()
        response = client.table("user_api_keys").insert(api_key_data).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to create API key: {str(e)}")
        return None


def get_api_keys_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Get all API keys for a user with provider information."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        # Simple direct select from user_api_keys. Some deployments do not have
        # an `llm_providers` table; avoid joins against a missing table.
        response = (
            client.table("user_api_keys")
            .select("*")
            .eq("student_id", user_id)
            .eq("is_active", True)
            .execute()
        )
        return safe_extract_data(response)
    except Exception as e:
        logger.error(f"Failed to get API keys by user_id: {str(e)}")
        return []


def get_active_api_key_by_provider(user_id: str, provider_id: str) -> Optional[Dict[str, Any]]:
    """Get active API key for a specific provider and user."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = (
            client.table("user_api_keys")
            .select("*")
            .eq("student_id", user_id)
            .eq("provider_id", provider_id)
            .eq("is_active", True)
            .execute()
        )
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get active API key: {str(e)}")
        return None


def set_default_api_key(user_id: str, api_key_id: str) -> bool:
    """Set an API key as default and unset others."""
    try:
        client = create_supabase_client()
        if not client:
            return False
        
        # First, unset all default flags for this user
        client.table("user_api_keys").update({
            "is_default": False,
            "updated_at": datetime.now().isoformat()
        }).eq("student_id", user_id).execute()
        
        # Then set the specified key as default
        response = (
            client.table("user_api_keys")
            .update({
                "is_default": True,
                "updated_at": datetime.now().isoformat()
            })
            .eq("id", api_key_id)
            .eq("student_id", user_id)
            .execute()
        )
        return len(safe_extract_data(response)) > 0
    except Exception as e:
        logger.error(f"Failed to set default API key: {str(e)}")
        return False


def deactivate_api_key(api_key_id: str) -> bool:
    """Deactivate an API key."""
    try:
        client = create_supabase_client()
        if not client:
            return False
        response = (
            client.table("user_api_keys")
            .update({"is_active": False, "updated_at": datetime.now().isoformat()})
            .eq("id", api_key_id)
            .execute()
        )
        return len(safe_extract_data(response)) > 0
    except Exception as e:
        logger.error(f"Failed to deactivate API key: {str(e)}")
        return False


def get_api_keys_by_user(student_id: str) -> List[Dict[str, Any]]:
    """Get all API keys for a user."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        response = client.table("user_api_keys").select("*").eq("student_id", student_id).execute()
        return safe_extract_data(response)
    except Exception as e:
        logger.error(f"Failed to get API keys: {str(e)}")
        return []


def update_api_key(api_key_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update an API key."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        update_data['updated_at'] = datetime.now().isoformat()
        response = (
            client.table("user_api_keys")
            .update(update_data)
            .eq("id", api_key_id)
            .execute()
        )
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to update API key: {str(e)}")
        return None


def delete_api_key(api_key_id: str) -> bool:
    """Delete an API key."""
    try:
        client = create_supabase_client()
        if not client:
            return False
        response = (
            client.table("user_api_keys")
            .delete()
            .eq("id", api_key_id)
            .execute()
        )
        return len(safe_extract_data(response)) > 0
    except Exception as e:
        logger.error(f"Failed to delete API key: {str(e)}")
        return False


def get_provider_id_by_name(provider_name: str) -> Optional[str]:
    """Get provider ID by name."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        # Query the user's stored API keys for a provider_id matching provider_name.
        # The older `llm_providers` table is not relied upon; we store provider
        # identifiers in `user_api_keys` per the new schema.
        try:
            resp = (
                client.table("user_api_keys")
                .select("provider_id")
                .eq("provider_name", provider_name)
                .eq("is_active", True)
                .limit(1)
                .execute()
            )
            provider_row = safe_extract_single(resp)
            if provider_row and provider_row.get("provider_id"):
                return provider_row.get("provider_id")
        except Exception as e:
            logger.debug("user_api_keys lookup for provider_id failed: %s", e)

        # As a last resort return provider_name itself so other code paths that
        # accept provider identifiers can still proceed and use provider_name.
        return provider_name
    except Exception as e:
        logger.error(f"Failed to get provider ID for {provider_name}: {str(e)}")
        return None


# Type definitions
UserData = Dict[str, Any]
StudentInfo = Dict[str, Any]
APIKeyInfo = Optional[Dict[str, Any]]
SigninResponse = Dict[str, Any]
# Pure functions for data validation
def validate_student_data(student_data: UserData) -> str:
    """Validate user data and extract student ID"""
    student_id = student_data.get("sub")
    if not student_id:
        raise ValueError("User ID (sub) is required")
    return str(student_id)

def extract_student_fields(student_data: UserData) -> Dict[str, str]:
    """Extract and validate student fields from user data"""
    return {
        "student_id": str(student_data.get("sub", "")),
        "username": student_data.get("email", "").split("@")[0],
        "email": student_data.get("email", ""),
        "full_name": student_data.get("name", ""),
        "grade_level": student_data.get("grade_level", "unknown"),
        "bio": student_data.get("bio", "")
    }

def create_update_data(student_data: UserData, existing_student: StudentInfo) -> Dict[str, Any]:
    """Create update data for existing student"""
    return {
        "email": student_data.get("email", existing_student.get("email", "")),
        "full_name": student_data.get("name", existing_student.get("full_name", "")),
        "updated_at": datetime.now().isoformat()
    }

def create_api_key_status(api_key_info: APIKeyInfo) -> Dict[str, Any]:
    """Create API key status information"""
    return {
        "has_api_key": api_key_info is not None,
        "service": api_key_info.get("service") if api_key_info else None,
        "created_at": api_key_info.get("created_at") if api_key_info else None,
        "updated_at": api_key_info.get("updated_at") if api_key_info else None
    }

def create_signin_response(
    student_info: StudentInfo,
    api_key_info: APIKeyInfo
) -> SigninResponse:
    """Create complete signin response"""
    return {
        "user": student_info,
        "api_key_status": create_api_key_status(api_key_info),
        "signin_timestamp": datetime.now().isoformat()
    }

# Database operations
async def get_existing_student(student_id: str) -> Optional[StudentInfo]:
    """Get existing student from database"""
    try:
        client = create_supabase_client()
        if not client:
            return None
            
        response = client.table("students").select("*").eq("student_id", student_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Error getting student {student_id}: {e}")
        return None

async def create_new_student(student_data: Dict[str, str]) -> StudentInfo:
    """Create new student in database"""
    client = create_supabase_client()
    if not client:
        raise ValueError("Database connection failed")
        
    response = client.table("students").insert(student_data).execute()
    result = safe_extract_single(response)
    if not result:
        raise ValueError("Failed to create student")
    return result

async def update_existing_student(student_id: str, update_data: Dict[str, Any]) -> StudentInfo:
    """Update existing student in database"""
    client = create_supabase_client()
    if not client:
        raise ValueError("Database connection failed")
        
    response = client.table("students").update(update_data).eq("student_id", student_id).execute()
    result = safe_extract_single(response)
    if not result:
        raise ValueError("Failed to update student")
    return result

async def get_api_key_info(student_id: str) -> APIKeyInfo:
    """Get API key information without the actual key"""
    try:
        client = create_supabase_client()
        if not client:
            return None
            
        # Get the user's API keys
        api_keys_response = client.table("user_api_keys").select("*").eq("student_id", student_id).eq("is_active", True).execute()
        api_keys = safe_extract_data(api_keys_response)
        
        if not api_keys:
            return None
            
        # Return basic info about the first active API key
        first_key = api_keys[0]
        return {
            "has_api_key": True,
            "service": "llm",  # Generic service indicator
            "created_at": first_key.get("created_at"),
            "updated_at": first_key.get("updated_at")
        }
    except Exception as e:
        logger.error(f"Error getting API key info for {student_id}: {e}")
        return None

async def get_encrypted_api_key(student_id: str) -> Optional[str]:
    """Get encrypted API key from database"""
    try:
        client = create_supabase_client()
        if not client:
            return None
            
        response = client.table("user_api_keys").select("encrypted_api_key").eq("student_id", student_id).execute()
        result = safe_extract_single(response)
        return result.get("encrypted_api_key") if result else None
    except Exception as e:
        logger.error(f"Error getting encrypted API key for {student_id}: {e}")
        return None











