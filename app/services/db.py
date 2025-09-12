"""
Pure functional database operations
"""
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


# Student operations
def create_student(student_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new student."""
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
    """Get student by ID."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("students").select("*").eq("id", student_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get student: {str(e)}")
        return None


def get_student_by_user_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get student by user ID."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("students").select("*").eq("user_id", user_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get student by user_id: {str(e)}")
        return None


def get_students_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Get all students by user ID."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        response = client.table("students").select("*").eq("user_id", user_id).execute()
        return safe_extract_data(response)
    except Exception as e:
        logger.error(f"Failed to get students by user_id: {str(e)}")
        return []


def update_student(student_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update student data."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        update_data['updated_at'] = datetime.now().isoformat()
        response = client.table("students").update(update_data).eq("id", student_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to update student: {str(e)}")
        return None


def delete_student(student_id: str) -> bool:
    """Delete a student."""
    try:
        client = create_supabase_client()
        if not client:
            return False
        response = client.table("students").delete().eq("id", student_id).execute()
        return len(safe_extract_data(response)) > 0
    except Exception as e:
        logger.error(f"Failed to delete student: {str(e)}")
        return False


# User operations
def create_user(user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new user."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("users").insert(user_data).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}")
        return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("users").select("*").eq("id", user_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get user: {str(e)}")
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("users").select("*").eq("email", email).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get user by email: {str(e)}")
        return None


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
        
        # Join with providers table to get provider information
        response = (
            client.table("user_api_keys")
            .select("""
                *,
                llm_providers!inner(
                    id,
                    name,
                    display_name,
                    is_active
                )
            """)
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


def get_api_keys_by_user(user_id: str) -> List[Dict[str, Any]]:
    """Get all API keys for a user."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        response = client.table("user_api_keys").select("*").eq("user_id", user_id).execute()
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


# Provider operations
def get_available_providers() -> List[Dict[str, Any]]:
    """Get all available LLM providers (alias for get_active_providers)."""
    return get_active_providers()


def get_active_providers() -> List[Dict[str, Any]]:
    """Get all active LLM providers."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        response = client.table("llm_providers").select("*").eq("is_active", True).execute()
        return safe_extract_data(response)
    except Exception as e:
        logger.error(f"Failed to get providers: {str(e)}")
        return []


def get_provider_by_name(provider_name: str) -> Optional[Dict[str, Any]]:
    """Get provider by name."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("llm_providers").select("*").eq("name", provider_name).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get provider by name: {str(e)}")
        return None


def get_models_by_provider(provider_id: str) -> List[Dict[str, Any]]:
    """Get all active models for a provider."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        response = (
            client.table("available_models")
            .select("*")
            .eq("provider_id", provider_id)
            .eq("is_active", True)
            .order("display_name")
            .execute()
        )
        return safe_extract_data(response)
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        return []


def get_models_by_type(model_type: str) -> List[Dict[str, Any]]:
    """Get all active models by type (chat, embeddings, etc.)."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        response = (
            client.table("available_models")
            .select("""
                *,
                llm_providers!inner(
                    id,
                    name,
                    display_name
                )
            """)
            .eq("model_type", model_type)
            .eq("is_active", True)
            .order("display_name")
            .execute()
        )
        return safe_extract_data(response)
    except Exception as e:
        logger.error(f"Failed to get models by type: {str(e)}")
        return []


# User Model Preferences operations
def create_user_model_preference(preference_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create user model preference."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        preference_data['created_at'] = datetime.now().isoformat()
        preference_data['updated_at'] = datetime.now().isoformat()
        response = client.table("user_model_preferences").insert(preference_data).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to create model preference: {str(e)}")
        return None


def get_user_model_preferences(user_id: str) -> List[Dict[str, Any]]:
    """Get user's model preferences with model and provider details."""
    try:
        client = create_supabase_client()
        if not client:
            return []
        response = (
            client.table("user_model_preferences")
            .select("""
                *,
                available_models!inner(
                    id,
                    model_name,
                    display_name,
                    model_type,
                    llm_providers!inner(
                        name,
                        display_name
                    )
                )
            """)
            .eq("student_id", user_id)
            .execute()
        )
        return safe_extract_data(response)
    except Exception as e:
        logger.error(f"Failed to get user model preferences: {str(e)}")
        return []


def update_user_model_preference(preference_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update user model preference."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        update_data['updated_at'] = datetime.now().isoformat()
        response = (
            client.table("user_model_preferences")
            .update(update_data)
            .eq("id", preference_id)
            .execute()
        )
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to update model preference: {str(e)}")
        return None


def set_default_model_for_use_case(user_id: str, model_id: str, use_case: str) -> bool:
    """Set a model as default for a specific use case (chat/embedding)."""
    try:
        client = create_supabase_client()
        if not client:
            return False
        
        # First, unset all defaults for this use case
        if use_case == "chat":
            client.table("user_model_preferences").update({
                "use_for_chat": False,
                "updated_at": datetime.now().isoformat()
            }).eq("student_id", user_id).execute()
        elif use_case == "embedding":
            client.table("user_model_preferences").update({
                "use_for_embedding": False,
                "updated_at": datetime.now().isoformat()
            }).eq("student_id", user_id).execute()
        
        # Then set the specified model as default for this use case
        update_data: Dict[str, Any] = {"updated_at": datetime.now().isoformat()}
        if use_case == "chat":
            update_data["use_for_chat"] = True
        elif use_case == "embedding":
            update_data["use_for_embedding"] = True
            
        response = (
            client.table("user_model_preferences")
            .update(update_data)
            .eq("student_id", user_id)
            .eq("model_id", model_id)
            .execute()
        )
        return len(safe_extract_data(response)) > 0
    except Exception as e:
        logger.error(f"Failed to set default model: {str(e)}")
        return False


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model by ID."""
    try:
        client = create_supabase_client()
        if not client:
            return None
        response = client.table("available_models").select("*").eq("id", model_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Failed to get model: {str(e)}")
        return None
