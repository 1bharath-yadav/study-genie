"""
Service layer for persisting and querying user model preferences.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.services.db import create_supabase_client
from app.services.student_service import safe_extract_data, safe_extract_single
import logging

logger = logging.getLogger(__name__)


def create_model_preference(student_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        client = create_supabase_client()
        if not client:
            return None
        data = {
            "student_id": student_id,
            "model_id": payload.get("model_id"),
            "provider_name": payload.get("provider_name"),
            "use_for_chat": payload.get("use_for_chat", False),
            "use_for_embedding": payload.get("use_for_embedding", False),
            "is_default": payload.get("is_default", False),
            "metadata": payload.get("metadata", {}),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        resp = client.table("model_preferences").insert(data).execute()
        return safe_extract_single(resp)
    except Exception as e:
        logger.error(f"Failed to create model preference: {e}")
        return None


def list_model_preferences(student_id: str) -> List[Dict[str, Any]]:
    try:
        client = create_supabase_client()
        if not client:
            return []
        resp = client.table("model_preferences").select("*").eq("student_id", student_id).execute()
        return safe_extract_data(resp)
    except Exception as e:
        logger.error(f"Failed to list model preferences: {e}")
        return []


def get_model_preference(pref_id: str) -> Optional[Dict[str, Any]]:
    try:
        client = create_supabase_client()
        if not client:
            return None
        resp = client.table("model_preferences").select("*").eq("id", pref_id).execute()
        return safe_extract_single(resp)
    except Exception as e:
        logger.error(f"Failed to get model preference: {e}")
        return None


def update_model_preference(pref_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        client = create_supabase_client()
        if not client:
            return None
        updates["updated_at"] = datetime.now().isoformat()
        resp = client.table("model_preferences").update(updates).eq("id", pref_id).execute()
        return safe_extract_single(resp)
    except Exception as e:
        logger.error(f"Failed to update model preference: {e}")
        return None


def delete_model_preference(pref_id: str) -> bool:
    try:
        client = create_supabase_client()
        if not client:
            return False
        resp = client.table("model_preferences").delete().eq("id", pref_id).execute()
        return len(safe_extract_data(resp)) > 0
    except Exception as e:
        logger.error(f"Failed to delete model preference: {e}")
        return False


def activate_model_for_user(student_id: str, model_id: str, provider_name: str, use_case: str = 'chat') -> Optional[Dict[str, Any]]:
    """Create or update a model preference for a user and mark it active for a use case.

    use_case: 'chat' or 'embedding'. When activating, this will unset the flag for all
    other preferences for the same student so only one model is active per use case.
    """
    try:
        client = create_supabase_client()
        if not client:
            return None
        # Unset the use_case flag for all existing preferences for this user
        if use_case == 'chat':
            client.table("model_preferences").update({"use_for_chat": False}).eq("student_id", student_id).execute()
        else:
            client.table("model_preferences").update({"use_for_embedding": False}).eq("student_id", student_id).execute()

        # Check if preference exists for this model
        resp = client.table("model_preferences").select("*").eq("student_id", student_id).eq("model_id", model_id).execute()
        existing = safe_extract_single(resp)
        now = datetime.now().isoformat()
        data = {
            "student_id": student_id,
            "model_id": model_id,
            "provider_name": provider_name,
            "use_for_chat": True if use_case == 'chat' else False,
            "use_for_embedding": True if use_case == 'embedding' else False,
            "is_default": False,
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        }
        if existing:
            # update
            updates = {
                "use_for_chat": True if use_case == 'chat' else False,
                "use_for_embedding": True if use_case == 'embedding' else False,
                "updated_at": now,
                "provider_name": provider_name,
            }
            resp2 = client.table("model_preferences").update(updates).eq("id", existing.get("id")).execute()
            return safe_extract_single(resp2)
        else:
            resp2 = client.table("model_preferences").insert(data).execute()
            return safe_extract_single(resp2)
    except Exception as e:
        logger.error(f"Failed to activate model preference: {e}")
        return None


def deactivate_model_for_user(student_id: str, model_id: str, use_case: str = 'chat') -> bool:
    """Mark a model preference inactive for a given use_case.

    We do not delete the row; instead unset the corresponding use_for_* flag. This is
    idempotent and preserves history/metadata.
    """
    try:
        client = create_supabase_client()
        if not client:
            return False

        # Find existing preference row for this model
        resp = client.table("model_preferences").select("*").eq("student_id", student_id).eq("model_id", model_id).execute()
        existing = safe_extract_single(resp)
        if not existing:
            # Nothing to update; treat as success
            return True

        updates = {"updated_at": datetime.now().isoformat()}
        if use_case == 'chat':
            updates["use_for_chat"] = False
        else:
            updates["use_for_embedding"] = False

        resp2 = client.table("model_preferences").update(updates).eq("id", existing.get("id")).execute()
        return len(safe_extract_data(resp2)) > 0
    except Exception as e:
        logger.error(f"Failed to deactivate model preference: {e}")
        return False


def set_default_model_for_use_case(student_id: str, model_id: str, use_case: str) -> bool:
    """Set a model as default for a use case ('chat' or 'embedding')."""
    try:
        client = create_supabase_client()
        if not client:
            return False
        # unset existing defaults for this user and use_case
        if use_case == "chat":
            client.table("model_preferences").update({"is_default": False}).eq("student_id", student_id).execute()
            resp = client.table("model_preferences").update({"is_default": True, "updated_at": datetime.now().isoformat()}).eq("student_id", student_id).eq("model_id", model_id).execute()
        else:
            # embedding
            client.table("model_preferences").update({"is_default": False}).eq("student_id", student_id).execute()
            resp = client.table("model_preferences").update({"is_default": True, "updated_at": datetime.now().isoformat()}).eq("student_id", student_id).eq("model_id", model_id).execute()
        return len(safe_extract_data(resp)) > 0
    except Exception as e:
        logger.error(f"Failed to set default model: {e}")
        return False
