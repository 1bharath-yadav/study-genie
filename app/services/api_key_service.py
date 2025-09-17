"""
Functional API Key service
Pure functional operations for managing user API keys
"""

#  app/services/api_key_service.py  

from typing import Optional, Dict, Any, List
from .student_service import (
    get_api_keys_by_user,
    get_active_api_key_by_provider,
    create_api_key,
    update_api_key,
    delete_api_key,
    get_provider_id_by_name
)
from app.services.db import create_supabase_client, safe_extract_single
from ..core.encryption import decrypt_api_key
import logging

logger = logging.getLogger(__name__)


async def get_api_key_for_provider(student_id: str, provider_name: str) -> Optional[str]:
    """Get decrypted API key for a user and provider."""
    try:
        # Get the provider ID first
        provider_id = get_provider_id_by_name(provider_name)
        # If get_provider_id_by_name falls back to returning the provider_name
        # string (a design decision in student_service), treat that as "no
        # canonical provider id" and fall back to searching by provider_name.
        use_provider_id_lookup = bool(provider_id) and (provider_id != provider_name)

        key_data = None
        if use_provider_id_lookup:
            # Try provider_id-based lookup first (preferred when an actual id exists)
            # Ensure provider_id is a string (student_service may return None on error)
            pid = str(provider_id) if provider_id is not None else None
            if pid:
                key_data = get_active_api_key_by_provider(student_id, pid)
            else:
                key_data = None
        else:
            keys = get_api_keys_by_user(student_id)
            logger.debug("Checking %d stored API keys for student=%s provider=%s", len(keys), student_id, provider_name)
            for k in keys:
                if k.get("provider_name") == provider_name and k.get("is_active", True):
                    key_data = k
                    break
            # If still not found, try a direct DB lookup by provider_name (handles provider_id=null)
            if not key_data:
                try:
                    client = create_supabase_client()
                    if client:
                        resp = client.table("user_api_keys").select("*").eq("student_id", student_id).eq("provider_name", provider_name).eq("is_active", True).execute()
                        key_data = safe_extract_single(resp)
                except Exception:
                    key_data = None
        
        if not key_data:
            # Nothing found using either lookup method
            logger.debug("No active API key found for student=%s provider=%s (provider_id_lookup=%s)", student_id, provider_name, use_provider_id_lookup)
            return None
        
        # Decrypt the API key (do not log secret values)
        encrypted_key = key_data.get("encrypted_api_key")
        if not encrypted_key:
            logger.debug("Found API key metadata but encrypted_api_key missing for key id=%s", key_data.get("id"))
            return None
        logger.debug("Using API key id=%s for student=%s provider=%s", key_data.get("id"), student_id, provider_name)
        return decrypt_api_key(encrypted_key)
    
    except Exception:
        logger.exception("Error resolving API key for student=%s provider=%s", student_id, provider_name)
        return None


async def store_api_key(student_id: str, provider_name: str, api_key: str, key_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Store an encrypted API key for a user."""
    from ..core.encryption import encrypt_api_key
    
    # Encrypt the API key
    encrypted_key = encrypt_api_key(api_key)
    
    # Create the API key record
    key_data = {
        "student_id": student_id,
        "provider_name": provider_name,
        "key_name": key_name or f"{provider_name} API Key",
        "encrypted_api_key": encrypted_key,
        "is_active": True
    }
    
    return create_api_key(key_data)


async def get_user_api_keys(student_id: str) -> List[Dict[str, Any]]:
    """Get all API keys for a user (without decrypting them)."""
    keys = get_api_keys_by_user(student_id)
    logger.debug("Retrieved %d API keys for student=%s", len(keys), student_id)
    return keys


async def update_user_api_key(key_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update an API key record."""
    if "api_key" in updates:
        # If updating the actual key, encrypt it
        from ..core.encryption import encrypt_api_key
        updates["encrypted_api_key"] = encrypt_api_key(updates["api_key"])
        del updates["api_key"]
    
    return update_api_key(key_id, updates)


async def delete_user_api_key(key_id: str) -> bool:
    """Delete an API key."""
    return delete_api_key(key_id)


async def get_providers_with_keys(student_id: str) -> List[str]:
    """Get list of provider names that the user has API keys for."""
    keys = get_api_keys_by_user(student_id)
    return [key["provider_name"] for key in keys if key.get("is_active", True)]




def deactivate_api_key_by_id(key_id: str) -> bool:
    """Deactivate an API key by setting is_active to False."""
    from .student_service import update_api_key
    result = update_api_key(key_id, {"is_active": False})
    logger.debug("Deactivated api key id=%s result=%s", key_id, bool(result))
    return result is not None


def activate_api_key_for_user(student_id: str, key_id: str) -> Optional[Dict[str, Any]]:
    """Activate the given API key for the student and deactivate other keys for the same provider.

    Returns the updated key record or None on failure.
    """
    # Get the key to activate
    keys = get_api_keys_by_user(student_id)
    logger.debug("activate_api_key_for_user called for student=%s key_id=%s (found %d keys)", student_id, key_id, len(keys))
    target = None
    for k in keys:
        if k.get("id") == key_id:
            target = k
            break
    if not target:
        logger.debug("activate_api_key_for_user: target key not found for student=%s key_id=%s", student_id, key_id)
        return None

    provider = target.get("provider_name")
    logger.debug("Activating key id=%s for student=%s provider=%s", key_id, student_id, provider)
    # Deactivate other keys for this student+provider
    for k in keys:
        kid = k.get("id")
        if kid and kid != key_id and k.get("provider_name") == provider and k.get("is_active", False):
            logger.debug("Deactivating key id=%s for student=%s provider=%s", kid, student_id, provider)
            update_api_key(kid, {"is_active": False})

    # Activate target key
    updated = update_api_key(key_id, {"is_active": True})
    logger.debug("Activation result for key id=%s: %s", key_id, bool(updated))
    return updated


def check_provider_api_key_exists(student_id: str, provider_id: str) -> bool:
    """Check if user has an active API key for a provider."""
    key_data = get_active_api_key_by_provider(student_id, provider_id)
    return key_data is not None
