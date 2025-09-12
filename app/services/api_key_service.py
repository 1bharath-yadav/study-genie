"""
Functional API Key service
Pure functional operations for managing user API keys
"""
from typing import Optional, Dict, Any, List
from .db import (
    get_api_keys_by_user_id,
    get_active_api_key_by_provider,
    create_api_key,
    update_api_key,
    delete_api_key
)
from ..core.encryption import decrypt_api_key


async def get_api_key_for_provider(user_id: str, provider_name: str) -> Optional[str]:
    """Get decrypted API key for a user and provider."""
    try:
        # Get the active API key for this provider
        key_data = get_active_api_key_by_provider(user_id, provider_name)
        
        if not key_data:
            return None
        
        # Decrypt the API key
        encrypted_key = key_data.get("encrypted_api_key")
        if not encrypted_key:
            return None
        
        return decrypt_api_key(encrypted_key)
    
    except Exception:
        return None


async def store_api_key(user_id: str, provider_name: str, api_key: str, key_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Store an encrypted API key for a user."""
    from ..core.encryption import encrypt_api_key
    
    # Encrypt the API key
    encrypted_key = encrypt_api_key(api_key)
    
    # Create the API key record
    key_data = {
        "user_id": user_id,
        "provider_name": provider_name,
        "key_name": key_name or f"{provider_name} API Key",
        "encrypted_api_key": encrypted_key,
        "is_active": True
    }
    
    return create_api_key(key_data)


async def get_user_api_keys(user_id: str) -> List[Dict[str, Any]]:
    """Get all API keys for a user (without decrypting them)."""
    return get_api_keys_by_user_id(user_id)


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


async def get_providers_with_keys(user_id: str) -> List[str]:
    """Get list of provider names that the user has API keys for."""
    keys = get_api_keys_by_user_id(user_id)
    return [key["provider_name"] for key in keys if key.get("is_active", True)]


# Additional functions needed by routes
def create_new_api_key(api_key_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create API key - delegates to db."""
    from .db import create_api_key as db_create_api_key
    return db_create_api_key(api_key_data)


def deactivate_api_key_by_id(key_id: str) -> bool:
    """Deactivate an API key by setting is_active to False."""
    from .db import update_api_key
    result = update_api_key(key_id, {"is_active": False})
    return result is not None


def check_provider_api_key_exists(user_id: str, provider_id: str) -> bool:
    """Check if user has an active API key for a provider."""
    key_data = get_active_api_key_by_provider(user_id, provider_id)
    return key_data is not None
