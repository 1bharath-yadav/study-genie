# Pure functional API key routes
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models import APIKeyCreate, APIKeyResponse, LLMProvider
from app.services.api_key_service import (
    create_new_api_key,
    get_user_api_keys,
    deactivate_api_key_by_id,
    check_provider_api_key_exists,
)
from app.core.security import get_current_user

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


@router.post("", response_model=APIKeyResponse)
async def create_api_key_endpoint(
    api_key_data: APIKeyCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new API key for a provider."""
    import logging
    from app.services.db import get_provider_by_name
    from app.core.encryption import encrypt_api_key
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating API key - received data: {api_key_data}")
    logger.info(f"Current user: {current_user}")
    
    user_id = current_user.get("student_id") or current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    logger.info(f"Extracted user_id: {user_id}")
    
    # Map provider names to correct database names
    provider_name_map = {
        "gemini": "google",
        "google": "google", 
        "openai": "openai",
        "anthropic": "anthropic"
    }
    
    # Get the correct provider name
    provider_name = provider_name_map.get(api_key_data.provider_id, api_key_data.provider_id)
    logger.info(f"Mapped provider '{api_key_data.provider_id}' to '{provider_name}'")
    
    # Get provider UUID from database
    provider = get_provider_by_name(provider_name)
    if not provider:
        raise HTTPException(
            status_code=400, 
            detail=f"Provider '{provider_name}' not found"
        )
    
    provider_uuid = provider["id"]
    logger.info(f"Found provider UUID: {provider_uuid}")
    
    # Check if user already has an API key for this provider  
    if check_provider_api_key_exists(str(user_id), provider_uuid):
        raise HTTPException(
            status_code=400, 
            detail=f"API key for {provider_name} already exists"
        )
    
    # Encrypt the API key
    encrypted_key = encrypt_api_key(api_key_data.api_key)
    
    # Create API key data dict with correct field names for database
    key_data = {
        "student_id": str(user_id),        # Database expects student_id
        "provider_id": provider_uuid,      # Database expects provider_id as UUID
        "encrypted_api_key": encrypted_key,
        "is_active": True
    }
    
    logger.info(f"Creating API key with data: {key_data}")
    
    api_key = create_new_api_key(key_data)
    if not api_key:
        raise HTTPException(status_code=400, detail="Failed to create API key")
    
    # Add provider information to the response
    response_data = {
        "id": api_key["id"],
        "provider_id": api_key["provider_id"],
        "provider_name": provider["name"],
        "provider_display_name": provider["display_name"],
        "is_active": api_key["is_active"],
        "is_default": api_key["is_default"],
        "student_id": api_key["student_id"],
        "created_at": api_key["created_at"]
    }
    
    return response_data


@router.get("", response_model=List[APIKeyResponse])
async def get_user_api_keys_endpoint(
    current_user: dict = Depends(get_current_user)
):
    """Get all API keys for the current user."""
    user_id = current_user.get("student_id") or current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return await get_user_api_keys(str(user_id))


@router.delete("/{api_key_id}")
async def deactivate_api_key_endpoint(
    api_key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Deactivate an API key."""
    user_id = current_user.get("student_id") or current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    # First, get user's API keys to verify ownership
    user_api_keys = await get_user_api_keys(str(user_id))
    user_key_ids = [key.get("id") for key in user_api_keys]
    
    # Check if the API key belongs to the user
    if api_key_id not in user_key_ids:
        raise HTTPException(status_code=404, detail="API key not found or access denied")
    
    success = deactivate_api_key_by_id(api_key_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to deactivate API key")
    
    return {"message": "API key deactivated successfully"}


@router.get("/providers/{provider}/status")
async def check_provider_api_key_endpoint(
    provider: LLMProvider,
    current_user: dict = Depends(get_current_user)
):
    """Check if user has an active API key for a provider."""
    user_id = current_user.get("student_id") or current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    has_key = check_provider_api_key_exists(str(user_id), provider)
    
    return {
        "provider": provider.value,
        "has_api_key": has_key
    }
