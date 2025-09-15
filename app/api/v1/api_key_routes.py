from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.core.security import get_current_user
from app.services import api_key_service

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


@router.get("/")
async def list_api_keys(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List API keys for the current user."""
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    keys = await api_key_service.get_user_api_keys(student_id)
    return keys


@router.post("/")
async def create_api_key(payload: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_user)):
    """Create/store a new API key for the current user.

    Expected payload: { provider_id: str, api_key: str }
    """
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    provider = payload.get("provider_id") or payload.get("provider")
    api_key = payload.get("api_key")
    if not provider or not api_key:
        raise HTTPException(status_code=400, detail="provider_id and api_key are required")

    stored = await api_key_service.store_api_key(student_id, provider, api_key)
    if not stored:
        raise HTTPException(status_code=500, detail="Failed to store API key")
    return stored


@router.delete("/{key_id}")
async def delete_api_key(key_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete an API key by id."""
    ok = await api_key_service.delete_user_api_key(key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found or could not be deleted")
    return {"success": True}


@router.get("/providers/{provider_name}/status")
async def provider_status(provider_name: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return whether the current user has an active API key for the provider."""
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    providers = await api_key_service.get_providers_with_keys(student_id)
    has = provider_name in providers
    return {"provider": provider_name, "has_api_key": has}


@router.post("/{key_id}/active")
async def activate_api_key(key_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Set the specified API key active for the current user and deactivate other keys for the same provider."""
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    updated = api_key_service.activate_api_key_for_user(student_id, key_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Key not found or could not be activated")
    return updated
