from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user
from app.llm.providers import get_provider_list, get_models_for_provider
from typing import Dict, Any
from app.services import model_preference_service

router = APIRouter(prefix="/providers", tags=["providers"])

@router.get("/")
async def list_providers(current_user: dict = Depends(get_current_user)):
    """Return the list of available providers from PROVIDERS_JSON."""
    return get_provider_list()

@router.get("/{provider_name}/models")
async def models_by_provider(provider_name: str, current_user: dict = Depends(get_current_user)):
    """Return models for a specific provider."""
    models = get_models_for_provider(provider_name)
    # If user available, annotate models with user's saved preferences
    student_id = None
    if current_user:
        student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        return models

    prefs = model_preference_service.list_model_preferences(student_id)
    # Build lookup maps for chat/embedding active flags
    chat_active = {p.get("model_id") for p in prefs if p and p.get("use_for_chat")}
    embed_active = {p.get("model_id") for p in prefs if p and p.get("use_for_embedding")}
    # Annotate models with per-use-case active flags
    for m in models:
        m_id = m.get("id")
        m["is_active_chat"] = m_id in chat_active
        m["is_active_embedding"] = m_id in embed_active
    return models


@router.post("/models/{model_id}/active")
async def activate_model(model_id: str, use_case: str = 'chat', current_user: Dict[str, Any] = Depends(get_current_user)):
    """Activate (persist) a model preference for the current user."""
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # model_id is formatted as provider-model_name (e.g., google-gemini-2.0-flash)
    parts = model_id.split('-', 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid model_id format")
    provider_name = parts[0]
    pref = model_preference_service.activate_model_for_user(student_id, model_id, provider_name, use_case)
    if not pref:
        raise HTTPException(status_code=500, detail="Failed to activate model")
    return pref


@router.delete("/models/{model_id}/active")
async def deactivate_model(model_id: str, use_case: str = 'chat', current_user: Dict[str, Any] = Depends(get_current_user)):
    """Deactivate (remove) a model preference for the current user."""
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    ok = model_preference_service.deactivate_model_for_user(student_id, model_id, use_case)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to deactivate model")
    return {"success": True}

@router.get("/models/{model_type}")
async def models_by_type(model_type: str, current_user: dict = Depends(get_current_user)):
    """Return models for a given type across providers. For now this delegates to available models endpoint in models router."""
    # Keep simple: reuse the models endpoints under /models/available/* which are handled elsewhere
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Use /models/available/{type} instead")
