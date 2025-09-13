from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.core.security import get_current_user
from app.services import model_preference_service

router = APIRouter(prefix="/model-preferences", tags=["model-preferences"])


@router.get("/")
async def list_model_preferences(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return model preferences for the current user."""
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return model_preference_service.list_model_preferences(student_id)


@router.post("/")
async def create_model_preference(payload: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_user)):
    """Create a model preference.

    Expected payload: { model_id: str, provider_name?: str, use_for_chat?: bool, use_for_embedding?: bool }
    """
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not payload.get("model_id"):
        raise HTTPException(status_code=400, detail="model_id is required")
    pref = model_preference_service.create_model_preference(student_id, payload)
    if not pref:
        raise HTTPException(status_code=500, detail="Failed to create model preference")
    return pref


@router.put("/{pref_id}")
async def update_model_preference(pref_id: str, payload: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_user)):
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    updated = model_preference_service.update_model_preference(pref_id, payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Preference not found or update failed")
    return updated


@router.delete("/{pref_id}")
async def delete_model_preference(pref_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    ok = model_preference_service.delete_model_preference(pref_id)
    return {"success": ok}


@router.post("/default/{model_id}/{use_case}")
async def set_default_model(model_id: str, use_case: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Set a default model for a use case ('chat' or 'embedding')."""
    student_id = current_user.get("sub") or current_user.get("student_id")
    if not student_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if use_case not in ("chat", "embedding"):
        raise HTTPException(status_code=400, detail="Invalid use_case")
    ok = model_preference_service.set_default_model_for_use_case(student_id, model_id, use_case)
    return {"success": ok, "model_id": model_id, "use_case": use_case}
