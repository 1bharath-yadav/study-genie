"""
Student API key management using Supabase backend
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any

from app.services_supabase import get_learning_progress_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/students/{student_id}/api-key")
async def store_api_key(student_id: str, api_key_data: dict):
    """Store encrypted API key for student."""
    try:
        service = get_learning_progress_service()
        encrypted_key = api_key_data.get('encrypted_api_key')

        if not encrypted_key:
            raise HTTPException(
                status_code=400, detail="encrypted_api_key is required")

        stored = await service.store_student_api_key(student_id, encrypted_key)
        if not stored:
            raise HTTPException(
                status_code=500, detail="Failed to store API key")

        return {"message": "API key stored successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/api-key")
async def get_api_key(student_id: str):
    """Get encrypted API key for student."""
    try:
        service = get_learning_progress_service()
        encrypted_key = await service.get_student_api_key(student_id)

        if not encrypted_key:
            raise HTTPException(status_code=404, detail="API key not found")

        return {"encrypted_api_key": encrypted_key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/students/{student_id}/api-key")
async def delete_api_key(student_id: str):
    """Delete API key for student."""
    try:
        service = get_learning_progress_service()
        deleted = await service.delete_student_api_key(student_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found")

        return {"message": "API key deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
