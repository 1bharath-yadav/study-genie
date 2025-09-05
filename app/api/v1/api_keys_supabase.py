from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime
import uuid

from app.core.security import get_current_user
from app.supabase_client import get_supabase_client

router = APIRouter()


class ApiKey(BaseModel):
    id: str
    name: str
    service: str  # 'gemini' | 'openai' | 'anthropic'
    created_at: str
    last_used: Optional[str] = None
    is_active: bool


class CreateApiKeyRequest(BaseModel):
    name: str
    service: str  # 'gemini' | 'openai' | 'anthropic'
    key: str


class ApiKeyStatus(BaseModel):
    hasActiveApiKey: bool
    service: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.get("/api-keys", response_model=List[ApiKey])
async def get_api_keys(
    current_user: dict = Depends(get_current_user)
):
    """Get user's API keys (without exposing the actual key values)"""
    try:
        supabase = get_supabase_client()
        student_id = current_user.get("student_id") or current_user.get("sub")

        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Get API key info from Supabase
        try:
            api_key_info = await supabase.get_api_key_info(str(student_id))
            if api_key_info:
                return [ApiKey(
                    id=str(uuid.uuid4()),
                    name=f"{api_key_info.get('service', 'Unknown').title()} API Key",
                    service=api_key_info.get('service', 'gemini'),
                    created_at=api_key_info.get(
                        'created_at', datetime.now().isoformat()),
                    is_active=True
                )]
            else:
                return []
        except Exception:
            return []

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving API keys: {str(e)}")


@router.get("/api-keys/status", response_model=ApiKeyStatus)
async def get_api_key_status(
    current_user: dict = Depends(get_current_user)
):
    """Check if user has an active API key and return basic info"""
    try:
        supabase = get_supabase_client()
        student_id = current_user.get("student_id") or current_user.get("sub")

        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Check if user has API key in Supabase
        try:
            api_key_info = await supabase.get_api_key_info(str(student_id))
            if api_key_info:
                return ApiKeyStatus(
                    hasActiveApiKey=True,
                    service=api_key_info.get('service'),
                    created_at=api_key_info.get('created_at'),
                    updated_at=api_key_info.get('updated_at')
                )
            else:
                return ApiKeyStatus(hasActiveApiKey=False)
        except Exception:
            return ApiKeyStatus(hasActiveApiKey=False)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error checking API key status: {str(e)}")


@router.post("/api-keys", response_model=ApiKey)
async def create_api_key(
    api_key_data: CreateApiKeyRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create or update user's API key with proper encryption"""
    try:
        supabase = get_supabase_client()
        student_id = current_user.get("student_id") or current_user.get("sub")

        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # First ensure the student exists
        try:
            student = await supabase.get_student(str(student_id))
            if not student:
                # Create student if doesn't exist
                student_data = {
                    "student_id": str(student_id),
                    "username": current_user.get("name", ""),
                    "email": current_user.get("email", ""),
                    "full_name": current_user.get("name", "")
                }
                await supabase.create_student(student_data)
        except Exception as e:
            print(f"Error checking/creating student: {e}")

        # Store the API key with encryption in Supabase
        success = await supabase.store_api_key(
            student_id=str(student_id),
            api_key=api_key_data.key,
            service=api_key_data.service
        )

        if success:
            return ApiKey(
                id=str(uuid.uuid4()),
                name=api_key_data.name,
                service=api_key_data.service,
                created_at=datetime.now().isoformat(),
                is_active=True
            )
        else:
            raise HTTPException(
                status_code=500, detail="Failed to save API key")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error setting up API key: {str(e)}")


@router.get("/api-keys/retrieve")
async def retrieve_api_key_for_llm(
    current_user: dict = Depends(get_current_user)
):
    """Retrieve the actual API key for use by LLM services (internal use only)"""
    try:
        supabase = get_supabase_client()
        student_id = current_user.get("student_id") or current_user.get("sub")

        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Get the decrypted API key
        api_key = await supabase.get_api_key(str(student_id))

        if api_key:
            return {"api_key": api_key}
        else:
            raise HTTPException(
                status_code=404, detail="No API key found for user")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving API key: {str(e)}")


@router.delete("/api-keys/{api_key_id}")
async def delete_api_key(
    api_key_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete user's API key"""
    try:
        supabase = get_supabase_client()
        student_id = current_user.get("student_id") or current_user.get("sub")

        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        success = await supabase.delete_api_key(str(student_id))

        if success:
            return {"message": "API key removed successfully"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to remove API key")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error removing API key: {str(e)}")
