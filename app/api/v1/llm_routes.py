"""
Functional LLM API routes
Pure functional endpoints for LLM operations
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ...core.security import get_current_user
from ...llm.types import LLMRequest, LLMResponse, StudyContent, QuestionAnswer
from ...llm.llm_orchestrator import (
    orchestrate_llm_request,
    create_study_content,
    create_qa_content,
    get_user_available_providers,
    get_user_available_models,
    get_system_providers
)

router = APIRouter(prefix="/llm", tags=["llm"])


# Request/Response Models
class GenerateTextRequest(BaseModel):
    prompt: str
    model_id: str
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None


class StudyContentRequest(BaseModel):
    prompt: str
    model_id: str


class QARequest(BaseModel):
    prompt: str
    model_id: str


# Routes
@router.post("/generate", response_model=LLMResponse)
async def generate_text(
    request: GenerateTextRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate text using LLM with user's API keys."""
    llm_request = LLMRequest(
        prompt=request.prompt,
        model_id=request.model_id,
        user_id=current_user["sub"],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        system_prompt=request.system_prompt
    )
    
    return await orchestrate_llm_request(llm_request)


@router.post("/study-content", response_model=StudyContent)
async def generate_study_content(
    request: StudyContentRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate structured study content."""
    try:
        return await create_study_content(
            user_id=current_user["sub"],
            model_id=request.model_id,
            prompt=request.prompt
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/qa", response_model=QuestionAnswer)
async def generate_qa(
    request: QARequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate Q&A pairs for study."""
    try:
        return await create_qa_content(
            user_id=current_user["sub"],
            model_id=request.model_id,
            prompt=request.prompt
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/providers", response_model=List[Dict])
async def get_user_providers(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get LLM providers available to the current user."""
    return await get_user_available_providers(current_user["sub"])


@router.get("/models", response_model=List[Dict])
async def get_user_models(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all LLM models available to the current user."""
    return await get_user_available_models(current_user["sub"])


@router.get("/system/providers", response_model=Dict)
async def get_system_supported_providers():
    """Get all system-supported providers and their capabilities."""
    return get_system_providers()
