# Pure functional provider and model routes
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from app.models import ProviderResponse, ModelResponse, LLMProvider, ModelType
from app.llm.provider_service import (
    get_available_providers,
    get_models_for_provider,
    get_model_by_id,
    get_chat_models,
    get_embedding_models,
    get_user_available_models,
)
from app.core.security import get_current_user

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("/", response_model=List[ProviderResponse])
async def get_providers():
    """Get all available LLM providers."""
    return get_available_providers()


@router.get("/{provider}/models", response_model=List[ModelResponse])
async def get_provider_models(provider: LLMProvider):
    """Get all models for a specific provider."""
    return get_models_for_provider(provider)


@router.get("/models/chat", response_model=List[ModelResponse])
async def get_chat_models_endpoint(provider: Optional[LLMProvider] = None):
    """Get all chat models, optionally filtered by provider."""
    return get_chat_models(provider)


@router.get("/models/embedding", response_model=List[ModelResponse])
async def get_embedding_models_endpoint(provider: Optional[LLMProvider] = None):
    """Get all embedding models, optionally filtered by provider."""
    return get_embedding_models(provider)


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model_by_id_endpoint(model_id: str):
    """Get a specific model by ID."""
    model = get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.get("/models/available/chat", response_model=List[ModelResponse])
async def get_user_chat_models(
    current_user: dict = Depends(get_current_user)
):
    """Get chat models available to the current user based on their API keys."""
    user_id = current_user["id"]
    return get_user_available_models(user_id, ModelType.CHAT)


@router.get("/models/available/embedding", response_model=List[ModelResponse])
async def get_user_embedding_models(
    current_user: dict = Depends(get_current_user)
):
    """Get embedding models available to the current user based on their API keys."""
    user_id = current_user["id"]
    return get_user_available_models(user_id, ModelType.EMBEDDING)


@router.get("/models/available", response_model=List[ModelResponse])
async def get_user_all_available_models(
    current_user: dict = Depends(get_current_user)
):
    """Get all models available to the current user based on their API keys."""
    user_id = current_user["id"]
    return get_user_available_models(user_id)
