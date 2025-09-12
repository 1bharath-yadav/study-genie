"""
Functional LLM orchestrator service
Coordinates LLM providers, API keys, and user requests
"""
from typing import Dict, List
from .providers import (
    generate_text, 
    generate_structured_content, 
    generate_qa_pair,
    get_all_providers
)
from .types import LLMRequest, LLMResponse, StudyContent, QuestionAnswer
from ..services.api_key_service import get_api_key_for_provider
from ..services.db import get_available_providers, get_models_by_provider


async def orchestrate_llm_request(request: LLMRequest) -> LLMResponse:
    """
    Orchestrate an LLM request by:
    1. Getting user's API key for the provider
    2. Routing to appropriate provider
    3. Returning structured response
    """
    # Extract provider from model_id
    provider = request.model_id.split('-')[0].lower() if '-' in request.model_id else 'openai'
    
    # Get user's API key for this provider
    api_key = await get_api_key_for_provider(request.user_id, provider)
    if not api_key:
        return LLMResponse(
            content=f"No API key found for provider: {provider}",
            model_id=request.model_id,
            provider=provider,
            usage=None,
            finish_reason="error"
        )
    
    # Generate text using the provider
    return await generate_text(request, api_key)


async def create_study_content(user_id: str, model_id: str, prompt: str) -> StudyContent:
    """Create structured study content from a prompt."""
    provider = model_id.split('-')[0].lower() if '-' in model_id else 'openai'
    
    api_key = await get_api_key_for_provider(user_id, provider)
    if not api_key:
        raise ValueError(f"No API key found for provider: {provider}")
    
    return await generate_structured_content(prompt, model_id, api_key)


async def create_qa_content(user_id: str, model_id: str, prompt: str) -> QuestionAnswer:
    """Create Q&A content from a prompt."""
    provider = model_id.split('-')[0].lower() if '-' in model_id else 'openai'
    
    api_key = await get_api_key_for_provider(user_id, provider)
    if not api_key:
        raise ValueError(f"No API key found for provider: {provider}")
    
    return await generate_qa_pair(prompt, model_id, api_key)


async def get_user_available_providers(user_id: str) -> List[Dict]:
    """Get providers that the user has API keys for."""
    # Get all configured providers from database
    all_providers = get_available_providers()
    
    # Filter to only providers where user has API keys
    user_providers = []
    for provider in all_providers:
        api_key = await get_api_key_for_provider(user_id, provider["name"])
        if api_key:
            # Get supported models for this provider
            models = get_models_by_provider(provider["name"])
            user_providers.append({
                "name": provider["name"],
                "display_name": provider["display_name"],
                "models": models,
                "capabilities": provider.get("capabilities", [])
            })
    
    return user_providers


async def get_user_available_models(user_id: str) -> List[Dict]:
    """Get all models available to the user across all their configured providers."""
    providers = await get_user_available_providers(user_id)
    models = []
    
    for provider in providers:
        for model in provider["models"]:
            models.append({
                "model_id": model["model_id"],
                "display_name": model["display_name"],
                "provider": provider["name"],
                "provider_display_name": provider["display_name"],
                "capabilities": model.get("capabilities", [])
            })
    
    return models


def get_system_providers() -> Dict:
    """Get all system-supported providers and their capabilities."""
    return get_all_providers()
