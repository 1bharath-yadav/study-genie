"""Pure functional provider and model service."""

import logging
from datetime import datetime
from typing import Optional, List

from app.models import (
    ProviderResponse, 
    ModelResponse, 
    LLMProvider, 
    ModelType
)
from app.services.db import (
    get_available_providers as db_get_providers,
    get_models_by_provider as db_get_models_by_provider,
    get_model_by_id as db_get_model_by_id,
)

logger = logging.getLogger(__name__)

# Constants
ALL_PROVIDERS = [
    LLMProvider.OPENAI, 
    LLMProvider.ANTHROPIC, 
    LLMProvider.GOOGLE, 
    LLMProvider.OLLAMA
]


def get_available_providers() -> List[ProviderResponse]:
    """Get all available LLM providers.
    
    Returns:
        List[ProviderResponse]: List of available providers, empty list if error occurs.
    """
    try:
        results = db_get_providers()
        return [
            ProviderResponse(
                id=provider["id"],
                name=provider["name"],
                display_name=provider["display_name"],
                base_url=provider.get("base_url"),
                is_active=provider["is_active"],
                created_at=provider.get("created_at", datetime.now())  # Provide default if missing
            )
            for provider in results
        ]
    except Exception as e:
        logger.error("Failed to get providers: %s", str(e))
        return []


def get_models_for_provider(provider: LLMProvider) -> List[ModelResponse]:
    """Get all available models for a specific provider.
    
    Args:
        provider (LLMProvider): The provider to get models for.
        
    Returns:
        List[ModelResponse]: List of models for the provider, empty list if error occurs.
    """
    try:
        results = db_get_models_by_provider(provider.value)
        return [
            ModelResponse(
                id=model["id"],
                provider_id=model.get("provider_id", model.get("provider", "")),
                model_name=model.get("model_name", model.get("name", "")),
                display_name=model["display_name"],
                model_type=ModelType(model["model_type"]),
                context_length=model.get("context_length"),
                supports_system_prompt=model.get("supports_system_prompt", True),
                supports_function_calling=model.get("supports_function_calling", False),
                max_tokens=model.get("max_tokens"),
                is_active=model.get("is_active", model.get("is_available", True)),
                features=model.get("features", {})
            )
            for model in results
        ]
    except Exception as e:
        logger.error("Failed to get models for provider %s: %s", provider, str(e))
        return []


def get_model_by_id(model_id: str) -> Optional[ModelResponse]:
    """Get a specific model by ID.
    
    Args:
        model_id (str): The ID of the model to retrieve.
        
    Returns:
        Optional[ModelResponse]: The model if found, None otherwise.
    """
    try:
        result = db_get_model_by_id(model_id)
        if not result:
            return None
            
        return ModelResponse(
            id=result["id"],
            provider_id=result.get("provider_id", result.get("provider", "")),
            model_name=result.get("model_name", result.get("name", "")),
            display_name=result["display_name"],
            model_type=ModelType(result["model_type"]),
            context_length=result.get("context_length"),
            supports_system_prompt=result.get("supports_system_prompt", True),
            supports_function_calling=result.get("supports_function_calling", False),
            max_tokens=result.get("max_tokens"),
            is_active=result.get("is_active", result.get("is_available", True)),
            features=result.get("features", {})
        )
    except Exception as e:
        logger.error("Failed to get model %s: %s", model_id, str(e))
        return None


def _get_all_models(providers: Optional[List[LLMProvider]] = None) -> List[ModelResponse]:
    """Helper function to get models from multiple providers.
    
    Args:
        providers (Optional[List[LLMProvider]]): List of providers to get models from.
                                               If None, uses all available providers.
                                               
    Returns:
        List[ModelResponse]: Combined list of models from all specified providers.
    """
    if providers is None:
        providers = ALL_PROVIDERS
        
    all_models = []
    for provider in providers:
        models = get_models_for_provider(provider)
        all_models.extend(models)
    
    return all_models


def get_chat_models(provider: Optional[LLMProvider] = None) -> List[ModelResponse]:
    """Get all available chat models, optionally filtered by provider.
    
    Args:
        provider (Optional[LLMProvider]): Specific provider to filter by.
                                        If None, gets models from all providers.
                                        
    Returns:
        List[ModelResponse]: List of chat models, empty list if error occurs.
    """
    try:
        if provider:
            models = get_models_for_provider(provider)
        else:
            models = _get_all_models()
        
        return [model for model in models if model.model_type == ModelType.CHAT]
    except Exception as e:
        logger.error("Failed to get chat models: %s", str(e))
        return []


def get_embedding_models(provider: Optional[LLMProvider] = None) -> List[ModelResponse]:
    """Get all available embedding models, optionally filtered by provider.
    
    Args:
        provider (Optional[LLMProvider]): Specific provider to filter by.
                                        If None, gets models from all providers.
                                        
    Returns:
        List[ModelResponse]: List of embedding models, empty list if error occurs.
    """
    try:
        if provider:
            models = get_models_for_provider(provider)
        else:
            models = _get_all_models()
        
        return [model for model in models if model.model_type == ModelType.EMBEDDING]
    except Exception as e:
        logger.error("Failed to get embedding models: %s", str(e))
        return []


def get_models_by_type(
    model_type: ModelType, 
    provider: Optional[LLMProvider] = None
) -> List[ModelResponse]:
    """Get all available models of a specific type, optionally filtered by provider.
    
    Args:
        model_type (ModelType): The type of models to retrieve.
        provider (Optional[LLMProvider]): Specific provider to filter by.
                                        If None, gets models from all providers.
                                        
    Returns:
        List[ModelResponse]: List of models of the specified type, empty list if error occurs.
    """
    try:
        if provider:
            models = get_models_for_provider(provider)
        else:
            models = _get_all_models()
        
        return [model for model in models if model.model_type == model_type]
    except Exception as e:
        logger.error("Failed to get %s models: %s", model_type, str(e))
        return []


async def get_user_available_models(
    user_id: str, 
    model_type: Optional[ModelType] = None
) -> List[ModelResponse]:
    """Get models available to a user based on their API keys.
    
    Args:
        user_id (str): The ID of the user.
        model_type (Optional[ModelType]): Filter by specific model type.
                                        If None, returns all available models.
                                        
    Returns:
        List[ModelResponse]: List of models available to the user, empty list if error occurs.
    """
    try:
        # Import here to avoid circular imports
        from app.services.api_key_service import get_user_api_keys
        
        # Get user's API keys - this is an async function
        api_keys = await get_user_api_keys(user_id)
        user_providers = [
            LLMProvider(key.get("provider_id", key.get("provider"))) 
            for key in api_keys 
            if key.get("is_active", True)
        ]
        
        if not user_providers:
            logger.info("User %s has no active API keys", user_id)
            return []
        
        # Get models for user's providers
        available_models = []
        for provider in user_providers:
            models = get_models_for_provider(provider)
            available_models.extend(models)
        
        # Filter by model type if specified
        if model_type:
            available_models = [
                model for model in available_models 
                if model.model_type == model_type
            ]
        
        return available_models
        
    except ImportError as e:
        logger.error("Failed to import API key service: %s", str(e))
        return []
    except Exception as e:
        logger.error("Failed to get user available models for user %s: %s", user_id, str(e))
        return []
