"""
Hardcoded provider definitions and Pydantic-AI agent factories using explicit model and provider classes.

This module uses the official pydantic-ai model and provider classes for each vendor.
"""

from typing import Dict
import logging

# Import model and provider classes
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider

logger = logging.getLogger(__name__)

PROVIDERS_JSON = {
  "openai": {
    "display_name": "OpenAI",
    "chat_models": ["gpt-5", "gpt-4.1", "gpt-4o"],
    "embed_models": ["text-embedding-3-small", "text-embedding-3-large"],
    "capabilities": ["text_generation", "function_calling", "vision", "embeddings"]
  },
  "anthropic": {
    "display_name": "Anthropic",
    "chat_models": ["claude-opus-4.1", "claude-sonnet-4", "claude-3.5-haiku"],
    "embed_models": [],
    "capabilities": ["text_generation", "vision", "embeddings"]
  },
  "google": {
    "display_name": "Google (Gemini)",
    "chat_models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "embed_models": ["gemini-embedding-001"],
    "capabilities": ["text_generation", "vision", "embeddings"]
  },
  "bedrock": {
    "display_name": "AWS Bedrock",
    "chat_models": ["amazon.nova-pro", "amazon.nova-lite"],
    "embed_models": ["amazon.titan-embed-text-v2"],
    "capabilities": ["text_generation", "embeddings"]
  },
  "cohere": {
    "display_name": "Cohere",
    "chat_models": ["command-r", "command-r-plus"],
    "embed_models": ["embed-english-v3.0", "embed-multilingual-v3.0"],
    "capabilities": ["text_generation", "embeddings"]
  },
  "groq": {
    "display_name": "Groq",
    "chat_models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "embed_models": [],
    "capabilities": ["text_generation"]
  },
  "mistral": {
    "display_name": "Mistral",
    "chat_models": ["mistral-large", "mistral-small-3.1", "mixtral-8x22b"],
    "embed_models": ["mistral-embed"],
    "capabilities": ["text_generation", "embeddings"]
  },
  "huggingface": {
    "display_name": "Hugging Face",
    "chat_models": ["deepseek-ai/DeepSeek-V3-0324"],
    "embed_models": ["BAAI/bge-m3"],
    "capabilities": ["text_generation", "embeddings"]
  }
}


def get_all_providers() -> Dict[str, Dict]:
    """Return the raw PROVIDERS_JSON mapping."""
    return PROVIDERS_JSON


def get_provider_list() -> list[dict]:
    """Return a list of provider metadata suitable for API responses.

    Each provider contains id, name, display_name, models and capabilities.
    """
    return [
        {
            "id": key,
            "name": key,
            "display_name": val.get("display_name", key),
            "chat_models": val.get("chat_models", []),
            "embed_models": val.get("embed_models", []),
            "capabilities": val.get("capabilities", []),
            "is_active": True,
        }
        for key, val in PROVIDERS_JSON.items()
    ]


def get_models_for_provider(provider_name: str) -> list[dict]:
    """Return model entries for a given provider.

    Models are returned as simple dicts consumable by the frontend. We generate
    an id using the provider and model name.
    """
    provider = PROVIDERS_JSON.get(provider_name)
    if not provider:
        return []
    models = []
    # chat models
    for model_name in provider.get("chat_models", []):
        models.append({
            "id": f"{provider_name}-{model_name}",
            "provider_id": provider_name,
            "model_name": model_name,
            "display_name": model_name,
            "model_type": "chat",
            "supports_embedding": False,
            "is_active": True,
        })
    # embedding models
    for model_name in provider.get("embed_models", []):
        models.append({
            "id": f"{provider_name}-{model_name}",
            "provider_id": provider_name,
            "model_name": model_name,
            "display_name": model_name,
            "model_type": "embedding",
            "supports_embedding": True,
            "is_active": True,
        })
    return models


def get_provider_by_id(provider_id: str) -> dict | None:
    """Return provider metadata (dict) for a given provider id/name, or None."""
    providers = get_provider_list()
    for p in providers:
        if p.get("id") == provider_id or p.get("name") == provider_id:
            return p
    return None


def get_user_model_preferences(student_id: str) -> list[dict]:
    """Return model preferences for a user using the persisted service."""
    try:
        from app.services.model_preference_service import list_model_preferences
        return list_model_preferences(student_id)
    except Exception:
        return []

def _create_model(provider: str, model_name: str, api_key: str):
    """Create a pydantic-ai model instance with its provider."""
    provider_lower = provider.lower()
    if provider_lower == "openai":
        provider_instance = OpenAIProvider(api_key=api_key)
        return OpenAIChatModel(model_name, provider=provider_instance)
    elif provider_lower == "anthropic":
        provider_instance = AnthropicProvider(api_key=api_key)
        return AnthropicModel(model_name, provider=provider_instance)
    elif provider_lower == "google":
        provider_instance = GoogleProvider(api_key=api_key)
        return GoogleModel(model_name, provider=provider_instance)
    elif provider_lower == "bedrock":
        # For Bedrock, assume AWS credentials are set in env
        provider_instance = BedrockProvider()
        return BedrockConverseModel(model_name, provider=provider_instance)
    elif provider_lower == "cohere":
        provider_instance = CohereProvider(api_key=api_key)
        return CohereModel(model_name, provider=provider_instance)
    elif provider_lower == "groq":
        provider_instance = GroqProvider(api_key=api_key)
        return GroqModel(model_name, provider=provider_instance)
    elif provider_lower == "mistral":
        provider_instance = MistralProvider(api_key=api_key)
        return MistralModel(model_name, provider=provider_instance)
    elif provider_lower == "huggingface":
        provider_instance = HuggingFaceProvider(api_key=api_key)
        return HuggingFaceModel(model_name, provider=provider_instance)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_learning_agent(provider: str, model_name: str, api_key: str):
    """Create an agent for generating learning content."""
    model = _create_model(provider, model_name, api_key)
    return Agent(model)

