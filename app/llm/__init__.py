"""
LLM module - Pure functional approach
"""

# Export main types and functions
from .types import (
    LLMRequest,
    LLMResponse,
    StudyContent,
    QuestionAnswer,
    StudyPlan,
    ProviderType,
    ModelCapability
)

from .providers import (
    generate_text,
    generate_structured_content,
    generate_qa_pair,
    get_all_providers,
    get_supported_models,
    get_provider_capabilities
)

__all__ = [
    "LLMRequest",
    "LLMResponse",
    "StudyContent",
    "QuestionAnswer",
    "StudyPlan",
    "ProviderType",
    "ModelCapability",
    "generate_text",
    "generate_structured_content",
    "generate_qa_pair",
    "get_all_providers",
    "get_supported_models",
    "get_provider_capabilities"
]
