"""
Pure functional LLM models and types
"""
from typing import Dict, Any, List, Optional, Union, Callable
from pydantic import BaseModel
from enum import Enum


class ProviderType(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"


class ModelCapability(str, Enum):
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS = "embeddings" 
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"


class LLMRequest(BaseModel):
    prompt: str
    model_id: str
    user_id: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    system_prompt: Optional[str] = None
    response_format: Optional[str] = "text"


class LLMResponse(BaseModel):
    content: str
    model_id: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]
    model_id: str
    user_id: str


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_id: str
    provider: str
    usage: Optional[Dict[str, int]] = None


# Structured output models for study content
class StudyContent(BaseModel):
    """Structured study content for learning."""
    title: str
    summary: str
    key_points: List[str]
    concepts: List[str]
    difficulty_level: str = "intermediate"


class QuestionAnswer(BaseModel):
    """Q&A structure for study sessions."""
    question: str
    answer: str
    explanation: str
    topic: str


class StudyPlan(BaseModel):
    """Structured study plan."""
    subject: str
    topic: str
    duration_minutes: int
    activities: List[str]
    learning_objectives: List[str]


# Type aliases for functional programming
LLMProvider = Callable[[LLMRequest], LLMResponse]
EmbeddingProvider = Callable[[EmbeddingRequest], EmbeddingResponse]
ProviderConfig = Dict[str, Any]
