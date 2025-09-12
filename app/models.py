# Pure functional data models - no classes with methods
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class ModelType(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    COMPLETION = "completion"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Pure data models for users
class UserCreate(BaseModel):
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    updated_at: datetime


# Pure data models for students
class StudentCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    grade_level: Optional[str] = None
    subjects: Optional[List[str]] = Field(default_factory=list)


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    grade_level: Optional[str] = None
    subjects: Optional[List[str]] = None


class StudentResponse(BaseModel):
    id: str
    name: str
    email: str
    grade_level: Optional[str]
    subjects: Optional[List[str]]
    user_id: str
    created_at: datetime
    updated_at: datetime


# Pure data models for API keys
class APIKeyCreate(BaseModel):
    provider_id: str  # Use provider_id instead of provider enum
    api_key: str = Field(..., min_length=20)
    # Note: key_name not supported by database schema


class APIKeyResponse(BaseModel):
    id: str
    provider_id: str
    provider_name: str
    provider_display_name: str
    is_active: bool
    is_default: bool
    student_id: str
    created_at: datetime


# Pure data models for LLM providers and models
class ProviderResponse(BaseModel):
    id: str
    name: str
    display_name: str
    base_url: Optional[str] = None
    is_active: bool
    created_at: datetime


class ModelResponse(BaseModel):
    id: str
    provider_id: str
    model_name: str
    display_name: str
    model_type: ModelType
    context_length: Optional[int] = None
    supports_system_prompt: bool = True
    supports_function_calling: bool = False
    max_tokens: Optional[int] = None
    is_active: bool
    features: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserModelPreference(BaseModel):
    id: str
    student_id: str
    model_id: str
    model_name: str
    model_display_name: str
    provider_name: str
    use_for_chat: bool = False
    use_for_embedding: bool = False
    is_default: bool = False
    created_at: datetime


class UserModelPreferenceCreate(BaseModel):
    model_id: str
    use_for_chat: bool = False
    use_for_embedding: bool = False
    is_default: bool = False


class UserModelPreferenceUpdate(BaseModel):
    use_for_chat: Optional[bool] = None
    use_for_embedding: Optional[bool] = None
    is_default: Optional[bool] = None
    is_available: bool


# Pure data models for study sessions
class StudySessionCreate(BaseModel):
    student_id: str
    session_name: str
    subject: Optional[str] = None
    embedding_model_id: Optional[str] = None
    llm_model_id: Optional[str] = None


class StudySessionResponse(BaseModel):
    id: str
    student_id: str
    session_name: str
    subject: Optional[str]
    embedding_model_id: Optional[str]
    llm_model_id: Optional[str]
    created_at: datetime


# Pure data models for content processing
class DocumentUpload(BaseModel):
    filename: str
    content: bytes
    content_type: str


class ProcessedContent(BaseModel):
    id: str
    filename: str
    content_text: str
    embeddings: Optional[List[float]]
    session_id: str
    created_at: datetime


# Pure data models for study materials
class FlashcardData(BaseModel):
    question: str
    answer: str
    difficulty: DifficultyLevel


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str


class StudyMaterial(BaseModel):
    flashcards: List[FlashcardData] = Field(default_factory=list)
    quiz_questions: List[QuizQuestion] = Field(default_factory=list)
    summary: str = ""
    learning_objectives: List[str] = Field(default_factory=list)
