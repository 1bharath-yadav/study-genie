# app/models.py
from __future__ import annotations

from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum





class ModelType(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    COMPLETION = "completion"


class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class StudentData(BaseModel):
    id: str
    name: str
    email: str
    grade_level: Optional[str] = None
    subjects: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


# Pure data models for API keys
class APIKeyCreate(BaseModel):
    provider_id: str
    api_key: str = Field(..., min_length=5)  # Reduced minimum length
    
    @validator('api_key')
    def validate_api_key(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('API key cannot be empty')
        # Remove any common prefixes or whitespace that might be accidentally copied
        if v.startswith('sk-') or v.startswith('AIza'):
            # These are valid prefixes, don't reject them
            return v
        return v

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
    # Optional capabilities and model list coming from provider definitions
    capabilities: Optional[List[str]] = Field(default_factory=list)
    models: Optional[List[str]] = Field(default_factory=list)


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
    is_available: Optional[bool] = None


# Pure data models for study sessions
class StudySessionCreate(BaseModel):
    student_id: str
    session_name: str
    subject: Optional[str] = None
    embedding_model_id: Optional[str] = None
    llm_model_id: Optional[str] = None


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


# Pure data models for LLM interactions
class GenerateTextRequest(BaseModel):
    prompt: str
    model_id: str
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None


class LLMResponse(BaseModel):
    text: str


class StudyContentRequest(BaseModel):
    prompt: str
    model_id: str


class StudyContent(BaseModel):
    key_concepts: List[str] = Field(default_factory=list)
    summary: str = ""
    important_figures: List[str] = Field(default_factory=list)


class QARequest(BaseModel):
    prompt: str
    model_id: str


class Question(BaseModel):
    question: str
    answer: str


class QuestionAnswer(BaseModel):
    questions: List[Question] = Field(default_factory=list)


class ConceptStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    MASTERED = "mastered"
    NEEDS_REVIEW = "needs_review"


class ActivityType(str, Enum):
    FLASHCARD_PRACTICE = "flashcard_practice"
    QUIZ_ATTEMPT = "quiz_attempt"
    CONTENT_STUDY = "content_study"
    CONCEPT_REVIEW = "concept_review"


class RecommendationType(str, Enum):
    CONCEPT_REVIEW = "concept_review"
    PRACTICE_MORE = "practice_more"
    ADVANCE_TOPIC = "advance_topic"
    MAINTENANCE_PRACTICE = "maintenance_practice"
    WEAKNESS_FOCUS = "weakness_focus"


# Student Models
class StudentCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    full_name: str = Field(..., min_length=2, max_length=100)
    learning_preferences: Dict[str, Any] = Field(default_factory=dict)
    gemini_api_key: Optional[str] = None

    @field_validator('username', 'full_name', mode='before')
    @classmethod
    def strip_names(cls, v: str):
        return v.strip() if isinstance(v, str) else v


class StudentResponse(BaseModel):
    student_id: str
    username: str
    email: str
    full_name: str
    message: str
    has_api_key: bool = False


class StudentUpdate(BaseModel):
    full_name: Optional[str] = None
    learning_preferences: Optional[Dict[str, Any]] = None

    @field_validator('full_name', mode='before')
    @classmethod
    def strip_full_name(cls, v: Optional[str]):
        return v.strip() if isinstance(v, str) else v


class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=30, max_length=100)

    @field_validator('api_key', mode='after')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v.startswith('AIza'):
            raise ValueError('Invalid Gemini API key format')
        return v.strip()


class ApiKeyResponse(BaseModel):
    has_api_key: bool
    message: str


class ApiKeyUpdateRequest(BaseModel):
    new_api_key: str = Field(..., min_length=30, max_length=100)

    @field_validator('new_api_key', mode='after')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v.startswith('AIza'):
            raise ValueError('Invalid Gemini API key format')
        return v.strip()


# Core LLM Response Models
class QuizData(BaseModel):
    question: str
    options: List[str] = Field(default_factory=list)
    correct_answer: str
    explanation: str


class LLMResponseContent(BaseModel):
    flashcards: Dict[str, FlashcardData] = Field(default_factory=dict)
    quiz: Dict[str, QuizData] = Field(default_factory=dict)
    match_the_following: Optional[Dict[str, Any]] = None
    summary: str = ""
    learning_objectives: List[str] = Field(default_factory=list)


class LLMResponseRequest(BaseModel):
    student_id: str
    subject_name: str
    chapter_name: str
    concept_name: str
    llm_response: LLMResponseContent
    user_query: str
    difficulty_level: Optional[DifficultyLevel] = DifficultyLevel.MEDIUM

    @field_validator('subject_name', 'chapter_name', 'concept_name', mode='after')
    @classmethod
    def validate_names(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters long')
        return v.strip().title()


class TrackingMetadata(BaseModel):
    student_id: str
    subject_id: int
    chapter_id: int
    concept_id: int
    subject_name: str
    chapter_name: str
    concept_name: str
    created_at: datetime


class CreatedEntities(BaseModel):
    subject_created: bool = False
    chapter_created: bool = False
    concept_created: bool = False


class ProcessedLLMResponse(BaseModel):
    enhanced_response: LLMResponseContent
    tracking_metadata: TrackingMetadata
    created_entities: CreatedEntities
    message: str


class ProcessFilesResponse(BaseModel):
    """Frontend-compatible response format"""
    task_id: str
    status: str = "completed"  # 'processing' | 'completed' | 'failed'
    content: Optional[LLMResponseContent] = None
    error: Optional[str] = None

    # Metadata from LLM response for context tracking
    metadata: Optional[Dict[str, Any]] = None
    subject_name: Optional[str] = None
    chapter_name: Optional[str] = None
    concept_name: Optional[str] = None
    difficulty_level: Optional[DifficultyLevel] = None
    estimated_study_time: Optional[str] = None


# Progress Models
class ConceptProgressUpdate(BaseModel):
    concept_id: int
    correct_answers: int = Field(..., ge=0)
    total_questions: int = Field(..., gt=0)
    time_spent: Optional[int] = Field(None, ge=0)  # in seconds
    activity_type: ActivityType = ActivityType.QUIZ_ATTEMPT

    @model_validator(mode='after')
    def validate_counts(self):
        if self.total_questions < self.correct_answers:
            raise ValueError('Total questions must be >= correct answers')
        return self


class ConceptProgress(BaseModel):
    concept_id: int
    concept_name: str
    status: ConceptStatus
    mastery_score: float = Field(..., ge=0, le=100)
    attempts_count: int = Field(..., ge=0)
    correct_answers: int = Field(..., ge=0)
    total_questions: int = Field(..., ge=0)
    last_practiced: Optional[datetime] = None
    first_learned: Optional[datetime] = None


class StudentProgressResponse(BaseModel):
    student_id: str
    overall_progress: Dict[str, Any]
    subject_progress: List[Dict[str, Any]]
    concept_progress: List[ConceptProgress]
    recent_activity: List[Dict[str, Any]]
    total_concepts: int
    mastered_concepts: int
    weak_concepts: int


# Learning Activity Models
class LearningActivityRequest(BaseModel):
    """Request model for learning activity tracking"""
    activity_type: ActivityType
    correct_answers: int = Field(..., ge=0)
    total_questions: int = Field(..., gt=0)
    time_spent: Optional[int] = Field(None, ge=0)  # in seconds
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM

    # Context information (should be provided by frontend)
    subject_name: Optional[str] = None
    chapter_name: Optional[str] = None
    concept_name: Optional[str] = None

    # Additional metadata
    session_id: Optional[str] = None
    # e.g., "llm_generated", "manual_upload"
    content_source: Optional[str] = None

    @model_validator(mode='after')
    def validate_counts(self):
        if self.total_questions < self.correct_answers:
            raise ValueError('Total questions must be >= correct answers')
        return self


# Weakness Models
class WeaknessRecord(BaseModel):
    concept_id: int
    weakness_type: str
    error_pattern: Optional[str] = None
    severity: Optional[float] = Field(0.3, ge=0.0, le=1.0)


class WeaknessAnalysis(BaseModel):
    weakness_id: int
    concept_name: str
    chapter_name: str
    subject_name: str
    weakness_type: str
    error_pattern: Optional[str] = None
    frequency_count: int
    severity_score: float
    last_occurrence: datetime
    is_resolved: bool
    recommended_actions: List[str] = Field(default_factory=list)


# Study Session Models
class StudySessionStart(BaseModel):
    subject_name: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class StudySessionEnd(BaseModel):
    session_data: Dict[str, Any]
    total_questions: int = Field(..., ge=0)
    correct_answers: int = Field(..., ge=0)
    duration: int = Field(..., ge=0)  # in seconds
    concepts_covered: List[int] = Field(default_factory=list)


class StudySessionResponse(BaseModel):
    session_id: int
    student_id: str
    status: str
    message: str


class StudySessionHistory(BaseModel):
    session_id: int
    subject_name: str
    total_questions: int
    correct_answers: int
    accuracy: float
    duration: int
    concepts_covered: int
    started_at: datetime
    completed_at: Optional[datetime] = None


# Recommendation Models
class RecommendationResponse(BaseModel):
    recommendation_id: int
    recommendation_type: RecommendationType
    title: str
    description: str
    priority_score: int = Field(..., ge=1, le=10)
    concept_id: Optional[int] = None
    concept_name: Optional[str] = None
    subject_name: Optional[str] = None
    is_active: bool = True
    is_completed: bool = False
    created_at: datetime
    expires_at: Optional[datetime] = None


# Analytics Models
class OverallStats(BaseModel):
    total_concepts: int = 0
    mastered_concepts: int = 0
    in_progress_concepts: int = 0
    weak_concepts: int = 0
    average_mastery_score: float = 0.0
    total_study_time: int = 0  # in minutes
    streak_days: int = 0
    last_active: Optional[datetime] = None


class SubjectProgressStats(BaseModel):
    subject_id: int
    subject_name: str
    total_concepts: int
    mastered_concepts: int
    average_score: float
    time_spent: int  # in minutes
    last_activity: Optional[datetime] = None
    progress_percentage: float


class ActivityStats(BaseModel):
    activity_type: ActivityType
    count: int
    average_score: Optional[float] = None
    total_time: int = 0  # in minutes
    last_activity: Optional[datetime] = None


class StudentAnalyticsResponse(BaseModel):
    student_id: str
    overall_stats: OverallStats
    subject_progress: List[SubjectProgressStats]
    activity_stats: List[ActivityStats]
    weekly_progress: List[Dict[str, Any]]  # Weekly activity data
    learning_velocity: float = 0.0  # Concepts mastered per week
    focus_areas: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)


# Subject and Content Models
class ConceptInfo(BaseModel):
    concept_id: int
    concept_name: str
    concept_order: int
    difficulty_level: DifficultyLevel
    description: Optional[str] = None


class ChapterInfo(BaseModel):
    chapter_id: int
    chapter_name: str
    chapter_order: int
    description: Optional[str] = None
    concepts: List[ConceptInfo] = Field(default_factory=list)
    concept_count: int = 0


class SubjectInfo(BaseModel):
    subject_id: int
    subject_name: str
    description: Optional[str] = None
    chapter_count: int = 0
    concept_count: int = 0


class SubjectStructure(BaseModel):
    subject_info: SubjectInfo
    chapters: List[ChapterInfo]


# Learning Path Models
class LearningPathStep(BaseModel):
    step_order: int
    concept_id: int
    concept_name: str
    chapter_name: str
    estimated_time: int  # in minutes
    prerequisites_met: bool
    difficulty_level: DifficultyLevel
    recommended_activities: List[str]


class LearningPathResponse(BaseModel):
    student_id: str
    subject_name: Optional[str] = None
    path_steps: List[LearningPathStep]
    total_estimated_time: int  # in minutes
    completion_percentage: float
    next_milestone: Optional[str] = None


# Batch Processing Models
class QuizResult(BaseModel):
    concept_id: int
    question_id: Optional[str] = None
    is_correct: bool
    time_taken: Optional[int] = None  # in seconds
    difficulty_level: Optional[DifficultyLevel] = None


class QuizResultsBatch(BaseModel):
    quiz_results: List[QuizResult]
    session_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FlashcardResult(BaseModel):
    concept_id: int
    card_id: Optional[str] = None
    confidence_level: int = Field(..., ge=1, le=5)  # 1=very hard, 5=very easy
    time_taken: Optional[int] = None  # in seconds
    difficulty_level: Optional[DifficultyLevel] = None


class FlashcardSessionResults(BaseModel):
    flashcard_results: List[FlashcardResult]
    session_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    total_session_time: Optional[int] = None  # in seconds


# Dashboard Models
class DashboardData(BaseModel):
    student_id: str
    analytics: StudentAnalyticsResponse
    recommendations: List[RecommendationResponse]
    recent_activity: List[Dict[str, Any]]
    upcoming_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    learning_streaks: Dict[str, int] = Field(default_factory=dict)


# Error Models
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# Validation Models
class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# Search and Filter Models
class ConceptSearchFilter(BaseModel):
    subject_id: Optional[int] = None
    chapter_id: Optional[int] = None
    difficulty_level: Optional[DifficultyLevel] = None
    status: Optional[ConceptStatus] = None
    search_term: Optional[str] = None


class ProgressFilter(BaseModel):
    time_period: Optional[str] = "30d"  # 7d, 30d, 90d, all
    subject_ids: Optional[List[int]] = None
    status_filter: Optional[List[ConceptStatus]] = None
    include_inactive: bool = False


# File Processing Models
class FileProcessRequest(BaseModel):
    student_id: Optional[str] = None
    subject_name: Optional[str] = None
    chapter_name: Optional[str] = None


class FileProcessResponse(BaseModel):
    filename: str
    file_type: str
    extracted_text: str
    text_length: int
    word_count: int
    processed_response: Optional[Dict[str, Any]] = None
    message: str
