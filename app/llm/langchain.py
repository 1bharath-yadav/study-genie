import os
import json
import logging
from typing import Dict, Any, List, Optional, Type, Union, Literal
from pydantic import BaseModel, ValidationError, Field
from enum import Enum

# PydanticAI imports
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.anthropic import AnthropicModel

# Google AI imports
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Supabase client
from app.db.db_client import get_supabase_client

# Langchain for embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

from app.llm.process_files.file_process import (
    enhance_retrieved_context,
    format_context_for_llm,
    load_documents_from_files,
    perform_document_chunking,
    setup_vector_store_and_retriever
)

load_dotenv()

logger = logging.getLogger("multi_llm_provider")
logging.basicConfig(level=logging.INFO)

# Constants
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEFAULT_PROVIDER = "google"
DEFAULT_MODEL = "gemini-2.0-flash"

# Pydantic Models for structured output
class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class ContentMetadata(BaseModel):
    """Automatically extracted metadata about the content"""
    subject_name: str = Field(description="The main subject/domain")
    chapter_name: str = Field(description="The chapter or topic name from the content")
    concept_name: str = Field(description="The specific concept being studied")
    difficulty_level: DifficultyLevel = Field(description="Assessed difficulty level based on content complexity")
    estimated_study_time: Optional[str] = Field(default=None, description="Estimated time needed to complete all materials")

class Flashcard(BaseModel):
    """Individual flashcard"""
    question: str = Field(description="The question for the flashcard")
    answer: str = Field(description="The answer to the question")
    key_concepts: Optional[str] = Field(default=None, description="Key concepts covered")
    difficulty: DifficultyLevel = Field(description="Difficulty level of this flashcard")

class QuizQuestion(BaseModel):
    """Individual quiz question"""
    question: str = Field(description="The quiz question")
    options: List[str] = Field(description="List of answer options")
    correct_answer: str = Field(description="The correct answer")
    explanation: str = Field(description="Explanation of why this is correct")

class MatchMapping(BaseModel):
    """Match the following mapping"""
    A: str = Field(description="Item from column A")
    B: str = Field(description="Corresponding item from column B")

class MatchTheFollowing(BaseModel):
    """Match the following exercise"""
    columnA: List[str] = Field(description="Items in column A")
    columnB: List[str] = Field(description="Items in column B")
    mappings: List[MatchMapping] = Field(description="Correct mappings between columns")

class LearningContent(BaseModel):
    """Complete learning content structure"""
    metadata: ContentMetadata
    content_type: Literal["flashcards", "quiz", "match_the_following", "summary", "all"]
    flashcards: Optional[List[Flashcard]] = None
    quiz: Optional[List[QuizQuestion]] = None
    match_the_following: Optional[MatchTheFollowing] = None
    summary: str = Field(description="Summary of the content")
    learning_objectives: List[str] = Field(description="Learning objectives for this content")

# Provider configuration mapping
PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model_factory": lambda api_key, model_name=None: OpenAIModel(
            model_name=model_name or "gpt-4", 
            provider="openai"
        ),
        "embedding_handler": None,
    },
    "google": {
        "model_factory": lambda api_key, model_name=None: GeminiModel(
            model_name=model_name or "gemini-2.0-flash"
        ),
        "embedding_handler": lambda api_key: GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        ),
    },
    "anthropic": {
        "model_factory": lambda api_key, model_name=None: AnthropicModel(
            model_name=model_name or "claude-3-5-sonnet-20241022"
        ),
        "embedding_handler": None,
    }
}


LEARNING_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        # Metadata extraction (always included)
        "metadata": {
            "type": "object",
            "description": "Automatically extracted metadata about the content",
            "properties": {
                "subject_name": {
                    "type": "string",
                    "description": "The main subject/domain (e.g., Mathematics, Physics, Computer Science, History, etc.)"
                },
                "chapter_name": {
                    "type": "string",
                    "description": "The chapter or topic name from the content (e.g., Neural Networks, Calculus, World War II, etc.)"
                },
                "concept_name": {
                    "type": "string",
                    "description": "The specific concept being studied (e.g., Perceptron, Derivatives, Treaty of Versailles, etc.)"
                },
                "difficulty_level": {
                    "type": "string",
                    "enum": ["Easy", "Medium", "Hard"],
                    "description": "Assessed difficulty level based on content complexity"
                },
                "estimated_study_time": {
                    "type": "string",
                    "description": "Estimated time needed to complete all materials (e.g., '2-3 hours', '45 minutes')"
                }
            },
            "required": ["subject_name", "chapter_name", "concept_name", "difficulty_level"]
        },

        # Content type requested by user
        "content_type": {
            "type": "string",
            "enum": ["flashcards", "quiz", "match_the_following", "summary", "all"],
            "description": "Type of content requested by the user"
        },

        # 15 flashcards (only when requested)
        "flashcards": {
            "type": "array",
            "description": "A set of exactly 15 flashcards summarizing important key concepts. Only include if user requests flashcards.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Front side of the flashcard: a concise, focused question."
                    },
                    "answer": {
                        "type": "string",
                        "description": "Back side of the flashcard: a clear and short answer."
                    },
                    "key_concepts": {
                        "type": "string",
                        "description": "Topic or key concept covered in this flashcard."
                    },
                    "key_concepts_data": {
                        "type": "string",
                        "description": "Detailed information about the concept to reinforce understanding."
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["Easy", "Medium", "Hard"],
                        "description": "Difficulty level of the flashcard."
                    }
                },
                "required": ["question", "answer", "difficulty"]
            }
        },

        # 10 quiz questions (only when requested)
        "quiz": {
            "type": "array",
            "description": "A set of exactly 10 quiz questions for practice. Only include if user requests quiz.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The quiz question."
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple-choice options for the question (3-5 options)."
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "The correct answer from the provided options."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for why the correct answer is right."
                    }
                },
                "required": ["question", "options", "correct_answer", "explanation"]
            }
        },

        # Match the following section (only when requested)
        "match_the_following": {
            "type": "object",
            "description": "A 'match the following' exercise with two columns (A and B) and correct mappings. Only include if user requests this.",
            "properties": {
                "columnA": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of terms, definitions, or entities in column A."
                },
                "columnB": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of corresponding matches for items in column A."
                },
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "A": {
                                "type": "string",
                                "description": "Item from column A."
                            },
                            "B": {
                                "type": "string",
                                "description": "Correctly matched item from column B."
                            }
                        },
                        "required": ["A", "B"]
                    },
                    "description": "Array of correct pairings between column A and column B."
                }
            },
            "required": ["columnA", "columnB", "mappings"]
        },

        # Summary (always included)
        "summary": {
            "type": "string",
            "description": "Comprehensive, concise summary of all key concepts covered."
        },

        # Learning objectives (always included)
        "learning_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of learning objectives for the given topic."
        }
    },
    "required": ["metadata", "content_type", "summary", "learning_objectives"]
}



# Helper alias
SupabaseRow = Dict[str, Any]
ResultOrError = Union[SupabaseRow, List[SupabaseRow], None]


def handle_supabase_response(response: Any) -> ResultOrError:
    """
    Extract data from supabase response or raise error.
    """
    if hasattr(response, "error") and response.error is not None:
        raise RuntimeError(f"Supabase error: {response.error}")
    return response.data


# Pure/fn‐style CRUD operations

async def supabase_select(
    table: str,
    filters: Optional[Dict[str, Any]] = None,
    columns: Union[str, List[str]] = "*",
    single: bool = False
) -> ResultOrError:
    """
    Select records from Supabase table with optional filters.
    If single=True, return maybe_single() or single()
    """
    supabase = get_supabase_client()
    # Convert columns to string if it's a list
    columns_str = ",".join(columns) if isinstance(columns, list) else columns
    query = supabase.table(table).select(columns_str)
    if filters:
        for col, val in filters.items():
            query = query.eq(col, val)
    if single:
        query = query.maybe_single()
    resp = query.execute()
    return handle_supabase_response(resp)


async def supabase_insert(
    table: str,
    record: SupabaseRow,
    returning: str = "representation"
) -> ResultOrError:
    """
    Insert a single record, return inserted row(s).
    """
    supabase = get_supabase_client()
    resp = supabase.table(table).insert(record).execute()
    return handle_supabase_response(resp)


async def supabase_update(
    table: str,
    updates: SupabaseRow,
    filters: Dict[str, Any]
) -> ResultOrError:
    """
    Update records matching filters, return updated rows.
    """
    if not filters:
        raise ValueError("Update requires filters to prevent full‐table update")
    supabase = get_supabase_client()
    query = supabase.table(table).update(updates)
    for col, val in filters.items():
        query = query.eq(col, val)
    resp = query.execute()
    return handle_supabase_response(resp)


async def supabase_upsert(
    table: str,
    record: SupabaseRow,
    on_conflict: Optional[Union[str, List[str]]] = None,
    returning: str = "representation"
) -> ResultOrError:
    """
    Upsert record (insert or update if conflict).
    """
    kwargs = {}
    if on_conflict is not None:
        kwargs["on_conflict"] = on_conflict
    supabase = get_supabase_client()
    resp = supabase.table(table).upsert(record, **kwargs).execute()
    return handle_supabase_response(resp)


async def supabase_delete(
    table: str,
    filters: Dict[str, Any]
) -> ResultOrError:
    """
    Delete records matching filters.
    """
    if not filters:
        raise ValueError("Delete requires filters to prevent full‐table deletion")
    supabase = get_supabase_client()
    query = supabase.table(table).delete()
    for col, val in filters.items():
        query = query.eq(col, val)
    resp = query.execute()
    return handle_supabase_response(resp)


# Unified provider response wrapper

def format_learning_content_response(
    content: Dict[str, Any],
    schema: Type[BaseModel]
) -> Dict[str, Any]:
    """
    Validate content against LEARNING_CONTENT_SCHEMA, return a standard format, or raise error.
    Always returns a dict with:
      - status: "success" or "error"
      - data: validated content if success, else None
      - errors: validation errors if any
    """
    try:
        validated = schema.parse_obj(content)
        return {
            "status": "success",
            "data": validated.dict(),
            "errors": None
        }
    except ValidationError as ve:
        logger.error(f"Validation error formatting content: {ve}")
        return {
            "status": "error",
            "data": None,
            "errors": ve.errors()
        }

# Structured generation function (slightly refactored for purity)

# Embeddings function
def create_structured_prompt(context: str, query: str, content_type: str, schema: Type[BaseModel]) -> str:
    """Create a structured prompt for content generation."""
    
    return f"""
You are an expert educational content creator. Based on the provided context and query, generate structured learning content.

Context:
{context}

User Query: {query}

Content Type Requested: {content_type}

Make sure to:
1. Extract appropriate metadata (subject, chapter, concept names) from the content
2. Create educational content suitable for the requested type ({content_type})
3. Ensure difficulty level is appropriate based on content complexity
4. Provide clear, accurate educational content
5. Include comprehensive summaries and learning objectives

Focus on creating high-quality educational content that helps students learn effectively.
"""


async def google_gemini_generate(
    prompt: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    response_mime_type: str = "application/json"
) -> Dict[str, Any]:
    """Generate content using Google Gemini API."""
    genai.configure(api_key=api_key)  # type: ignore
    
    model = genai.GenerativeModel(model_name)  # type: ignore
    
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type=response_mime_type
        )
    )
    
    return {
        "content": response.text,
        "usage": {
            "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0
        }
    }


def get_embeddings_provider(provider_name: str, api_key: str):
    """Get embeddings function for specified provider."""
    config = PROVIDER_CONFIGS.get(provider_name)
    if not config or not config["embedding_handler"]:
        raise ValueError(f"Embeddings not supported for {provider_name}")
    
    return config["embedding_handler"](api_key)



# LLM Provider functions
def create_llm_agent(provider_name: str, api_key: str, model_name: Optional[str] = None) -> Agent:
    """Create an LLM agent for the specified provider."""
    config = PROVIDER_CONFIGS.get(provider_name)
    if not config:
        raise ValueError(f"Unsupported provider: {provider_name}")
    
    model = config["model_factory"](api_key, model_name)
    
    # Create agent with the model
    agent = Agent(model=model)
    return agent
   

async def generate_structured_response(
    context: str,
    query: str,
    user_id: str,
    schema: Type[BaseModel],
    provider_name: str = DEFAULT_PROVIDER,
    model_name: Optional[str] = None,
    content_type: str = "all"
) -> Dict[str, Any]:
    """
    Given context, query etc., generate content that fits the schema,
    then format it into a uniform response.
    """
    api_key = await get_user_api_key(user_id, provider_name)
    if not api_key:
        return {"status": "error", "error": f"No API key found for {provider_name}", "error_type": "missing_api_key"}

    prompt = create_structured_prompt(context, query, content_type, schema)
    raw_resp: Dict[str, Any]
    if provider_name in ("openai", "anthropic"):
        agent = create_llm_agent(provider_name, api_key, model_name)
        # Use PydanticAI properly with output_type
        agent = Agent(model=agent.model, output_type=schema)
        result = await agent.run(prompt)
        raw_resp = result.output.model_dump()
    elif provider_name == "google":
        resp = await google_gemini_generate(
            prompt,
            api_key,
            model_name or DEFAULT_MODEL,
            response_mime_type="application/json"
        )
        raw_resp = json.loads(resp["content"])
    else:
        return {"status": "error", "error": f"Unsupported provider {provider_name}", "error_type": "unsupported_provider"}

    return format_learning_content_response(raw_resp, schema)


# Non‐pure pieces (IO, orchestrator)

async def get_llm_response(
    uploaded_files_paths: List[str],
    userprompt: str,
    temp_dir: str,
    user_id: str,
    provider_name: str = DEFAULT_PROVIDER,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestration: document loading → context → generation.
    """
    # 1. Get API key
    api_key = await get_user_api_key(user_id, provider_name)
    if not api_key:
        return {"status": "error", "error": f"API key is required for {provider_name}", "error_type": "missing_api_key"}

    # 2. Load and process documents
    documents = await load_documents_from_files(uploaded_files_paths, temp_dir)
    chunks = await perform_document_chunking(documents)

    # 3. Setup embeddings and vector store  
    # Use the API key directly - the setup function will create embeddings internally
    embedding_api_key = api_key
    if provider_name not in ["google"]:
        # For non-Google providers, try to get a Google key for embeddings
        google_key = await get_user_api_key(user_id, "google")
        if google_key:
            embedding_api_key = google_key
        else:
            return {"status": "error", "error": "Google API key required for embeddings", "error_type": "missing_embeddings_provider"}

    vector_store, hybrid_retriever = await setup_vector_store_and_retriever(chunks, embedding_api_key)

    # 4. Retrieve & prepare context
    retrieved_docs = await hybrid_retriever.ainvoke(userprompt)
    enhanced_docs = enhance_retrieved_context(retrieved_docs, chunks)
    formatted_context = format_context_for_llm(enhanced_docs)

    # 5. Detect content type
    content_type = detect_content_type(userprompt)

    # 6. Generate structured response
    resp = await generate_structured_response(
        context=formatted_context,
        query=userprompt,
        user_id=user_id,
        schema=LearningContent,
        provider_name=provider_name,
        model_name=model_name,
        content_type=content_type
    )

    return resp


# Utility: get_user_api_key with pure signature
# Pure functions for database operations
async def get_user_api_key(user_id: str, provider_name: str) -> Optional[str]:
    """Get user's API key for a specific provider from Supabase."""
    try:
        supabase = get_supabase_client()
        response = supabase.table('user_api_keys') \
            .select('encrypted_api_key') \
            .eq('student_id', user_id) \
            .eq('provider_id', provider_name) \
            .eq('is_active', True) \
            .maybe_single() \
            .execute()
        
        if response and hasattr(response, 'data') and response.data:
            return response.data.get('encrypted_api_key')
        return None
    except Exception as e:
        logger.error(f"Error fetching API key for {provider_name}: {e}")
        return None
# Helper functions
def detect_content_type(query: str) -> str:
    """Detect content type from query."""
    query_lower = query.lower()
    content_types = {
        "flashcards": "flashcard" in query_lower,
        "quiz": "quiz" in query_lower,
        "match_the_following": any(x in query_lower for x in ["match", "matching"]),
        "summary": "summary" in query_lower
    }
    
    # Count true values
    true_count = sum(content_types.values())
    
    if true_count > 1:
        return "all"
    elif true_count == 1:
        return next(key for key, value in content_types.items() if value)
    else:
        return "all"

# API endpoints (FastAPI routers would use these functions)
async def add_api_key(user_id: str, provider_name: str, api_key: str, is_default: bool = False) -> bool:
    """Add API key for user."""
    try:
        supabase = get_supabase_client()
        response = supabase.table('user_api_keys').insert({
            'student_id': user_id,
            'provider_id': provider_name,
            'encrypted_api_key': api_key,  # Note: encrypt in production
            'is_default': is_default,
            'is_active': True
        }).execute()
        
        return bool(response.data)
    except Exception as e:
        logger.error(f"Error adding API key: {e}")
        return False

async def get_user_providers(user_id: str) -> List[Dict[str, Any]]:
    """Get all providers with API keys for user."""
    try:
        supabase = get_supabase_client()
        response = supabase.table('user_api_keys') \
            .select('provider_id, is_default') \
            .eq('student_id', user_id) \
            .eq('is_active', True) \
            .execute()
        
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error fetching user providers: {e}")
        return []

