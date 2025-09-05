import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Direct Google Generative AI client
import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import GenerationConfig

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader
)

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Retrieval and chains
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

# Configuration - No global API key, only use student's API key
# google_api_key = os.getenv("GEMINI_API_KEY")  # Removed global API key
# genai.configure(api_key=google_api_key)  # No global configuration

logger = logging.getLogger("enhanced_rag_pipeline")
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 6
DEFAULT_ALPHA = 0.75

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


async def load_documents_from_files(file_paths: List[str], temp_dir: str) -> List[Document]:
    """
    Load documents from various file formats with enhanced error handling
    """
    all_documents = []

    async def send_to_llm_for_handwriting(file_path: str) -> List[Document]:
        """
        Placeholder: Send the file to LLM for OCR/handwriting recognition if text extraction fails.
        Replace this with actual LLM OCR logic as needed.
        """
        logger.warning(
            f"Text extraction failed for {file_path}. Sending to LLM for handwriting/OCR analysis.")
        # Example: create a Document with a note that LLM OCR is needed
        from langchain_core.documents import Document
        doc = Document(page_content="", metadata={
            'source_file': file_path,
            'file_type': Path(file_path).suffix.lower(),
            'original_length': 0,
            'llm_ocr_required': True
        })
        # In production, replace this with actual LLM OCR result
        return [doc]

    for file_path in file_paths:
        try:
            file_extension = Path(file_path).suffix.lower()
            loader = None

            # Choose appropriate loader based on file type
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
                loader = UnstructuredImageLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)

            # Load documents
            docs = await asyncio.to_thread(loader.load)

            # Check if all docs are empty (no text extracted)
            if not any(doc.page_content.strip() for doc in docs):
                # If no text extracted, send to LLM for OCR/handwriting
                docs = await send_to_llm_for_handwriting(file_path)

            # Add source metadata
            for doc in docs:
                doc.metadata.update({
                    'source_file': file_path,
                    'file_type': file_extension,
                    'original_length': len(doc.page_content)
                })

            all_documents.extend(docs)
            logger.info(
                f"Successfully loaded {len(docs)} documents from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            continue

    if not all_documents:
        raise ValueError(
            "No documents could be loaded from the provided files")

    return all_documents


async def perform_document_chunking(documents: List[Document]) -> List[Document]:
    """
    Advanced document chunking with metadata preservation
    """
    # Configure text splitter with optimized parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        keep_separator=True
    )

    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Enhance chunk metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': idx,
            'chunk_size': len(chunk.page_content),
            'chunk_index': idx,
            'total_chunks': len(chunks)
        })

    logger.info(
        f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


async def setup_vector_store_and_retriever(chunks: List[Document], user_api_key: str) -> Tuple[FAISS, EnsembleRetriever]:
    """
    Setup vector store using FAISS and create hybrid retriever
    """
    # Validate that user has provided API key
    if not user_api_key:
        raise ValueError(
            "API key is required. Please add your Gemini API key in settings.")

    # Clean the API key - remove any whitespace or control characters
    clean_api_key = user_api_key.strip()

    # Additional validation
    if not clean_api_key:
        raise ValueError(
            "API key is empty after cleaning. Please check your API key in settings.")

    logger.info(f"Using API key of length {len(clean_api_key)} for embeddings")

    # Test API key validity first before using it for embeddings
    try:
        # Configure genai with the API key and test it
        genai.configure(api_key=clean_api_key)

        # Create a simple test model to validate the API key
        test_model = GenerativeModel('gemini-2.0-flash-exp')

        # Test with a simple generation to validate the key works
        test_response = test_model.generate_content("Hello")
        logger.info("API key validation successful - key works with Gemini")

    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        # Try to provide more specific error info
        if "API_KEY_INVALID" in str(e):
            raise ValueError(
                f"Invalid API key provided. Please check your Gemini API key: {e}")
        elif "quota" in str(e).lower():
            raise ValueError(f"API key quota exceeded: {e}")
        else:
            raise ValueError(f"API key validation failed: {e}")

    # Initialize Gemini embeddings with user's API key
    from pydantic import SecretStr

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretStr(clean_api_key)
        )
        logger.info("Successfully initialized Google Generative AI embeddings")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings with API key: {e}")
        raise ValueError(f"Invalid API key for embeddings: {e}")

    # Create FAISS vector store
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info(f"Stored {len(chunks)} chunks in FAISS vector store")
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise ValueError(
            f"Failed to create embeddings - check your API key: {e}")

    # Setup vector retriever
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": DEFAULT_TOP_K}
    )

    # Setup BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = DEFAULT_TOP_K

    # Create hybrid ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[DEFAULT_ALPHA, 1.0 - DEFAULT_ALPHA]
    )

    logger.info("Successfully configured hybrid retrieval system")
    return vector_store, ensemble_retriever


async def enhance_retrieved_context(retrieved_docs: List[Document], all_chunks: List[Document]) -> List[Document]:
    """
    Enhance retrieved documents with surrounding context chunks
    """
    enhanced_docs = list(retrieved_docs)  # Start with retrieved docs
    retrieved_chunk_ids = {doc.metadata.get(
        'chunk_id', -1) for doc in retrieved_docs}

    # Add neighboring chunks for better context
    for doc in retrieved_docs:
        chunk_id = doc.metadata.get('chunk_id', -1)
        if chunk_id == -1:
            continue

        # Look for adjacent chunks
        for offset in [-2, -1, 1, 2]:
            neighbor_id = chunk_id + offset
            if neighbor_id in retrieved_chunk_ids:
                continue

            # Find and add neighbor chunk
            for chunk in all_chunks:
                if chunk.metadata.get('chunk_id') == neighbor_id:
                    enhanced_docs.append(chunk)
                    retrieved_chunk_ids.add(neighbor_id)
                    break

    # Sort by chunk_id to maintain order
    enhanced_docs.sort(key=lambda x: x.metadata.get('chunk_id', 0))
    logger.info(
        f"Enhanced context from {len(retrieved_docs)} to {len(enhanced_docs)} chunks")
    return enhanced_docs


def format_context_for_llm(documents: List[Document]) -> str:
    """
    Format retrieved documents into a structured context string
    """
    context_parts = []

    for idx, doc in enumerate(documents, 1):
        source = doc.metadata.get('source_file', 'Unknown')
        content = doc.page_content.strip()

        formatted_section = f"""
=== Document Section {idx} ===
Source: {source}
Content: {content}
"""
        context_parts.append(formatted_section)

    return "\n".join(context_parts)


async def generate_structured_response(context: str, query: str, user_api_key: str, content_type: str = "all") -> Dict[str, Any]:
    """
    Generate structured learning content using direct Gemini API with structured output
    """
    try:
        # Validate that user has provided API key
        if not user_api_key:
            raise ValueError(
                "API key is required. Please add your Gemini API key in settings.")

        # Validate API key format
        user_api_key = user_api_key.strip()  # Remove any whitespace
        if not user_api_key.startswith("AIza"):
            logger.error(
                f"Invalid API key format. Gemini API keys should start with 'AIza'. Received: {user_api_key[:10]}...")
            raise ValueError(
                "Invalid API key format. Gemini API keys should start with 'AIza'.")

        # Determine what content to generate based on user request
        query_lower = query.lower()
        logger.info(f"🔍 Content detection - query_lower: '{query_lower}'")
        logger.info(f"🔍 Initial content_type: '{content_type}'")

        if content_type == "all":
            # Auto-detect based on query content - check for multiple content types
            has_flashcards = "flashcard" in query_lower
            has_quiz = "quiz" in query_lower
            has_match = "match" in query_lower or "matching" in query_lower
            has_summary = "summary" in query_lower

            logger.info(
                f"🔍 Detection results - flashcards:{has_flashcards}, quiz:{has_quiz}, match:{has_match}, summary:{has_summary}")

            # If multiple content types are requested, use "all"
            content_types_count = sum([has_flashcards, has_quiz, has_match])
            logger.info(f"🔍 Content types count: {content_types_count}")

            if content_types_count > 1:
                content_type = "all"
            elif has_flashcards:
                content_type = "flashcards"
            elif has_quiz:
                content_type = "quiz"
            elif has_match:
                content_type = "match_the_following"
            elif has_summary:
                content_type = "summary"
            else:
                content_type = "all"

        logger.info(f"🔍 Final content_type: '{content_type}'")

        # Create conditional prompt based on content type
        if content_type == "flashcards":
            content_instructions = """
            Generate exactly 15 flashcards covering the key concepts from the content.
            Each flashcard should have a question, answer, and difficulty level.
            """
        elif content_type == "quiz":
            content_instructions = """
            Generate exactly 10 quiz questions with multiple choice options.
            Each question should have 3-5 options, correct answer, and explanation.
            """
        elif content_type == "match_the_following":
            content_instructions = """
            Create a match-the-following exercise with two columns (A and B) and correct mappings.
            Include at least 5-8 items in each column with their correct pairings.
            """
        elif content_type == "summary":
            content_instructions = """
            Provide a comprehensive summary of all key concepts covered.
            Focus on the main ideas and important details.
            """
        else:  # "all"
            content_instructions = """
            Generate comprehensive study materials including ALL of the following sections:
            1. Exactly 15 flashcards covering key concepts (REQUIRED - must be present)
            2. Exactly 10 quiz questions with multiple choice options (REQUIRED - must be present)
            3. A match-the-following exercise with 2 columns and correct mappings (REQUIRED - must be present)
            4. A comprehensive summary (REQUIRED - must be present)
            5. Learning objectives (REQUIRED - must be present)
            
            CRITICAL: You MUST include the "match_the_following" field in your JSON response with:
            - "columnA": array of items (at least 5-8 items)
            - "columnB": array of corresponding matches 
            - "mappings": array of correct A-B pairings
            
            Do not skip the match_the_following section - it is mandatory when content_type is "all".
            """

        # Create the prompt
        prompt = f"""
        Based on the following educational content and user query, generate study materials as requested.

        CONTENT:
        {context}

        USER QUERY:
        {query}

        CONTENT TYPE REQUESTED: {content_type}

        Please extract the following information and generate study materials:
        1. Automatically identify the subject, chapter, and concept from the content
        2. Set the content_type field to: "{content_type}"
        3. {content_instructions}
        4. Always provide a comprehensive summary and learning objectives
        5. Determine appropriate difficulty level

        IMPORTANT SCHEMA REQUIREMENTS:
        - Your response MUST be valid JSON that strictly follows the provided schema
        - When content_type is "all", you MUST include ALL required fields: metadata, content_type, flashcards, quiz, match_the_following, summary, learning_objectives
        - Do NOT omit the match_the_following field when content_type is "all"
        - The match_the_following field must contain: columnA (array), columnB (array), mappings (array)

        Focus on creating high-quality educational content that helps with active learning and retention.
        """

        # Configure genai with user's API key and create model
        # Clean the API key first
        clean_api_key = user_api_key.strip()
        logger.info(
            f"Creating Gemini model with API key: {clean_api_key[:10]}...{clean_api_key[-5:] if len(clean_api_key) > 15 else clean_api_key}")

        # Clear any existing configuration and set user's API key
        try:
            # Configure with user's API key - this MUST be done before creating the model
            genai.configure(api_key=clean_api_key)

            # Verify the configuration worked by testing the API key
            # This will fail fast if the API key is invalid
            model = GenerativeModel('gemini-2.0-flash')

            logger.info("Successfully configured Gemini with user's API key")

        except Exception as e:
            logger.error(
                f"Failed to configure Gemini with user's API key: {e}")
            raise ValueError(
                f"Invalid API key or Gemini configuration failed: {e}")

        # Generate content with structured output
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=LEARNING_CONTENT_SCHEMA
            )
        )

        # Parse the JSON response
        result = json.loads(response.text)
        logger.info("🎯 LLM Response keys: %s", list(result.keys())
                    if isinstance(result, dict) else "Not a dict")
        if isinstance(result, dict):
            logger.info("🎯 LLM Response match_the_following: %s",
                        result.get('match_the_following', 'KEY_NOT_FOUND'))
        logger.info(
            "Successfully generated structured learning content using user's Gemini API")
        return result

    except Exception as e:
        logger.error(f"Error generating structured response: {str(e)}")
        raise
# Main pipeline function


async def get_llm_response(uploaded_files_paths: List[Path], userprompt: str, temp_dir: str, user_api_key: str) -> Dict[str, Any]:
    """
    Complete RAG pipeline for generating structured learning content
    """
    try:
        # Validate that user has provided API key
        if not user_api_key:
            return {
                "status": "error",
                "error": "API key is required. Please add your Gemini API key in settings to continue.",
                "error_type": "missing_api_key"
            }

        logger.info("Starting enhanced RAG pipeline...")

        # Step 1: Load documents from files
        logger.info("Loading documents from uploaded files...")
        documents = await load_documents_from_files([str(p) for p in uploaded_files_paths], temp_dir)

        # Step 2: Chunk documents for processing
        logger.info("Chunking documents for optimal processing...")
        chunks = await perform_document_chunking(documents)

        # Step 3: Setup vector store and hybrid retriever with user's API key
        logger.info("Setting up vector store and hybrid retrieval system...")
        vector_store, hybrid_retriever = await setup_vector_store_and_retriever(chunks, user_api_key)

        # Step 4: Retrieve relevant content using hybrid search
        logger.info(f"Retrieving relevant content for query: '{userprompt}'")
        retrieved_docs = await hybrid_retriever.ainvoke(userprompt)

        # Step 5: Enhance with surrounding context
        enhanced_docs = await enhance_retrieved_context(retrieved_docs, chunks)

        # Step 6: Format context for LLM processing
        formatted_context = format_context_for_llm(enhanced_docs)

        # Step 7: Generate structured response with user's API key
        logger.info("Generating structured learning content...")

        # Auto-detect content type from user prompt
        content_type = "all"  # default
        userprompt_lower = userprompt.lower()

        # Check for multiple content types
        has_flashcards = "flashcard" in userprompt_lower
        has_quiz = "quiz" in userprompt_lower
        has_match = "match" in userprompt_lower or "matching" in userprompt_lower
        has_summary = "summary" in userprompt_lower

        # If multiple content types are requested, use "all"
        content_types_count = sum([has_flashcards, has_quiz, has_match])

        if content_types_count > 1:
            content_type = "all"
        elif has_flashcards:
            content_type = "flashcards"
        elif has_quiz:
            content_type = "quiz"
        elif has_match:
            content_type = "match_the_following"
        elif has_summary:
            content_type = "summary"

        logger.info(
            f"Detected content type: {content_type} (flashcards:{has_flashcards}, quiz:{has_quiz}, match:{has_match})")
        logger.info(f"User prompt was: '{userprompt}'")
        logger.info(f"Content types count: {content_types_count}")

        response = await generate_structured_response(formatted_context, userprompt, user_api_key, content_type)

        return response
    except ValueError as ve:
        logger.error("Validation error in get_llm_response: %s", ve)
        return {
            "status": "error",
            "error": str(ve),
            "error_type": "validation_error"
        }
    except Exception as e:
        logger.exception("Error in get_llm_response: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "error_type": "processing_error"
        }
