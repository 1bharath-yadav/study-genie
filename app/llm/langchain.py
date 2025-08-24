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

# Configuration
google_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=google_api_key)

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
        # Metadata extraction
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

        # 15 flashcards
        "flashcards": {
            "type": "array",
            "description": "A set of exactly 15 flashcards summarizing important key concepts.",
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

        # 10 quiz questions
        "quiz": {
            "type": "array",
            "description": "A set of exactly 10 quiz questions for practice, each with options, correct answers, and explanations.",
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

        # Match the following section
        "match_the_following": {
            "type": "object",
            "description": "A 'match the following' exercise with two columns (A and B) and correct mappings.",
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

        # Summary
        "summary": {
            "type": "string",
            "description": "Comprehensive, concise summary of all key concepts covered."
        },

        # Learning objectives
        "learning_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of learning objectives for the given topic."
        }
    },
    "required": ["metadata", "flashcards", "quiz", "match_the_following", "summary", "learning_objectives"]
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


async def setup_vector_store_and_retriever(chunks: List[Document]) -> Tuple[FAISS, EnsembleRetriever]:
    """
    Setup vector store using FAISS and create hybrid retriever
    """
    # Initialize Gemini embeddings
    from pydantic import SecretStr

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(
            google_api_key) if google_api_key is not None else None
    )

    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info(f"Stored {len(chunks)} chunks in FAISS vector store")

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


async def generate_structured_response(context: str, query: str) -> Dict[str, Any]:
    """
    Generate structured learning content using direct Gemini API with structured output
    """
    try:
        # Create the prompt
        prompt = f"""
        Based on the following educational content and user query, generate comprehensive study materials including flashcards, quiz questions, match-the-following exercises, and metadata.

        CONTENT:
        {context}

        USER QUERY:
        {query}

        Please extract the following information and generate study materials:
        1. Automatically identify the subject, chapter, and concept from the content
        2. Create exactly 15 flashcards covering key concepts
        3. Create exactly 10 quiz questions with multiple choice options
        4. Create a match-the-following exercise
        5. Provide a comprehensive summary
        6. List learning objectives
        7. Determine appropriate difficulty level

        Focus on creating high-quality educational content that helps with active learning and retention.
        """

        # Initialize Gemini model
        model = GenerativeModel('gemini-2.0-flash-exp')

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
        logger.info(
            "Successfully generated structured learning content using direct Gemini API")
        return result

    except Exception as e:
        logger.error(f"Error generating structured response: {str(e)}")
        raise


# Main pipeline function


async def get_llm_response(uploaded_files_paths: List[Path], userprompt: str, temp_dir: str) -> Dict[str, Any]:
    """
    Complete RAG pipeline for generating structured learning content
    """
    try:
        logger.info("Starting enhanced RAG pipeline...")

        # Step 1: Load documents from files
        logger.info("Loading documents from uploaded files...")
        documents = await load_documents_from_files([str(p) for p in uploaded_files_paths], temp_dir)

        # Step 2: Chunk documents for processing
        logger.info("Chunking documents for optimal processing...")
        chunks = await perform_document_chunking(documents)

        # Step 3: Setup vector store and hybrid retriever
        logger.info("Setting up vector store and hybrid retrieval system...")
        vector_store, hybrid_retriever = await setup_vector_store_and_retriever(chunks)

        # Step 4: Retrieve relevant content using hybrid search
        logger.info(f"Retrieving relevant content for query: '{userprompt}'")
        retrieved_docs = await hybrid_retriever.ainvoke(userprompt)

        # Step 5: Enhance with surrounding context
        enhanced_docs = await enhance_retrieved_context(retrieved_docs, chunks)

        # Step 6: Format context for LLM processing
        formatted_context = format_context_for_llm(enhanced_docs)

        # Step 7: Generate structured response
        logger.info("Generating structured learning content...")
        response = await generate_structured_response(formatted_context, userprompt)

        return response
    except Exception as e:
        logger.exception("Error in get_llm_response: %s", e)
        return {"status": "error", "error": str(e)}
# Utility functions for additional functionality


async def get_retrieval_metrics(retriever_response: List[Document]) -> Dict[str, Any]:
    """
    Calculate metrics about the retrieval process
    """
    return {
        "retrieved_chunks": len(retriever_response),
        "average_chunk_size": sum(len(doc.page_content) for doc in retriever_response) / len(retriever_response),
        "unique_sources": len(set(doc.metadata.get('source_file') for doc in retriever_response)),
        "content_types": list(set(doc.metadata.get('file_type') for doc in retriever_response))
    }


async def optimize_chunk_parameters(documents: List[Document]) -> Tuple[int, int]:
    """
    Dynamically optimize chunking parameters based on document characteristics
    """
    total_length = sum(len(doc.page_content) for doc in documents)
    avg_doc_length = total_length / len(documents) if documents else 0

    if avg_doc_length < 2000:
        return 800, 100  # Smaller chunks for short documents
    elif avg_doc_length > 10000:
        return 1500, 300  # Larger chunks for long documents
    else:
        return DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
