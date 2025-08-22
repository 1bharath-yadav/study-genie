import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Core LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader
)

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore

# Retrieval and chains
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv

load_dotenv()

# Configuration
google_api_key = os.getenv("GEMINI_API_KEY")
astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
astra_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

logger = logging.getLogger("enhanced_rag_pipeline")
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 6
DEFAULT_ALPHA = 0.75

# JSON Schema for structured output
LEARNING_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "flashcards": {
            "type": "object",
            "patternProperties": {
                "^card\\d+$": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The flashcard question"},
                        "answer": {"type": "string", "description": "The flashcard answer"},
                        "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]}
                    },
                    "required": ["question", "answer", "difficulty"],
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "quiz": {
            "type": "object",
            "patternProperties": {
                "^Q\\d+$": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The quiz question"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 5,
                            "description": "Multiple choice options"
                        },
                        "correct_answer": {"type": "string", "description": "The correct answer"},
                        "explanation": {"type": "string", "description": "Explanation of the correct answer"}
                    },
                    "required": ["question", "options", "correct_answer", "explanation"],
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "summary": {
            "type": "string",
            "description": "Comprehensive summary of key concepts"
        },
        "learning_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of learning objectives"
        }
    },
    "required": ["flashcards", "quiz", "summary", "learning_objectives"],
    "additionalProperties": False
}


async def load_documents_from_files(file_paths: List[str], temp_dir: str) -> List[Document]:
    """
    Load documents from various file formats with enhanced error handling
    """
    all_documents = []

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


async def setup_vector_store_and_retriever(chunks: List[Document]) -> Tuple[AstraDBVectorStore, EnsembleRetriever]:
    """
    Setup vector store in AstraDB and create hybrid retriever
    """
    # Initialize Gemini embeddings
    from pydantic import SecretStr

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(
            google_api_key) if google_api_key is not None else None
    )

    # Create and populate AstraDB vector store
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="learning_documents",
        token=astra_token,
        api_endpoint=astra_api_endpoint,
    )

    # Add documents to vector store
    await vector_store.aadd_documents(chunks)
    logger.info(f"Stored {len(chunks)} chunks in AstraDB vector store")

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


def create_learning_content_prompt() -> ChatPromptTemplate:
    """
    Create structured prompt for learning content generation
    """
    system_prompt = """You are an expert educational content creator specializing in personalized learning materials.

Your task is to analyze the provided study material and create comprehensive learning content that helps students understand and retain the information effectively.

Guidelines:
- Create diverse flashcards covering key concepts, definitions, and applications
- Design quiz questions that test understanding at different cognitive levels
- Provide clear explanations that enhance learning
- Ensure content is academically rigorous yet accessible
- Focus on the most important concepts from the material

Generate content that promotes active learning and knowledge retention."""

    human_prompt = """Based on the following study materials, create comprehensive learning content:

STUDY MATERIALS:
{context}

USER QUERY: {query}

Please generate structured learning content including:
1. Flashcards for key concepts and definitions
2. Quiz questions with multiple choice answers and explanations
3. A comprehensive summary of the main topics
4. Clear learning objectives

Ensure all content is educationally sound and directly related to the provided materials."""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])


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
    Generate structured learning content using LangChain's structured output
    """
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=google_api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )

    # Create structured output chain using JSON schema
    structured_llm = llm.with_structured_output(
        schema=LEARNING_CONTENT_SCHEMA,
        method="json_mode"
    )

    # Create prompt template
    prompt = create_learning_content_prompt()

    # Build the chain
    chain = (
        {
            "context": RunnablePassthrough(),
            "query": RunnablePassthrough()
        }
        | prompt
        | structured_llm
    )

    # Execute chain with input

    response = await chain.ainvoke({
        "context": context,
        "query": query
    })

    logger.info("Successfully generated structured learning content")
    return response


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
