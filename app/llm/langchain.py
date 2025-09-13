import asyncio
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Direct Google Generative AI client
# We no longer use the direct Google Generative AI client for chat generation.
# Generation is done via per-user provider agents (pydantic-ai). Keep provider-specific
# embedding implementations where available.

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

# Retrieval and chains
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from dotenv import load_dotenv
from pydantic_ai import StructuredDict

from app.llm.types import LEARNING_CONTENT_SCHEMA

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


def get_embeddings(provider: str, model: str, api_key: str | None = None, **kwargs):
    """
    Return a LangChain Embeddings instance for the given provider and model.
    """
    provider_l = (provider or "").lower()

    if provider_l == "google":
        from pydantic import SecretStr
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if not api_key:
            raise ValueError("Google embeddings require an API key")
        mapped = model if model.startswith("models/") else f"models/{model}"
        return GoogleGenerativeAIEmbeddings(model=mapped, google_api_key=SecretStr(api_key))

    if provider_l == "openai":
        try:
            from langchain.embeddings import OpenAIEmbeddings
        except Exception:
            from langchain.embeddings import OpenAIEmbeddings
        params: Dict[str, Any] = {}
        if api_key:
            params["openai_api_key"] = api_key
        params.update(kwargs)
        return OpenAIEmbeddings(model=model, **params)

    if provider_l == "cohere":
        from pydantic import SecretStr
        from langchain_cohere import CohereEmbeddings
        key_secret = SecretStr(api_key) if api_key else None
        return CohereEmbeddings(model=model, cohere_api_key=key_secret, **kwargs)

    if provider_l == "bedrock":
        from langchain_community.embeddings import BedrockEmbeddings
        mapped = model if ":" in model else f"{model}:0"
        region_name = kwargs.pop("region_name", None)
        return BedrockEmbeddings(model_id=mapped, region_name=region_name, **kwargs)

    if provider_l == "mistral":
        from pydantic import SecretStr
        from langchain_mistralai import MistralAIEmbeddings
        if not api_key:
            raise ValueError("Mistral embeddings require an API key")
        key_secret = SecretStr(api_key)
        return MistralAIEmbeddings(model=model, api_key=key_secret, **kwargs)

    if provider_l == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model_kwargs = kwargs.pop("model_kwargs", {"trust_remote_code": True})
        encode_kwargs = kwargs.pop("encode_kwargs", {"normalize_embeddings": True})
        return HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, **kwargs)

    raise ValueError(f"Unsupported provider for embeddings: {provider}")


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


async def setup_vector_store_and_retriever(chunks: List[Document], embedding_provider: str, embedding_model: str, embedding_api_key: str | None) -> Tuple[FAISS, EnsembleRetriever]:
    """
    Setup vector store using FAISS and create hybrid retriever.
    embedding_provider/model/key are chosen from per-user model preferences.
    """
    if not embedding_provider or not embedding_model:
        raise ValueError("Embedding provider and model must be provided")

    key = embedding_api_key.strip() if embedding_api_key else None
    if not key:
        logger.debug("No explicit embedding API key provided; factory may still initialize embeddings if supported by env")

    # Create embeddings via factory
    try:
        embeddings = get_embeddings(embedding_provider, embedding_model, key)
        logger.info("Initialized embeddings for provider=%s model=%s", embedding_provider, embedding_model)
    except Exception as e:
        logger.error("Failed to initialize embeddings for provider=%s model=%s: %s", embedding_provider, embedding_model, e)
        raise ValueError(f"Failed to initialize embeddings: {e}")

    # Create FAISS vector store
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Stored %d chunks in FAISS vector store", len(chunks))
    except Exception as e:
        logger.error("Failed to create vector store: %s", e)
        raise ValueError(f"Failed to create embeddings - check your API key or model: {e}")

    vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": DEFAULT_TOP_K})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = DEFAULT_TOP_K

    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[DEFAULT_ALPHA, 1.0 - DEFAULT_ALPHA])

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

logger = logging.getLogger(__name__)

# ----- Pure helpers -----

def detect_content_type(query: str, content_type: str) -> str:
    q = (query or "").lower().strip()
    if content_type != "all":
        return content_type or "all"

    has_flashcards = "flashcard" in q
    has_quiz = "quiz" in q
    has_match = "match" in q or "matching" in q
    has_summary = "summary" in q

    count = sum([has_flashcards, has_quiz, has_match])
    if count > 1:
        return "all"
    if has_flashcards:
        return "flashcards"
    if has_quiz:
        return "quiz"
    if has_match:
        return "match_the_following"
    if has_summary:
        return "summary"
    return "all"


def instructions_for(ct: str) -> str:
    return {
        "flashcards": """
Generate exactly 15 flashcards covering the key concepts from the content.
Each flashcard should have a question, answer, and difficulty level.
""",
        "quiz": """
Generate exactly 10 quiz questions with multiple choice options.
Each question should have 3-5 options, correct answer, and explanation.
""",
        "match_the_following": """
Create a match-the-following exercise with two columns (A and B) and correct mappings.
Include at least 5-8 items in each column with their correct pairings.
""",
        "summary": """
Provide a comprehensive summary of all key concepts covered.
Focus on the main ideas and important details.
""",
        "all": """
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
""",
    }.get(ct, "")


def build_prompt(context: str, query: str, ct: str, content_instructions: str) -> str:
    return f"""
Based on the following educational content and user query, generate study materials as requested.

CONTENT:
{context}

USER QUERY:
{query}

CONTENT TYPE REQUESTED: {ct}

Please extract the following information and generate study materials:
1. Automatically identify the subject, chapter, and concept from the content
2. Set the content_type field to: "{ct}"
3. {content_instructions}
4. Always provide a comprehensive summary and learning objectives
5. Determine appropriate difficulty level

IMPORTANT SCHEMA REQUIREMENTS:
- Your response MUST be valid JSON that strictly follows the provided schema
- When content_type is "all", you MUST include ALL required fields: metadata, content_type, flashcards, quiz, match_the_following, summary, learning_objectives
- Do NOT omit the match_the_following field when content_type is "all"
- The match_the_following field must contain: columnA (array), columnB (array), mappings (array)

Focus on creating high-quality educational content that helps with active learning and retention.

Return the response as JSON conforming to the LearningContent Pydantic model.
""".strip()


# ----- Main function -----

async def generate_structured_response(
    context: str,
    query: str,
    provider: str,
    model_name: str,
    api_key: str,
    content_type: str = "all",
):
    """
    Generate structured learning content using a PydanticAI Agent with output_type=LearningContent.
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key is required for the provider agent.")

    from app.llm.providers import create_learning_agent
    # Decide content type and build prompt (pure functions)
    final_ct = detect_content_type(query, content_type)
    prompt = build_prompt(context, query, final_ct, instructions_for(final_ct))

    logger.info("Creating learning agent for provider=%s model=%s", provider, model_name)

    try:
        agent = await create_learning_agent(provider, model_name, api_key)
    except Exception as e:
        logger.error("Failed to create learning agent: %s", e)
        raise ValueError(f"Failed to initialize model/provider: {e}")

    # Correct PydanticAI call — no invoke/ainvoke
    logger.info(f"model_prompt{prompt}")
    # Non-streaming invocation: use the async context manager and read final output
    async with agent.run_stream(prompt, output_type=StructuredDict(LEARNING_CONTENT_SCHEMA)) as res:
        llm_response = getattr(res, 'output', None)
        # If output is not present, try to collect streamed output into one dict
        if llm_response is None:
            collected = None
            try:
                # Attempt to consume stream_output to get final assembled result
                async for partial in res.stream_output(debounce_by=0.01):
                    collected = partial
                llm_response = collected
            except Exception:
                llm_response = None
    with open("./../../llm_response.exampl", "a") as f:
        f.write(str(llm_response) + "\n")

    logger.info(f"llm_response{llm_response}")

    # Return JSON-compatible dict regardless of pydantic version
    return llm_response


async def generate_structured_response_stream(
    context: str,
    query: str,
    provider: str,
    model_name: str,
    api_key: str,
    content_type: str = "all",
):
    """
    Async generator that yields partial, validated structured outputs from Pydantic-AI as they stream.
    Each yielded item is a Python object (dict/TypedDict) representing the partial validated data.
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key is required for the provider.")

    from app.llm.providers import create_learning_agent

    final_ct = detect_content_type(query, content_type)
    prompt = build_prompt(context, query, final_ct, instructions_for(final_ct))

    try:
        agent = await create_learning_agent(provider, model_name, api_key)
    except Exception as e:
        logger.error("Failed to create learning agent for streaming: %s", e)
        raise ValueError(f"Failed to initialize model/provider for streaming: {e}")

    logger.info("Starting streaming structured response")
    # Use the agent as an async context manager to stream partial validated output
    async with agent.run_stream(prompt, output_type=StructuredDict(LEARNING_CONTENT_SCHEMA)) as result:
        # `result.stream_output()` yields partial validated chunks (typed)
        async for partial in result.stream_output(debounce_by=0.01):
            # Each `partial` is already a validated Python object (TypedDict or Pydantic model)
            try:
                yield partial
            except Exception:
                # If serialization/processing fails for a partial, continue streaming
                logger.exception("Failed to yield partial output, skipping chunk")


async def get_llm_response_stream(uploaded_files_paths: List[Path], userprompt: str, temp_dir: str, user_api_key: str, user_id: str, provider_name: str, model_name: str):
    """
    Streamed variant of get_llm_response that yields partial structured outputs as they are produced.
    This mirrors the synchronous pipeline but returns an async generator of partial outputs.
    """
    if not user_api_key:
        raise ValueError("API key is required. Please add your Gemini API key in settings to continue.")

    # Reuse same pipeline: load docs, chunk, setup retriever, retrieve, enhance, format
    documents = await load_documents_from_files([str(p) for p in uploaded_files_paths], temp_dir)
    chunks = await perform_document_chunking(documents)

    from app.llm.providers import get_user_model_preferences
    from app.services.api_key_service import get_api_key_for_provider

    embedding_provider = provider_name
    embedding_model = model_name
    try:
        prefs = get_user_model_preferences(user_id)
        emb_pref = None
        for p in prefs:
            if p and p.get("use_for_embedding"):
                emb_pref = p
                break
        if emb_pref:
            mid = emb_pref.get("model_id")
            prov = emb_pref.get("provider_name") or None
            if mid and "-" in mid:
                parts = mid.split("-", 1)
                embedding_provider = prov or parts[0]
                embedding_model = parts[1]
            else:
                embedding_provider = prov or embedding_provider
                embedding_model = mid or embedding_model
    except Exception:
        logger.debug("Failed to read embedding preferences; falling back to chat model")

    try:
        embedding_api_key = await get_api_key_for_provider(user_id, embedding_provider)
    except Exception:
        embedding_api_key = None

    vector_store, hybrid_retriever = await setup_vector_store_and_retriever(chunks, embedding_provider, embedding_model, embedding_api_key)
    retrieved_docs = await hybrid_retriever.ainvoke(userprompt)
    enhanced_docs = await enhance_retrieved_context(retrieved_docs, chunks)
    formatted_context = format_context_for_llm(enhanced_docs)

    # Now call the streaming generator and yield partials
    async for partial in generate_structured_response_stream(formatted_context, userprompt, provider_name, model_name, user_api_key):
        yield partial


async def get_llm_response(uploaded_files_paths: List[Path], userprompt: str, temp_dir: str, user_api_key: str, user_id: str, provider_name: str, model_name: str) -> Dict[str, Any]:
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

        # Step 3: Determine embedding preference (per-user) and setup vector store
        logger.info("Determining embedding model preference for user and configuring retrieval...")
        from app.llm.providers import get_user_model_preferences
        from app.services.api_key_service import get_api_key_for_provider

        embedding_provider = provider_name
        embedding_model = model_name
        # Try to find an explicit embedding preference
        try:
            prefs = get_user_model_preferences(user_id)
            emb_pref = None
            for p in prefs:
                if p and p.get("use_for_embedding"):
                    emb_pref = p
                    break
            if emb_pref:
                mid = emb_pref.get("model_id")
                prov = emb_pref.get("provider_name") or None
                if mid and "-" in mid:
                    parts = mid.split("-", 1)
                    embedding_provider = prov or parts[0]
                    embedding_model = parts[1]
                else:
                    # fallback to provided provider_name if available
                    embedding_provider = prov or embedding_provider
                    embedding_model = mid or embedding_model
        except Exception:
            logger.debug("Failed to read embedding preferences; falling back to chat model")

        # Fetch embedding API key for the embedding provider (may be None)
        try:
            embedding_api_key = await get_api_key_for_provider(user_id, embedding_provider)
        except Exception:
            embedding_api_key = None

        logger.info("Using embedding provider=%s model=%s (key present=%s)", embedding_provider, embedding_model, bool(embedding_api_key))
        vector_store, hybrid_retriever = await setup_vector_store_and_retriever(chunks, embedding_provider, embedding_model, embedding_api_key)

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
        response = await generate_structured_response(formatted_context, userprompt, provider_name, model_name, user_api_key, content_type)
        # Ensure we always return a dict
        return response if isinstance(response, dict) else (response or {})
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
