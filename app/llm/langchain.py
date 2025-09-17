from __future__ import annotations

import logging
from typing import AsyncGenerator, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
from typing import Optional
import faiss
# Direct Google Generative AI client
# We no longer use the direct Google Generative AI client for chat generation.
# Generation is done via per-user provider agents (pydantic-ai). Keep provider-specific
# embedding implementations where available.

# Document loaders are imported lazily inside helper functions to keep optional deps local
from app.llm.providers import create_learning_agent

# Text processing
import uuid
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from app.llm.process_files.process_files import (
    load_documents_from_files,
    perform_document_chunking,
)

from app.llm.types import LEARNING_CONTENT_SCHEMA

# embedding factory helper placeholder. Providers module may expose a get_embeddings factory.
# If not present, get_embeddings remains None and setup_vector_store_and_retriever will raise at runtime.
get_embeddings = None

# retrieval defaults
DEFAULT_TOP_K = 4
DEFAULT_ALPHA = 0.7

# (imports above are sufficient)


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

    # Create embeddings via factory (providers should expose `get_embeddings`)
    if not callable(get_embeddings):
        raise ValueError("Embedding factory `get_embeddings` is not available. Ensure app.llm.providers exposes get_embeddings for provider/model.")
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


def _faiss_session_dir(base_dir: str, session_id: str) -> Path:
    p = Path(base_dir) / f"faiss_{session_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_vector_store_and_chunks(vector_store: FAISS, chunks: List[Document], base_dir: str, session_id: str) -> None:
    """Persist FAISS index and serialized chunks to disk under base_dir/faiss_{session_id}.

    This allows subsequent requests to reuse the vector index without recomputing embeddings.
    """
    try:
        # If base_dir omitted or points to a system temp, prefer configured persistent dir
        from app.config import settings
        base = base_dir or getattr(settings, 'PERSIST_DIR', None) or base_dir
        session_path = _faiss_session_dir(base, session_id)
        # Save FAISS index and also write a raw faiss index file for mmap loading
        try:
            vector_store.save_local(str(session_path))
        except Exception:
            # Best-effort; continue to write raw faiss index below
            pass

        # If the FAISS wrapper has an underlying index, also write it as a raw file for mmap reads
        try:
            idx = getattr(vector_store, 'index', None)
            if idx is not None:
                idx_path = str(session_path / 'index.faiss')
                faiss.write_index(idx, idx_path)
        except Exception as e:
            logger.debug(f"Could not write raw faiss index for session={session_id}: {e}")

        # Serialize chunks (page_content + metadata) to a JSON file
        serial = []
        for d in chunks:
            serial.append({
                'page_content': d.page_content,
                'metadata': d.metadata or {}
            })
        with open(session_path / 'chunks.json', 'w', encoding='utf-8') as fh:
            json.dump(serial, fh)
        logger.info(f"Saved FAISS store and {len(chunks)} chunks for session={session_id} at {session_path}")
    except Exception as e:
        logger.warning(f"Failed to persist FAISS store for session={session_id}: {e}")


def load_vector_store_and_chunks(base_dir: str, session_id: str, embeddings) -> Tuple[Optional[FAISS], Optional[List[Document]]]:
    """Attempt to load a persisted FAISS store and chunks for a session. Returns (vector_store, chunks) or (None, None) on miss.
    """
    try:
        from app.config import settings
        base = base_dir or getattr(settings, 'PERSIST_DIR', None) or base_dir
        session_path = Path(base) / f"faiss_{session_id}"
        if not session_path.exists():
            return None, None

        # Load FAISS index. Prefer memory-mapped raw index if available for lower RAM usage.
        vs = None
        try:
            idx_path = session_path / 'index.faiss'
            if idx_path.exists():
                # Memory-map the raw FAISS index to warm the OS page cache and allow read-only mmap access.
                try:
                    flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                    _ = faiss.read_index(str(idx_path), flags)
                    logger.info(f"Memory-mapped raw FAISS index for session={session_id} to warm OS cache")
                except Exception as _e:
                    logger.debug(f"Failed to mmap raw faiss index for session={session_id}: {_e}")

                # Construct LangChain FAISS wrapper from the saved local directory (this should read auxiliary files)
                vs = FAISS.load_local(str(session_path), embeddings)
            else:
                # Fallback to LangChain's saved directory format
                vs = FAISS.load_local(str(session_path), embeddings)
        except Exception as e:
            logger.warning(f"Failed to load FAISS index for session={session_id}: {e}")
            return None, None

        # Load chunks
        chunks_file = session_path / 'chunks.json'
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as fh:
                serial = json.load(fh)
            docs = []
            for item in serial:
                docs.append(Document(page_content=item.get('page_content', ''), metadata=item.get('metadata', {})))
        else:
            docs = None

        logger.info(f"Loaded FAISS store and {len(docs) if docs else 0} chunks for session={session_id} from {session_path}")
        return vs, docs
    except Exception as e:
        logger.warning(f"Error loading persisted FAISS store for session={session_id}: {e}")
        return None, None


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




def build_prompt(context: str, query: str, ct: str,) -> str:
    return f"""
Based on the following educational content and user query, generate study materials as requested.

CONTENT:
{context}

USER QUERY:
{query}

CONTENT TYPE REQUESTED: {ct}

Respect user query generate content on the basis of what he asked.

Please extract the following information and generate study materials:
1. Automatically identify the subject, chapter, and concept from the content
2. Set the content_type field to: "{ct}"
4. Always provide a comprehensive summary and learning objectives
5. Determine appropriate difficulty level

IMPORTANT SCHEMA REQUIREMENTS:
- Your response MUST be valid JSON that strictly follows the provided schema
- When content_type is "all", you MUST include ALL required fields: metadata, content_type, flashcards, quiz, match_the_following, summary, learning_objectives
- Do NOT omit the match_the_following field when content_type is "all"
- The match_the_following field must contain: columnA (array), columnB (array), mappings (array)

Focus on creating high-quality educational content that helps with active learning and retention.

""".strip()


# ----- Main function -----

"""
Refactored RAG pipeline with streaming responses and functional programming principles
"""


logger = logging.getLogger(__name__)


# ----- Pure Functions -----
"""
Refactored RAG pipeline with streaming responses and functional programming principles
"""
# ----- Pure Functions -----


def should_skip_embedding(combined_text: str, threshold: int = 1300) -> bool:
    """Pure function to determine if embedding should be skipped."""
    return len(combined_text.split()) < threshold





def create_session_name(metadata: Dict[str, Any]) -> Optional[str]:
    """Pure function to create session name from metadata."""
    if not isinstance(metadata, dict):
        return None
        
    subject = (metadata.get('subject_name') or metadata.get('subject') or '').strip()
    chapter = (metadata.get('chapter_name') or metadata.get('chapter') or '').strip()
    concept = (metadata.get('concept_name') or metadata.get('concept') or '').strip()
    
    if subject and chapter and concept:
        return f"{subject}-{chapter}-{concept}"
    elif subject and concept:
        return f"{subject}-{concept}"
    elif subject:
        return subject
    elif concept:
        return concept
    
    return None


# ----- Streaming Functions -----

async def generate_streaming_response(
    context: str,
    query: str,
    provider: str,
    model_name: str,
    api_key: str,
    content_type: str = "all"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate streaming structured learning content using PydanticAI.
    Yields partial responses as they're generated.
    """
    if not api_key or not api_key.strip():
        yield {
            'status': 'error',
            'error': 'API key is required for the provider agent.',
            'error_type': 'missing_api_key'
        }
        return

    try:
      
        from pydantic_ai import StructuredDict
        
        # Build components
        final_content_type = detect_content_type(query, content_type)
        prompt = build_prompt(context, query, final_content_type)
        
        logger.info(f'prompt for structured_llm{prompt}')
        logger.info(f"Creating learning agent for provider={provider} model={model_name}")
        agent = create_learning_agent(provider, model_name, api_key)
        
        def _to_primitive(obj):
            """Convert model response objects into JSON-serializable primitives."""
            try:
                if obj is None:
                    return None
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, (list, dict)):
                    return obj
                # Pydantic / dataclass style
                if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
                    try:
                        return obj.dict()
                    except Exception:
                        pass
                if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                    try:
                        return obj.to_dict()
                    except Exception:
                        pass
                if hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
                    try:
                        return obj.model_dump()
                    except Exception:
                        pass
                if hasattr(obj, 'json') and callable(getattr(obj, 'json')):
                    try:
                        import json as _json
                        return _json.loads(obj.json())
                    except Exception:
                        pass
                # Fallback to string
                return str(obj)
            except Exception:
                return str(obj)

        # Use streaming run
        async with agent.run_stream(
            prompt,
            output_type=StructuredDict(LEARNING_CONTENT_SCHEMA)
        ) as result:
            async for partial_response in result.stream_responses():
                ser = _to_primitive(partial_response)
                # Derive a short text delta if possible to help frontend incremental rendering

                logger.info("LLM stream partial (serialized): %s", repr(ser))
                yield {
                    'status': 'streaming',
                    'data': ser,
                    'is_final': False
                }

            # Final response
            final_output = await result.get_output()
            final_ser = _to_primitive(final_output)
            # For final payload, include a textual fallback as well
            final_text = None
            try:
                if isinstance(final_ser, str):
                    final_text = final_ser
                elif isinstance(final_ser, dict):
                    final_text = final_ser.get('summary') or final_ser.get('metadata', {}).get('summary') or None
                    if final_text is None:
                        import json as _json
                        try:
                            final_text = _json.dumps(final_ser)
                        except Exception:
                            final_text = str(final_ser)
            except Exception:
                final_text = str(final_ser)

            logger.info("LLM stream final (serialized): %s", repr(final_ser))


            # Finally emit the complete object
            yield {
                'status': 'complete',
                'data': final_ser,
                'is_final': True
            }
            
    except Exception as e:
        logger.exception(f"LLM provider error in generate_streaming_response: {e}")
        error_msg = str(e).lower()
        suggestion = None
        
        if any(term in error_msg for term in ['token', 'exhaust', 'context']):
            suggestion = 'Consider switching to a model with a larger context window or using a different provider.'
        
        yield {
            'status': 'error',
            'error': str(e),
            'error_type': 'llm_error',
            'suggestion': suggestion,
            'is_final': True
        }


# ----- Additional Database Helper Functions -----

async def upsert_learning_session_service(
    user_id: str,
    session_id: Optional[str],
    session_name: Optional[str],
    user_prompt: str,
    assistant_response: Dict[str, Any]
) -> Optional[str]:
    """
    Service layer function that mirrors the original upsert_chat_history from learning_history_service.
    Provides the same interface as the original service.
    """
    try:
        from app.services.learning_history_service import upsert_chat_history
        return upsert_chat_history(user_id, session_id, session_name, user_prompt, assistant_response)
    except ImportError:
        # Fallback to our implementation if service doesn't exist
        logger.warning("learning_history_service not available, using built-in implementation")
        return await persist_chat_history(user_id, session_id, session_name, user_prompt, assistant_response)
    except Exception as e:
        logger.error(f"Error in upsert_learning_session_service: {e}")
        return await persist_chat_history(user_id, session_id, session_name, user_prompt, assistant_response)


async def get_embedding_preferences(user_id: str) -> Tuple[str, str, str]:
    """
    Get user's embedding model preferences with proper fallback handling.
    Returns (provider, model, api_key) tuple.
    """
    embedding_provider = None
    embedding_model = None
    embedding_api_key = None
    
    try:
        from app.llm.providers import get_user_model_preferences
        from app.services.api_key_service import get_api_key_for_provider
        
        prefs = get_user_model_preferences(user_id)
        for p in prefs:
            if p and p.get("use_for_embedding"):
                model_id = p.get("model_id")
                provider_name = p.get("provider_name")
                
                if model_id and "-" in model_id:
                    parts = model_id.split("-", 1)
                    embedding_provider = provider_name or parts[0]
                    embedding_model = parts[1]
                else:
                    embedding_provider = provider_name
                    embedding_model = model_id
                break
        
        if embedding_provider:
            try:
                embedding_api_key = await get_api_key_for_provider(user_id, embedding_provider)
            except Exception as e:
                logger.debug(f"Could not get API key for embedding provider {embedding_provider}: {e}")
                
    except Exception as e:
        logger.debug(f"Failed to get embedding preferences: {e}")
    
    return (
        embedding_provider or "",
        embedding_model or "",
        embedding_api_key or ""
    )

async def persist_metadata(
    user_id: str, 
    metadata: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    """
    Complete metadata persistence with proper ID tracking and error handling.
    Returns dict with created IDs for referencing.
    """
    try:
        from app.db.db_client import get_supabase_client
        
        if not isinstance(metadata, dict):
            return {'subject_id': None, 'chapter_id': None, 'concept_id': None}
        
        client = get_supabase_client()
        now_iso = datetime.now().isoformat()
        
        subject_name = metadata.get('subject_name')
        chapter_name = metadata.get('chapter_name')
        concept_name = metadata.get('concept_name')
        difficulty = metadata.get('difficulty_level') or metadata.get('difficulty') or 'Medium'
        
        subject_id = None
        chapter_id = None
        concept_id = None
        
        # Subject persistence with proper error handling
        if subject_name:
            try:
                existing = client.table('subjects').select('*').eq(
                    'student_id', user_id
                ).eq('llm_suggested_subject_name', subject_name).execute()
                
                if existing and getattr(existing, 'data', None):
                    rows = existing.data if isinstance(existing.data, list) else [existing.data]
                    if rows:
                        subject_id = rows[0].get('subject_id')
                        logger.info(f'Found existing subject id={subject_id} for user={user_id} name={subject_name}')
                
                if not subject_id:
                    # Allow optional session_id to be attached to the subject so that
                    # we can deep-link or restore sessions directly from subjects.
                    subject_session_id = metadata.get('session_id') or metadata.get('sessionId')
                    insert_payload = {
                        'student_id': user_id,
                        'llm_suggested_subject_name': subject_name,
                        'created_at': now_iso,
                        'updated_at': now_iso
                    }
                    if subject_session_id:
                        insert_payload['session_id'] = subject_session_id

                    resp = client.table('subjects').insert(insert_payload).execute()
                    
                    if resp and getattr(resp, 'data', None):
                        inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                        subject_id = inserted.get('subject_id')
                        logger.info(f'Created subject id={subject_id} for user={user_id} name={subject_name}')
                        # If the client passed a session_id and an existing subject row didn't have it,
                        # ensure the subject record includes the session id for future lookups.
                        if subject_session_id:
                            try:
                                client.table('subjects').update({'session_id': subject_session_id}).eq('subject_id', subject_id).execute()
                            except Exception:
                                logger.debug('Failed to attach session_id to subject after insert')
            except Exception as e:
                logger.error(f'Failed to persist subject: {e}')
        
        # Chapter persistence with proper error handling
        if subject_id and chapter_name:
            try:
                existing = client.table('chapters').select('*').eq(
                    'student_id', user_id
                ).eq('subject_id', subject_id).eq(
                    'llm_suggested_chapter_name', chapter_name
                ).execute()

                if existing and getattr(existing, 'data', None):
                    rows = existing.data if isinstance(existing.data, list) else [existing.data]
                    if rows:
                        chapter_id = rows[0].get('chapter_id')
                        logger.info(f'Found existing chapter id={chapter_id} for subject_id={subject_id} name={chapter_name}')

                if not chapter_id:
                    # Determine safe chapter_order to avoid unique constraint collisions
                    try:
                        top = client.table('chapters').select('chapter_order').eq('subject_id', subject_id).order('chapter_order', desc=True).limit(1).execute()
                        max_order = None
                        if top and getattr(top, 'data', None):
                            top_rows = top.data if isinstance(top.data, list) else [top.data]
                            if top_rows and isinstance(top_rows[0], dict):
                                max_order = top_rows[0].get('chapter_order')
                        new_order = (max_order + 1) if isinstance(max_order, int) else 0
                    except Exception:
                        new_order = 0

                    try:
                        resp = client.table('chapters').insert({
                            'student_id': user_id,
                            'subject_id': subject_id,
                            'llm_suggested_chapter_name': chapter_name,
                            'chapter_order': new_order,
                            'description': None,
                            'created_at': now_iso,
                            'updated_at': now_iso
                        }).execute()

                        if resp and getattr(resp, 'data', None):
                            inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                            chapter_id = inserted.get('chapter_id')
                            logger.info(f'Created chapter id={chapter_id} for subject_id={subject_id} name={chapter_name} order={new_order}')
                    except Exception as insert_exc:
                        logger.warning(f"Chapter insert failed, attempting to recover: {insert_exc}")
                        try:
                            recheck = client.table('chapters').select('*').eq('student_id', user_id).eq('subject_id', subject_id).eq('llm_suggested_chapter_name', chapter_name).execute()
                            if recheck and getattr(recheck, 'data', None):
                                rows = recheck.data if isinstance(recheck.data, list) else [recheck.data]
                                if rows:
                                    chapter_id = rows[0].get('chapter_id')
                                    logger.info(f'Recovered existing chapter id={chapter_id} after insert conflict for subject_id={subject_id} name={chapter_name}')
                        except Exception as re_exc:
                            logger.error(f'Failed to persist chapter after retry: {re_exc}')
            except Exception as e:
                logger.error(f'Failed to persist chapter: {e}')
        
        # Concept persistence with proper error handling
        if chapter_id and concept_name:
            try:
                existing = client.table('concepts').select('*').eq(
                    'student_id', user_id
                ).eq('chapter_id', chapter_id).eq(
                    'llm_suggested_concept_name', concept_name
                ).execute()

                if existing and getattr(existing, 'data', None):
                    rows = existing.data if isinstance(existing.data, list) else [existing.data]
                    if rows:
                        concept_id = rows[0].get('concept_id')
                        logger.info(f'Found existing concept id={concept_id} for chapter_id={chapter_id} name={concept_name}')
                else:
                    # Determine a safe concept_order to avoid unique constraint collisions
                    try:
                        top = client.table('concepts').select('concept_order').eq('chapter_id', chapter_id).order('concept_order', desc=True).limit(1).execute()
                        max_order = None
                        if top and getattr(top, 'data', None):
                            top_rows = top.data if isinstance(top.data, list) else [top.data]
                            if top_rows and isinstance(top_rows[0], dict):
                                max_order = top_rows[0].get('concept_order')
                        new_order = (max_order + 1) if isinstance(max_order, int) else 0
                    except Exception:
                        new_order = 0

                    # Try insert; if a unique constraint race occurs, attempt to recover by re-querying
                    try:
                        resp = client.table('concepts').insert({
                            'student_id': user_id,
                            'chapter_id': chapter_id,
                            'llm_suggested_concept_name': concept_name,
                            'concept_order': new_order,
                            'description': None,
                            'difficulty_level': difficulty,
                            'created_at': now_iso,
                            'updated_at': now_iso
                        }).execute()

                        if resp and getattr(resp, 'data', None):
                            inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                            concept_id = inserted.get('concept_id')
                            logger.info(f'Created concept id={concept_id} for chapter_id={chapter_id} name={concept_name} order={new_order}')
                    except Exception as insert_exc:
                        # handle duplicate key race: try to fetch the existing row again
                        logger.warning(f"Concept insert failed, attempting to recover: {insert_exc}")
                        try:
                            recheck = client.table('concepts').select('*').eq('student_id', user_id).eq('chapter_id', chapter_id).eq('llm_suggested_concept_name', concept_name).execute()
                            if recheck and getattr(recheck, 'data', None):
                                rows = recheck.data if isinstance(recheck.data, list) else [recheck.data]
                                if rows:
                                    concept_id = rows[0].get('concept_id')
                                    logger.info(f'Recovered existing concept id={concept_id} after insert conflict for chapter_id={chapter_id} name={concept_name}')
                                    
                        except Exception as re_exc:
                            logger.error(f'Failed to persist concept after retry: {re_exc}')
            except Exception as e:
                logger.error(f'Failed to persist concept: {e}')
        
        return {
            'subject_id': subject_id,
            'chapter_id': chapter_id, 
            'concept_id': concept_id
        }
        
    except Exception as e:
        logger.error(f'Failed to persist metadata: {e}')
        return {'subject_id': None, 'chapter_id': None, 'concept_id': None}


async def persist_chat_history(
    user_id: str,
    session_id: Optional[str],
    session_name: Optional[str],
    user_prompt: str,
    assistant_response: Dict[str, Any]
) -> Optional[str]:
    """
    Complete chat history and session management with all database operations.
    Returns session_id (existing or newly created).
    """
    try:
        from app.db.db_client import get_supabase_client
        
        client = get_supabase_client()
        now_iso = datetime.now().isoformat()
        
        # Create conversation entries
        user_entry = {
            'role': 'user', 
            'content': user_prompt, 
            'timestamp': now_iso
        }
        
        assistant_content = (
            assistant_response.get('summary') or 
            assistant_response.get('metadata', {}).get('summary') or
            json.dumps(assistant_response) if isinstance(assistant_response, dict) else str(assistant_response)
        )
        
        assistant_entry = {
            'role': 'assistant',
            'content': assistant_content,
            'timestamp': now_iso
        }
        
        # Handle existing session update
        if session_id:
            existing = client.table('chat_history').select('*').eq(
                'session_id', session_id
            ).execute()
            
            if existing and existing.data:
                record = existing.data[0]
                
                # Handle llm_response_history properly
                history = record.get('llm_response_history') or []
                if isinstance(history, str):
                    try:
                        history = json.loads(history)
                        history = history if isinstance(history, list) else [history]
                    except Exception:
                        history = [history]
                elif not isinstance(history, list):
                    history = [] if history is None else [history]
                
                history.extend([user_entry, assistant_entry])
                
                # Handle study_material_history properly
                materials = record.get('study_material_history') or []
                if isinstance(materials, str):
                    try:
                        materials = json.loads(materials)
                        materials = materials if isinstance(materials, list) else [materials]
                    except Exception:
                        materials = [materials]
                elif not isinstance(materials, list):
                    materials = [] if materials is None else [materials]
                
                if isinstance(assistant_response, dict):
                    materials.append(assistant_response)
                
                # Update with complete data
                update_data = {
                    'llm_response_history': history,
                    'study_material_history': materials,
                    'session_name': session_name or record.get('session_name'),
                    'updated_at': now_iso
                }
                
                client.table('chat_history').update(update_data).eq('session_id', session_id).execute()
                logger.info(f'Updated chat_history for session_id={session_id} user={user_id}')
                return session_id
            else:
                # Session ID provided but doesn't exist - create new with provided ID
                client.table('chat_history').insert({
                    'session_id': session_id,
                    'student_id': user_id,
                    'session_name': session_name,
                    'llm_response_history': [user_entry, assistant_entry],
                    'study_material_history': [assistant_response] if isinstance(assistant_response, dict) else [],
                    'created_at': now_iso,
                    'updated_at': now_iso
                }).execute()
                logger.info(f'Created new chat_history with provided session_id={session_id}')
                return session_id
        
        # Create completely new session
        new_session_id = str(uuid.uuid4())
        
        client.table('chat_history').insert({
            'session_id': new_session_id,
            'student_id': user_id,
            'session_name': session_name,
            'llm_response_history': [user_entry, assistant_entry],
            'study_material_history': [assistant_response] if isinstance(assistant_response, dict) else [],
            'created_at': now_iso,
            'updated_at': now_iso
        }).execute()
        
        logger.info(f'Created new chat_history session_id={new_session_id} for user={user_id}')
        return new_session_id
        
    except Exception as e:
        logger.error(f'Failed to persist chat history: {e}')
        return session_id


# ----- Main Pipeline Function -----

async def stream_structured_content(
    uploaded_files_paths: List[Path],
    user_prompt: str,
    temp_dir: str,
    user_api_key: str,
    user_id: str,
    provider_name: str,
    model_name: str,
    session_id: Optional[str] = None,
    session_name: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Complete streaming RAG pipeline for generating structured learning content.
    Yields partial responses as they're generated.
    """
    try:
        if not user_api_key:
            yield {
                "status": "error",
                "error": "API key is required. Please add your API key in settings to continue.",
                "error_type": "missing_api_key"
            }
            return

        logger.info("Starting streaming RAG pipeline...")


        
        documents = await load_documents_from_files(
            [str(p) for p in uploaded_files_paths], 
            temp_dir, 
            user_api_key, 
            provider_name, 
            model_name
        )
        
        combined_text = "\n\n".join(documents)
        content_type = detect_content_type(user_prompt, "all")
        
        # Check if we should skip embedding for small documents
        if should_skip_embedding(combined_text):
            logger.info(f"Document small (words={len(combined_text.split())}) — using direct LLM approach")
            
            async for response in generate_streaming_response(
                combined_text, user_prompt, provider_name, model_name, user_api_key, content_type
            ):
                # Handle streaming responses
                if response.get('status') == 'error':
                    yield response
                    return
                elif response.get('status') in ['streaming', 'complete']:
                    yield response
                    
                    # Persist final response with complete session management
                    if response.get('is_final') and response.get('status') == 'complete':
                        final_data = response.get('data')
                        if isinstance(final_data, dict):
                            # Create session name from metadata
                            metadata = final_data.get('metadata', {})
                            final_session_name = session_name or create_session_name(metadata)
                            
                            # Use service layer for session management if available
                            try:
                                created_session_id = await upsert_learning_session_service(
                                    user_id, session_id, final_session_name, user_prompt, final_data
                                )
                            except Exception as e:
                                logger.warning(f"Service layer failed, using direct persistence: {e}")
                                created_session_id = await persist_chat_history(
                                    user_id, session_id, final_session_name, user_prompt, final_data
                                )
                            
                            # Persist metadata with complete tracking
                            metadata_ids = await persist_metadata(user_id, metadata)
                            
                            yield {
                                'status': 'persisted',
                                'session_id': created_session_id,
                                'metadata_ids': metadata_ids,
                                'is_final': True
                            }
        
        else:
            # Full RAG pipeline with embedding
            logger.info("Using full RAG pipeline with embeddings...")
            
            chunks = await perform_document_chunking(documents)
            
            # Get embedding preferences with proper fallback
            embedding_provider, embedding_model, embedding_api_key = await get_embedding_preferences(user_id)
            
            # Fallback to chat model if no embedding preferences found
            if not embedding_provider:
                embedding_provider = provider_name
                embedding_model = model_name
                embedding_api_key = user_api_key
            
            # Use user's API key if embedding API key not available
            if not embedding_api_key:
                embedding_api_key = user_api_key
            
            logger.info(f"Using embedding provider={embedding_provider} model={embedding_model} (key_available={bool(embedding_api_key)})")
            
            # Setup vector store and retrieve context
            # Attempt to reuse a persisted vector store for this session to speed up repeated queries
            vector_store = None
            saved_chunks = None
            try:
                # embeddings factory must be available to load a persisted FAISS index
                if callable(get_embeddings):
                    emb_obj = get_embeddings(embedding_provider, embedding_model, embedding_api_key)
                    vs_loaded, saved_chunks = load_vector_store_and_chunks(temp_dir, session_id, emb_obj) if session_id else (None, None)
                    if vs_loaded:
                        vector_store = vs_loaded
                        logger.info(f"Reusing persisted FAISS store for session={session_id}")
                else:
                    emb_obj = None
            except Exception as e:
                logger.debug(f"Error while attempting to load persisted vector store: {e}")

            if vector_store is None:
                vector_store, hybrid_retriever = await setup_vector_store_and_retriever(
                    chunks, embedding_provider, embedding_model, embedding_api_key
                )
                # Persist the vector store and chunks for subsequent requests, if session_id provided
                if session_id:
                    try:
                        save_vector_store_and_chunks(vector_store, chunks, temp_dir, session_id)
                    except Exception as e:
                        logger.debug(f"Could not persist vector store: {e}")
            else:
                # Create a retriever from the loaded store
                hybrid_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": DEFAULT_TOP_K})
                bm25_retriever = BM25Retriever.from_documents(saved_chunks or chunks)
                bm25_retriever.k = DEFAULT_TOP_K
                hybrid_retriever = EnsembleRetriever(retrievers=[hybrid_retriever, bm25_retriever], weights=[DEFAULT_ALPHA, 1.0 - DEFAULT_ALPHA])
            
            retrieved_docs = await hybrid_retriever.ainvoke(user_prompt)
            enhanced_docs = await enhance_retrieved_context(retrieved_docs, chunks)
            formatted_context = format_context_for_llm(enhanced_docs)
            
            # Stream the response
            async for response in generate_streaming_response(
                formatted_context, user_prompt, provider_name, model_name, user_api_key, content_type
            ):
                if response.get('status') == 'error':
                    yield response
                    return
                elif response.get('status') in ['streaming', 'complete']:
                    yield response
                    
                    # Persist final response with complete session management
                    if response.get('is_final') and response.get('status') == 'complete':
                        final_data = response.get('data')
                        if isinstance(final_data, dict):
                            metadata = final_data.get('metadata', {})
                            final_session_name = session_name or create_session_name(metadata)
                            
                            # Use service layer for session management if available
                            try:
                                created_session_id = await upsert_learning_session_service(
                                    user_id, session_id, final_session_name, user_prompt, final_data
                                )
                            except Exception as e:
                                logger.warning(f"Service layer failed, using direct persistence: {e}")
                                created_session_id = await persist_chat_history(
                                    user_id, session_id, final_session_name, user_prompt, final_data
                                )
                            
                            # Persist metadata with complete tracking
                            metadata_ids = await persist_metadata(user_id, metadata)
                            
                            yield {
                                'status': 'persisted', 
                                'session_id': created_session_id,
                                'metadata_ids': metadata_ids,
                                'is_final': True
                            }

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        yield {
            "status": "error",
            "error": str(ve),
            "error_type": "validation_error"
        }
    except Exception as e:
        logger.exception(f"Error in streaming pipeline: {e}")
        yield {
            "status": "error",
            "error": str(e),
            "error_type": "processing_error"
        }


# ----- Non-streaming wrapper for backward compatibility -----

async def get_llm_response(
    uploaded_files_paths: List[Path],
    user_prompt: str,
    temp_dir: str,
    user_api_key: str,
    user_id: str,
    provider_name: str,
    model_name: str,
    session_id: Optional[str] = None,
    session_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Non-streaming wrapper that collects all streaming responses into final result.
    Maintains backward compatibility while using streaming under the hood.
    """
    final_response = None
    final_session_id = None
    
    async for response in stream_structured_content(
        uploaded_files_paths, user_prompt, temp_dir, user_api_key, 
        user_id, provider_name, model_name, session_id, session_name
    ):
        if response.get('status') == 'error':
            return response
        elif response.get('status') == 'complete':
            final_response = response.get('data')
        elif response.get('status') == 'persisted':
            final_session_id = response.get('session_id')
    
    return {
        'session_id': final_session_id,
        'llm_response': final_response
    } if final_response else {'status': 'error', 'error': 'No response generated'}