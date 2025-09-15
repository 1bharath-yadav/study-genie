from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

from typing import Optional
import mimetypes
import json
import os
import tarfile
import zipfile
from typing import Iterable
# Direct Google Generative AI client
# We no longer use the direct Google Generative AI client for chat generation.
# Generation is done via per-user provider agents (pydantic-ai). Keep provider-specific
# embedding implementations where available.

# Document loaders are imported lazily inside helper functions to keep optional deps local
from app.llm.providers import create_learning_agent

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

# Configuration - No global API key; per-user API keys are used at runtime

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



# --- small utils

def _ext(p: Path) -> str:
    return p.suffix.lower()

def _fmt(name: str, content: str) -> str:
    return f"# {name}\n{content.strip()}\n"

def _is_archive(p: Path) -> bool:
    return _ext(p) in {".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".rar", ".7z"}

def _is_image(p: Path) -> bool:
    return _ext(p) in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}

def _is_audio(p: Path) -> bool:
    return _ext(p) in {".mp3", ".wav", ".aac", ".ogg", ".m4a", ".flac"}

def _is_video(p: Path) -> bool:
    return _ext(p) in {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def _is_text_like(p: Path) -> bool:
    return _ext(p) in {".txt", ".md", ".tex", ".rmd", ".bib", ".csv", ".tsv", ".py", ".html", ".htm", ".svg"}

def _flatten(iters: Iterable[Iterable[Path]]) -> List[Path]:
    out: List[Path] = []
    for it in iters:
        out.extend(list(it))
    return out

# --- archive expansion (pure; returns extracted file paths)

def _expand_archive(p: Path, temp_dir: Path) -> List[Path]:
    tdir = temp_dir / (p.stem + "_extracted")
    tdir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    try:
        if _ext(p) == ".zip":
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(tdir)
        elif _ext(p) in {".tar", ".gz", ".tgz", ".tar.gz"}:
            mode = "r:gz" if _ext(p) in {".gz", ".tgz", ".tar.gz"} else "r"
            with tarfile.open(p, mode) as t:
                t.extractall(tdir)
        else:
            # minimal fallback: treat as no-op if unsupported archive
            return []
        for root, _, files in os.walk(tdir):
            for f in files:
                out.append(Path(root) / f)
    except Exception:
        return []
    return out

# --- LangChain helpers (attempt LC first, then fallbacks)
# LC API: loaders expose .load() / .aload() and return Documents with .page_content [1][21][22]

async def _lc_load(path: Path) -> str:
    try:
        # prioritize specific, then general
        if _ext(path) == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader  # [22][21]
            docs = await PyPDFLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".docx", ".doc"}:
            from langchain_community.document_loaders import Docx2txtLoader  # [33][36]
            docs = await Docx2txtLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".ppt", ".pptx", ".odp"}:
            from langchain_community.document_loaders import UnstructuredPowerPointLoader  # [27]
            docs = await UnstructuredPowerPointLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".xlsx", ".xls"}:
            from langchain_community.document_loaders import UnstructuredExcelLoader  # [27]
            docs = await UnstructuredExcelLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".epub"}:
            from langchain_community.document_loaders import UnstructuredEPubLoader  # [1]
            docs = await UnstructuredEPubLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".html", ".htm"}:
            from langchain_community.document_loaders import BSHTMLLoader  # [1]
            docs = await BSHTMLLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".csv", ".tsv"}:
            from langchain_community.document_loaders import CSVLoader  # [1]
            docs = await CSVLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        # general fallback that auto-detects many file types via Unstructured [32][29]
        from langchain_community.document_loaders.unstructured import UnstructuredFileLoader  # [32]
        docs = await UnstructuredFileLoader(str(path), mode="single").aload()  # single = one combined doc [32]
        return "\n\n".join(d.page_content for d in docs) or ""
    except Exception:
        return ""

# --- OCR helpers

def _ocr_image_sync(path: Path) -> str:
    try:
        import pytesseract  # [10]
        from PIL import Image
        img = Image.open(path)
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""

def _ocr_pdf_sync(path: Path) -> str:
    # try text with pypdf first; if empty, try pdf2image + pytesseract
    try:
        from pypdf import PdfReader
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import pytesseract  # [10]
        from pdf2image import convert_from_path
        pages = convert_from_path(str(path))
        return "\n\n".join(pytesseract.image_to_string(p) or "" for p in pages)
    except Exception:
        return ""

# --- simple parsers for text-ish formats

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

def _html_to_text(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        html = _read_text(path)
        return BeautifulSoup(html, "html.parser").get_text(" ").strip()
    except Exception:
        return _read_text(path)

def _svg_text(path: Path) -> str:
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(_read_text(path))
        return " ".join("".join(el.itertext()).strip() for el in root.iter() if el.tag.endswith("text"))
    except Exception:
        return _read_text(path)

def _table_to_text(path: Path) -> str:
    try:
        import pandas as pd
        if _ext(path) in {".csv", ".tsv"}:
            sep = "\t" if _ext(path) == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, dtype=str, engine="python")
        elif _ext(path) in {".xlsx", ".xls", ".ods"}:
            df = pd.read_excel(path, dtype=str, engine=None)
        else:
            return ""
        return df.to_csv(index=False)
    except Exception:
        return ""

def _ipynb_to_text(path: Path) -> str:
    try:
        import nbformat as nbf
        nb = nbf.read(path.open("r", encoding="utf-8"), as_version=4)
        parts = []
        for c in nb.cells:
            if c.cell_type == "markdown":
                parts.append(c.source)
            elif c.cell_type == "code":
                parts.append("``````")
        return "\n\n".join(parts)
    except Exception:
        try:
            raw = json.loads(_read_text(path))
            return json.dumps(raw.get("cells", raw), indent=2)
        except Exception:
            return ""

# --- transcription (optional)  TODO

def _transcribe_sync(path: Path) -> str:
    # try local whisper if available; otherwise empty
    try:
        import whisper
        model = whisper.load_model("base")  # type: ignore[attr-defined]
        result = model.transcribe(str(path))
        return result.get("text", "").strip()
    except Exception:
        return ""

# --- Pydantic AI fallback (summarize/describe when no text) [6]



async def _agent_fallback(
    path: Path,
    meta_text: str,
    api_key,
    provider,
    model_name,
    file_url: Optional[str] = None,
) -> str:
    """
    Send local bytes via BinaryContent or a remote URL via the appropriate FileUrl part to a PydanticAI Agent.
    Returns a compact, readable summary extracted by the model.
    """
    try:
        from pydantic_ai import (
            BinaryContent,
            ImageUrl,
            AudioUrl,
            VideoUrl,
            DocumentUrl,
        )

        mtype, _ = mimetypes.guess_type(str(file_url or path))
        prompt = (
            "Extract ALL available information from the attached file.\n"
            "Summarize purpose, structure, key points, and any available metadata.\n"
            f"Filename: {path.name}\n"
            f"MIME: {mtype or 'application/octet-stream'}\n"
            f"Context:\n{meta_text[:6000]}"
        )

        user_part = None
        if file_url:
            if mtype and mtype.startswith("image/"):
                user_part = ImageUrl(url=file_url)
            elif mtype and mtype.startswith("audio/"):
                user_part = AudioUrl(url=file_url)
            elif mtype and mtype.startswith("video/"):
                user_part = VideoUrl(url=file_url)
            else:
                user_part = DocumentUrl(url=file_url)
        else:
            data = b""
            try:
                data = path.read_bytes()
            except Exception:
                data = b""
            user_part = BinaryContent(data=data, media_type=mtype or "application/octet-stream")
        agent = create_learning_agent( provider ,model_name,api_key)
        res = await agent.run([prompt, user_part])
        return getattr(res, "output", "") or ""
    except Exception:
        return f"Could not parse content; file name suggests: {path.suffix}."

# --- one file to text

async def _file_to_text(path: Path,api_key,provider_name,model_name) -> str:
    # 1) try LangChain loaders
    text = await _lc_load(path)
    if text.strip():
        return text

    # 2) fallback by type
    if _ext(path) == ".pdf":
        text = await asyncio.to_thread(_ocr_pdf_sync, path)
    elif _is_image(path):
        text = await asyncio.to_thread(_ocr_image_sync, path)
    elif _is_audio(path) or _is_video(path):
        text = await asyncio.to_thread(_transcribe_sync, path)
    elif _ext(path) == ".ipynb":
        text = await asyncio.to_thread(_ipynb_to_text, path)
    elif _ext(path) in {".xlsx", ".xls", ".ods", ".csv", ".tsv"}:
        text = await asyncio.to_thread(_table_to_text, path)
    elif _ext(path) in {".html", ".htm"}:
        text = await asyncio.to_thread(_html_to_text, path)
    elif _ext(path) == ".svg":
        text = await asyncio.to_thread(_svg_text, path)
    elif _is_text_like(path) or _ext(path) in {".tex", ".bib", ".rmd"}:
        text = await asyncio.to_thread(_read_text, path)
    else:
        text = ""

    if text.strip():
        return text

    # 3) final fallback: Pydantic AI agent on metadata and any scraps
    meta = f"size={path.stat().st_size} bytes; ext={path.suffix}; name={path.name}"
    return await _agent_fallback(path, meta,api_key,provider_name,model_name)

# --- public API

async def load_documents_from_files(file_paths: List[str], temp_dir: str, api_key: str, provider_name: str, model_name: str) -> List[str]:
    """
    Load documents from various file formats with enhanced error handling.
    Performs OCR/transcription when possible; falls back to a Pydantic AI agent when no text is found.
    Returns a single string formatted as:
    # filename
    content

    # filename2
    content
    """
    temp = Path(temp_dir)
    temp.mkdir(parents=True, exist_ok=True)

    in_paths = [Path(p) for p in file_paths if Path(p).exists()]
    # expand archives into temp_dir
    expanded_lists = await asyncio.gather(*[asyncio.to_thread(_expand_archive, p, temp) if _is_archive(p) else asyncio.to_thread(lambda x: [x], p) for p in in_paths])
    worklist = _flatten(expanded_lists)

    # dedupe and preserve order
    seen, ordered = set(), []
    for p in worklist:
        if p.is_file() and p not in seen:
            seen.add(p)
            ordered.append(p)

    # process concurrently (pass api_key/provider/model to file processor)
    texts = await asyncio.gather(*[_file_to_text(p, api_key, provider_name, model_name) for p in ordered])

    # format
    parts = [_fmt(p.name, t if t.strip() else "(no extractable text)") for p, t in zip(ordered, texts)]
    return parts



async def perform_document_chunking(documents: List[str]) -> List[Document]:
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
    
    

    # Convert raw document strings into LangChain Documents (preserve basic metadata)
    lc_docs: List[Document] = [Document(page_content=d) for d in documents]

    # Split documents into chunks
    chunks = text_splitter.split_documents(lc_docs)

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

    # Decide content type and build prompt (pure functions)
    final_ct = detect_content_type(query, content_type)
    prompt = build_prompt(context, query, final_ct,)

    logger.info("Creating learning agent for provider=%s model=%s", provider, model_name)

    try:
        agent = create_learning_agent(provider, model_name, api_key)
    except Exception as e:
        logger.error("Failed to create learning agent: %s", e)
        raise ValueError(f"Failed to initialize model/provider: {e}")

    # Correct PydanticAI call — no invoke/ainvoke
    logger.info(f"model_prompt{prompt}")
    lc = await agent.run(prompt, output_type=StructuredDict(LEARNING_CONTENT_SCHEMA))

    llm_response = lc.output 

 
    logger.info(f"llm_response{llm_response}")

    return llm_response                                       



# Streaming support removed: the codebase now exposes only the non-streaming
# `get_llm_response` function which performs ingestion, chunking, embeddings and
# calls the model synchronously (returning a final structured JSON-compatible dict).


async def get_llm_response(uploaded_files_paths: List[Path], userprompt: str, temp_dir: str, user_api_key: str, user_id: str, provider_name: str, model_name: str, session_id: str | None = None, session_name: str | None = None) -> Dict[str, Any]:
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
        documents = await load_documents_from_files([str(p) for p in uploaded_files_paths], temp_dir,user_api_key,provider_name,model_name)
        #  if no of (tokens words) is less than 1300 skip embedding send direct to llm
        # Auto-detect content type from user prompt early so both paths can use it
        content_type = detect_content_type(userprompt, "all")

        # If the uploaded documents are small (few words/tokens), skip embedding/retrieval
        # and send the combined text directly to the LLM. This avoids unnecessary embedding
        # calls for short inputs. Heuristic: use word count as a proxy for tokens.
        combined_text = "\n\n".join(documents)
        word_count = len(combined_text.split())
        if word_count < 1300:
            logger.info("Document small (words=%d) — skipping embedding/retrieval and sending direct to LLM", word_count)
            formatted_context = combined_text

            # Generate structured response directly without vector store
            response = await generate_structured_response(formatted_context, userprompt, provider_name, model_name, user_api_key, content_type)

            # Persist the LLM response into chat_history via service helper
            out_session_id = None
            try:
                from app.services.learning_history_service import upsert_chat_history

                # Build a guaranteed session name from LLM metadata: Subject | Chapter | Concept
                # This ensures the session_name is never optional and is consistent across writes.
                composed_session_name = None
                try:
                    if isinstance(response, dict):
                        meta = response.get('metadata') or {}
                        subject_name = (meta.get('subject_name') or meta.get('subject') or '').strip()
                        chapter_name = (meta.get('chapter_name') or meta.get('chapter') or '').strip()
                        concept_name = (meta.get('concept_name') or meta.get('concept') or '').strip()

                        composed_session_name = f"{subject_name}-{chapter_name}-{concept_name}"
                except Exception:
                    composed_session_name = None

                # prefer an explicit composed name, then caller-provided session_name, then fallback
                final_session_name = composed_session_name or session_name or f"Uncategorized | {datetime.now().date()}"

                # upsert_chat_history will write conversational entries into chat_history
                out_session_id = upsert_chat_history(user_id, session_id, final_session_name, userprompt, response)

                # Also, since this is a small-doc path where the full structured response is available,
                # persist the full structured learning content into study_material_history for the session.
                if isinstance(response, dict) and out_session_id:
                    try:
                        from app.db.db_client import get_supabase_client as _get_supabase_client
                        client = _get_supabase_client()
                        existing = client.table('chat_history').select('*').eq('session_id', out_session_id).execute()
                        if existing and getattr(existing, 'data', None) and len(existing.data) > 0:
                            rec = existing.data[0]
                            materials = rec.get('study_material_history') or []
                            materials.append(response)
                            now_iso_small = datetime.now().isoformat()
                            client.table('chat_history').update({'study_material_history': materials, 'updated_at': now_iso_small}).eq('session_id', out_session_id).execute()
                    except Exception:
                        logger.debug('Failed to persist study_material_history for small-doc path')
            except Exception as e:
                logger.debug('Skipping persistence to chat_history due to error: %s', e)

            # Also persist metadata (subject/chapter/concept) for small-doc path
            try:
                if isinstance(response, dict):
                    from app.db.db_client import get_supabase_client as _get_supabase_client

                    meta = response.get('metadata') or {}
                    subject_name = meta.get('subject_name')
                    chapter_name = meta.get('chapter_name')
                    concept_name = meta.get('concept_name')
                    difficulty = meta.get('difficulty_level') or meta.get('difficulty')

                    now_iso_local = datetime.now().isoformat()

                    if subject_name:
                        client = _get_supabase_client()

                        existing = client.table('subjects').select('*').eq('student_id', user_id).eq('llm_suggested_subject_name', subject_name).execute()
                        subject_id_val = None
                        if existing and getattr(existing, 'data', None):
                            rows = existing.data if isinstance(existing.data, list) else [existing.data]
                            if len(rows) > 0:
                                subject_id_val = rows[0].get('subject_id')
                        if not subject_id_val:
                            ins = {
                                'student_id': user_id,
                                'llm_suggested_subject_name': subject_name,
                                'created_at': now_iso_local,
                                'updated_at': now_iso_local
                            }
                            resp = client.table('subjects').insert(ins).execute()
                            if resp and getattr(resp, 'data', None):
                                inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                                subject_id_val = inserted.get('subject_id')

                        chapter_id_val = None
                        if subject_id_val and chapter_name:
                            existing = client.table('chapters').select('*').eq('student_id', user_id).eq('subject_id', subject_id_val).eq('llm_suggested_chapter_name', chapter_name).execute()
                            if existing and getattr(existing, 'data', None):
                                rows = existing.data if isinstance(existing.data, list) else [existing.data]
                                if len(rows) > 0:
                                    chapter_id_val = rows[0].get('chapter_id')
                            if not chapter_id_val:
                                ins = {
                                    'student_id': user_id,
                                    'subject_id': subject_id_val,
                                    'llm_suggested_chapter_name': chapter_name,
                                    'chapter_order': 0,
                                    'description': None,
                                    'created_at': now_iso_local,
                                    'updated_at': now_iso_local
                                }
                                resp = client.table('chapters').insert(ins).execute()
                                if resp and getattr(resp, 'data', None):
                                    inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                                    chapter_id_val = inserted.get('chapter_id')

                        if chapter_id_val and concept_name:
                            existing = client.table('concepts').select('*').eq('student_id', user_id).eq('chapter_id', chapter_id_val).eq('llm_suggested_concept_name', concept_name).execute()
                            if existing and getattr(existing, 'data', None):
                                pass
                            else:
                                ins = {
                                    'student_id': user_id,
                                    'chapter_id': chapter_id_val,
                                    'llm_suggested_concept_name': concept_name,
                                    'concept_order': 0,
                                    'description': None,
                                    'difficulty_level': difficulty or 'Medium',
                                    'created_at': now_iso_local,
                                    'updated_at': now_iso_local
                                }
                                resp = client.table('concepts').insert(ins).execute()
            except Exception as e:
                logger.error('Failed to persist small-doc LLM metadata: %s', e)

            return {
                'session_id': out_session_id,
                'llm_response': response
            } if isinstance(response, dict) else (response or {})
        else:
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
    # Persist the LLM response into chat_history for session/history tracking
    # Table schema: chat_history (id PK), session_id UUID, student_id, session_name,
    # llm_response_history (JSONB), study_material_history (JSONB), created_at, updated_at
        created_session_id = None
        try:
            from app.db.db_client import get_supabase_client
            import uuid

            client = get_supabase_client()

            # Determine session_name from response metadata when available when not provided
            try:
                meta = response.get('metadata') if isinstance(response, dict) else None
                subj = meta.get('subject_name') if meta else None
                conc = meta.get('concept_name') if meta else None
                if not session_name:
                    if subj and conc:
                        session_name = f"{subj} - {conc}"
                    elif subj:
                        session_name = subj
                    elif conc:
                        session_name = conc
            except Exception:
                pass

            now_iso = datetime.now().isoformat()

            # Append a user+assistant pair to chat_history so sessions have conversational memory
            try:
                user_entry = {'role': 'user', 'content': userprompt, 'timestamp': now_iso}
                assistant_content = None
                if isinstance(response, dict):
                    assistant_content = response.get('summary')
                    meta = response.get('metadata')
                    if not assistant_content and isinstance(meta, dict):
                        assistant_content = meta.get('summary')
                    if not assistant_content:
                        try:
                            assistant_content = json.dumps(response)
                        except Exception:
                            assistant_content = str(response)
                else:
                    assistant_content = response
                assistant_entry = {'role': 'assistant', 'content': assistant_content, 'timestamp': now_iso}

                if session_id:
                    existing = client.table('chat_history').select('*').eq('session_id', session_id).execute()
                    if existing and getattr(existing, 'data', None) and len(existing.data) > 0:
                        rec = existing.data[0]
                        history = rec.get('llm_response_history') or []
                        history.append(user_entry)
                        history.append(assistant_entry)
                        update_obj = {
                            'llm_response_history': history,
                            'session_name': session_name or rec.get('session_name'),
                            'updated_at': now_iso
                        }
                        # if we have a structured response, also append to study_material_history
                        if isinstance(response, dict):
                            materials = rec.get('study_material_history') or []
                            materials.append(response)
                            update_obj['study_material_history'] = materials

                        client.table('chat_history').update(update_obj).eq('session_id', session_id).execute()
                        created_session_id = session_id
                        logger.info('Appended to chat_history session_id=%s student=%s', session_id, user_id)
                    else:
                        row = {
                            'session_id': session_id,
                            'student_id': user_id,
                            'session_name': session_name,
                            'llm_response_history': [user_entry, assistant_entry],
                            'study_material_history': [response] if isinstance(response, dict) else [],
                            'created_at': now_iso,
                            'updated_at': now_iso
                        }
                        client.table('chat_history').insert(row).execute()
                        created_session_id = session_id
                        logger.info('Inserted new chat_history row with session_id=%s', session_id)
                else:
                    new_session_id = str(uuid.uuid4())
                    row = {
                        'session_id': new_session_id,
                        'student_id': user_id,
                        'session_name': session_name,
                        'llm_response_history': [user_entry, assistant_entry],
                        'study_material_history': [response] if isinstance(response, dict) else [],
                        'created_at': now_iso,
                        'updated_at': now_iso
                    }
                    client.table('chat_history').insert(row).execute()
                    created_session_id = new_session_id
                    logger.info('Inserted new chat_history row for student=%s session_id=%s', user_id, new_session_id)
            except Exception as e:
                logger.error('Failed to upsert chat_history for session: %s', e)
        except Exception as e:
            logger.debug('Skipping persistence to chat_history due to error: %s', e)

        # Persist extracted metadata (subject -> chapter -> concept) into DB so frontend can render
        try:
            # Only persist when response is a dict and contains metadata
            if isinstance(response, dict):
                # local imports to avoid top-level dependency in module import time
                from app.db.db_client import get_supabase_client as _get_supabase_client

                meta = response.get('metadata') or {}
                subject_name = meta.get('subject_name')
                chapter_name = meta.get('chapter_name')
                concept_name = meta.get('concept_name')
                difficulty = meta.get('difficulty_level') or meta.get('difficulty')

                now_iso_local = datetime.now().isoformat()

                if subject_name:
                    client = _get_supabase_client()

                    # Try to find existing subject for this student
                    existing = client.table('subjects').select('*').eq('student_id', user_id).eq('llm_suggested_subject_name', subject_name).execute()
                    subject_id_val = None
                    if existing and getattr(existing, 'data', None):
                        rows = existing.data if isinstance(existing.data, list) else [existing.data]
                        if len(rows) > 0:
                            subject_id_val = rows[0].get('subject_id')
                            logger.info('Found existing subject id=%s for student=%s name=%s', subject_id_val, user_id, subject_name)
                    if not subject_id_val:
                        ins = {
                            'student_id': user_id,
                            'llm_suggested_subject_name': subject_name,
                            'created_at': now_iso_local,
                            'updated_at': now_iso_local
                        }
                        resp = client.table('subjects').insert(ins).execute()
                        if resp and getattr(resp, 'data', None):
                            inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                            subject_id_val = inserted.get('subject_id')
                            logger.info('Inserted subject id=%s for student=%s name=%s', subject_id_val, user_id, subject_name)

                    # Chapters
                    chapter_id_val = None
                    if subject_id_val and chapter_name:
                        existing = client.table('chapters').select('*').eq('student_id', user_id).eq('subject_id', subject_id_val).eq('llm_suggested_chapter_name', chapter_name).execute()
                        if existing and getattr(existing, 'data', None):
                            rows = existing.data if isinstance(existing.data, list) else [existing.data]
                            if len(rows) > 0:
                                chapter_id_val = rows[0].get('chapter_id')
                                logger.info('Found existing chapter id=%s for subject_id=%s name=%s', chapter_id_val, subject_id_val, chapter_name)
                        if not chapter_id_val:
                            ins = {
                                'student_id': user_id,
                                'subject_id': subject_id_val,
                                'llm_suggested_chapter_name': chapter_name,
                                'chapter_order': 0,
                                'description': None,
                                'created_at': now_iso_local,
                                'updated_at': now_iso_local
                            }
                            resp = client.table('chapters').insert(ins).execute()
                            if resp and getattr(resp, 'data', None):
                                inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                                chapter_id_val = inserted.get('chapter_id')
                                logger.info('Inserted chapter id=%s for subject_id=%s name=%s', chapter_id_val, subject_id_val, chapter_name)

                    # Concepts
                    if chapter_id_val and concept_name:
                        existing = client.table('concepts').select('*').eq('student_id', user_id).eq('chapter_id', chapter_id_val).eq('llm_suggested_concept_name', concept_name).execute()
                        if existing and getattr(existing, 'data', None):
                            rows = existing.data if isinstance(existing.data, list) else [existing.data]
                            if len(rows) > 0:
                                logger.info('Found existing concept for chapter_id=%s name=%s', chapter_id_val, concept_name)
                        else:
                            ins = {
                                'student_id': user_id,
                                'chapter_id': chapter_id_val,
                                'llm_suggested_concept_name': concept_name,
                                'concept_order': 0,
                                'description': None,
                                'difficulty_level': difficulty or 'Medium',
                                'created_at': now_iso_local,
                                'updated_at': now_iso_local
                            }
                            resp = client.table('concepts').insert(ins).execute()
                            if resp and getattr(resp, 'data', None):
                                logger.info('Inserted concept for chapter_id=%s name=%s', chapter_id_val, concept_name)
        except Exception as e:
            logger.error('Failed to persist LLM metadata (subjects/chapters/concepts): %s', e)

        # Return the generated response along with the session_id used/created so clients can persist it
        out_session_id = created_session_id or session_id

        return {
            'session_id': out_session_id,
            'llm_response': response
        } if isinstance(response, dict) else (response or {})
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
