from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

from typing import Optional
import mimetypes
import json
import os
import tarfile
import uuid
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
# StructuredDict is imported locally where needed to avoid optional dependency errors

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
    logger.info(f'Loading documents from files: {in_paths}')
    for p in in_paths:
        try:
            logger.debug(f'File exists: {p} size={p.stat().st_size}')
        except Exception as e:
            logger.debug(f'Could not stat file {p}: {e}')
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
    results = await asyncio.gather(*[_file_to_text(p, api_key, provider_name, model_name) for p in ordered], return_exceptions=True)
    texts: List[str] = []
    for p, r in zip(ordered, results):
        if isinstance(r, Exception):
            logger.exception(f'Failed to extract text from file {p}: {r}')
            texts.append(f"(error reading file: {r})")
        else:
            texts.append(str(r))

    # format
    parts = [_fmt(p.name, t if isinstance(t, str) and t.strip() else "(no extractable text)") for p, t in zip(ordered, texts)]
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
                    resp = client.table('subjects').insert({
                        'student_id': user_id,
                        'llm_suggested_subject_name': subject_name,
                        'created_at': now_iso,
                        'updated_at': now_iso
                    }).execute()
                    
                    if resp and getattr(resp, 'data', None):
                        inserted = resp.data[0] if isinstance(resp.data, list) else resp.data
                        subject_id = inserted.get('subject_id')
                        logger.info(f'Created subject id={subject_id} for user={user_id} name={subject_name}')
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

async def get_llm_streaming_response(
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
            vector_store, hybrid_retriever = await setup_vector_store_and_retriever(
                chunks, embedding_provider, embedding_model, embedding_api_key
            )
            
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
    
    async for response in get_llm_streaming_response(
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