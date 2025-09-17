import asyncio
import json
import logging
import mimetypes
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from app.llm.providers import create_learning_agent

logger = logging.getLogger(__name__)

# --- small helpers

def _ext(p: Path) -> str:
    return p.suffix.lower()


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


def _fmt(name: str, content: str) -> str:
    return f"# {name}\n{content.strip()}\n"


def _is_archive(p: Path) -> bool:
    return _ext(p) in {".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".rar", ".7z"}


async def _lc_load(path: Path) -> str:
    try:
        # prioritize specific, then general
        if _ext(path) == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            docs = await PyPDFLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".docx", ".doc"}:
            from langchain_community.document_loaders import Docx2txtLoader
            docs = await Docx2txtLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".ppt", ".pptx", ".odp"}:
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            docs = await UnstructuredPowerPointLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".xlsx", ".xls"}:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            docs = await UnstructuredExcelLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".epub"}:
            from langchain_community.document_loaders import UnstructuredEPubLoader
            docs = await UnstructuredEPubLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".html", ".htm"}:
            from langchain_community.document_loaders import BSHTMLLoader
            docs = await BSHTMLLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        if _ext(path) in {".csv", ".tsv"}:
            from langchain_community.document_loaders import CSVLoader
            docs = await CSVLoader(str(path)).aload()
            return "\n\n".join(d.page_content for d in docs) or ""
        # Prefer the new langchain-unstructured package's loader. Fall back to the
        # older UnstructuredFileLoader when the newer package is not available.
        
        # New package: import from langchain_unstructured
        from langchain_unstructured import UnstructuredLoader

        loader = UnstructuredLoader(str(path))
        # prefer async AIO interface if provided
        if hasattr(loader, 'aload'):
            docs = await loader.aload()
        else:
            docs = await asyncio.to_thread(lambda: loader.load())
        return "\n\n".join(getattr(d, 'page_content', str(d)) for d in docs) or ""
    
    except Exception:
        return ""


# --- archive expansion

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
            return []
        for root, _, files in os.walk(tdir):
            for f in files:
                out.append(Path(root) / f)
    except Exception:
        return []
    return out


# --- loaders & simple parsers

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
        return "\n\n".join(parts).replace("``````", "")
    except Exception:
        try:
            raw = json.loads(_read_text(path))
            return json.dumps(raw.get("cells", raw), indent=2)
        except Exception:
            return ""
    return ""


# --- OCR / transcription fallbacks

def _ocr_image_sync(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def _ocr_pdf_sync(path: Path) -> str:
    try:
        from pypdf import PdfReader
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import pytesseract
        from pdf2image import convert_from_path
        pages = convert_from_path(str(path))
        return "\n\n".join(pytesseract.image_to_string(p) or "" for p in pages)
    except Exception:
        return ""


def _transcribe_sync(path: Path) -> str:
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(path))
        return result.get("text", "").strip()
    except Exception:
        return ""


# --- agent fallback (pydantic-ai)

async def _agent_fallback(
    path: Path,
    meta_text: str,
    api_key,
    provider,
    model_name,
    file_url: Optional[str] = None,
) -> str:
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


# --- single file to text

async def _file_to_text(path: Path, api_key, provider_name, model_name) -> str:
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
    return await _agent_fallback(path, meta, api_key, provider_name, model_name)


async def load_documents_from_files(file_paths: List[str], temp_dir: str, api_key: str, provider_name: str, model_name: str) -> List[str]:
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


async def perform_document_chunking(documents: List[str]) -> List:
    # minimal chunking wrapper for reuse
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        keep_separator=True
    )

    lc_docs: List[Document] = [Document(page_content=d) for d in documents]
    chunks = text_splitter.split_documents(lc_docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': idx,
            'chunk_size': len(chunk.page_content),
            'chunk_index': idx,
            'total_chunks': len(chunks)
        })
    return chunks
