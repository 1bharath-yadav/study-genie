"""
Functional LLM API routes
Pure functional endpoints for LLM operations
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional

from app.llm.orchestrator import orchestrate_prompt, orchestrate_prompt_stream

from ...core.security import get_current_user
# Note: this module delegates file processing to get_llm_response in langchain
# from ...llm.langchain import get_llm_response
 
from fastapi import File, UploadFile, Form
import json
import logging
import time
import tempfile
import os
import shutil
from pathlib import Path

# Max total upload size in bytes (25 MiB). Adjust as needed for your deployment.
MAX_TOTAL_UPLOAD_BYTES = 25 * 1024 * 1024

# Router and request models expected by the module
router = APIRouter(prefix="/llm", tags=["llm"])

logger = logging.getLogger(__name__)

@router.post("/process-files")
async def process_files_endpoint(
    # Make files optional so prompt-only requests (no file uploads) are accepted.
    files: Optional[List[UploadFile]] = File(None),
    user_query: Optional[str] = Form(None),
    selected_content_types: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    session_name: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Accept uploaded files, run the processing pipeline, and return extracted documents."""
    # Delegate model/key resolution and processing to orchestrator
    student_id = current_user["sub"]

    # Save uploaded files to a temp directory and collect paths, then delegate
    # the rest of the processing to `get_llm_response` which performs ingestion,
    # chunking, embeddings and generation.
    tmp_dir = tempfile.mkdtemp(prefix="uploaded_files_")
    saved_paths: List[str] = []
    total_bytes = 0
    try:
        # If files were provided, save them. Otherwise proceed with an empty list
        if files:
            for up in files:
                # Sanitize filename to avoid directory traversal and other issues
                raw_name = up.filename or f"upload_{int(time.time()*1000)}"
                filename = Path(raw_name).name
                # Prevent empty or unsafe filenames
                if not filename:
                    filename = f"upload_{int(time.time()*1000)}"
                dest = os.path.join(tmp_dir, filename)

                # Stream the upload in chunks to avoid loading the whole file into memory
                with open(dest, "wb") as f:
                    while True:
                        chunk = await up.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        total_bytes += len(chunk)
                        if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                            raise ValueError(f"Total upload size exceeds limit of {MAX_TOTAL_UPLOAD_BYTES} bytes")

                saved_paths.append(dest)

        # Parse selected_content_types which may be JSON-encoded or a comma-separated string
        parsed_selected: List[str] = []
        if selected_content_types:
            try:
                val = json.loads(selected_content_types)
                if isinstance(val, list):
                    parsed_selected = [str(x) for x in val if x]
                elif isinstance(val, str):
                    parsed_selected = [s.strip() for s in val.split(',') if s.strip()]
            except Exception:
                # Fallback: treat as comma-separated list
                parsed_selected = [s.strip() for s in (selected_content_types or '').split(',') if s.strip()]

        # Delegate to orchestrator which resolves provider/model/key and runs the flow
        resp = await orchestrate_prompt(
            prompt=user_query or "Please analyze this content and create study materials.",
            student_id=student_id,
            selected_content_types=parsed_selected,
            uploaded_files_paths=[Path(p) for p in saved_paths],
            session_id=session_id,
            session_name=session_name,
        )

        return resp
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files and directory robustly
        try:
            if saved_paths:
                for p in saved_paths:
                    try:
                        os.remove(p)
                    except Exception:
                        logger.debug("Failed to remove uploaded file %s", p, exc_info=True)
            # Remove the directory and any leftover files
            try:
                shutil.rmtree(tmp_dir)
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug("Failed to remove temp dir %s", tmp_dir, exc_info=True)
        except Exception:
            # Best-effort cleanup only; do not mask original exceptions
            logger.debug("Error during cleanup of uploaded files", exc_info=True)


# Streaming endpoint removed. Use /process-files for synchronous processing only.


@router.post('/chat-stream')
async def chat_stream_endpoint(
    user_prompt: str = Form(...),
    session_id: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream a regular chat response (token-wise) and persist message history.

    This endpoint is intended for conversational flows when the user has not
    requested structured outputs like quizzes/flashcards. It streams newline
    delimited JSON events (ndjson) to the client.
    """
    student_id = current_user['sub']

    # Delegate streaming orchestration to central orchestrator
    return await orchestrate_prompt_stream(user_prompt, student_id, session_id=session_id)
