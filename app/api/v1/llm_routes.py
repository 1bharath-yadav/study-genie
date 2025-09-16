"""
Enhanced LLM API routes with streaming support
"""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import json
import logging
import time
import tempfile
import os
import shutil
from pathlib import Path

from app.llm.orchestrator import orchestrate_prompt, orchestrate_prompt_stream
from ...core.security import get_current_user

# Constants
MAX_TOTAL_UPLOAD_BYTES = 25 * 1024 * 1024
router = APIRouter(prefix="/llm", tags=["llm"])
logger = logging.getLogger(__name__)


# ----- Helper Functions -----

def parse_content_types(selected_content_types: Optional[str]) -> List[str]:
    """Parse content types from JSON or comma-separated string."""
    if not selected_content_types:
        return []
    
    try:
        val = json.loads(selected_content_types)
        if isinstance(val, list):
            return [str(x) for x in val if x]
        elif isinstance(val, str):
            return [s.strip() for s in val.split(',') if s.strip()]
    except Exception:
        return [s.strip() for s in selected_content_types.split(',') if s.strip()]
    
    return []


async def save_uploaded_files(files: List[UploadFile], tmp_dir: str) -> List[str]:
    """Save uploaded files and return paths."""
    saved_paths = []
    total_bytes = 0
    
    for up in files:
        # Sanitize filename
        raw_name = up.filename or f"upload_{int(time.time()*1000)}"
        filename = Path(raw_name).name or f"upload_{int(time.time()*1000)}"
        dest = os.path.join(tmp_dir, filename)
        
        # Stream file in chunks (defensive: attempt to reset file pointer and log progress)
        try:
            try:
                # If underlying file object supports seek, reset to start
                if hasattr(up, 'file') and hasattr(up.file, 'seek'):
                    try:
                        up.file.seek(0)
                    except Exception:
                        pass
            except Exception:
                pass

            logger.info(f"Saving uploaded file to {dest} (orig_name={raw_name})")
            with open(dest, "wb") as f:
                while True:
                    chunk = await up.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                        raise ValueError(f"Total upload exceeds {MAX_TOTAL_UPLOAD_BYTES} bytes")
        except Exception as e:
            logger.exception(f"Failed while saving uploaded file {raw_name}: {e}")
            # attempt to close the upload to free resources then re-raise
            try:
                await up.close()
            except Exception:
                pass
            raise

        # Close the UploadFile to ensure no dangling file handles
        try:
            await up.close()
        except Exception:
            pass
        
        saved_paths.append(dest)
    
    return saved_paths


def cleanup_temp_files(saved_paths: List[str], tmp_dir: str):
    """Clean up temporary files and directory."""
    try:
        for p in saved_paths:
            try:
                os.remove(p)
            except Exception:
                logger.debug(f"Failed to remove file {p}")
        
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        logger.debug("Error during cleanup", exc_info=True)


# ----- Endpoints -----

@router.post("/process-files")
async def process_files_endpoint(
    files: Optional[List[UploadFile]] = File(None),
    user_query: Optional[str] = Form(None),
    selected_content_types: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    session_name: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Process uploaded files and generate learning materials."""
    student_id = current_user["sub"]
    tmp_dir = tempfile.mkdtemp(prefix="uploaded_files_")
    saved_paths = []
    
    try:
        # Save uploaded files if provided
        if files:
            saved_paths = await save_uploaded_files(files, tmp_dir)
        
        # Parse content types
        parsed_selected = parse_content_types(selected_content_types)
        
        # Delegate to orchestrator
        return await orchestrate_prompt(
            prompt=user_query or "Please analyze this content and create study materials.",
            student_id=student_id,
            selected_content_types=parsed_selected,
            uploaded_files_paths=[Path(p) for p in saved_paths],
            session_id=session_id,
            session_name=session_name,
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_files(saved_paths, tmp_dir)


@router.post("/process-files-stream")
async def process_files_stream_endpoint(
    files: Optional[List[UploadFile]] = File(None),
    user_query: Optional[str] = Form(None),
    selected_content_types: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    session_name: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream processed files and learning material generation in real-time."""
    student_id = current_user["sub"]
    tmp_dir = tempfile.mkdtemp(prefix="uploaded_files_")
    saved_paths = []
    
    # Important: save uploaded files before returning the StreamingResponse. UploadFile
    # objects are tied to the request lifecycle and may be closed once the endpoint
    # returns; reading them inside the streaming generator can lead to 'read of closed file'.
    try:
        if files:
            saved_paths = await save_uploaded_files(files, tmp_dir)
    except Exception as e:
        # If saving fails, return a short NDJSON stream that reports the error.
        msg = str(e) if e is not None else "Failed to save uploaded files"

        async def error_stream():
            yield json.dumps({'status': 'error', 'error': msg, 'error_type': 'upload_error'}) + "\n"
            yield json.dumps({'status': 'cleanup_complete'}) + "\n"
        cleanup_temp_files(saved_paths, tmp_dir)
        return StreamingResponse(error_stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    async def generate_stream():
        try:
            # Saved files are available on disk now
            if saved_paths:
                payload = {'status': 'files_saved', 'count': len(saved_paths)}
                logger.info('Emitting stream event: %s', payload)
                # emit as NDJSON: one JSON object per line
                yield json.dumps(payload) + "\n"
            
            # Parse content types
            parsed_selected = parse_content_types(selected_content_types)
            payload = {'status': 'processing_started', 'content_types': parsed_selected}
            logger.info('Emitting stream event: %s', payload)
            yield json.dumps(payload) + "\n"
            
            # Resolve user's preferred provider/model and API key for generation (prefer chat-capable models)
            # Resolve user's preferred provider/model and API key.
            # For generation we must use a chat-capable model; prefer_chat=True to avoid embedding-only models.
            try:
                from app.llm.orchestrator import resolve_model_and_key_for_user
                provider_name, model_name, api_key = await resolve_model_and_key_for_user(student_id, prefer_chat=True)
            except Exception as e:
                # Surface a JSON error line for the client and stop streaming
                err = {'status': 'error', 'error': str(e), 'error_type': 'model_resolution'}
                yield json.dumps(err) + "\n"
                return

            # Import streaming function
            from app.llm.langchain import get_llm_streaming_response

            # Stream the RAG pipeline results as NDJSON objects per line
            async for response in get_llm_streaming_response(
                uploaded_files_paths=[Path(p) for p in saved_paths],
                user_prompt=user_query or "Please analyze this content and create study materials.",
                temp_dir=tmp_dir,
                user_api_key=api_key,
                user_id=student_id,
                provider_name=provider_name,
                model_name=model_name,
                session_id=session_id,
                session_name=session_name
            ):
                # Each yielded `response` is a JSON-serializable dict; emit as a single NDJSON line
                line = json.dumps(response)
                logger.info('Emitting stream event: %s', line)
                yield line + "\n"
                
        except Exception as e:
            error_response = {
                'status': 'error',
                'error': str(e),
                'error_type': 'processing_error'
            }
            # emit as NDJSON
            yield json.dumps(error_response) + "\n"
        finally:
            cleanup_temp_files(saved_paths, tmp_dir)
            # emit lifecycle completion as NDJSON
            yield json.dumps({'status': 'cleanup_complete'}) + "\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.post('/chat-stream')
async def chat_stream_endpoint(
    user_prompt: str = Form(...),
    session_id: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream a regular chat response with message history persistence."""
    student_id = current_user['sub']
    return await orchestrate_prompt_stream(user_prompt, student_id, session_id=session_id)

