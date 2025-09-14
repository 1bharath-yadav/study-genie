"""
Functional LLM API routes
Pure functional endpoints for LLM operations
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional

from app.llm.langchain import get_llm_response

from ...core.security import get_current_user
# Note: this module delegates file processing to get_llm_response in langchain
# from ...llm.langchain import get_llm_response
from ...services.api_key_service import get_api_key_for_provider
from ...llm.providers import get_user_model_preferences
from fastapi import File, UploadFile, Form
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
    session_id: Optional[str] = Form(None),
    session_name: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Accept uploaded files, run the processing pipeline, and return extracted documents."""
    # Get student's preferred model for chat
    student_id = current_user["sub"]
    preferences = get_user_model_preferences(student_id)
    preferred_model = None
    for pref in preferences:
        if pref.get("use_for_chat"):
            preferred_model = pref
            break
    if not preferred_model:
        # Do NOT auto-default a model. Require users to activate a model.
        logger.debug("No preferred model found for student_id=%s, prefs=%s", student_id, preferences)
        raise HTTPException(status_code=400, detail=(
            "No preferred model found for the user. Please activate a model for chat in Settings "
            "and ensure an API key exists for that provider before processing files."
        ))
    # preferred_model comes from the persisted `model_preferences` row and contains
    # `model_id` and optional `provider_name`. `model_id` is formatted as
    # "{provider}-{model_name}". Parse those values safely.
    model_id = preferred_model.get("model_id")
    provider_name = preferred_model.get("provider_name") or None
    if not model_id:
        logger.error("Invalid model preference for student %s: missing model_id. pref=%s", student_id, preferred_model)
        raise HTTPException(status_code=500, detail={
            "error": "Invalid model preference (missing model_id)",
            "preference": preferred_model
        })
    parts = model_id.split("-", 1)
    if len(parts) != 2:
        # Fallback: if provider_name is provided separately use it, otherwise error
        logger.warning("Unexpected model_id format for student %s: %s", student_id, model_id)
        if provider_name:
            model_name = model_id
        else:
            raise HTTPException(status_code=400, detail={
                "error": "Malformed model_id in preferences",
                "preference": preferred_model
            })
    else:
        provider_name = provider_name or parts[0]
        model_name = parts[1]

    # Get API key (this may raise/return None if no key exists)
    # Log the preference and provider_name to aid debugging
    logger.debug("Preference for student=%s resolved to provider=%s model=%s pref=%s", student_id, provider_name, model_name, preferred_model)
    api_key = await get_api_key_for_provider(student_id, provider_name)
    if not api_key:
        logger.error("Missing API key for student=%s provider=%s pref=%s", student_id, provider_name, preferred_model)
        raise HTTPException(status_code=400, detail={
            "error": f"No API key found for provider: {provider_name}",
            "preference": preferred_model
        })

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

        # Call get_llm_response with whatever files were saved (may be empty list)
        resp = await get_llm_response(
            uploaded_files_paths=[Path(p) for p in saved_paths],
            userprompt=user_query or "Please analyze this content and create study materials.",
            temp_dir=tmp_dir,
            user_api_key=api_key,
            user_id=student_id,
            provider_name=provider_name,
            model_name=model_name,
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
