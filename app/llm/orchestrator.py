from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path
import logging
import json

from fastapi.responses import StreamingResponse

from app.llm.langchain import stream_structured_content
from app.llm.chat import stream_chat_response
from app.llm.providers import get_user_model_preferences
from app.services.api_key_service import get_api_key_for_provider

logger = logging.getLogger(__name__)


async def resolve_model_and_key_for_user(student_id: str, prefer_chat: bool = True, provider_hint: Optional[str] = None) -> Tuple[str, str, str]:
    """Return (provider_name, model_name, api_key) for the student.

    Strategy:
    - If provider_hint is given, prefer it (but still require a user API key for that provider).
    - Otherwise, prefer a model preference where use_for_chat/use_for_embedding matches prefer_chat.
    - If no explicit preference is found, raise a ValueError so callers can surface a clear error.
    """
    prefs = get_user_model_preferences(student_id)
    candidate = None
    if provider_hint:
        # Find preference that matches provider_hint if possible
        for p in prefs:
            if p.get('provider_name') == provider_hint or (p.get('model_id') or '').startswith(f"{provider_hint}-"):
                candidate = p
                break

    if not candidate:
        for p in prefs:
            if prefer_chat and p.get('use_for_chat'):
                candidate = p
                break
            if not prefer_chat and p.get('use_for_embedding'):
                candidate = p
                break

    if not candidate:
        # Last resort: take first preference if any
        if prefs:
            candidate = prefs[0]

    if not candidate:
        raise ValueError('No model preference found for user. Please enable a model in Settings.')

    model_id = candidate.get('model_id')
    provider_name = candidate.get('provider_name') or None
    if not model_id:
        raise ValueError('Model preference missing model_id.')

    # model_id is formatted as "{provider}-{model_name}" in our app
    parts = model_id.split('-', 1)
    if len(parts) == 2:
        provider_name = provider_name or parts[0]
        model_name = parts[1]
    else:
        model_name = model_id

    # Ensure we have a provider name now
    if not provider_name:
        raise ValueError('Cannot determine provider name for model preference.')

    # fetch user's API key for that provider
    api_key = await get_api_key_for_provider(student_id, provider_name)
    if not api_key:
        raise ValueError(f'No API key found for provider: {provider_name}. Please add it in Settings.')

    return provider_name, model_name, api_key


async def orchestrate_prompt_stream(
    prompt: str,
    student_id: str,
    session_id: Optional[str] = None,
    uploaded_files_paths: Optional[List[Path]] = None,
    selected_content_types: Optional[str] = None,
    temp_dir: Optional[str] = None,
    provider_hint: Optional[str] = None,
) -> StreamingResponse:
    """Return a StreamingResponse for live streaming flows.

    Behavior:
    - If `selected_content_types` is provided (non-empty) or `uploaded_files_paths` is not empty,
      route to the structured RAG/learning-materials streaming pipeline.
    - Otherwise, delegate to normal chat streaming.

    The structured pipeline returns an async generator of dicts; this function wraps those
    dicts into NDJSON lines and performs filesystem cleanup for `temp_dir` when streaming finishes.
    """
    # Resolve user's preferred model/key (prefer chat-capable models)
    try:
        provider_name, model_name, api_key = await resolve_model_and_key_for_user(student_id, prefer_chat=True, provider_hint=provider_hint)
    except Exception:
        # Return a short NDJSON error stream
        async def err_gen():
            yield json.dumps({'status': 'error', 'error': 'model resolution failed', 'error_type': 'model_resolution'}) + "\n"
        return StreamingResponse(err_gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    # Decide whether to use structured pipeline
    use_structured = False
    if selected_content_types:
        # non-empty string or JSON list indicates structured request
        try:
            parsed = json.loads(selected_content_types)
            if isinstance(parsed, list) and len(parsed) > 0:
                use_structured = True
            elif isinstance(parsed, str) and parsed.strip():
                use_structured = True
        except Exception:
            # fallback: comma-separated values
            if isinstance(selected_content_types, str) and selected_content_types.strip():
                use_structured = True

    # NOTE: uploaded files alone should NOT force structured pipeline. Only use structured
    # when `selected_content_types` explicitly contains items. Files can be passed into
    # the chat streaming flow when content types are off.

    from app.config import settings

    if use_structured:
        # Ensure we have data shapes expected by the RAG pipeline
        paths = uploaded_files_paths or []

        async def structured_gen():
            try:
                # Provide a sensible temp_dir: prefer temp configured in settings, or provided temp_dir
                base_temp = temp_dir or getattr(settings, 'TEMP_DIR', '/tmp')
                async for resp in stream_structured_content(
                    paths,
                    prompt,
                    base_temp,
                    api_key,
                    student_id,
                    provider_name,
                    model_name,
                    session_id,
                    None
                ):
                    yield json.dumps(resp) + "\n"
            finally:
                # Best-effort cleanup of temp_dir if provided
                # Only cleanup if the temp_dir is not the persistent PERSIST_DIR uploads location
                try:
                    persist_root = getattr(settings, 'PERSIST_DIR', None)
                    if temp_dir and persist_root and not str(temp_dir).startswith(str(persist_root)):
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    logger.debug("Failed to cleanup temp_dir %s", temp_dir, exc_info=True)

        return StreamingResponse(structured_gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    # Default: chat streaming
    # Forward uploaded file paths and temp_dir so chat can extract file text and include it in the prompt.
    return await stream_chat_response(
        prompt,
        student_id,
        provider_name,
        model_name,
        api_key,
        session_id,
        uploaded_files_paths=uploaded_files_paths,
        temp_dir=temp_dir,
    )
