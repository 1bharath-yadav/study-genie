from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging

from fastapi.responses import StreamingResponse

from app.llm.langchain import get_llm_response
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


async def orchestrate_prompt(
    prompt: str,
    student_id: str,
    selected_content_types: Optional[List[str]] = None,
    uploaded_files_paths: Optional[List[Path]] = None,
    session_id: Optional[str] = None,
    session_name: Optional[str] = None,
    provider_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Main orchestrator for non-streaming flows.

    - If structured content is requested (selected_content_types non-empty) or uploaded files exist,
      this will call the RAG/structured pipeline `get_llm_response` and return its dict result.
    - Otherwise, it will call the structured pipeline as a fallback (non-streaming chat) and return its result.
    """
    selected = selected_content_types or []
    uploaded = uploaded_files_paths or []

    # Resolve model and api key (prefer chat use case)
    provider_name, model_name, api_key = await resolve_model_and_key_for_user(student_id, prefer_chat=True, provider_hint=provider_hint)

    # If user requested structured types or uploaded files exist -> structured pipeline (non-streaming wrapper)
    if len(selected) > 0 or len(uploaded) > 0:
        paths = uploaded or []
        # get_llm_response returns a finalized dict (backwards compatible)
        resp = await get_llm_response(
            uploaded_files_paths=paths,
            user_prompt=prompt,
            temp_dir='/tmp',
            user_api_key=api_key,
            user_id=student_id,
            provider_name=provider_name,
            model_name=model_name,
            session_id=session_id,
            session_name=session_name,
        )
        return resp

    # No structured request and no files -> use the non-streaming wrapper as a quick reply
    resp = await get_llm_response(
        uploaded_files_paths=[],
        user_prompt=prompt,
        temp_dir='/tmp',
        user_api_key=api_key,
        user_id=student_id,
        provider_name=provider_name,
        model_name=model_name,
        session_id=session_id,
        session_name=session_name,
    )
    return resp


async def orchestrate_prompt_stream(
    prompt: str,
    student_id: str,
    session_id: Optional[str] = None,
    provider_hint: Optional[str] = None,
) -> StreamingResponse:
    """Return a StreamingResponse for live chat streaming when the user wants a conversational flow.

    This resolves the user's model and API key then delegates to `stream_chat_response` which
    returns a StreamingResponse that yields newline-delimited JSON events.
    """
    provider_name, model_name, api_key = await resolve_model_and_key_for_user(student_id, prefer_chat=True, provider_hint=provider_hint)
    return await stream_chat_response(prompt, student_id, provider_name, model_name, api_key, session_id)
