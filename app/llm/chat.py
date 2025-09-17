# chat_stream.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.db.db_client import get_supabase_client
from app.llm.providers import create_chat_agent
from app.llm.process_files.process_files import load_documents_from_files

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
TABLE = "chat_history"
COLUMN = "llm_response_history"

# --------------------------------------------------------------------
# Message transforms
# --------------------------------------------------------------------
def _to_pydantic_messages(raw_history: List[Dict[str, Any]]):
    """
    Convert [{role, content, timestamp}] into Pydantic AI ModelMessage list.
    - user    -> ModelRequest(parts=[UserPromptPart]), timestamp on the part
    - others  -> ModelResponse(parts=[TextPart]), timestamp on the message
    """
    msgs: List[Dict[str, Any]] = []
    for m in raw_history or []:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        ts = m.get("timestamp")
        dt = None
        try:
            dt = datetime.fromisoformat(ts) if ts else None
        except Exception:
            dt = None

        if role == "user":
            req = {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": content,
                        **({"timestamp": dt} if dt else {}),
                    }
                ],
            }
            msgs.append(req)
        else:
            resp = {
                "kind": "response",
                "parts": [{"part_kind": "text", "content": content}],
                **({"timestamp": dt} if dt else {}),
            }
            msgs.append(resp)

    return ModelMessagesTypeAdapter.validate_python(msgs)  # -> List[ModelMessage] [20]

def _dt_to_iso(x: Any) -> Optional[str]:
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, str):
        return x
    return None

def _to_simple_history_from_messages_json(messages_json: Union[List[Any], Any]) -> List[Dict[str, Any]]:
    """
    Convert Pydantic-AI messages JSON (result.new_messages_json()) into
    [{'role': 'user'|'assistant', 'content': str, 'timestamp': iso}] items.
    """
    if not isinstance(messages_json, list):
        messages_json = [messages_json]

    out: List[Dict[str, Any]] = []
    for m in messages_json:
        kind = m.get("kind")
        if kind == "request":
            for p in m.get("parts", []):
                if p.get("part_kind") == "user-prompt":
                    out.append(
                        {
                            "role": "user",
                            "content": p.get("content") or "",
                            "timestamp": _dt_to_iso(p.get("timestamp")),
                        }
                    )
        elif kind == "response":
            ts = _dt_to_iso(m.get("timestamp"))
            for p in m.get("parts", []):
                if p.get("part_kind") == "text":
                    out.append(
                        {
                            "role": "assistant",
                            "content": p.get("content") or "",
                            "timestamp": ts,
                        }
                    )
    return out

# --------------------------------------------------------------------
# Database helpers (Supabase)
# --------------------------------------------------------------------
def _select_latest_history_row(
    student_id: str, session_id: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (row, llm_response_history[]) for the latest chat_history row
    filtered by session_id or student_id.
    """
    client = get_supabase_client()
    q = client.table(TABLE).select("id, updated_at, " + COLUMN)
    q = q.eq("session_id", session_id) if session_id else q.eq("student_id", student_id)
    q = q.order("updated_at", desc=True).limit(1)
    resp = q.execute()
    rows = getattr(resp, "data", None) or []
    if not rows:
        return None, []
    row = rows[0]  # ✅ pick the first row instead of assigning the whole list
    history = row.get(COLUMN) or []
    return row, history


def _update_history_row(row_id: Any, new_history: List[Dict[str, Any]]):
    """
    Update a single chat_history row's llm_response_history by id.
    """
    client = get_supabase_client()
    (
        client.table(TABLE)
        .update({COLUMN: new_history})  # update must use filters (eq id) [1]
        .eq("id", row_id)
        .execute()
    )

def _insert_history_row(student_id: str, session_id: Optional[str], history: List[Dict[str, Any]]):
    """
    Insert a new chat_history row if none exists.
    """
    client = get_supabase_client()
    row = {"student_id": student_id, COLUMN: history}
    if session_id:
        row["session_id"] = session_id
    client.table(TABLE).insert(row).execute()  # standard insert pattern [15]

def _append_to_llm_history(student_id: str, session_id: Optional[str], additions: List[Dict[str, Any]]):
    """
    Append new entries to the latest row, or create a new row if none exists.
    """
    row, existing = _select_latest_history_row(student_id, session_id)
    merged = list(existing) + list(additions)
    if row and row.get("id") is not None:
        _update_history_row(row["id"], merged)  # update by primary key [1]
    else:
        _insert_history_row(student_id, session_id, merged)  # insert new row [15]

# --------------------------------------------------------------------
# Streaming endpoint
# --------------------------------------------------------------------
async def stream_chat_response(
    prompt: str,
    student_id: str,
    provider: str,
    model_name: str,
    api_key: str,
    session_id: Optional[str] = None,
    uploaded_files_paths: Optional[List[Path]] = None,
    temp_dir: Optional[str] = None,
):
    """
    Streams NDJSON with deltas and persists new messages into chat_history.llm_response_history.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing API key for provider")  # basic validation [11]

    # 1) Load prior history from chat_history -> llm_response_history
    row, raw_history = _select_latest_history_row(student_id, session_id)  # single-row fetch with filters [11]
    message_history = _to_pydantic_messages(raw_history)  # validated ModelMessage list [20]

    # If files were uploaded, extract their text and prepend as context to the prompt
    if uploaded_files_paths and len(uploaded_files_paths) > 0:
        try:
            temp = temp_dir or "/tmp"
            docs = await load_documents_from_files([str(p) for p in uploaded_files_paths], temp, api_key, provider, model_name)
            if docs:
                file_context = "\n\n".join(docs)
                prompt = f"Attached documents:\n{file_context}\n\nUser prompt:\n{prompt}"
        except Exception:
            logger.exception("Failed to extract or include uploaded file context into chat prompt")

    # 2) Create agent and run with streaming
    agent: Agent = create_chat_agent(provider, model_name, api_key)  # construct agent per provider+model [10]

    logger.info(f"composed chat prompt:{prompt}")
    async def event_generator():
        try:
            async with agent.run_stream(prompt, message_history=message_history) as result:  # run_stream ctx manager [10]
                # Stream text growth events; stream_text yields growing text until completion
                async for text in result.stream_text():  # documented streaming pattern [8][6]
                    yield (json.dumps({"type": "delta", "text": text}).encode("utf-8") + b"\n")

                # Persist newly created messages for this run
                new_messages_json = None
                try:
                    nm = getattr(result, "new_messages_json", None)
                    raw_json = nm() if callable(nm) else nm
                    if raw_json:
                        if isinstance(raw_json, (bytes, bytearray)):
                            new_messages_json = json.loads(raw_json.decode("utf-8"))
                        elif isinstance(raw_json, str):
                            new_messages_json = json.loads(raw_json)
                        else:
                            new_messages_json = raw_json
                except Exception:
                    new_messages_json = None

                if new_messages_json:
                    additions = _to_simple_history_from_messages_json(new_messages_json)
                    if additions:
                        _append_to_llm_history(student_id, session_id, additions)

                                # Optionally emit a final "done" record
                # yield (json.dumps({"type": "done"}).encode("utf-8") + b"\n")

        except Exception as e:
            logger.exception("Error while streaming chat response: %s", e)
            err_payload = {"type": "error", "error": str(e)}
            yield json.dumps(err_payload).encode("utf-8") + b"\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")  # NDJSON stream response [10]
