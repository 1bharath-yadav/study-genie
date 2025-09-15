from __future__ import annotations

import json
import logging
from typing import List, Optional, Any, Dict
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.db.db_client import get_supabase_client
from app.llm.providers import create_chat_agent
# We intentionally avoid importing ModelMessage here to keep persistence format flexible

logger = logging.getLogger(__name__)


def _load_messages_for_session(student_id: str, session_id: Optional[str]) -> List[Dict[str, Any]]:
    """Load stored ModelMessage list from `chat_messages` table for a session.

    We expect each row to have `message_json` which is the serialized ModelMessage
    from pydantic-ai (i.e. result.new_messages_json()). We reconstruct the list
    ordered by created_at ascending.
    """
    client = get_supabase_client()
    try:
        # Load chat_messages
        q = client.table('chat_messages').select('*')
        q = q.eq('session_id', session_id) if session_id else q.eq('student_id', student_id)
        q = q.order('created_at', desc=False)
        resp = q.execute()
        rows = resp.data if getattr(resp, 'data', None) else []

        messages: List[Dict[str, Any]] = []
        for r in rows:
            mj = r.get('message_json')
            if mj:
                if isinstance(mj, list):
                    messages.extend(mj)
                else:
                    messages.append(mj)
            else:
                messages.append({'role': r.get('role') or 'user', 'content': r.get('content'), 'timestamp': r.get('created_at')})

        # Load chat_history entries and append their llm_response_history items (assistant/user pairs)
        ch_q = client.table('chat_history').select('*')
        ch_q = ch_q.eq('session_id', session_id) if session_id else ch_q.eq('student_id', student_id)
        ch_q = ch_q.order('updated_at', desc=False)
        ch_resp = ch_q.execute()
        ch_rows = ch_resp.data if getattr(ch_resp, 'data', None) else []
        for lr in ch_rows:
            history = lr.get('llm_response_history') or []
            for entry in history:
                if isinstance(entry, dict):
                    messages.append(entry)
                else:
                    messages.append({'role': 'assistant', 'content': entry})

        # Sort chronologically by timestamp (if present)
        messages.sort(key=lambda m: m.get('timestamp') or '')
        return messages
    except Exception as e:
        logger.debug('Failed to load messages for session: %s', e)
        return []


def _persist_messages(student_id: str, session_id: Optional[str], messages_json: Any, role: str = 'assistant', content: Optional[str] = None):
    """Persist one or more messages into chat_messages table."""
    client = get_supabase_client()
    now = datetime.now().isoformat()
    try:
        # messages_json may be a list or a single message
        if isinstance(messages_json, list):
            for mj in messages_json:
                # skip system messages (these are agent instructions, not user/assistant content)
                if isinstance(mj, dict) and mj.get('role') == 'system':
                    continue
                row = {
                    'session_id': session_id,
                    'student_id': student_id,
                    'role': (mj.get('role') if isinstance(mj, dict) else role) or role,
                    'content': (mj.get('content') if isinstance(mj, dict) else None) or content,
                    'message_json': mj,
                    'created_at': now,
                }
                client.table('chat_messages').insert(row).execute()
        else:
            # If single message is a system role, skip persisting
            if isinstance(messages_json, dict) and messages_json.get('role') == 'system':
                return
            row = {
                'session_id': session_id,
                'student_id': student_id,
                'role': (messages_json.get('role') if isinstance(messages_json, dict) else role) or role,
                'content': (messages_json.get('content') if isinstance(messages_json, dict) else content),
                'message_json': messages_json,
                'created_at': now,
            }
            client.table('chat_messages').insert(row).execute()
    except Exception as e:
        logger.debug('Failed to persist chat_messages: %s', e)


async def stream_chat_response(prompt: str, student_id: str, provider: str, model_name: str, api_key: str, session_id: Optional[str] = None):
    """Return a StreamingResponse that yields newline-delimited JSON chunks.

    It will load prior messages as message_history, call agent.run_stream, stream text
    deltas to the client, and persist new messages on completion.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail='Missing API key for provider')

    agent = create_chat_agent(provider, model_name, api_key)

    # Load previous messages (if any) to pass as message_history
    message_history: Any = _load_messages_for_session(student_id, session_id)
    

    async def event_generator():
        try:
            async with agent.run_stream(prompt, message_history=message_history) as result:
                # Stream textual deltas as they arrive
                async for delta in result.stream_output(debounce_by=0.03):
                    # delta can be a partial validated object or text depending on output_type
                    # We'll send a simple JSON object with type and text
                    try:
                        out_text = delta if isinstance(delta, str) else (getattr(delta, 'text', str(delta)) or '')
                    except Exception:
                        out_text = str(delta)
                    payload = { 'type': 'delta', 'text': out_text }
                    yield json.dumps(payload).encode('utf-8') + b"\n"

                # After streaming text deltas, attempt to emit final structured output if available
                final = getattr(result, 'output', None)
                if final is not None:
                    payload = { 'type': 'final', 'output': final }
                    yield json.dumps(payload).encode('utf-8') + b"\n"

                # Persist new messages returned by the run (if available)
                new_messages_json = None
                try:
                    # Prefer callable new_messages_json()
                    nm = getattr(result, 'new_messages_json', None)
                    if callable(nm):
                        new_messages_json = nm()
                    else:
                        new_messages_json = nm
                except Exception:
                    new_messages_json = None
                if new_messages_json:
                    _persist_messages(student_id, session_id, new_messages_json, role='assistant')

        except Exception as e:
            logger.exception('Error while streaming chat response: %s', e)
            err_payload = { 'type': 'error', 'error': str(e) }
            yield json.dumps(err_payload).encode('utf-8') + b"\n"

    return StreamingResponse(event_generator(), media_type='application/x-ndjson')
