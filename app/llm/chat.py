"""
Optimized chat module with pure functional programming approach.
Database operations are deferred until after streaming response completes.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

from app.db.db_client import get_supabase_client
from app.llm.providers import create_chat_agent
from app.llm.process_files.process_files import load_documents_from_files

logger = logging.getLogger(__name__)

# Constants
CHAT_HISTORY_TABLE = "chat_history"
HISTORY_COLUMN = "llm_response_history"

# Pure data structures
class MessageEntry(NamedTuple):
    role: str
    content: str
    timestamp: str

class ChatContext(NamedTuple):
    student_id: str
    session_id: Optional[str]
    prompt: str
    file_context: Optional[str]
    message_history: List[Any]

class DatabaseUpdate(NamedTuple):
    student_id: str
    session_id: Optional[str]
    new_messages: List[MessageEntry]

# Pure helper functions
def create_timestamp() -> str:
    """Create ISO timestamp string."""
    return datetime.now().isoformat()

def safe_parse_datetime(timestamp_str: Optional[str]) -> Optional[datetime]:
    """Safely parse datetime from string."""
    if not timestamp_str:
        return None
    try:
        return datetime.fromisoformat(timestamp_str)
    except Exception:
        return None

def create_message_entry(role: str, content: str, timestamp: Optional[str] = None) -> MessageEntry:
    """Create a message entry with timestamp."""
    return MessageEntry(
        role=role,
        content=content,
        timestamp=timestamp or create_timestamp()
    )

def extract_content_from_response(response_data: Any) -> str:
    """Extract text content from various response formats."""
    if isinstance(response_data, str):
        return response_data
    elif isinstance(response_data, dict):
        return response_data.get('content', str(response_data))
    else:
        return str(response_data)

# Pure functions for message transformation
def transform_raw_history_to_pydantic_messages(raw_history: List[Dict[str, Any]]) -> List[Any]:
    """Transform raw message history to Pydantic AI ModelMessage format."""
    messages = []
    
    for msg in raw_history or []:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        timestamp_str = msg.get("timestamp")
        dt = safe_parse_datetime(timestamp_str)
        
        if role == "user":
            request_msg = {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": content,
                        **({"timestamp": dt} if dt else {}),
                    }
                ],
            }
            messages.append(request_msg)
        else:
            response_msg = {
                "kind": "response",
                "parts": [{"part_kind": "text", "content": content}],
                **({"timestamp": dt} if dt else {}),
            }
            messages.append(response_msg)
    
    return ModelMessagesTypeAdapter.validate_python(messages)

def transform_pydantic_messages_to_simple_history(messages_json: Any) -> List[MessageEntry]:
    """Transform Pydantic AI messages JSON to simple message entries."""
    if not isinstance(messages_json, list):
        messages_json = [messages_json] if messages_json else []
    
    entries = []
    
    for msg in messages_json:
        kind = msg.get("kind")
        
        if kind == "request":
            for part in msg.get("parts", []):
                if part.get("part_kind") == "user-prompt":
                    timestamp = part.get("timestamp")
                    if isinstance(timestamp, datetime):
                        timestamp = timestamp.isoformat()
                    
                    entries.append(create_message_entry(
                        role="user",
                        content=part.get("content", ""),
                        timestamp=timestamp
                    ))
        
        elif kind == "response":
            msg_timestamp = msg.get("timestamp")
            if isinstance(msg_timestamp, datetime):
                msg_timestamp = msg_timestamp.isoformat()
                
            for part in msg.get("parts", []):
                if part.get("part_kind") == "text":
                    entries.append(create_message_entry(
                        role="assistant",
                        content=part.get("content", ""),
                        timestamp=msg_timestamp
                    ))
    
    return entries

def compose_prompt_with_files(base_prompt: str, file_context: Optional[str]) -> str:
    """Compose final prompt with file context if available."""
    if not file_context:
        return base_prompt
    
    return f"Attached documents:\n{file_context}\n\nUser prompt:\n{base_prompt}"

# Pure async functions for file processing
async def extract_file_context(
    file_paths: List[Path], 
    temp_dir: str, 
    api_key: str,
    provider: str,
    model_name: str
) -> Optional[str]:
    """Extract text content from uploaded files."""
    if not file_paths:
        return None
    
    try:
        documents = await load_documents_from_files(
            [str(p) for p in file_paths], 
            temp_dir, 
            api_key, 
            provider, 
            model_name
        )
        return "\n\n".join(documents) if documents else None
    except Exception:
        logger.exception("Failed to extract file context")
        return None

# Pure async functions for database operations
async def fetch_chat_history(student_id: str, session_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch latest chat history row and messages."""
    try:
        client = get_supabase_client()
        
        query = client.table(CHAT_HISTORY_TABLE).select(f"id, updated_at, {HISTORY_COLUMN}")
        
        if session_id:
            query = query.eq("session_id", session_id)
        else:
            query = query.eq("student_id", student_id)
        
        query = query.order("updated_at", desc=True).limit(1)
        
        response = query.execute()
        rows = getattr(response, "data", None) or []
        
        if not rows:
            return None, []
        
        row = rows[0]
        history = row.get(HISTORY_COLUMN) or []
        
        return row, history
        
    except Exception:
        logger.error("Failed to fetch chat history")
        return None, []

async def update_chat_history_row(row_id: Any, new_history: List[Dict[str, Any]]) -> bool:
    """Update existing chat history row with new messages."""
    try:
        client = get_supabase_client()
        
        client.table(CHAT_HISTORY_TABLE).update({
            HISTORY_COLUMN: new_history,
            "updated_at": create_timestamp()
        }).eq("id", row_id).execute()
        
        return True
        
    except Exception:
        logger.error("Failed to update chat history row")
        return False

async def insert_chat_history_row(
    student_id: str, 
    session_id: Optional[str], 
    history: List[Dict[str, Any]]
) -> bool:
    """Insert new chat history row."""
    try:
        client = get_supabase_client()
        
        row_data = {
            'student_id': student_id,
            HISTORY_COLUMN: history,
            'created_at': create_timestamp(),
            'updated_at': create_timestamp()
        }
        
        if session_id:
            row_data['session_id'] = session_id
        
        client.table(CHAT_HISTORY_TABLE).insert(row_data).execute()
        return True
        
    except Exception:
        logger.error("Failed to insert chat history row")
        return False

async def persist_new_messages_async(update_data: DatabaseUpdate) -> bool:
    """Persist new messages to database asynchronously."""
    try:
        # Convert MessageEntry tuples to dicts for database storage
        new_message_dicts = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in update_data.new_messages
        ]
        
        # Fetch existing history
        row, existing_history = await fetch_chat_history(
            update_data.student_id, 
            update_data.session_id
        )
        
        # Merge with existing messages
        merged_history = list(existing_history) + new_message_dicts
        
        # Update or insert
        if row and row.get("id"):
            success = await update_chat_history_row(row["id"], merged_history)
        else:
            success = await insert_chat_history_row(
                update_data.student_id, 
                update_data.session_id, 
                merged_history
            )
        
        if success:
            logger.info(f"Successfully persisted {len(update_data.new_messages)} new messages")
        
        return success
        
    except Exception:
        logger.error("Failed to persist messages")
        return False

# Pure async function for building chat context
async def build_chat_context(
    prompt: str,
    student_id: str,
    session_id: Optional[str],
    file_paths: Optional[List[Path]] = None,
    temp_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> ChatContext:
    """Build complete chat context with history and files."""
    
    # Extract file context if files provided
    file_context = None
    if file_paths and temp_dir and api_key and provider and model_name:
        file_context = await extract_file_context(
            file_paths, temp_dir, api_key, provider, model_name
        )
    
    # Fetch message history
    _, raw_history = await fetch_chat_history(student_id, session_id)
    
    # Transform to Pydantic AI format
    message_history = transform_raw_history_to_pydantic_messages(raw_history)
    
    # Compose final prompt
    final_prompt = compose_prompt_with_files(prompt, file_context)
    
    return ChatContext(
        student_id=student_id,
        session_id=session_id,
        prompt=final_prompt,
        file_context=file_context,
        message_history=message_history
    )

# Main streaming function
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
    """Stream chat response with deferred database operations."""
    
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing API key for provider")
    
    # Build complete chat context
    chat_context = await build_chat_context(
        prompt=prompt,
        student_id=student_id,
        session_id=session_id,
        file_paths=uploaded_files_paths,
        temp_dir=temp_dir,
        api_key=api_key,
        provider=provider,
        model_name=model_name
    )
    
    # Create agent
    agent: Agent = create_chat_agent(provider, model_name, api_key)
    
    logger.info(f"Composed chat prompt: {chat_context.prompt}")
    
    async def event_generator():
        new_messages_for_db = []
        
        try:
            async with agent.run_stream(
                chat_context.prompt, 
                message_history=chat_context.message_history
            ) as result:
                
                # Stream text deltas
                async for text in result.stream_text():
                    yield (json.dumps({"type": "delta", "text": text}).encode("utf-8") + b"\n")
                
                # After streaming completes, prepare database update
                try:
                    new_messages_method = getattr(result, "new_messages_json", None)
                    if callable(new_messages_method):
                        raw_json = new_messages_method()
                        
                        # Parse new messages
                        if isinstance(raw_json, (bytes, bytearray)):
                            new_messages_json = json.loads(raw_json.decode("utf-8"))
                        elif isinstance(raw_json, str):
                            new_messages_json = json.loads(raw_json)
                        else:
                            new_messages_json = raw_json
                        
                        # Transform to MessageEntry format
                        if new_messages_json:
                            new_messages_for_db = transform_pydantic_messages_to_simple_history(
                                new_messages_json
                            )
                
                except Exception as e:
                    logger.warning(f"Failed to extract new messages: {e}")
                
                # Schedule async database update without blocking response
                if new_messages_for_db:
                    update_data = DatabaseUpdate(
                        student_id=chat_context.student_id,
                        session_id=chat_context.session_id,
                        new_messages=new_messages_for_db
                    )
                    
                    # Create background task for database persistence
                    async def persist_messages():
                        try:
                            await persist_new_messages_async(update_data)
                        except Exception as e:
                            logger.error(f"Background message persistence failed: {e}")
                    
                    # Schedule persistence task without awaiting
                    asyncio.create_task(persist_messages())
                
        except Exception as e:
            logger.exception(f"Error during chat streaming: {e}")
            error_payload = {"type": "error", "error": str(e)}
            yield json.dumps(error_payload).encode("utf-8") + b"\n"
    
    return StreamingResponse(
        event_generator(), 
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# Pure utility functions for testing and debugging
def validate_message_entry(entry: MessageEntry) -> bool:
    """Validate message entry structure."""
    return (
        isinstance(entry.role, str) and 
        entry.role in ["user", "assistant"] and
        isinstance(entry.content, str) and
        isinstance(entry.timestamp, str)
    )

def count_messages_by_role(messages: List[MessageEntry]) -> Dict[str, int]:
    """Count messages by role for analytics."""
    counts = {"user": 0, "assistant": 0}
    for msg in messages:
        if msg.role in counts:
            counts[msg.role] += 1
    return counts

def get_latest_message_timestamp(messages: List[MessageEntry]) -> Optional[str]:
    """Get the latest message timestamp."""
    if not messages:
        return None
    
    return max(messages, key=lambda m: m.timestamp).timestamp

# Export main functions
__all__ = [
    "stream_chat_response",
    "build_chat_context", 
    "ChatContext",
    "MessageEntry",
    "DatabaseUpdate"
]
