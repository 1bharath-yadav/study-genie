from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from app.db.db_client import get_supabase_client
from ...core.security import get_current_user
import logging

router = APIRouter(prefix="/session", tags=["session"])
logger = logging.getLogger(__name__)

@router.get("", include_in_schema=False)
@router.get("/")
async def list_sessions(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List recent learning sessions for the current user (limit 25)."""
    try:
        client = get_supabase_client()
        student_id = current_user.get('sub')

        # Select session_id and map to session_id for the frontend (table: chat_history)
        resp = client.table('chat_history').select('session_id, session_name, created_at, updated_at').eq('student_id', student_id).order('updated_at', desc=True).limit(25).execute()
        data = resp.data or []
        # Deduplicate by id while preserving order and map id -> session_id for frontend
        seen = set()
        sessions = []
        for r in data:
            sid = r.get('session_id')
            if not sid or sid in seen:
                continue
            seen.add(sid)
            sessions.append({
                'session_id': sid,
                'session_name': r.get('session_name'),
                'created_at': r.get('created_at'),
                'updated_at': r.get('updated_at')
            })

        return {'sessions': sessions}
    except Exception as e:
        logger.error('Failed to list sessions: %s', e)
        raise HTTPException(status_code=500, detail='Failed to list sessions')


@router.get("/{session_id}")
async def get_session(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return full history for a specific session."""
    try:
        client = get_supabase_client()
        student_id = current_user.get('sub')

        # Query by session_id in chat_history table
        resp = client.table('chat_history').select('*').eq('student_id', student_id).eq('session_id', session_id).order('created_at', desc=False).execute()
        data = resp.data or []
        if not data:
            raise HTTPException(status_code=404, detail='Session not found')
        # Return combined session object
        # We expect llm_response_history and study_material_history to be present on the rows
        # If multiple rows exist for same session_id, merge their arrays
        combined_history = []
        combined_materials = []
        for r in data:
            h = r.get('llm_response_history') or []
            combined_history.extend(h)
            mats = r.get('study_material_history') or []
            combined_materials.extend(mats)

        return {
            'session_id': session_id,
            'session_name': data[0].get('session_name'),
            'history': combined_history,
            'study_materials': combined_materials
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to fetch session %s: %s', session_id, e)
        raise HTTPException(status_code=500, detail='Failed to fetch session')


@router.delete("/{session_id}")
async def delete_session(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete all chat_history rows for a session (soft delete could be added later)."""
    try:
        client = get_supabase_client()
        student_id = current_user.get('sub')

        client.table('chat_history').delete().eq('student_id', student_id).eq('session_id', session_id).execute()
        return {'deleted': True}
    except Exception as e:
        logger.error('Failed to delete session %s: %s', session_id, e)
        raise HTTPException(status_code=500, detail='Failed to delete session')


@router.patch("/{session_id}")
async def update_session(session_id: str, payload: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_user)):
    """Update session metadata (currently supports renaming session_name)."""
    try:
        client = get_supabase_client()
        student_id = current_user.get('sub')
        new_name = (payload.get('session_name') or '').strip()
        if not new_name:
            raise HTTPException(status_code=400, detail='session_name is required')

        # Update all rows for this session
        now_iso = __import__('datetime').datetime.now().isoformat()
        client.table('chat_history').update({'session_name': new_name, 'updated_at': now_iso}).eq('student_id', student_id).eq('session_id', session_id).execute()
        return {'updated': True, 'session_name': new_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to update session %s: %s', session_id, e)
        raise HTTPException(status_code=500, detail='Failed to update session')
