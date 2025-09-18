from typing import Optional, Any
from datetime import datetime
import json

from app.db.db_client import get_supabase_client
from app.models import LearningActivityRequest
from app.services.db import safe_extract_data


def _assistant_content_from_response(response: Any) -> str:
    if isinstance(response, dict):
        summary = response.get('summary')
        if summary is not None:
            return str(summary)
        meta = response.get('metadata')
        if isinstance(meta, dict) and meta.get('summary') is not None:
            return str(meta.get('summary'))
        try:
            return json.dumps(response)
        except Exception:
            return str(response)
    return str(response)


def upsert_chat_history(student_id: str, session_id: Optional[str], session_name: Optional[str], userprompt: str, response: Any) -> Optional[str]:
    """Insert or append a user+assistant pair into chat_history.

    Returns the session_id used/created, or None on failure.
    """
    try:
        # Ensure session_name is a non-empty, trimmed string
        def _normalize_name(name: Optional[str]) -> str:
            if not name:
                return ''
            return ' '.join(name.split())

        session_name = _normalize_name(session_name) or ''

        client = get_supabase_client()
        now_iso = datetime.now().isoformat()

        user_entry = {'role': 'user', 'content': userprompt, 'timestamp': now_iso}
        assistant_entry = {'role': 'assistant', 'content': _assistant_content_from_response(response), 'timestamp': now_iso}

        if session_id:
            existing = client.table('chat_history').select('*').eq('session_id', session_id).execute()
            if existing and getattr(existing, 'data', None) and len(existing.data) > 0:
                rec = existing.data[0]
                history = rec.get('llm_response_history') or []
                history.extend([user_entry, assistant_entry])
                # preserve existing study_material_history if present
                # Prefer provided session_name, otherwise keep existing session_name
                update_obj = {'llm_response_history': history, 'session_name': session_name or rec.get('session_name') or '', 'updated_at': now_iso}
                client.table('chat_history').update(update_obj).eq('session_id', session_id).execute()
                return session_id
            else:
                row = {'session_id': session_id, 'student_id': student_id, 'session_name': session_name, 'llm_response_history': [user_entry, assistant_entry], 'study_material_history': [], 'created_at': now_iso, 'updated_at': now_iso}
                client.table('chat_history').insert(row).execute()
                return session_id

        # create new session id
        import uuid

        new_session_id = str(uuid.uuid4())
        row = {'session_id': new_session_id, 'student_id': student_id, 'session_name': session_name, 'llm_response_history': [user_entry, assistant_entry], 'study_material_history': [], 'created_at': now_iso, 'updated_at': now_iso}
        client.table('chat_history').insert(row).execute()
        return new_session_id
    except Exception:
        return None


async def get_learning_history_for_user(student_id: str):
    """Return all chat_history rows for a student as a list of dicts.

    This is an async-friendly wrapper used by export endpoints.
    """
    client = get_supabase_client()
    try:
        resp = client.table('chat_history').select('*').eq('student_id', student_id).order('created_at', desc=False).execute()
        if getattr(resp, 'data', None) is None:
            return []
        return resp.data
    except Exception:
        return []


def save_learning_activity(student_id: str, activity: LearningActivityRequest) -> dict:
    """Persist a learning activity row into the learning_activities table.

    Resolves concept_id by matching llm_suggested_concept_name (if provided) for the student.
    Returns the inserted row (as dict) on success or raises an exception on failure.
    """
    client = get_supabase_client()
    try:
        # Best-effort resolution of subject -> chapter -> concept IDs using
        # the llm_suggested_* name columns.
        subject_id = None
        chapter_id = None
        concept_id = None

        if activity.subject_name:
            try:
                resp = client.table('subjects').select('subject_id').eq('student_id', student_id).ilike('llm_suggested_subject_name', activity.subject_name).limit(1).execute()
                rows = safe_extract_data(resp)
                if rows:
                    subject_id = rows[0].get('subject_id')
            except Exception:
                subject_id = None

        if activity.chapter_name:
            try:
                qb = client.table('chapters').select('chapter_id')
                qb = qb.eq('student_id', student_id)
                if subject_id:
                    qb = qb.eq('subject_id', subject_id)
                qb = qb.ilike('llm_suggested_chapter_name', activity.chapter_name).limit(1)
                resp = qb.execute()
                rows = safe_extract_data(resp)
                if rows:
                    chapter_id = rows[0].get('chapter_id')
            except Exception:
                chapter_id = None

        if activity.concept_name:
            try:
                resp = client.table('concepts').select('concept_id').eq('student_id', student_id).ilike('llm_suggested_concept_name', activity.concept_name).limit(1).execute()
                rows = safe_extract_data(resp)
                if rows:
                    concept_id = rows[0].get('concept_id')
            except Exception:
                concept_id = None

        score = None
        if activity.total_questions and activity.total_questions > 0:
            score = round((activity.correct_answers / activity.total_questions) * 100, 2)

        payload = {
            'subject_name': activity.subject_name,
            'chapter_name': activity.chapter_name,
            'concept_name': activity.concept_name,
            'session_id': activity.session_id,
            'content_source': activity.content_source,
            'difficulty_level': activity.difficulty_level.value if hasattr(activity.difficulty_level, 'value') else str(activity.difficulty_level),
            'correct_answers': activity.correct_answers,
            'total_questions': activity.total_questions,
        }

        row = {
            'student_id': student_id,
            'activity_type': activity.activity_type.value if hasattr(activity.activity_type, 'value') else str(activity.activity_type),
            'related_subject_id': subject_id,
            'related_chapter_id': chapter_id,
            'related_concept_id': concept_id,
            'payload': payload,
            'score': score,
            'time_spent_seconds': activity.time_spent,
            'created_at': datetime.utcnow().isoformat()
        }

        insert_resp = client.table('student_activity').insert(row).execute()
        inserted = safe_extract_data(insert_resp)
        if inserted and len(inserted) > 0:
            return inserted[0]
        return row
    except Exception:
        raise
