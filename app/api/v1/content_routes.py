from typing import List
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user
from app.db.db_client import get_supabase_client
from app.models import SubjectStructure, SubjectInfo, ChapterInfo, ConceptInfo, DifficultyLevel

router = APIRouter(prefix="/subjects", tags=["content"])

logger = logging.getLogger(__name__)


@router.get("/", response_model=List[SubjectStructure])
async def list_subjects(current_user: dict = Depends(get_current_user)):
    """Return subjects with nested chapters and concepts for the current user."""
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client = get_supabase_client()
    try:
        # Fetch subjects
        subj_resp = client.table('subjects').select('*').eq('student_id', user_id).execute()
        subjects = getattr(subj_resp, 'data', []) or []

        out: List[SubjectStructure] = []
        for s in subjects:
            sid = s.get('subject_id')
            # prefer suggested name but fall back to other fields or a fallback label
            subject_name = s.get('llm_suggested_subject_name') or s.get('subject_name') or f"Subject {sid}"
            desc = s.get('description') if isinstance(s.get('description'), str) and s.get('description') != '' else None

            # Fetch chapters for subject
            ch_resp = client.table('chapters').select('*').eq('student_id', user_id).eq('subject_id', sid).order('chapter_order', desc=False).execute()
            chapters = getattr(ch_resp, 'data', []) or []
            chapter_objs = []
            total_concepts = 0
            for c in chapters:
                cid = c.get('chapter_id')
                ch_name = c.get('llm_suggested_chapter_name') or c.get('chapter_name') or f"Chapter {cid}"
                ch_desc = c.get('description') if isinstance(c.get('description'), str) and c.get('description') != '' else None

                # Fetch concepts for chapter
                con_resp = client.table('concepts').select('*').eq('student_id', user_id).eq('chapter_id', cid).order('concept_order', desc=False).execute()
                concepts = getattr(con_resp, 'data', []) or []
                concept_objs = []
                for co in concepts:
                    # Normalize concept name and difficulty
                    concept_name = co.get('llm_suggested_concept_name') or co.get('concept_name') or f"Concept {co.get('concept_id')}"
                    raw_diff = co.get('difficulty_level') or 'Medium'
                    # Map to DifficultyLevel safely
                    try:
                        if isinstance(raw_diff, str):
                            # Accept common casing variations
                            diff_val = raw_diff.strip().title()
                        else:
                            diff_val = str(raw_diff)
                        difficulty = DifficultyLevel(diff_val)
                    except Exception:
                        difficulty = DifficultyLevel.MEDIUM

                    concept_objs.append(ConceptInfo(
                        concept_id=co.get('concept_id'),
                        concept_name=concept_name,
                        concept_order=co.get('concept_order') or 0,
                        difficulty_level=difficulty,
                        description=co.get('description') if isinstance(co.get('description'), str) and co.get('description') != '' else None
                    ))
                total_concepts += len(concept_objs)
                chapter_objs.append(ChapterInfo(
                    chapter_id=cid,
                    chapter_name=ch_name,
                    chapter_order=c.get('chapter_order') or 0,
                    description=ch_desc,
                    concepts=concept_objs,
                    concept_count=len(concept_objs)
                ))

            subj_info = SubjectInfo(
                subject_id=sid,
                subject_name=subject_name,
                description=desc,
                chapter_count=len(chapter_objs),
                concept_count=total_concepts,
                session_id=s.get('session_id') if s.get('session_id') else None,
            )

            out.append(SubjectStructure(subject_info=subj_info, chapters=chapter_objs))

        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{subject_id}")
async def patch_subject(subject_id: str, payload: dict, current_user: dict = Depends(get_current_user)):
    """Patch subject fields (e.g., subject_name)."""
    user_id = current_user.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client = get_supabase_client()
    try:
        # Try lookup with provided id, then string form, then int form
        existing = client.table('subjects').select('*').eq('subject_id', subject_id).eq('student_id', user_id).execute()
        data = getattr(existing, 'data', []) or []
        if not data:
            # try string/int variants
            alt = None
            try:
                alt = client.table('subjects').select('*').eq('subject_id', str(subject_id)).eq('student_id', user_id).execute()
                data = getattr(alt, 'data', []) or []
            except Exception:
                data = []
            if not data:
                try:
                    alt2 = client.table('subjects').select('*').eq('subject_id', int(subject_id)).eq('student_id', user_id).execute()
                    data = getattr(alt2, 'data', []) or []
                except Exception:
                    data = []
        if not data:
            raise HTTPException(status_code=404, detail='Subject not found')

        # sanitize payload to only allow known subject columns
        allowed = {'subject_name', 'description', 'llm_suggested_subject_name', 'session_id'}
        payload_clean = {k: v for k, v in (payload or {}).items() if k in allowed}
        # Translate frontend-friendly key to actual DB column if necessary
        if 'subject_name' in payload_clean:
            payload_clean['llm_suggested_subject_name'] = payload_clean.pop('subject_name')
        logger.debug('Attempting to update subject %s with payload: %s', subject_id, payload_clean)
        if not payload_clean:
            raise HTTPException(status_code=400, detail='No valid fields to update')

        try:
            upd = client.table('subjects').update(payload_clean).eq('subject_id', subject_id).eq('student_id', user_id).execute()
            # log response details for debugging
            logger.info('Supabase update response: %s', getattr(upd, '__dict__', str(upd)))
            status = getattr(upd, 'status_code', None)
            if status and status >= 400:
                # try to extract more info
                err = getattr(upd, 'error', None)
                raise HTTPException(status_code=502, detail=f'Supabase update failed: {err or getattr(upd, "data", None)}')
            return {'updated': getattr(upd, 'data', [])}
        except Exception as e:
            # try to extract httpx response content if present
            logger.exception('Supabase update exception for subject %s: %s', subject_id, e)
            resp = getattr(e, 'response', None)
            if resp is not None:
                try:
                    body = resp.text
                except Exception:
                    body = str(resp)
                logger.error('Supabase HTTP response body: %s', body)
            raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{subject_id}")
async def delete_subject(subject_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a subject (and optionally cascade) for the current user."""
    user_id = current_user.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client = get_supabase_client()
    try:
        existing = client.table('subjects').select('*').eq('subject_id', subject_id).eq('student_id', user_id).execute()
        data = getattr(existing, 'data', []) or []
        if not data:
            # try alternate forms
            try:
                alt = client.table('subjects').select('*').eq('subject_id', str(subject_id)).eq('student_id', user_id).execute()
                data = getattr(alt, 'data', []) or []
            except Exception:
                data = []
            if not data:
                try:
                    alt2 = client.table('subjects').select('*').eq('subject_id', int(subject_id)).eq('student_id', user_id).execute()
                    data = getattr(alt2, 'data', []) or []
                except Exception:
                    data = []
        if not data:
            raise HTTPException(status_code=404, detail='Subject not found')

        resp = client.table('subjects').delete().eq('subject_id', subject_id).eq('student_id', user_id).execute()
        logger.info('Supabase delete response: %s', getattr(resp, '__dict__', str(resp)))
        status = getattr(resp, 'status_code', None)
        if status and status >= 400:
            raise HTTPException(status_code=502, detail=f'Supabase delete failed: {getattr(resp, "error", None) or getattr(resp, "data", None)}')
        return {'deleted': getattr(resp, 'data', [])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
