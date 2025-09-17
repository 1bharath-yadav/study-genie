from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict
import os
import io
import json
import zipfile

from app.core.security import get_current_user, get_current_student_id
from app.config import settings
from app.services.learning_history_service import get_learning_history_for_user

router = APIRouter(prefix="/export", tags=["export"])


def _stream_zip(files: Dict[str, bytes]):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    buf.seek(0)
    # Yield the full zip as a single chunk
    yield buf.read()


@router.get("/learning-history")
async def export_learning_history(current_user: Dict = Depends(get_current_user), student_id: str = Depends(get_current_student_id)):
    user_id = student_id
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user")

    # Fetch learning history from service (returns list of dicts)
    records = await get_learning_history_for_user(user_id)
    payload = json.dumps(records, indent=2).encode("utf-8")

    filename = f"learning_history_{user_id}.zip"
    stream = _stream_zip({f"learning_history_{user_id}.json": payload})
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})


@router.get("/session/{session_id}/faiss")
async def export_session_faiss(session_id: str, current_user: Dict = Depends(get_current_user), student_id: str = Depends(get_current_student_id)):
    # Look for persisted faiss folder under settings.PERSIST_DIR
    base = settings.PERSIST_DIR
    session_dir = os.path.join(base, str(student_id), f"faiss_{session_id}")
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="FAISS store not found for session")

    # Collect files
    files = {}
    for root, _, filenames in os.walk(session_dir):
        for fname in filenames:
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, session_dir)
            with open(path, "rb") as f:
                files[rel] = f.read()

    filename = f"session_{session_id}_faiss.zip"
    stream = _stream_zip(files)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})


@router.get("/session/{session_id}/uploads")
async def export_session_uploads(session_id: str, current_user: Dict = Depends(get_current_user), student_id: str = Depends(get_current_student_id)):
    base = settings.PERSIST_DIR
    upload_dir = os.path.join(base, str(student_id), f"uploads_{session_id}")
    if not os.path.exists(upload_dir):
        raise HTTPException(status_code=404, detail="Uploads not found for session")

    files = {}
    for root, _, filenames in os.walk(upload_dir):
        for fname in filenames:
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, upload_dir)
            with open(path, "rb") as f:
                files[rel] = f.read()

    filename = f"session_{session_id}_uploads.zip"
    stream = _stream_zip(files)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})


@router.get("/student/all/faiss")
async def export_all_student_faiss(current_user: Dict = Depends(get_current_user)):
    """Export all FAISS session stores for the current student as a zip."""
    user_id = current_user.get('sub') or current_user.get('id')
    base = settings.PERSIST_DIR
    student_dir = os.path.join(base, str(user_id))
    if not os.path.exists(student_dir):
        raise HTTPException(status_code=404, detail="No persisted data for student")

    files = {}
    # Walk only faiss_ directories under the student's dir
    for root, dirs, filenames in os.walk(student_dir):
        for fname in filenames:
            rel = os.path.relpath(os.path.join(root, fname), student_dir)
            with open(os.path.join(root, fname), 'rb') as f:
                files[rel] = f.read()

    filename = f"student_{user_id}_faiss_all.zip"
    stream = _stream_zip(files)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})


@router.get("/student/all/uploads")
async def export_all_student_uploads(current_user: Dict = Depends(get_current_user)):
    """Export all uploaded files for the current student as a zip."""
    user_id = current_user.get('sub') or current_user.get('id')
    base = settings.PERSIST_DIR
    student_dir = os.path.join(base, str(user_id))
    if not os.path.exists(student_dir):
        raise HTTPException(status_code=404, detail="No persisted data for student")

    files = {}
    for root, dirs, filenames in os.walk(student_dir):
        for fname in filenames:
            rel = os.path.relpath(os.path.join(root, fname), student_dir)
            with open(os.path.join(root, fname), 'rb') as f:
                files[rel] = f.read()

    filename = f"student_{user_id}_uploads_all.zip"
    stream = _stream_zip(files)
    return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})


@router.delete("/student/all")
async def delete_all_student_data(current_user: Dict = Depends(get_current_user), student_id: str = Depends(get_current_student_id)):
    """Delete all persisted files and key DB rows for the current student.

    This is a destructive, irreversible operation. It will remove the student's
    persisted directory under PERSIST_DIR/<student_id> and attempt to delete rows
    in chat_history and learning_activities. Additional tables may be removed as
    a best-effort cleanup.
    """
    base = settings.PERSIST_DIR
    student_dir = os.path.join(base, str(student_id))

    # Remove persisted files directory
    try:
        if os.path.exists(student_dir):
            for root, dirs, files in os.walk(student_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except OSError:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass
            try:
                os.rmdir(student_dir)
            except OSError:
                pass
    except Exception as e:
        # log and continue with DB cleanup
        import logging
        logging.getLogger(__name__).warning(f"Failed to remove files for student {student_id}: {e}")

    # Attempt to delete DB rows related to the student
    try:
        from app.db.db_client import get_supabase_client
        client = get_supabase_client()
        # Delete chat_history rows
        try:
            client.table('chat_history').delete().eq('student_id', student_id).execute()
        except Exception:
            pass
        # Delete learning activities
        try:
            client.table('learning_activities').delete().eq('student_id', student_id).execute()
        except Exception:
            pass
        # Optionally delete subjects/chapters/concepts created by the user
        try:
            client.table('concepts').delete().eq('student_id', student_id).execute()
        except Exception:
            pass
        try:
            client.table('chapters').delete().eq('student_id', student_id).execute()
        except Exception:
            pass
        try:
            client.table('subjects').delete().eq('student_id', student_id).execute()
        except Exception:
            pass
    except Exception:
        # best-effort: ignore DB cleanup failures
        pass

    return {"status": "deleted"}


@router.delete("/session/{session_id}/faiss")
async def delete_session_faiss(session_id: str, current_user: Dict = Depends(get_current_user), student_id: str = Depends(get_current_student_id)):
    base = settings.PERSIST_DIR
    session_dir = os.path.join(base, str(student_id), f"faiss_{session_id}")
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="FAISS store not found for session")

    # Remove directory
    for root, dirs, files in os.walk(session_dir, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass
    try:
        os.rmdir(session_dir)
    except OSError:
        pass

    return {"status": "deleted"}


@router.delete("/session/{session_id}/uploads")
async def delete_session_uploads(session_id: str, current_user: Dict = Depends(get_current_user), student_id: str = Depends(get_current_student_id)):
    base = settings.PERSIST_DIR
    upload_dir = os.path.join(base, str(student_id), f"uploads_{session_id}")
    if not os.path.exists(upload_dir):
        raise HTTPException(status_code=404, detail="Uploads not found for session")

    for root, dirs, files in os.walk(upload_dir, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except OSError:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass
    try:
        os.rmdir(upload_dir)
    except OSError:
        pass

    return {"status": "deleted"}
