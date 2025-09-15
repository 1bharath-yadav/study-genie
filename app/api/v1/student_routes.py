# app/api/v1/student_routes.py
# logger not used in this module
from fastapi import APIRouter, HTTPException, Depends
from app.models import  StudentData, StudentUpdate
from app.models import LearningActivityRequest
from app.services.learning_history_service import save_learning_activity
from app.services.student_service import (
    get_student_by_id,
    update_student_data,
    delete_student_by_id,
)
# analytics_service references removed
from app.core.security import get_current_user

router = APIRouter(prefix="/student", tags=["student"])

@router.get("/", response_model=StudentData)
async def get_student_endpoint(
    current_user: dict = Depends(get_current_user)
):
    """Get the current student's data."""
    student_id = current_user["sub"]
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@router.put("/", response_model=StudentData)
async def update_student_endpoint(
    student_update: StudentUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update the current student's data.

    The frontend must send fields matching the students table schema:
    - full_name, grade_level, bio, learning_preferences (JSON object or array)
    """
    student_id = current_user["sub"]

    # Pydantic will validate the incoming payload. No normalization is performed here;
    # the frontend must send the correct shape.
    student = update_student_data(student_id, student_update)
    if not student:
        raise HTTPException(status_code=400, detail="Failed to update student")
    return student

@router.delete("/")
async def delete_student_endpoint(
    current_user: dict = Depends(get_current_user)
):
    """Delete the current student's data."""
    student_id = current_user["sub"]
    success = delete_student_by_id(student_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete student")
    return {"message": "Student deleted successfully"}


@router.post('/students/{student_id}/learning-activity')
async def save_learning_activity_endpoint(
    student_id: str,
    activity: LearningActivityRequest,
    current_user: dict = Depends(get_current_user)
):
    """Persist a learning activity for the student. The authenticated user must match the student_id."""
    # Enforce that the caller is the same student (or an admin; admin logic not implemented)
    if current_user.get('sub') != student_id:
        raise HTTPException(status_code=403, detail='Forbidden')

    try:
        inserted = save_learning_activity(student_id, activity)
        return { 'success': True, 'data': inserted }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

