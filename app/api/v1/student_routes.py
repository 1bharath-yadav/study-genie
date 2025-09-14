# app/api/v1/student_routes.py
from asyncio.log import logger
from fastapi import APIRouter, HTTPException, Depends
from app.models import  StudentData, StudentUpdate
from app.services.student_service import (
    get_student_by_id,
    update_student_data,
    delete_student_by_id,
)
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
