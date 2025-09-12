# Pure functional student routes
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models import StudentCreate, StudentUpdate, StudentResponse
from app.services.student_service import (
    create_student,
    get_student_by_id,
    get_students_by_user,
    update_student_data,
    delete_student_by_id,
    check_student_ownership,
)
from app.core.security import get_current_user

router = APIRouter(prefix="/students", tags=["students"])


@router.post("/", response_model=StudentResponse)
async def create_student_endpoint(
    student_data: StudentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new student."""
    user_id = current_user["id"]
    
    student = create_student(student_data, user_id)
    if not student:
        raise HTTPException(status_code=400, detail="Failed to create student")
    
    return student


@router.get("/", response_model=List[StudentResponse])
async def get_user_students(
    current_user: dict = Depends(get_current_user)
):
    """Get all students for the current user."""
    user_id = current_user["id"]
    return get_students_by_user(user_id)


@router.get("/{student_id}", response_model=StudentResponse)
async def get_student_by_id_endpoint(
    student_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific student by ID."""
    user_id = current_user["id"]
    
    # Check ownership
    if not check_student_ownership(student_id, user_id):
        raise HTTPException(status_code=404, detail="Student not found")
    
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    return student


@router.put("/{student_id}", response_model=StudentResponse)
async def update_student_endpoint(
    student_id: str,
    student_data: StudentUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a student."""
    user_id = current_user["id"]
    
    # Check ownership
    if not check_student_ownership(student_id, user_id):
        raise HTTPException(status_code=404, detail="Student not found")
    
    student = update_student_data(student_id, student_data)
    if not student:
        raise HTTPException(status_code=400, detail="Failed to update student")
    
    return student


@router.delete("/{student_id}")
async def delete_student_endpoint(
    student_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a student."""
    user_id = current_user["id"]
    
    # Check ownership
    if not check_student_ownership(student_id, user_id):
        raise HTTPException(status_code=404, detail="Student not found")
    
    success = delete_student_by_id(student_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete student")
    
    return {"message": "Student deleted successfully"}
