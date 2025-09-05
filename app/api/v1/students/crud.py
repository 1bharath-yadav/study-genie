"""
Student CRUD operations using Supabase backend
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any

from app.services_supabase import get_learning_progress_service
from app.models import StudentCreate, StudentResponse, StudentUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/students", response_model=StudentResponse)
async def create_student(student_data: StudentCreate):
    """Create a new student or return existing one."""
    try:
        service = get_learning_progress_service()
        student_id = await service.create_or_get_student(
            student_id=student_data.username,  # or generate a unique id if needed
            username=student_data.username,
            email=student_data.email,
            full_name=student_data.full_name
        )

        return StudentResponse(
            student_id=student_id,
            username=student_data.username,
            email=student_data.email,
            full_name=student_data.full_name,
            message="Student created/retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Error creating student: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students", response_model=List[StudentResponse])
async def get_all_students():
    """Retrieve a list of all students."""
    try:
        service = get_learning_progress_service()
        students = await service.get_all_students()
        return students
    except Exception as e:
        logger.error(f"Error getting students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}", response_model=StudentResponse)
async def get_student_by_id(student_id: str):
    """Retrieve a single student's details by ID."""
    try:
        service = get_learning_progress_service()
        student = await service.get_student_by_id(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        return student
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/students/{student_id}", response_model=StudentResponse)
async def update_student(student_id: str, student_update: StudentUpdate):
    """Update a student's details."""
    try:
        service = get_learning_progress_service()

        # Update learning preferences if provided
        if student_update.learning_preferences:
            await service.update_learning_preferences(student_id, student_update.learning_preferences)

        # Get updated student data
        student = await service.get_student_by_id(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        return student
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/students/{student_id}")
async def delete_student(student_id: str):
    """Delete a student by ID."""
    try:
        service = get_learning_progress_service()
        deleted = await service.delete_student(student_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Student not found")
        return {"message": "Student deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
