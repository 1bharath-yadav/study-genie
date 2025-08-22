from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from app.core import get_db_manager
from app.services import LearningProgressService
from app.models import StudentCreate, StudentResponse, ConceptProgressUpdate, StudentProgressResponse

router = APIRouter()


@router.post("/students", response_model=StudentResponse)
async def create_student(student_data: StudentCreate, db=Depends(get_db_manager)):
    """Create a new student or return existing one."""
    try:
        service = LearningProgressService(db)
        student_id = await service.create_or_get_student(
            student_data.username, student_data.email, student_data.full_name
        )

        return StudentResponse(
            student_id=student_id,
            username=student_data.username,
            email=student_data.email,
            full_name=student_data.full_name,
            message="Student created/retrieved successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/students/{student_id}/progress")
async def update_concept_progress(
    student_id: int, progress_data: ConceptProgressUpdate, db=Depends(get_db_manager)
):
    try:
        service = LearningProgressService(db)
        await service.update_concept_progress(
            student_id,
            progress_data.concept_id,
            progress_data.correct_answers,
            progress_data.total_questions,
        )
        return {"message": "Progress updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/progress", response_model=StudentProgressResponse)
async def get_student_progress(student_id: int, subject_id: Optional[int] = None, db=Depends(get_db_manager)):
    try:
        service = LearningProgressService(db)
        progress_data = await service.get_student_progress(student_id, subject_id)
        return StudentProgressResponse(**progress_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
