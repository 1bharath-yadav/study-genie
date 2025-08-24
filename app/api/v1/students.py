
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any

from app.core import get_db_manager
from app.services import LearningProgressService
from app.services import RecommendationService
from app.models import StudentCreate, StudentResponse, StudentUpdate, ConceptProgressUpdate, StudentProgressResponse

router = APIRouter()


@router.post("/students", response_model=StudentResponse)
async def create_student(student_data: StudentCreate, db=Depends(get_db_manager)):
    """Create a new student or return existing one."""
    try:
        service = LearningProgressService(db)
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students", response_model=List[StudentResponse])
async def get_all_students(db=Depends(get_db_manager)):
    """Retrieve a list of all students."""
    try:
        service = LearningProgressService(db)
        students = await service.get_all_students()
        return students
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}", response_model=StudentResponse)
async def get_student_by_id(student_id: str, db=Depends(get_db_manager)):
    """Retrieve a single student's details by ID."""
    try:
        service = LearningProgressService(db)
        student = await service.get_student_by_id(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        return student
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/students/{student_id}", response_model=StudentResponse)
async def update_student(student_id: str, student_update: StudentUpdate, db=Depends(get_db_manager)):
    """Update an existing student's details."""
    try:
        service = LearningProgressService(db)
        updated_student = await service.update_student(student_id, student_update)
        if not updated_student:
            raise HTTPException(status_code=404, detail="Student not found or no changes applied")
        return updated_student
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/students/{student_id}")
async def delete_student(student_id: str, db=Depends(get_db_manager)):
    """Delete a student by ID."""
    try:
        service = LearningProgressService(db)
        deleted = await service.delete_student(student_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Student not found")
        return {"message": "Student deleted successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/students/{student_id}/progress")
async def update_concept_progress(
    student_id: str, progress_data: ConceptProgressUpdate, db=Depends(get_db_manager)
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
async def get_student_progress(student_id: str, subject_id: Optional[int] = None, db=Depends(get_db_manager)):
    try:
        service = LearningProgressService(db)
        progress_data = await service.get_student_progress(student_id, subject_id)
        return StudentProgressResponse(**progress_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- New endpoint: Get recommendations for a student ---
@router.get("/students/{student_id}/recommendations", response_model=List[Dict[str, Any]])
async def get_student_recommendations(student_id: str, active_only: bool = True, db=Depends(get_db_manager)):
    """Get personalized recommendations for a student."""
    try:
        service = RecommendationService(db)
        recommendations = await service.get_recommendations(student_id, active_only=active_only)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
