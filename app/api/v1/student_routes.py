# app/api/v1/student_routes.py
from asyncio.log import logger
from fastapi import APIRouter, HTTPException, Depends
from app.models import  ConceptProgressUpdate, StudentData, LearningActivityRequest ,LLMResponse, StudentProgressResponse
from app.services.student_service import (
    get_student_by_id,
    update_student_data,
    delete_student_by_id,
)
from app.core.security import get_current_user
from app.services.students.progress_saving import get_learning_progress_service

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
    student_data: StudentData,
    current_user: dict = Depends(get_current_user)
):
    """Update the current student's data."""
    student_id = current_user["sub"]
    student = update_student_data(student_id, student_data)
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




@router.post("/students/{student_id}/learning-activity")
async def save_learning_activity(student_id: str, activity_request: LearningActivityRequest):
    """Save learning activity results (quiz, flashcard session, etc.)."""
    try:
        # Log the incoming request for debugging
        logger.info(
            f"Received learning activity request for student: {student_id}")
        logger.info(f"Activity request data: {activity_request.dict()}")

        service = get_learning_progress_service()

        # Convert student_id if it's an email
        resolved_student_id = student_id
        if "@" in student_id:
            
                
                    # If no student found by email, create new one
            resolved_student_id = await service.create_or_get_student(
                student_id=student_id,
                username=student_id,
                email=student_id,
                full_name=f"User {student_id}"
            )
            # Continue with original student_id

        # Extract activity details from the structured request
        subject_name = activity_request.subject_name or 'General'
        chapter_name = activity_request.chapter_name or 'General Concepts'
        concept_name = activity_request.concept_name or activity_request.activity_type.value.replace(
            '_', ' ').title()

        activity_type = activity_request.activity_type.value
        correct_answers = activity_request.correct_answers
        total_questions = activity_request.total_questions
        time_spent = activity_request.time_spent or 0
        difficulty_level = activity_request.difficulty_level.value

        # Additional validation
        if correct_answers > total_questions:
            raise HTTPException(
                status_code=422, detail="Correct answers cannot exceed total questions")
        if total_questions <= 0:
            raise HTTPException(
                status_code=422, detail="Total questions must be greater than 0")
        if correct_answers < 0:
            raise HTTPException(
                status_code=422, detail="Correct answers cannot be negative")

        # Log the activity details for debugging
        logger.info(
            f"Processing learning activity: {activity_type} for {concept_name} in {subject_name}/{chapter_name}")

        # Calculate mastery level
        mastery_level = (correct_answers / total_questions *
                         100) if total_questions > 0 else 0

        # Save progress to database
        await service.save_concept_progress(
            student_id=resolved_student_id,
            subject_name=subject_name,
            concept_name=concept_name,
            mastery_level=mastery_level,
            correct_answers=correct_answers,
            total_questions=total_questions,
            time_spent=time_spent,
            difficulty_level=difficulty_level,
            activity_type=activity_type
        )

        # Generate recommendations based on performance
        if mastery_level < 70:
            recommendations = [
                {
                    "recommendation_type": "practice_more",
                    "title": f"Practice {concept_name}",
                    "description": f"You scored {mastery_level:.1f}% on {concept_name}. Try reviewing the fundamentals.",
                    "priority_score": 9 if mastery_level < 50 else 6
                }
            ]
            await service.save_recommendations(resolved_student_id, recommendations)

        return {
            "message": "Learning activity saved successfully",
            "mastery_level": mastery_level,
            "progress_saved": True,
            "subject_name": subject_name,
            "chapter_name": chapter_name,
            "concept_name": concept_name
        }
    except ValueError as ve:
        logger.error(f"Validation error in learning activity: {str(ve)}")
        raise HTTPException(
            status_code=422, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"Error saving learning activity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/students/{student_id}/progress")
async def update_concept_progress(student_id: str, progress_update: ConceptProgressUpdate):
    """Update a student's progress on a specific concept."""
    try:
        service = get_learning_progress_service()

        await service.update_concept_progress(
            student_id=student_id,
            concept_id=progress_update.concept_id,
            correct_answers=progress_update.correct_answers,
            total_questions=progress_update.total_questions,
            time_spent=progress_update.time_spent
        )

        return {"message": "Progress updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/{student_id}/progress", response_model=StudentProgressResponse)
async def get_student_progress(student_id: str):
    """Get a student's learning progress across all subjects."""
    try:
        service = get_learning_progress_service()
        progress = await service.get_student_progress(student_id)

        # Convert Supabase progress format to expected model format
        subjects = progress.get('subjects', {})
        overall_stats = progress.get('overall_stats', {})

        # Convert subjects to subject_progress format
        subject_progress = []
        concept_progress = []

        for subject_name, subject_data in subjects.items():
            subject_info = {
                'subject_name': subject_name,
                'total_concepts': subject_data.get('total_concepts', 0),
                'mastered_concepts': subject_data.get('mastered_concepts', 0),
                'chapters': subject_data.get('chapters', {})
            }
            subject_progress.append(subject_info)

            # Extract individual concept progress
            for chapter_name, chapter_data in subject_data.get('chapters', {}).items():
                for concept in chapter_data.get('concepts', []):
                    concept_progress.append({
                        'concept_id': 0,  # Would need to be mapped from concept name
                        'concept_name': concept.get('concept_name', ''),
                        'status': concept.get('status', 'not_started'),
                        'mastery_score': concept.get('mastery_score', 0),
                        'attempts_count': concept.get('attempts_count', 0),
                        'correct_answers': 0,  # Would need actual data
                        'total_questions': 0,  # Would need actual data
                        'last_practiced': concept.get('last_practiced'),
                        'first_learned': concept.get('first_learned')
                    })

        return StudentProgressResponse(
            student_id=student_id,
            overall_progress=overall_stats,
            subject_progress=subject_progress,
            concept_progress=concept_progress,
            recent_activity=[],  # Would need to implement
            total_concepts=overall_stats.get('total_concepts', 0),
            mastered_concepts=overall_stats.get('mastered_concepts', 0),
            weak_concepts=overall_stats.get(
                'total_concepts', 0) - overall_stats.get('mastered_concepts', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))