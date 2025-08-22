"""Problem Statement : StudyGenie - Personalized Study Guide Generator

Description:
Turn raw learning input (books, PDFs, handwritten notes) into quizzes, flashcards, and

interactive study material. Use RAG/GenAl to personalize learning, add multilingual support, and

track student progress.
Expected Outcomes (Detailed):
* Upload PDF/notes — App extracts text (OCR if needed) - Summaries + quizzes auto-
generated.
* Personalized study flow based on weals/strong subjects logged over time.
*  Flashcard system with active recall and spaced repetition built-in.

* Interactive RAG-powered Al tutor that answers student questions directly from provided

materials.

* Multilingual support (English, Hindi, Marathi, regional). Example: Convert a math
summary into Marathi for regional students.

+ Dashboard with progress bars, study strealts, knowledge heatmaps.

* Example Demo Flow: Student uploads Physics Chapter PDF -» app outputs 15 flashcards +
10 MCQs — student takes quiz - dashboard updates progress — Al tutor explains wrong

answers in simple words.

Technology: LangChain, OpenAl API, Hugging Face Transformers, Tesseract OCR / Google Vision
AP, React, Nodes/Python, Pinecone/FAISS, Firebase/PostgresQL, D3.s/Chartjs
"""


from .db.db import DatabaseManager, initialize_database
from .services import (
    LearningProgressService,
    RecommendationService,
    AnalyticsService,
    StudySessionService
)
from .models import *
import asyncio
from typing import Dict, Any, List, Optional
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.v1.routes import router  # Your custom routes

from configs import API_HOST, API_PORT  # Configuration for host and port
app = FastAPI(
    title="Study-genie",
    description="""Interactive RAG-powered Al tutor that answers student questions directly from provided materials""",
    version="1.0.0",
)

# progress_tracker/main.py


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("progress_tracker_api")

# Global database manager
db_manager: Optional[DatabaseManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup database connections"""
    global db_manager
    try:
        db_manager = await initialize_database()
        logger.info("Database initialized successfully")
        yield
    finally:
        if db_manager:
            await db_manager.close_pool()
            logger.info("Database connections closed")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database manager


async def get_db_manager() -> DatabaseManager:
    if db_manager is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db_manager

# Health check endpoint


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Progress tracker API is running"}

# Student Management Endpoints


@app.post("/api/students", response_model=StudentResponse)
async def create_student(
    student_data: StudentCreate,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Create a new student or get existing one"""
    try:
        service = LearningProgressService(db)
        student_id = await service.create_or_get_student(
            student_data.username,
            student_data.email,
            student_data.full_name
        )

        return StudentResponse(
            student_id=student_id,
            username=student_data.username,
            email=student_data.email,
            full_name=student_data.full_name,
            message="Student created/retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error creating student: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/students/{student_id}/preferences")
async def update_student_preferences(
    student_id: int,
    preferences: Dict[str, Any],
    db: DatabaseManager = Depends(get_db_manager)
):
    """Update student learning preferences"""
    try:
        service = LearningProgressService(db)
        await service.update_learning_preferences(student_id, preferences)
        return {"message": "Preferences updated successfully"}
    except Exception as e:
        logger.error(f"Error updating preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main RAG Response Processing Endpoint


@app.post("/api/process-llm-response", response_model=ProcessedLLMResponse)
async def process_llm_response(
    request: LLMResponseRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager)
):
    """
    Process LLM response and update student progress
    This is the main endpoint that integrates with your RAG system
    """
    try:
        service = LearningProgressService(db)

        # Process the LLM response and update tracking
        result = await service.process_llm_response(
            student_id=request.student_id,
            subject_name=request.subject_name,
            chapter_name=request.chapter_name,
            concept_name=request.concept_name,
            llm_response=request.llm_response,
            user_query=request.user_query
        )

        # Generate recommendations in background
        background_tasks.add_task(
            generate_recommendations_background,
            request.student_id,
            db
        )

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message="LLM response processed and progress updated successfully"
        )

    except Exception as e:
        logger.error(f"Error processing LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Progress Tracking Endpoints


@app.post("/api/students/{student_id}/progress")
async def update_concept_progress(
    student_id: int,
    progress_data: ConceptProgressUpdate,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Update student progress on a specific concept"""
    try:
        service = LearningProgressService(db)
        await service.update_concept_progress(
            student_id=student_id,
            concept_id=progress_data.concept_id,
            correct_answers=progress_data.correct_answers,
            total_questions=progress_data.total_questions,
            time_spent=progress_data.time_spent
        )
        return {"message": "Progress updated successfully"}
    except Exception as e:
        logger.error(f"Error updating progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/students/{student_id}/weaknesses")
async def record_weakness(
    student_id: int,
    weakness_data: WeaknessRecord,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Record a student weakness"""
    try:
        service = LearningProgressService(db)
        await service.record_weakness(
            student_id=student_id,
            concept_id=weakness_data.concept_id,
            weakness_type=weakness_data.weakness_type,
            error_pattern=weakness_data.error_pattern,
            severity=weakness_data.severity
        )
        return {"message": "Weakness recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording weakness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/{student_id}/progress", response_model=StudentProgressResponse)
async def get_student_progress(
    student_id: int,
    subject_id: Optional[int] = None,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get comprehensive student progress"""
    try:
        service = LearningProgressService(db)
        progress_data = await service.get_student_progress(student_id, subject_id)
        return StudentProgressResponse(**progress_data)
    except Exception as e:
        logger.error(f"Error fetching student progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Study Session Endpoints


@app.post("/api/students/{student_id}/sessions/start", response_model=StudySessionResponse)
async def start_study_session(
    student_id: int,
    session_data: StudySessionStart,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Start a new study session"""
    try:
        service = StudySessionService(db)
        session_id = await service.start_session(
            student_id=student_id,
            subject_name=session_data.subject_name,
            session_metadata=session_data.metadata
        )
        return StudySessionResponse(
            session_id=session_id,
            student_id=student_id,
            status="started",
            message="Study session started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting study session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/students/{student_id}/sessions/{session_id}/end")
async def end_study_session(
    student_id: int,
    session_id: int,
    session_results: StudySessionEnd,
    db: DatabaseManager = Depends(get_db_manager)
):
    """End a study session with results"""
    try:
        service = StudySessionService(db)
        await service.end_session(
            session_id=session_id,
            results=session_results
        )
        return {"message": "Study session ended successfully"}
    except Exception as e:
        logger.error(f"Error ending study session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/{student_id}/sessions", response_model=List[StudySessionHistory])
async def get_study_sessions(
    student_id: int,
    limit: int = 20,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get student's study session history"""
    try:
        service = StudySessionService(db)
        sessions = await service.get_session_history(student_id, limit)
        return [StudySessionHistory(**session) for session in sessions]
    except Exception as e:
        logger.error(f"Error fetching study sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Recommendations Endpoints


@app.get("/api/students/{student_id}/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    student_id: int,
    active_only: bool = True,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get personalized recommendations for a student"""
    try:
        service = RecommendationService(db)
        recommendations = await service.get_recommendations(student_id, active_only)
        return [RecommendationResponse(**rec) for rec in recommendations]
    except Exception as e:
        logger.error(f"Error fetching recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/students/{student_id}/recommendations/generate")
async def generate_recommendations(
    student_id: int,
    force_regenerate: bool = False,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Generate new personalized recommendations"""
    try:
        service = RecommendationService(db)
        count = await service.generate_recommendations(student_id, force_regenerate)
        return {"message": f"Generated {count} new recommendations"}
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/recommendations/{recommendation_id}/complete")
async def complete_recommendation(
    recommendation_id: int,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Mark a recommendation as completed"""
    try:
        service = RecommendationService(db)
        await service.complete_recommendation(recommendation_id)
        return {"message": "Recommendation marked as completed"}
    except Exception as e:
        logger.error(f"Error completing recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints


@app.get("/api/students/{student_id}/analytics", response_model=StudentAnalyticsResponse)
async def get_student_analytics(
    student_id: int,
    time_period: Optional[str] = "30d",  # 7d, 30d, 90d, all
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get comprehensive student analytics"""
    try:
        service = AnalyticsService(db)
        analytics = await service.get_student_analytics(student_id, time_period)
        return StudentAnalyticsResponse(**analytics)
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/{student_id}/weaknesses", response_model=List[WeaknessAnalysis])
async def get_weakness_analysis(
    student_id: int,
    resolved: Optional[bool] = False,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get detailed weakness analysis"""
    try:
        service = AnalyticsService(db)
        weaknesses = await service.get_weakness_analysis(student_id, resolved)
        return [WeaknessAnalysis(**weakness) for weakness in weaknesses]
    except Exception as e:
        logger.error(f"Error fetching weakness analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/{student_id}/learning-path", response_model=LearningPathResponse)
async def get_learning_path(
    student_id: int,
    subject_id: Optional[int] = None,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get suggested learning path based on progress"""
    try:
        service = AnalyticsService(db)
        learning_path = await service.generate_learning_path(student_id, subject_id)
        return LearningPathResponse(**learning_path)
    except Exception as e:
        logger.error(f"Error generating learning path: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Subject and Content Management


@app.get("/api/subjects", response_model=List[SubjectInfo])
async def get_subjects(db: DatabaseManager = Depends(get_db_manager)):
    """Get all subjects"""
    try:
        service = LearningProgressService(db)
        subjects = await service.get_all_subjects()
        return [SubjectInfo(**subject) for subject in subjects]
    except Exception as e:
        logger.error(f"Error fetching subjects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/subjects/{subject_id}/structure", response_model=SubjectStructure)
async def get_subject_structure(
    subject_id: int,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get complete subject structure with chapters and concepts"""
    try:
        service = LearningProgressService(db)
        structure = await service.get_subject_structure(subject_id)
        return SubjectStructure(**structure)
    except Exception as e:
        logger.error(f"Error fetching subject structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/subjects/{subject_id}/enroll/{student_id}")
async def enroll_student_in_subject(
    student_id: int,
    subject_id: int,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Enroll a student in a subject"""
    try:
        service = LearningProgressService(db)
        await service.enroll_student_in_subject(student_id, subject_id)
        return {"message": "Student enrolled successfully"}
    except Exception as e:
        logger.error(f"Error enrolling student: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch Operations


@app.post("/api/students/{student_id}/quiz-results")
async def process_quiz_results(
    student_id: int,
    quiz_results: QuizResultsBatch,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Process batch quiz results and update progress"""
    try:
        service = LearningProgressService(db)
        processed_count = await service.process_quiz_results(student_id, quiz_results)
        return {"message": f"Processed {processed_count} quiz results"}
    except Exception as e:
        logger.error(f"Error processing quiz results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/students/{student_id}/flashcard-session")
async def process_flashcard_session(
    student_id: int,
    flashcard_session: FlashcardSessionResults,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Process flashcard session results"""
    try:
        service = LearningProgressService(db)
        await service.process_flashcard_session(student_id, flashcard_session)
        return {"message": "Flashcard session processed successfully"}
    except Exception as e:
        logger.error(f"Error processing flashcard session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions


async def generate_recommendations_background(student_id: int, db: DatabaseManager):
    """Background task to generate recommendations"""
    try:
        service = RecommendationService(db)
        await service.generate_recommendations(student_id, force_regenerate=False)
        logger.info(f"Generated recommendations for student {student_id}")
    except Exception as e:
        logger.error(
            f"Error generating recommendations in background: {str(e)}")

# Utility endpoints


@app.get("/api/students/{student_id}/dashboard", response_model=DashboardData)
async def get_dashboard_data(
    student_id: int,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get all data needed for student dashboard"""
    try:
        analytics_service = AnalyticsService(db)
        recommendation_service = RecommendationService(db)
        progress_service = LearningProgressService(db)

        # Fetch all dashboard data concurrently
        analytics, recommendations, recent_progress = await asyncio.gather(
            analytics_service.get_student_analytics(student_id, "30d"),
            recommendation_service.get_recommendations(
                student_id, active_only=True),
            progress_service.get_recent_activity(student_id, limit=10)
        )

        return DashboardData(
            analytics=analytics,
            recommendations=recommendations,
            recent_activity=recent_progress,
            student_id=student_id
        )
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount core API routes
app.include_router(router, prefix="/api")

# Run with: python app/main.py OR uvicorn app.main:app --reload
if __name__ == "__main__":
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
