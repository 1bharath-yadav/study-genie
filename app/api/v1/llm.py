from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, File, Form, UploadFile
from typing import List, Dict, Any, Optional
import os
import shutil
import tempfile
import json
import logging
import traceback
from pathlib import Path
from app.core import get_db_manager
from app.services import LearningProgressService
from app.models import LLMResponseRequest, ProcessedLLMResponse, LLMResponseContent, FlashcardData, QuizData, DifficultyLevel
from app.llm.langchain import get_llm_response
from app.utils.file_utils import save_uploaded_files_to_temp, cleanup_temp_dir

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


def transform_langchain_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform LangChain response format to dictionary format compatible with LLMResponseContent
    """
    try:
        if "error" in raw_response:
            # Handle error response
            return {
                "flashcards": {},
                "quiz": {},
                "summary": f"Error processing files: {raw_response.get('error', 'Unknown error')}",
                "learning_objectives": []
            }

        # Transform flashcards from array to dict
        flashcards = {}
        if "flashcards" in raw_response and isinstance(raw_response["flashcards"], list):
            for i, card_data in enumerate(raw_response["flashcards"], 1):
                try:
                    if isinstance(card_data, dict):
                        flashcards[f"card{i}"] = {
                            "question": str(card_data.get("question", "")),
                            "answer": str(card_data.get("answer", "")),
                            "difficulty": str(card_data.get("difficulty", "Medium"))
                        }
                except Exception as e:
                    logger.error(f"Error processing flashcard {i}: {e}")
                    continue

        # Transform quiz questions from array to dict
        quiz = {}
        if "quiz" in raw_response and isinstance(raw_response["quiz"], list):
            for i, q_data in enumerate(raw_response["quiz"], 1):
                try:
                    if isinstance(q_data, dict):
                        quiz[f"Q{i}"] = {
                            "question": str(q_data.get("question", "")),
                            "options": list(q_data.get("options", [])),
                            "correct_answer": str(q_data.get("correct_answer", "")),
                            "explanation": str(q_data.get("explanation", ""))
                        }
                except Exception as e:
                    logger.error(f"Error processing quiz question {i}: {e}")
                    continue

        return {
            "flashcards": flashcards,
            "quiz": quiz,
            "summary": str(raw_response.get("summary", "")),
            "learning_objectives": list(raw_response.get("learning_objectives", [])),
            # Include additional fields from the response
            "match_the_following": raw_response.get("match_the_following", {}),
            "metadata": raw_response.get("metadata", {})
        }
    except Exception as e:
        logger.error(f"Error in transform_langchain_response: {e}")
        return {
            "flashcards": {},
            "quiz": {},
            "summary": f"Error transforming response: {str(e)}",
            "learning_objectives": []
        }


@router.post("/upload-simple", response_model=ProcessedLLMResponse)
async def upload_and_process_simple(
    student_id: str = Form(...),
    user_query: str = Form(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db=Depends(get_db_manager)
):
    """
    Simplified upload endpoint that only requires student_id, prompt and files.
    All metadata (subject, chapter, concept, difficulty) is automatically extracted by the LLM.
    """
    tmp_dir = None
    uploaded_files_paths = []

    try:
        logger.info(f"Processing upload for student_id: {student_id}")
        logger.info(f"User query: {user_query}")
        logger.info(f"Number of files: {len(files) if files else 0}")

        # Validate input parameters
        if not student_id or not student_id.strip():
            raise HTTPException(
                status_code=400, detail="student_id is required and cannot be empty")

        if not user_query or not user_query.strip():
            raise HTTPException(
                status_code=400, detail="user_query is required and cannot be empty")

        # Save uploaded files to the temporary directory
        tmp_dir, uploaded_files_paths = await save_uploaded_files_to_temp(files)

        # Ensure student exists in DB first
        try:
            service = LearningProgressService(db)
            # Create student with Google ID as student_id, and generate placeholder username/email
            await service.create_or_get_student(
                student_id=student_id,
                username=f"user_{student_id[:10]}",  # Truncate for username
                email=f"{student_id}@placeholder.com",
                full_name=f"User {student_id[:8]}"  # Truncate for display name
            )
            logger.info(
                f"Student created/retrieved successfully: {student_id}")
        except Exception as e:
            logger.error(f"Error creating/getting student: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error managing student record: {str(e)}"
            )

        # Process files using LangChain LLM with metadata extraction
        try:
            logger.info("Starting LLM processing...")
            raw_llm_response = await get_llm_response(
                uploaded_files_paths=uploaded_files_paths,
                userprompt=user_query,
                temp_dir=tmp_dir
            )
            logger.info(
                f"LLM processing completed. Response type: {type(raw_llm_response)}")
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing files with LLM: {str(e)}"
            )

        # Debug: print the type and content of raw_llm_response
        logger.info(f"DEBUG: raw_llm_response type: {type(raw_llm_response)}")
        if isinstance(raw_llm_response, dict):
            logger.info(
                f"DEBUG: raw_llm_response keys: {list(raw_llm_response.keys())}")
        else:
            response_preview = str(raw_llm_response)[
                :200] if raw_llm_response else "None"
            logger.info(f"DEBUG: raw_llm_response preview: {response_preview}")

        # Handle case where response might be a string (JSON string)
        if isinstance(raw_llm_response, str):
            try:
                raw_llm_response = json.loads(raw_llm_response)
                logger.info("Successfully parsed JSON string response")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                response_str = str(raw_llm_response)
                response_preview = response_str[:200] if len(
                    response_str) > 200 else response_str
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM returned invalid JSON string: {response_preview}..."
                )

        # Ensure we have a dictionary
        if not isinstance(raw_llm_response, dict):
            raise HTTPException(
                status_code=500,
                detail=f"LLM response is not a dictionary. Got: {type(raw_llm_response)}"
            )

        # Extract metadata from LLM response
        metadata = raw_llm_response.get("metadata", {})
        subject_name = str(metadata.get("subject_name", "General Studies"))
        chapter_name = str(metadata.get("chapter_name", "Study Material"))
        concept_name = str(metadata.get("concept_name", "Learning Content"))

        logger.info(
            f"Extracted metadata - Subject: {subject_name}, Chapter: {chapter_name}, Concept: {concept_name}")

        # Transform the LangChain response to dictionary format
        try:
            llm_response_dict = transform_langchain_response(raw_llm_response)
            logger.info("Successfully transformed LangChain response")
        except Exception as e:
            logger.error(f"Error transforming response: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error transforming LLM response: {str(e)}"
            )

        # Process the LLM response through the existing service
        try:
            result = await service.process_llm_response(
                student_id=student_id,
                subject_name=subject_name,
                chapter_name=chapter_name,
                concept_name=concept_name,
                llm_response=llm_response_dict,  # Pass dictionary instead of Pydantic model
                user_query=user_query,
            )
            logger.info("Successfully processed LLM response through service")
        except Exception as e:
            logger.error(f"Error in service.process_llm_response: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing LLM response: {str(e)}"
            )

        # Schedule background recommendation generation
        try:
            background_tasks.add_task(
                service.generate_and_save_recommendations, student_id
            )
            logger.info("Scheduled background recommendation generation")
        except Exception as e:
            logger.warning(f"Error scheduling background task: {e}")
            # Don't fail the request for this

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message=f"Files processed successfully. Detected: {subject_name} - {chapter_name} - {concept_name}",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_and_process_simple: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing files: {str(e)}"
        )

    finally:
        # Clean up temporary files
        cleanup_temp_dir(tmp_dir)


@router.post("/upload-and-process", response_model=ProcessedLLMResponse)
async def upload_and_process_files(
    student_id: str = Form(...),
    subject_name: str = Form(...),
    chapter_name: str = Form(...),
    concept_name: str = Form(...),
    user_query: str = Form(...),
    difficulty_level: str = Form("Medium"),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db=Depends(get_db_manager)
):
    """
    Upload files (PDFs, images, etc.) and process them to generate study materials
    """
    tmp_dir = None
    uploaded_files_paths = []

    try:
        logger.info(f"Processing upload for student_id: {student_id}")
        logger.info(
            f"Subject: {subject_name}, Chapter: {chapter_name}, Concept: {concept_name}")

        # Validate input parameters
        if not student_id or not student_id.strip():
            raise HTTPException(
                status_code=400, detail="student_id is required and cannot be empty")

        # Save uploaded files to the temporary directory
        tmp_dir, uploaded_files_paths = await save_uploaded_files_to_temp(files)

        # Ensure student exists in DB first
        service = LearningProgressService(db)
        await service.create_or_get_student(
            student_id=student_id,
            username=f"user_{student_id[:10]}",  # Truncate for username
            email=f"{student_id}@placeholder.com",
            full_name=f"User {student_id[:8]}"  # Truncate for display name
        )

        # Process files using LangChain LLM
        raw_llm_response = await get_llm_response(
            uploaded_files_paths=uploaded_files_paths,
            userprompt=user_query,
            temp_dir=tmp_dir
        )

        # Debug: print the type and content of raw_llm_response
        logger.info(f"DEBUG: raw_llm_response type: {type(raw_llm_response)}")

        # Handle case where response might be a string (JSON string)
        if isinstance(raw_llm_response, str):
            try:
                raw_llm_response = json.loads(raw_llm_response)
            except json.JSONDecodeError:
                response_str = str(raw_llm_response)
                response_preview = response_str[:200] if len(
                    response_str) > 200 else response_str
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM returned invalid JSON string: {response_preview}..."
                )

        # Transform the LangChain response to dictionary format
        llm_response_dict = transform_langchain_response(raw_llm_response)

        # Process the LLM response through the existing service
        result = await service.process_llm_response(
            student_id=student_id,
            subject_name=subject_name,
            chapter_name=chapter_name,
            concept_name=concept_name,
            llm_response=llm_response_dict,  # Pass dictionary instead of Pydantic model
            user_query=user_query,
        )

        # Schedule background recommendation generation
        background_tasks.add_task(
            service.generate_and_save_recommendations, student_id
        )

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message="Files processed and study materials generated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Error processing files: {str(e)}")

    finally:
        # Clean up temporary files
        cleanup_temp_dir(tmp_dir)


@router.post("/process-llm-response", response_model=ProcessedLLMResponse)
async def process_llm_response(request: LLMResponseRequest, background_tasks: BackgroundTasks, db=Depends(get_db_manager)):
    try:
        service = LearningProgressService(db)

        # Ensure student exists first
        await service.create_or_get_student(
            student_id=request.student_id,
            username=f"user_{request.student_id[:10]}",
            email=f"{request.student_id}@placeholder.com",
            full_name=f"User {request.student_id[:8]}"
        )

        # Convert Pydantic model to dict for the service
        llm_response_dict = request.llm_response.dict() if hasattr(
            request.llm_response, 'dict') else request.llm_response

        result = await service.process_llm_response(
            student_id=request.student_id,
            subject_name=request.subject_name,
            chapter_name=request.chapter_name,
            concept_name=request.concept_name,
            llm_response=llm_response_dict,
            user_query=request.user_query,
        )

        # Schedule background recommendation generation
        background_tasks.add_task(
            service.generate_and_save_recommendations, request.student_id
        )

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message="LLM response processed and progress updated successfully",
        )
    except Exception as e:
        logger.error(f"Error in process_llm_response: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Add this new endpoint to get student progress
@router.get("/student/{student_id}/progress")
async def get_student_progress(
    student_id: str,
    subject_id: Optional[int] = None,
    db=Depends(get_db_manager)
):
    """
    Get comprehensive progress data for a student
    """
    try:
        service = LearningProgressService(db)
        progress_data = await service.get_student_progress(student_id, subject_id)
        return progress_data
    except Exception as e:
        logger.error(f"Error getting student progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add this new endpoint to update learning preferences
@router.post("/student/{student_id}/preferences")
async def update_student_preferences(
    student_id: str,
    preferences: Dict[str, Any],
    db=Depends(get_db_manager)
):
    """
    Update student learning preferences
    """
    try:
        service = LearningProgressService(db)
        await service.update_learning_preferences(student_id, preferences)
        return {"message": "Learning preferences updated successfully"}
    except Exception as e:
        logger.error(f"Error updating student preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))
