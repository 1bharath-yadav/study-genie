from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, File, Form, UploadFile
from typing import List, Dict, Any, Optional
import shutil
import tempfile
import json
import uuid
import logging
from pathlib import Path
from app.services_supabase import get_learning_progress_service
from app.core.security import get_current_user
from app.core.llm_key_service import get_llm_key_service
from app.models import LLMResponseRequest, ProcessedLLMResponse, ProcessFilesResponse, LLMResponseContent, FlashcardData, QuizData, DifficultyLevel
from app.llm.langchain import get_llm_response

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants
ALLOWED_FILE_EXTENSIONS = {'.pdf', '.txt',
                           '.doc', '.docx', '.jpg', '.jpeg', '.png'}


async def validate_and_save_files(files: List[UploadFile], tmp_dir: str) -> List[Path]:
    """
    Validate uploaded files and save them to temporary directory.
    Returns list of file paths.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploaded_files_paths = []

    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=400, detail="Uploaded file missing filename")

        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
            )

        # Save file
        file_path = Path(tmp_dir) / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        uploaded_files_paths.append(file_path)

    return uploaded_files_paths


async def process_llm_response_common(
    raw_llm_response: Dict[str, Any],
    student_id: str,
    subject_name: str,
    chapter_name: str,
    concept_name: str,
    user_query: str
) -> tuple[LLMResponseContent, Dict[str, Any]]:
    """
    Common LLM response processing logic.
    Returns (llm_response, metadata)
    """
    # Check for API key error
    if raw_llm_response.get("error_type") == "missing_api_key":
        raise HTTPException(status_code=400, detail=raw_llm_response["error"])

    # Check for other errors
    if raw_llm_response.get("status") == "error":
        raise HTTPException(status_code=500, detail=raw_llm_response["error"])

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

    # Extract metadata
    metadata = raw_llm_response.get("metadata", {})

    # Transform the LangChain response to LLMResponseContent format
    llm_response = transform_langchain_response(raw_llm_response)

    # Process the LLM response through the existing service
    service = get_learning_progress_service()
    await service.process_llm_response(
        student_id=str(student_id),
        subject_name=subject_name,
        chapter_name=chapter_name,
        concept_name=concept_name,
        llm_response=llm_response.dict() if hasattr(
            llm_response, "dict") else dict(llm_response),
        user_query=user_query,
    )

    return llm_response, metadata


def transform_langchain_response(raw_response: Dict[str, Any]) -> LLMResponseContent:
    """
    Transform LangChain response format to LLMResponseContent format
    """
    if "error" in raw_response:
        # Handle error response
        return LLMResponseContent(
            flashcards={},
            quiz={},
            summary=f"Error processing files: {raw_response.get('error', 'Unknown error')}",
            learning_objectives=[]
        )

    # Transform flashcards from array to dict
    flashcards = {}
    if "flashcards" in raw_response and isinstance(raw_response["flashcards"], list):
        for i, card_data in enumerate(raw_response["flashcards"], 1):
            try:
                flashcards[f"card{i}"] = FlashcardData(
                    question=card_data.get("question", ""),
                    answer=card_data.get("answer", ""),
                    difficulty=DifficultyLevel(
                        card_data.get("difficulty", "Medium"))
                )
            except Exception as e:
                print(f"Error processing flashcard {i}: {e}")
                continue

    # Transform quiz questions from array to dict
    quiz = {}
    if "quiz" in raw_response and isinstance(raw_response["quiz"], list):
        for i, q_data in enumerate(raw_response["quiz"], 1):
            try:
                quiz[f"Q{i}"] = QuizData(
                    question=q_data.get("question", ""),
                    options=q_data.get("options", []),
                    correct_answer=q_data.get("correct_answer", ""),
                    explanation=q_data.get("explanation", "")
                )
            except Exception as e:
                print(f"Error processing quiz question {i}: {e}")
                continue

    return LLMResponseContent(
        flashcards=flashcards,
        quiz=quiz,
        match_the_following=raw_response.get("match_the_following"),
        summary=raw_response.get("summary", ""),
        learning_objectives=raw_response.get("learning_objectives", [])
    )


@router.post("/process-files", response_model=ProcessFilesResponse)
async def process_files(
    user_query: str = Form(...),
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Process files endpoint - maps to upload-simple for compatibility
    """
    return await upload_and_process_simple(user_query, files, current_user)


@router.post("/upload-simple", response_model=ProcessFilesResponse)
async def upload_and_process_simple(
    user_query: str = Form(...),
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Simplified upload endpoint that only requires user query and files.
    All metadata (subject, chapter, concept, difficulty) is automatically extracted by the LLM.
    """
    tmp_dir = tempfile.mkdtemp(prefix="study_session_")

    try:
        # Validate and save files
        uploaded_files_paths = await validate_and_save_files(files, tmp_dir)

        # Get authenticated user ID
        student_id = current_user.get("student_id") or current_user.get("sub")
        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Get user's API key
        llm_key_service = get_llm_key_service()
        user_api_key = await llm_key_service.get_api_key_for_user(str(student_id))
        if not user_api_key:
            raise HTTPException(
                status_code=400,
                detail="API key not found. Please add your Gemini API key in settings before uploading files."
            )

        # Process files using LangChain LLM with metadata extraction
        raw_llm_response = await get_llm_response(
            uploaded_files_paths=uploaded_files_paths,
            userprompt=user_query,
            temp_dir=tmp_dir,
            user_api_key=user_api_key
        )

        # Extract metadata from LLM response
        metadata = raw_llm_response.get("metadata", {})
        subject_name = metadata.get("subject_name", "General Studies")
        chapter_name = metadata.get("chapter_name", "Study Material")
        concept_name = metadata.get("concept_name", "Learning Content")

        # Log metadata extraction for debugging
        logger.info(f"LLM metadata extracted: {metadata}")
        logger.info(
            f"Final values - Subject: {subject_name}, Chapter: {chapter_name}, Concept: {concept_name}")

        # Process LLM response
        llm_response, metadata = await process_llm_response_common(
            raw_llm_response, student_id, subject_name, chapter_name, concept_name, user_query
        )

        # Return response with extracted metadata
        return ProcessFilesResponse(
            task_id=str(uuid.uuid4()),
            status="completed",
            content=llm_response,
            metadata=metadata,
            subject_name=subject_name,
            chapter_name=chapter_name,
            concept_name=concept_name,
            difficulty_level=metadata.get("difficulty_level"),
            estimated_study_time=metadata.get("estimated_study_time")
        )

    except HTTPException:
        raise
    except Exception as e:
        return ProcessFilesResponse(
            task_id=str(uuid.uuid4()),
            status="failed",
            error=str(e)
        )
    finally:
        # Clean up temporary files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/upload-and-process", response_model=ProcessFilesResponse)
async def upload_and_process_files(
    subject_name: str = Form(...),
    chapter_name: str = Form(...),
    concept_name: str = Form(...),
    user_query: str = Form(...),
    difficulty_level: str = Form("Medium"),
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload files with explicit metadata (subject, chapter, concept, difficulty).
    This endpoint is used when the user provides specific context information.
    """
    tmp_dir = tempfile.mkdtemp(prefix="study_session_")

    try:
        # Validate and save files
        uploaded_files_paths = await validate_and_save_files(files, tmp_dir)

        # Get authenticated user ID
        student_id = current_user.get("student_id") or current_user.get("sub")
        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Get user's API key
        llm_key_service = get_llm_key_service()
        user_api_key = await llm_key_service.get_api_key_for_user(str(student_id))
        if not user_api_key:
            raise HTTPException(
                status_code=400,
                detail="API key not found. Please add your Gemini API key in settings before uploading files."
            )

        # Process files using LangChain LLM
        raw_llm_response = await get_llm_response(
            uploaded_files_paths=uploaded_files_paths,
            userprompt=user_query,
            temp_dir=tmp_dir,
            user_api_key=user_api_key
        )

        # Process LLM response using provided form data
        llm_response, metadata = await process_llm_response_common(
            raw_llm_response, student_id, subject_name, chapter_name, concept_name, user_query
        )

        # Return response with form data and any additional metadata from LLM
        return ProcessFilesResponse(
            task_id=str(uuid.uuid4()),
            status="completed",
            content=llm_response,
            metadata=metadata,
            subject_name=subject_name,
            chapter_name=chapter_name,
            concept_name=concept_name,
            difficulty_level=difficulty_level,
            estimated_study_time=metadata.get("estimated_study_time")
        )

    except HTTPException:
        raise
    except Exception as e:
        return ProcessFilesResponse(
            task_id=str(uuid.uuid4()),
            status="failed",
            error=str(e)
        )
    finally:
        # Clean up temporary files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/process-llm-response", response_model=ProcessedLLMResponse)
async def process_llm_response(request: LLMResponseRequest, background_tasks: BackgroundTasks):
    try:
        service = get_learning_progress_service()
        result = await service.process_llm_response(
            student_id=request.student_id,
            subject_name=request.subject_name,
            chapter_name=request.chapter_name,
            concept_name=request.concept_name,
            llm_response=request.llm_response.dict() if hasattr(
                request.llm_response, "dict") else dict(request.llm_response),
            user_query=request.user_query,
        )

        # schedule background recommendation generation
        # background_tasks.add_task(lambda sid, d: None, request.student_id, db)

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message="LLM response processed and progress updated successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
