from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, File, Form, UploadFile
from typing import List, Dict, Any
import os
import shutil
import tempfile
import json
from pathlib import Path
from app.core import get_db_manager
from app.services import LearningProgressService
from app.models import LLMResponseRequest, ProcessedLLMResponse, LLMResponseContent, FlashcardData, QuizData, DifficultyLevel
from app.llm.langchain import get_llm_response

router = APIRouter()


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
        summary=raw_response.get("summary", ""),
        learning_objectives=raw_response.get("learning_objectives", [])
    )


@router.post("/upload-simple", response_model=ProcessedLLMResponse)
async def upload_and_process_simple(
    student_id: int = Form(...),
    user_query: str = Form(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db=Depends(get_db_manager)
):
    """
    Simplified upload endpoint that only requires student_id, prompt and files.
    All metadata (subject, chapter, concept, difficulty) is automatically extracted by the LLM.
    """
    # Create a temporary directory for this request
    tmp_dir = tempfile.mkdtemp(prefix="study_session_")
    uploaded_files_paths = []

    try:
        # Validate files
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Save uploaded files to the temporary directory
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400, detail="Uploaded file missing filename")

            # Check file type (optional - you can add more validation)
            allowed_extensions = {'.pdf', '.txt',
                                  '.doc', '.docx', '.jpg', '.jpeg', '.png'}
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_extensions)}"
                )

            file_path = Path(tmp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            uploaded_files_paths.append(file_path)

        # Process files using LangChain LLM with metadata extraction
        raw_llm_response = await get_llm_response(
            uploaded_files_paths=uploaded_files_paths,
            userprompt=user_query,
            temp_dir=tmp_dir
        )

        # Debug: print the type and content of raw_llm_response
        print(f"DEBUG: raw_llm_response type: {type(raw_llm_response)}")
        print(f"DEBUG: raw_llm_response content: {raw_llm_response}")

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

        # Extract metadata from LLM response
        metadata = raw_llm_response.get("metadata", {})
        subject_name = metadata.get("subject_name", "General Studies")
        chapter_name = metadata.get("chapter_name", "Study Material")
        concept_name = metadata.get("concept_name", "Learning Content")

        # Transform the LangChain response to LLMResponseContent format
        llm_response = transform_langchain_response(raw_llm_response)

        # Process the LLM response through the existing service
        service = LearningProgressService(db)
        result = await service.process_llm_response(
            student_id=student_id,
            subject_name=subject_name,
            chapter_name=chapter_name,
            concept_name=concept_name,
            llm_response=llm_response,
            user_query=user_query,
        )

        # Schedule background recommendation generation
        background_tasks.add_task(lambda sid, d: None, student_id, db)

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message=f"Files processed successfully. Detected: {subject_name} - {chapter_name} - {concept_name}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing files: {str(e)}")

    finally:
        # Clean up temporary files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/upload-and-process", response_model=ProcessedLLMResponse)
async def upload_and_process_files(
    student_id: int = Form(...),
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
    # Create a temporary directory for this request
    tmp_dir = tempfile.mkdtemp(prefix="study_session_")
    uploaded_files_paths = []

    try:
        # Validate files
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Save uploaded files to the temporary directory
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400, detail="Uploaded file missing filename")

            # Check file type (optional - you can add more validation)
            allowed_extensions = {'.pdf', '.txt',
                                  '.doc', '.docx', '.jpg', '.jpeg', '.png'}
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_extensions)}"
                )

            file_path = Path(tmp_dir) / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            uploaded_files_paths.append(file_path)

        # Process files using LangChain LLM
        raw_llm_response = await get_llm_response(
            uploaded_files_paths=uploaded_files_paths,
            userprompt=user_query,
            temp_dir=tmp_dir
        )

        # Debug: print the type and content of raw_llm_response
        print(f"DEBUG: raw_llm_response type: {type(raw_llm_response)}")
        print(f"DEBUG: raw_llm_response content: {raw_llm_response}")

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

        # Transform the LangChain response to LLMResponseContent format
        llm_response = transform_langchain_response(raw_llm_response)

        # Process the LLM response through the existing service
        service = LearningProgressService(db)
        result = await service.process_llm_response(
            student_id=student_id,
            subject_name=subject_name,
            chapter_name=chapter_name,
            concept_name=concept_name,
            llm_response=llm_response,
            user_query=user_query,
        )

        # Schedule background recommendation generation
        background_tasks.add_task(lambda sid, d: None, student_id, db)

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message="Files processed and study materials generated successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing files: {str(e)}")

    finally:
        # Clean up temporary files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/process-llm-response", response_model=ProcessedLLMResponse)
async def process_llm_response(request: LLMResponseRequest, background_tasks: BackgroundTasks, db=Depends(get_db_manager)):
    try:
        service = LearningProgressService(db)
        result = await service.process_llm_response(
            student_id=request.student_id,
            subject_name=request.subject_name,
            chapter_name=request.chapter_name,
            concept_name=request.concept_name,
            llm_response=request.llm_response,
            user_query=request.user_query,
        )

        # schedule background recommendation generation
        background_tasks.add_task(lambda sid, d: None, request.student_id, db)

        return ProcessedLLMResponse(
            enhanced_response=result["enhanced_response"],
            tracking_metadata=result["tracking_metadata"],
            created_entities=result["created_entities"],
            message="LLM response processed and progress updated successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
