from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from app.core import get_db_manager
from app.services import LearningProgressService
from app.models import LLMResponseRequest, ProcessedLLMResponse

router = APIRouter()


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
