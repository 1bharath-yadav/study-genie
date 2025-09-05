"""
Student recommendations using Supabase backend
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List, Dict, Any

from app.services_supabase import get_learning_progress_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/students/{student_id}/recommendations")
async def get_student_recommendations(student_id: str):
    """Get personalized recommendations for a student."""
    try:
        logger.info(f"API: Getting recommendations for student {student_id}")
        service = get_learning_progress_service()
        recommendations = await service.get_recommendations(student_id)

        logger.info(
            f"API: Returning {len(recommendations)} recommendations for student {student_id}")
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(
            f"API Error getting recommendations for {student_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/students/{student_id}/recommendations/refresh")
async def refresh_student_recommendations(student_id: str):
    """Force refresh recommendations for a student."""
    try:
        logger.info(
            f"API: Force refreshing recommendations for student {student_id}")
        service = get_learning_progress_service()

        # Generate fresh recommendations
        fresh_recommendations = await service.supabase_service.recommendation_engine.generate_personalized_recommendations(student_id)

        # Save them
        if fresh_recommendations:
            await service.supabase_service.recommendation_engine.save_recommendations(student_id, fresh_recommendations)

        logger.info(
            f"API: Generated and saved {len(fresh_recommendations)} fresh recommendations for student {student_id}")
        return {"recommendations": fresh_recommendations, "message": f"Generated {len(fresh_recommendations)} fresh recommendations"}
    except Exception as e:
        logger.error(
            f"API Error refreshing recommendations for {student_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
