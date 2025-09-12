"""
Functional Analytics API Routes
Pure functional approach with Supabase JWT integration
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any
import logging

from app.core.security import get_current_user_id
from app.services.analytics_service import (
    resolve_student_id,
    get_dashboard_analytics,
    get_weekly_trends_analytics,
    get_weakness_analysis
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard")
async def get_dashboard_analytics_endpoint(
    days: int = Query(default=30, description="Number of days to analyze"),
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get dashboard analytics data for the current authenticated user"""
    try:
        # Resolve user_id to student_id
        student_id = await resolve_student_id(current_user_id)
        if not student_id:
            raise HTTPException(
                status_code=404, 
                detail="Student profile not found for current user"
            )

        analytics_data = await get_dashboard_analytics(student_id, days)

        return {
            "success": True,
            "data": analytics_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dashboard analytics: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching dashboard analytics: {str(e)}"
        )


@router.get("/{student_identifier}/dashboard")
async def get_student_dashboard_analytics_endpoint(
    student_identifier: str,
    days: int = Query(default=30, description="Number of days to analyze"),
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get dashboard analytics data for a specific student (admin/teacher access)"""
    try:
        # Resolve student identifier to actual student_id
        resolved_student_id = await resolve_student_id(student_identifier)
        if not resolved_student_id:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {student_identifier}"
            )

        analytics_data = await get_dashboard_analytics(resolved_student_id, days)

        return {
            "success": True,
            "data": analytics_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching student dashboard analytics: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching dashboard analytics: {str(e)}"
        )


@router.get("/{student_identifier}/subjects")
async def get_subject_wise_analytics_endpoint(
    student_identifier: str,
    days: int = Query(default=30, description="Number of days to analyze"),
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get detailed subject-wise analytics for a student"""
    try:
        # Resolve student identifier to actual student_id
        resolved_student_id = await resolve_student_id(student_identifier)
        if not resolved_student_id:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {student_identifier}"
            )

        analytics_data = await get_dashboard_analytics(resolved_student_id, days)

        # Extract and format subject-wise data
        subjects_data = analytics_data.get('subjects_analytics', {})
        subjects_summary = analytics_data.get('subjects_summary', [])

        # Calculate overall statistics
        total_subjects = len(subjects_data)
        total_concepts = sum(s.get('concepts_total', 0) for s in subjects_summary)
        total_mastered = sum(s.get('concepts_mastered', 0) for s in subjects_summary)
        average_mastery = (
            sum(s.get('mastery_percentage', 0) for s in subjects_summary) / len(subjects_summary)
            if subjects_summary else 0
        )
        total_time_spent = sum(s.get('time_spent', 0) for s in subjects_summary)
        
        most_studied_subject = (
            max(subjects_summary, key=lambda x: x.get('time_spent', 0)).get('subject_name')
            if subjects_summary else None
        )
        best_performing_subject = (
            max(subjects_summary, key=lambda x: x.get('mastery_percentage', 0)).get('subject_name')
            if subjects_summary else None
        )

        return {
            "success": True,
            "data": {
                "student_id": student_identifier,
                "resolved_student_id": resolved_student_id,
                "period_days": days,
                "subjects_summary": subjects_summary,
                "subjects_detail": subjects_data,
                "overall_stats": {
                    "total_subjects": total_subjects,
                    "total_concepts": total_concepts,
                    "total_mastered": total_mastered,
                    "average_mastery": round(average_mastery, 2),
                    "total_time_spent": total_time_spent,
                    "most_studied_subject": most_studied_subject,
                    "best_performing_subject": best_performing_subject
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching subject analytics: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching subject analytics: {str(e)}"
        )


@router.get("/{student_identifier}/progress")
async def get_progress_analytics_endpoint(
    student_identifier: str,
    subject_id: Optional[int] = None,
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get detailed progress analytics for subjects and concepts"""
    try:
        # Resolve student identifier to actual student_id
        resolved_student_id = await resolve_student_id(student_identifier)
        if not resolved_student_id:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {student_identifier}"
            )

        analytics_data = await get_dashboard_analytics(resolved_student_id, 30)
        
        # Extract concept progress data
        concept_progress = analytics_data.get('concept_progress', [])

        # Group by subject
        subjects_data = {}
        for concept in concept_progress:
            subject_name = concept.get('subject_name', 'General')
            if subject_name not in subjects_data:
                subjects_data[subject_name] = []
            subjects_data[subject_name].append(concept)

        # Filter by subject_id if provided
        if subject_id:
            # Filter concepts that belong to specific subject
            filtered_concepts = [
                concept for concept in concept_progress
                if concept.get('subject_id') == subject_id
            ]
            concept_progress = filtered_concepts

        return {
            "success": True,
            "data": {
                "progress": {
                    "concept_progress": concept_progress,
                    "overall_mastery_percentage": analytics_data.get('overall_mastery_percentage', 0)
                },
                "subjects_breakdown": subjects_data,
                "total_concepts": len(concept_progress),
                "subjects_count": len(subjects_data)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching progress analytics: {e}")
        return {
            "success": False,
            "data": {
                "progress": {"concept_progress": [], "overall_mastery_percentage": 0},
                "subjects_breakdown": {},
                "total_concepts": 0,
                "subjects_count": 0
            }
        }


@router.get("/{student_identifier}/achievements")
async def get_achievements_endpoint(
    student_identifier: str,
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get student achievements and milestones"""
    try:
        # Resolve student identifier to actual student_id
        resolved_student_id = await resolve_student_id(student_identifier)
        if not resolved_student_id:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {student_identifier}"
            )

        analytics_data = await get_dashboard_analytics(resolved_student_id, 30)
        
        achievements = []
        concepts_completed = analytics_data.get('total_concepts', 0)
        overall_mastery = analytics_data.get('overall_mastery_percentage', 0)

        # Generate achievements based on progress
        if concepts_completed >= 5:
            achievements.append({
                "title": "Quick Learner",
                "icon": "Zap",
                "color": "text-accent",
                "points": 50,
                "description": f"Completed {concepts_completed} concepts"
            })

        if overall_mastery >= 70:
            achievements.append({
                "title": "High Achiever",
                "icon": "Trophy",
                "color": "text-yellow-500",
                "points": 100,
                "description": f"Achieved {overall_mastery:.1f}% overall mastery"
            })

        if concepts_completed >= 10:
            achievements.append({
                "title": "Knowledge Seeker",
                "icon": "BookOpen",
                "color": "text-blue-500",
                "points": 75,
                "description": f"Explored {concepts_completed} different concepts"
            })

        if overall_mastery >= 90:
            achievements.append({
                "title": "Master Scholar",
                "icon": "Crown",
                "color": "text-purple-500",
                "points": 150,
                "description": f"Achieved exceptional {overall_mastery:.1f}% mastery"
            })

        return {
            "success": True,
            "data": {
                "achievements": achievements,
                "total_points": sum(a.get('points', 0) for a in achievements),
                "badges_earned": len(achievements),
                "student_stats": {
                    "concepts_completed": concepts_completed,
                    "overall_mastery": overall_mastery
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching achievements: {e}")
        return {
            "success": False,
            "data": {
                "achievements": [],
                "total_points": 0,
                "badges_earned": 0,
                "student_stats": {"concepts_completed": 0, "overall_mastery": 0}
            }
        }


@router.get("/{student_identifier}/weekly-trends")
async def get_weekly_trends_endpoint(
    student_identifier: str,
    weeks: int = Query(4, description="Number of weeks to look back"),
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get weekly performance trends"""
    try:
        # Resolve student identifier to actual student_id
        resolved_student_id = await resolve_student_id(student_identifier)
        if not resolved_student_id:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {student_identifier}"
            )

        trends_data = await get_weekly_trends_analytics(resolved_student_id, weeks)

        return {
            "success": True,
            "data": trends_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching weekly trends: {e}")
        return {
            "success": False,
            "data": {
                "weekly_progress": [],
                "trend": "stable",
                "improvement_rate": 0
            }
        }


@router.get("/{student_identifier}/weaknesses")
async def get_weakness_analysis_endpoint(
    student_identifier: str,
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get student's weak areas and improvement suggestions"""
    try:
        # Resolve student identifier to actual student_id
        resolved_student_id = await resolve_student_id(student_identifier)
        if not resolved_student_id:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {student_identifier}"
            )

        weakness_data = await get_weakness_analysis(resolved_student_id)

        return {
            "success": True,
            "data": weakness_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching weakness analysis: {e}")
        return {
            "success": False,
            "data": {
                "weak_concepts": [],
                "weak_subjects": [],
                "improvement_areas": [],
                "needs_attention": 0,
                "priority_actions": [],
                "overall_weakness_score": 0
            }
        }


@router.get("/{student_identifier}/study-patterns")
async def get_study_patterns_endpoint(
    student_identifier: str,
    days: int = Query(30, description="Number of days to analyze"),
    current_user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Get study patterns and behavior analytics for a student"""
    try:
        # Resolve student identifier to actual student_id
        resolved_student_id = await resolve_student_id(student_identifier)
        if not resolved_student_id:
            raise HTTPException(
                status_code=404,
                detail=f"Student not found: {student_identifier}"
            )

        analytics_data = await get_dashboard_analytics(resolved_student_id, days)
        
        # Extract study patterns from analytics data
        total_time = analytics_data.get('total_time_spent', 0)
        total_concepts = analytics_data.get('total_concepts', 0)
        mastered_concepts = analytics_data.get('mastered_concepts', 0)
        
        # Calculate derived metrics
        daily_average_time = total_time / max(1, days)  # minutes per day
        concepts_per_hour = total_concepts / max(1, total_time / 60) if total_time > 0 else 0
        retention_rate = mastered_concepts / max(1, total_concepts)
        consistency_score = min(100, (mastered_concepts / max(1, days)) * 10)

        study_patterns = {
            "daily_study_time": round(daily_average_time, 1),
            "peak_study_hours": [14, 16, 20],  # Default afternoon and evening peaks
            "consistency_score": round(consistency_score, 1),
            "preferred_study_duration": 45,  # Default 45 minutes
            "break_patterns": {
                "frequency": "every_45_minutes",
                "duration": 15
            },
            "learning_velocity": {
                "concepts_per_hour": round(concepts_per_hour, 2),
                "retention_rate": round(retention_rate, 2)
            },
            "study_habits": {
                "total_study_time": total_time,
                "average_session_length": 45,  # Default
                "most_active_period": "afternoon",
                "consistency": round(consistency_score, 1)
            },
            "performance_metrics": {
                "total_concepts_studied": total_concepts,
                "concepts_mastered": mastered_concepts,
                "overall_mastery_rate": round(retention_rate * 100, 1),
                "study_efficiency": round(concepts_per_hour, 2)
            }
        }

        return {
            "success": True,
            "data": study_patterns
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching study patterns: {e}")
        return {
            "success": False,
            "data": {
                "daily_study_time": 0,
                "peak_study_hours": [14, 16, 20],
                "consistency_score": 0,
                "preferred_study_duration": 45,
                "break_patterns": {"frequency": "every_45_minutes", "duration": 15},
                "learning_velocity": {"concepts_per_hour": 0, "retention_rate": 0},
                "study_habits": {"total_study_time": 0, "average_session_length": 0, "most_active_period": "unknown", "consistency": 0},
                "performance_metrics": {"total_concepts_studied": 0, "concepts_mastered": 0, "overall_mastery_rate": 0, "study_efficiency": 0}
            }
        }
