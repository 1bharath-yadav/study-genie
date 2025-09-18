from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from app.services import activity_service
from app.core.security import get_current_student_id
from app.services.db import create_supabase_client

router = APIRouter(prefix="/analytics", tags=["analytics"])


class ActivityIn(BaseModel):
	activity_type: str
	related_subject_id: int | None = None
	related_chapter_id: int | None = None
	related_concept_id: int | None = None
	payload: Dict[str, Any] = {}
	score: float | None = None
	time_spent_seconds: int | None = None


@router.post('/activity')
async def post_activity(activity: ActivityIn, student_id: str = Depends(get_current_student_id)):
	try:
		obj = activity.dict()
		obj['student_id'] = student_id
		rec = await activity_service.insert_activity(obj)
		if rec is None:
			raise HTTPException(status_code=500, detail='Failed to save activity')
		return {'success': True, 'data': rec}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get('/{student_identifier}/summary')
async def get_summary(student_identifier: str, days: int = 30, current_student_id: str = Depends(get_current_student_id)):
	# Only allow requesting own data for now
	if student_identifier != current_student_id:
		raise HTTPException(status_code=403, detail='Forbidden')
	summary = await activity_service.get_activity_summary(current_student_id, days)
	return {'success': True, 'data': summary}


# @router.get("/{student_identifier}/progress")
# async def get_progress_analytics_endpoint(
#     student_identifier: str,
#     subject_id: Optional[int] = None,
#     current_student_id: str = Depends(get_current_student_id)
# ) -> Dict[str, Any]:
#     """Get detailed progress analytics for subjects and concepts"""
#     try:
#         # Resolve student identifier to actual student_id
#         resolved_student_id = await resolve_student_id(student_identifier)
#         if not resolved_student_id:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Student not found: {student_identifier}"
#             )

#         analytics_data = await get_dashboard_analytics(resolved_student_id, 30)
        
#         # Extract concept progress data
#         concept_progress = analytics_data.get('concept_progress', [])

#         # Group by subject
#         subjects_data = {}
#         for concept in concept_progress:
#             subject_name = concept.get('subject_name', 'General')
#             if subject_name not in subjects_data:
#                 subjects_data[subject_name] = []
#             subjects_data[subject_name].append(concept)

#         # Filter by subject_id if provided
#         if subject_id:
#             # Filter concepts that belong to specific subject
#             filtered_concepts = [
#                 concept for concept in concept_progress
#                 if concept.get('subject_id') == subject_id
#             ]
#             concept_progress = filtered_concepts

#         return {
#             "success": True,
#             "data": {
#                 "progress": {
#                     "concept_progress": concept_progress,
#                     "overall_mastery_percentage": analytics_data.get('overall_mastery_percentage', 0)
#                 },
#                 "subjects_breakdown": subjects_data,
#                 "total_concepts": len(concept_progress),
#                 "subjects_count": len(subjects_data)
#             }
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching progress analytics: {e}")
#         return {
#             "success": False,
#             "data": {
#                 "progress": {"concept_progress": [], "overall_mastery_percentage": 0},
#                 "subjects_breakdown": {},
#                 "total_concepts": 0,
#                 "subjects_count": 0
#             }
#         }


# @router.get("/{student_identifier}/achievements")
# async def get_achievements_endpoint(
#     student_identifier: str,
#     current_student_id: str = Depends(get_current_student_id)
# ) -> Dict[str, Any]:
#     """Get student achievements and milestones"""
#     try:
#         # Resolve student identifier to actual student_id
#         resolved_student_id = await resolve_student_id(student_identifier)
#         if not resolved_student_id:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Student not found: {student_identifier}"
#             )

#         analytics_data = await get_dashboard_analytics(resolved_student_id, 30)
        
#         achievements = []
#         concepts_completed = analytics_data.get('total_concepts', 0)
#         overall_mastery = analytics_data.get('overall_mastery_percentage', 0)

#         # Generate achievements based on progress
#         if concepts_completed >= 5:
#             achievements.append({
#                 "title": "Quick Learner",
#                 "icon": "Zap",
#                 "color": "text-accent",
#                 "points": 50,
#                 "description": f"Completed {concepts_completed} concepts"
#             })

#         if overall_mastery >= 70:
#             achievements.append({
#                 "title": "High Achiever",
#                 "icon": "Trophy",
#                 "color": "text-yellow-500",
#                 "points": 100,
#                 "description": f"Achieved {overall_mastery:.1f}% overall mastery"
#             })

#         if concepts_completed >= 10:
#             achievements.append({
#                 "title": "Knowledge Seeker",
#                 "icon": "BookOpen",
#                 "color": "text-blue-500",
#                 "points": 75,
#                 "description": f"Explored {concepts_completed} different concepts"
#             })

#         if overall_mastery >= 90:
#             achievements.append({
#                 "title": "Master Scholar",
#                 "icon": "Crown",
#                 "color": "text-purple-500",
#                 "points": 150,
#                 "description": f"Achieved exceptional {overall_mastery:.1f}% mastery"
#             })

#         return {
#             "success": True,
#             "data": {
#                 "achievements": achievements,
#                 "total_points": sum(a.get('points', 0) for a in achievements),
#                 "badges_earned": len(achievements),
#                 "student_stats": {
#                     "concepts_completed": concepts_completed,
#                     "overall_mastery": overall_mastery
#                 }
#             }
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching achievements: {e}")
#         return {
#             "success": False,
#             "data": {
#                 "achievements": [],
#                 "total_points": 0,
#                 "badges_earned": 0,
#                 "student_stats": {"concepts_completed": 0, "overall_mastery": 0}
#             }
#         }


# @router.get("/{student_identifier}/weekly-trends")
# Weakness analysis
@router.get('/{student_identifier}/weaknesses')
async def get_weaknesses(student_identifier: str, days: int = 30, current_student_id: str = Depends(get_current_student_id)):
	if student_identifier != current_student_id:
		raise HTTPException(status_code=403, detail='Forbidden')

	raw = await activity_service.get_weakness_analysis(current_student_id, days)

	# Normalize to frontend-expected shape: { subjects: [...], concepts: [...] }
	subjects_raw = raw.get('subject_weaknesses', []) or []
	concepts_raw = raw.get('concept_weaknesses', []) or []

	client = create_supabase_client()

	# Helper to resolve names for ids (best-effort)
	def resolve_subject_name(sid: str) -> str:
		try:
			if not client:
				return str(sid)
			resp = client.table('subjects').select('llm_suggested_subject_name, subject_id').eq('subject_id', int(sid)).limit(1).execute()
			if getattr(resp, 'data', None):
				row = resp.data[0]
				return row.get('llm_suggested_subject_name') or str(sid)
			return str(sid)
		except Exception:
			return str(sid)

	def resolve_concept_name(cid: str) -> str:
		try:
			if not client:
				return str(cid)
			resp = client.table('concepts').select('llm_suggested_concept_name, concept_id').eq('concept_id', int(cid)).limit(1).execute()
			if getattr(resp, 'data', None):
				row = resp.data[0]
				return row.get('llm_suggested_concept_name') or str(cid)
			return str(cid)
		except Exception:
			return str(cid)

	subjects: List[Dict[str, Any]] = []
	for s in subjects_raw:
		sid = s.get('subject_id')
		subjects.append({
			'subject_id': sid,
			'name': resolve_subject_name(sid) if sid is not None else None,
			'weakness_score': s.get('weakness_score'),
			'avg_score': s.get('avg_score'),
			'attempts': s.get('count')
		})

	concepts: List[Dict[str, Any]] = []
	for c in concepts_raw:
		cid = c.get('concept_id')
		concepts.append({
			'concept_id': cid,
			'name': resolve_concept_name(cid) if cid is not None else None,
			'avg_score': c.get('avg_score'),
			'attempts': c.get('count'),
			'weakness_score': c.get('weakness_score'),
		})

	return {'success': True, 'data': {'subjects': subjects, 'concepts': concepts}}


@router.get('/{student_identifier}/weekly-trends')
async def get_weekly_trends_endpoint(student_identifier: str, weeks: int = 4, current_student_id: str = Depends(get_current_student_id)):
	if student_identifier != current_student_id:
		raise HTTPException(status_code=403, detail='Forbidden')

	data = await activity_service.get_weekly_trends(current_student_id, weeks)
	# activity_service returns {'weeks': [...]}; frontend expects an array
	weeks_list = data.get('weeks', []) if isinstance(data, dict) else []
	return {'success': True, 'data': weeks_list}


@router.get('/{student_identifier}/monthly-trends')
async def get_monthly_trends_endpoint(student_identifier: str, months: int = 3, current_student_id: str = Depends(get_current_student_id)):
	if student_identifier != current_student_id:
		raise HTTPException(status_code=403, detail='Forbidden')

	data = await activity_service.get_monthly_trends(current_student_id, months)
	return {'success': True, 'data': data}
# async def get_weekly_trends_endpoint(
#     student_identifier: str,
#     weeks: int = Query(4, description="Number of weeks to look back"),
#     current_student_id: str = Depends(get_current_student_id)
# ) -> Dict[str, Any]:
#     """Get weekly performance trends"""
#     try:
#         # Resolve student identifier to actual student_id
#         resolved_student_id = await resolve_student_id(student_identifier)
#         if not resolved_student_id:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Student not found: {student_identifier}"
#             )

#         trends_data = await get_weekly_trends_analytics(resolved_student_id, weeks)

#         return {
#             "success": True,
#             "data": trends_data
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching weekly trends: {e}")
#         return {
#             "success": False,
#             "data": {
#                 "weekly_progress": [],
#                 "trend": "stable",
#                 "improvement_rate": 0
#             }
#         }


# @router.get("/{student_identifier}/weaknesses")
# async def get_weakness_analysis_endpoint(
#     student_identifier: str,
#     current_student_id: str = Depends(get_current_student_id)
# ) -> Dict[str, Any]:
#     """Get student's weak areas and improvement suggestions"""
#     try:
#         # Resolve student identifier to actual student_id
#         resolved_student_id = await resolve_student_id(student_identifier)
#         if not resolved_student_id:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Student not found: {student_identifier}"
#             )

#         weakness_data = await get_weakness_analysis(resolved_student_id)

#         return {
#             "success": True,
#             "data": weakness_data
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching weakness analysis: {e}")
#         return {
#             "success": False,
#             "data": {
#                 "weak_concepts": [],
#                 "weak_subjects": [],
#                 "improvement_areas": [],
#                 "needs_attention": 0,
#                 "priority_actions": [],
#                 "overall_weakness_score": 0
#             }
#         }


# @router.get("/{student_identifier}/study-patterns")
# async def get_study_patterns_endpoint(
#     student_identifier: str,
#     days: int = Query(30, description="Number of days to analyze"),
#     current_student_id: str = Depends(get_current_student_id)
# ) -> Dict[str, Any]:
#     """Get study patterns and behavior analytics for a student"""
#     try:
#         # Resolve student identifier to actual student_id
#         resolved_student_id = await resolve_student_id(student_identifier)
#         if not resolved_student_id:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Student not found: {student_identifier}"
#             )

#         analytics_data = await get_dashboard_analytics(resolved_student_id, days)
        
#         # Extract study patterns from analytics data
#         total_time = analytics_data.get('total_time_spent', 0)
#         total_concepts = analytics_data.get('total_concepts', 0)
#         mastered_concepts = analytics_data.get('mastered_concepts', 0)
        
#         # Calculate derived metrics
#         daily_average_time = total_time / max(1, days)  # minutes per day
#         concepts_per_hour = total_concepts / max(1, total_time / 60) if total_time > 0 else 0
#         retention_rate = mastered_concepts / max(1, total_concepts)
#         consistency_score = min(100, (mastered_concepts / max(1, days)) * 10)

#         study_patterns = {
#             "daily_study_time": round(daily_average_time, 1),
#             "peak_study_hours": [14, 16, 20],  # Default afternoon and evening peaks
#             "consistency_score": round(consistency_score, 1),
#             "preferred_study_duration": 45,  # Default 45 minutes
#             "break_patterns": {
#                 "frequency": "every_45_minutes",
#                 "duration": 15
#             },
#             "learning_velocity": {
#                 "concepts_per_hour": round(concepts_per_hour, 2),
#                 "retention_rate": round(retention_rate, 2)
#             },
#             "study_habits": {
#                 "total_study_time": total_time,
#                 "average_session_length": 45,  # Default
#                 "most_active_period": "afternoon",
#                 "consistency": round(consistency_score, 1)
#             },
#             "performance_metrics": {
#                 "total_concepts_studied": total_concepts,
#                 "concepts_mastered": mastered_concepts,
#                 "overall_mastery_rate": round(retention_rate * 100, 1),
#                 "study_efficiency": round(concepts_per_hour, 2)
#             }
#         }

#         return {
#             "success": True,
#             "data": study_patterns
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching study patterns: {e}")
#         return {
#             "success": False,
#             "data": {
#                 "daily_study_time": 0,
#                 "peak_study_hours": [14, 16, 20],
#                 "consistency_score": 0,
#                 "preferred_study_duration": 45,
#                 "break_patterns": {"frequency": "every_45_minutes", "duration": 15},
#                 "learning_velocity": {"concepts_per_hour": 0, "retention_rate": 0},
#                 "study_habits": {"total_study_time": 0, "average_session_length": 0, "most_active_period": "unknown", "consistency": 0},
#                 "performance_metrics": {"total_concepts_studied": 0, "concepts_mastered": 0, "overall_mastery_rate": 0, "study_efficiency": 0}
#             }
#         }
