"""Core service implementations moved into a package for modularization.

This file contains the original service classes with imports adjusted to
use absolute imports so it works when placed under the `app.services`
package.
"""
# progress_tracker/services core implementation
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math

from app.database import (
    DatabaseManager, StudentManager, SubjectManager,
    ProgressTracker, RecommendationEngine, AnalyticsManager, StudySessionManager
)
from app.models import *

logger = logging.getLogger("progress_tracker_services")


class LearningProgressService:
    """Service for managing student learning progress and LLM integration"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.student_manager = StudentManager(db_manager)
        self.subject_manager = SubjectManager(db_manager)
        self.progress_tracker = ProgressTracker(db_manager)

    async def create_or_get_student(self, username: str, email: str, full_name: str) -> int:
        return await self.student_manager.create_or_get_student(username, email, full_name)

    async def update_learning_preferences(self, student_id: int, preferences: Dict[str, Any]):
        await self.student_manager.update_learning_preferences(student_id, preferences)

    async def process_llm_response(self, student_id: int, subject_name: str,
                                   chapter_name: str, concept_name: str,
                                   llm_response: LLMResponseContent, user_query: str) -> Dict[str, Any]:
        try:
            subject_id = await self.subject_manager.create_or_get_subject(
                subject_name, f"Auto-created from LLM response"
            )
            chapter_id = await self.subject_manager.create_or_get_chapter(
                subject_id, chapter_name, f"Auto-created from LLM response"
            )
            concept_id = await self.subject_manager.create_or_get_concept(
                chapter_id, concept_name, "Medium", f"Auto-created from LLM response"
            )

            await self._ensure_student_enrollment(student_id, subject_id)

            activity_data = {
                'llm_response': llm_response.dict(),
                'user_query': user_query,
                'timestamp': datetime.now().isoformat(),
                'response_type': 'structured_content'
            }

            await self.progress_tracker.record_learning_activity(
                student_id, concept_id, ActivityType.CONTENT_STUDY.value,
                activity_data, time_spent=None
            )

            await self._analyze_and_update_progress(student_id, concept_id, llm_response)

            enhanced_response = llm_response.copy()

            tracking_metadata = TrackingMetadata(
                student_id=student_id,
                subject_id=subject_id,
                chapter_id=chapter_id,
                concept_id=concept_id,
                subject_name=subject_name,
                chapter_name=chapter_name,
                concept_name=concept_name,
                created_at=datetime.now()
            )

            created_entities = CreatedEntities(
                subject_created=True,
                chapter_created=True,
                concept_created=True
            )

            return {
                "enhanced_response": enhanced_response,
                "tracking_metadata": tracking_metadata,
                "created_entities": created_entities
            }

        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            raise

    async def update_concept_progress(self, student_id: int, concept_id: int,
                                      correct_answers: int, total_questions: int,
                                      time_spent: Optional[int] = None):
        await self.progress_tracker.update_concept_progress(
            student_id, concept_id, correct_answers, total_questions
        )

        activity_data = {
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'accuracy': (correct_answers / total_questions) * 100,
            'time_spent': time_spent
        }

        await self.progress_tracker.record_learning_activity(
            student_id, concept_id, ActivityType.QUIZ_ATTEMPT.value,
            activity_data, score=(correct_answers / total_questions) * 100,
            time_spent=time_spent
        )

    async def record_weakness(self, student_id: int, concept_id: int,
                              weakness_type: str, error_pattern: Optional[str] = None,
                              severity: float = 0.3):
        await self.progress_tracker.record_weakness(
            student_id, concept_id, weakness_type, error_pattern
        )

    async def get_student_progress(self, student_id: int, subject_id: Optional[int] = None) -> Dict[str, Any]:
        async with self.db.pool.acquire() as conn:
            overall_query = """
                SELECT 
                    COUNT(*) as total_concepts,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_concepts,
                    COUNT(CASE WHEN status = 'needs_review' THEN 1 END) as weak_concepts,
                    AVG(mastery_score) as avg_mastery_score,
                    SUM(attempts_count) as total_attempts
                FROM student_concept_progress 
                WHERE student_id = $1
            """

            if subject_id:
                overall_query += " AND concept_id IN (SELECT concept_id FROM concepts c JOIN chapters ch ON c.chapter_id = ch.chapter_id WHERE ch.subject_id = $2)"
                overall_stats = await conn.fetchrow(overall_query, student_id, subject_id)
            else:
                overall_stats = await conn.fetchrow(overall_query, student_id)

            subject_query = """
                SELECT 
                    s.subject_id, s.subject_name,
                    COUNT(scp.concept_id) as total_concepts,
                    COUNT(CASE WHEN scp.status = 'mastered' THEN 1 END) as mastered_concepts,
                    AVG(scp.mastery_score) as avg_score,
                    MAX(scp.last_practiced) as last_activity
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE scp.student_id = $1
                GROUP BY s.subject_id, s.subject_name
                ORDER BY s.subject_name
            """


*** End Patch
