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

            subject_progress = await conn.fetch(subject_query, student_id)

            concept_query = """
                SELECT
                    scp.concept_id, c.concept_name, scp.status, scp.mastery_score,
                    scp.attempts_count, scp.correct_answers, scp.total_questions,
                    scp.last_practiced, scp.first_learned
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                WHERE scp.student_id = $1
            """

            if subject_id:
                concept_query += " AND c.chapter_id IN (SELECT chapter_id FROM chapters WHERE subject_id = $2)"
                concept_progress = await conn.fetch(concept_query, student_id, subject_id)
            else:
                concept_progress = await conn.fetch(concept_query, student_id)

            recent_activity_query = """
                SELECT activity_type, activity_data, completed_at
                FROM learning_activities
                WHERE student_id = $1
                ORDER BY completed_at DESC
                LIMIT 20
            """

            recent_activity = await conn.fetch(recent_activity_query, student_id)

            return {
                'student_id': student_id,
                'overall_progress': dict(overall_stats) if overall_stats else {},
                'subject_progress': [dict(row) for row in subject_progress],
                'concept_progress': [
                    ConceptProgress(
                        concept_id=row['concept_id'],
                        concept_name=row['concept_name'],
                        status=ConceptStatus(row['status']),
                        mastery_score=row['mastery_score'] or 0.0,
                        attempts_count=row['attempts_count'] or 0,
                        correct_answers=row['correct_answers'] or 0,
                        total_questions=row['total_questions'] or 0,
                        last_practiced=row['last_practiced'],
                        first_learned=row['first_learned']
                    ) for row in concept_progress
                ],
                'recent_activity': [dict(row) for row in recent_activity],
                'total_concepts': overall_stats['total_concepts'] if overall_stats else 0,
                'mastered_concepts': overall_stats['mastered_concepts'] if overall_stats else 0,
                'weak_concepts': overall_stats['weak_concepts'] if overall_stats else 0
            }

    async def _ensure_student_enrollment(self, student_id: int, subject_id: int):
        """Ensure student is enrolled in a subject (creates enrollment if missing)."""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO student_subjects (student_id, subject_id) VALUES ($1, $2) ON CONFLICT (student_id, subject_id) DO NOTHING""",
                student_id, subject_id
            )

    async def _analyze_and_update_progress(self, student_id: int, concept_id: int,
                                           llm_response: LLMResponseContent):
        """Analyze LLM response to derive engagement score and update concept progress."""
        engagement_score = 50

        if getattr(llm_response, 'flashcards', None):
            engagement_score += len(llm_response.flashcards) * 5
        if getattr(llm_response, 'quiz', None):
            engagement_score += len(llm_response.quiz) * 10
        if getattr(llm_response, 'summary', None):
            engagement_score += min(len(llm_response.summary.split()) // 10, 20)
        if getattr(llm_response, 'learning_objectives', None):
            engagement_score += len(llm_response.learning_objectives) * 3

        engagement_score = min(engagement_score, 100)

        # Treat as percentage out of 100
        await self.progress_tracker.update_concept_progress(
            student_id, concept_id, correct_answers=int(engagement_score), total_questions=100
        )
