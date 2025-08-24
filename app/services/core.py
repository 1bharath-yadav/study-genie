"""Core service implementations with Google ID support."""
# progress_tracker/services core implementation
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math
from sqlalchemy import text

from app.database import (
    DatabaseManager,
    StudentManager,
    SubjectManager,
    ProgressTracker,
    RecommendationEngine,
    AnalyticsManager,
    StudySessionManager,
)

logger = logging.getLogger("progress_tracker_services")


class LearningProgressService:
    """Service for managing student learning progress and LLM integration"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.student_manager = StudentManager(db_manager)
        self.subject_manager = SubjectManager(db_manager)
        self.progress_tracker = ProgressTracker(db_manager)

    async def generate_and_save_recommendations(self, student_id: str):
        """
        Generate and save personalized learning recommendations for a student
        This runs as a background task after processing LLM responses
        """
        try:
            logger.info(
                f"Generating recommendations for student: {student_id}")

            # Get student's learning progress and performance data
            progress_data = await self.get_student_progress(student_id)

            if not progress_data:
                logger.warning(
                    f"No progress data found for student: {student_id}")
                return

            # Analyze student's performance patterns
            recommendations = await self._analyze_and_generate_recommendations(progress_data)

            # Save recommendations to database
            if recommendations:
                await self._save_recommendations(student_id, recommendations)
                logger.info(
                    f"Successfully generated {len(recommendations)} recommendations for student: {student_id}")
            else:
                logger.info(
                    f"No new recommendations generated for student: {student_id}")

        except Exception as e:
            logger.error(
                f"Error generating recommendations for student {student_id}: {e}")
            # Don't re-raise since this is a background task

    async def _analyze_and_generate_recommendations(self, progress_data: dict) -> List[dict]:
        """
        Analyze student progress and generate personalized recommendations
        """
        recommendations = []

        try:
            # Extract key metrics from progress data
            subjects = progress_data.get('subject_progress', [])
            overall_performance = progress_data.get('overall_progress', {})

            for subject in subjects:
                subject_name = subject.get('subject_name', '')
                avg_score = subject.get('avg_score', 0)

                # Generate recommendations based on performance
                if avg_score < 60:  # Struggling in this subject
                    recommendations.append({
                        'type': 'remedial_study',
                        'subject': subject_name,
                        'priority': 'high',
                        'message': f"Consider reviewing fundamentals in {subject_name}. Current average: {avg_score}%",
                        'suggested_actions': [
                            'Review basic concepts',
                            'Practice more flashcards',
                            'Take additional quizzes'
                        ]
                    })
                elif avg_score > 85:  # Excelling in this subject
                    recommendations.append({
                        'type': 'advanced_study',
                        'subject': subject_name,
                        'priority': 'medium',
                        'message': f"Great progress in {subject_name}! Consider advancing to more complex topics.",
                        'suggested_actions': [
                            'Explore advanced concepts',
                            'Try challenging problems',
                            'Help peers with this subject'
                        ]
                    })

            # Analyze individual concepts for targeted recommendations
            weak_concepts = [cp for cp in progress_data.get('concept_progress', [])
                             if cp.get('mastery_score', 0) < 70]

            if weak_concepts:
                # Focus on top 2 weak concepts
                for concept in weak_concepts[:2]:
                    recommendations.append({
                        'type': 'concept_focus',
                        'subject': 'General',  # You might want to get this from concept metadata
                        'concept': concept.get('concept_name', ''),
                        'priority': 'medium',
                        'message': f"Focus on improving {concept.get('concept_name', '')}",
                        'suggested_actions': [
                            'Review concept materials',
                            'Practice related problems',
                            'Create summary notes'
                        ]
                    })

            # General study pattern recommendations
            recent_activity = progress_data.get('recent_activity', [])
            if len(recent_activity) < 3:  # Low activity
                recommendations.append({
                    'type': 'study_consistency',
                    'priority': 'high',
                    'message': "Try to maintain regular study sessions for better retention",
                    'suggested_actions': [
                        'Set daily study goals',
                        'Create a study schedule',
                        'Use spaced repetition'
                    ]
                })

            return recommendations[:10]  # Limit to top 10 recommendations

        except Exception as e:
            logger.error(f"Error analyzing progress data: {e}")
            return []

    async def _save_recommendations(self, student_id: str, recommendations: List[dict]):
        """
        Save recommendations to the database
        """
        try:
            async with self.db.pool.acquire() as conn:
                # Clear old recommendations (optional - you might want to keep history)
                await conn.execute(
                    "DELETE FROM student_recommendations WHERE student_id = $1",
                    student_id
                )

                # Insert new recommendations
                for i, rec in enumerate(recommendations):
                    await conn.execute(
                        """
                        INSERT INTO student_recommendations 
                        (student_id, recommendation_type, subject, chapter, priority, 
                         message, suggested_actions, created_at, order_index)
                        VALUES 
                        ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        student_id,
                        rec.get('type', 'general'),
                        rec.get('subject', ''),
                        rec.get('chapter', ''),
                        rec.get('priority', 'medium'),
                        rec.get('message', ''),
                        json.dumps(rec.get('suggested_actions', [])),
                        datetime.utcnow(),
                        i
                    )

                logger.info(
                    f"Saved {len(recommendations)} recommendations for student: {student_id}")

        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")

    async def get_student_recommendations(self, student_id: str, limit: int = 10) -> List[dict]:
        """
        Get personalized recommendations for a student
        """
        try:
            async with self.db.pool.acquire() as conn:
                result = await conn.fetch(
                    """
                    SELECT recommendation_type, subject, chapter, priority, 
                           message, suggested_actions, created_at
                    FROM student_recommendations 
                    WHERE student_id = $1 
                    ORDER BY order_index ASC, created_at DESC
                    LIMIT $2
                    """,
                    student_id, limit
                )

                recommendations = []
                for row in result:
                    recommendations.append({
                        'type': row['recommendation_type'],
                        'subject': row['subject'],
                        'chapter': row['chapter'],
                        'priority': row['priority'],
                        'message': row['message'],
                        'suggested_actions': json.loads(row['suggested_actions']) if row['suggested_actions'] else [],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None
                    })

                return recommendations

        except Exception as e:
            logger.error(
                f"Error getting recommendations for student {student_id}: {e}")
            return []

    async def create_or_get_student(self, student_id: str, username: str, email: str, full_name: str) -> str:
        return await self.student_manager.create_or_get_student(student_id, username, email, full_name)

    async def update_learning_preferences(self, student_id: str, preferences: Dict[str, Any]):
        await self.student_manager.update_learning_preferences(student_id, preferences)

    async def process_llm_response(self, student_id: str, subject_name: str,
                                   chapter_name: str, concept_name: str,
                                   llm_response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
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
                'llm_response': llm_response,
                'user_query': user_query,
                'timestamp': datetime.now().isoformat(),
                'response_type': 'structured_content'
            }

            await self.progress_tracker.record_learning_activity(
                student_id, concept_id, "content_study",
                activity_data, time_spent=None
            )

            await self._analyze_and_update_progress(student_id, concept_id, llm_response)

            enhanced_response = llm_response.copy()

            tracking_metadata = {
                'student_id': student_id,
                'subject_id': subject_id,
                'chapter_id': chapter_id,
                'concept_id': concept_id,
                'subject_name': subject_name,
                'chapter_name': chapter_name,
                'concept_name': concept_name,
                'created_at': datetime.now()
            }

            created_entities = {
                'subject_created': True,
                'chapter_created': True,
                'concept_created': True
            }

            return {
                "enhanced_response": enhanced_response,
                "tracking_metadata": tracking_metadata,
                "created_entities": created_entities
            }

        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            raise

    async def update_concept_progress(self, student_id: str, concept_id: int,
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
            student_id, concept_id, "quiz_attempt",
            activity_data, score=(correct_answers / total_questions) * 100,
            time_spent=time_spent
        )

    async def record_weakness(self, student_id: str, concept_id: int,
                              weakness_type: str, error_pattern: Optional[str] = None,
                              severity: float = 0.3):
        await self.progress_tracker.record_weakness(
            student_id, concept_id, weakness_type, error_pattern
        )

    async def get_student_progress(self, student_id: str, subject_id: Optional[int] = None) -> Dict[str, Any]:
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
                'concept_progress': [dict(row) for row in concept_progress],
                'recent_activity': [dict(row) for row in recent_activity],
                'total_concepts': overall_stats['total_concepts'] if overall_stats else 0,
                'mastered_concepts': overall_stats['mastered_concepts'] if overall_stats else 0,
                'weak_concepts': overall_stats['weak_concepts'] if overall_stats else 0
            }

    async def _ensure_student_enrollment(self, student_id: str, subject_id: int):
        """Ensure student is enrolled in a subject (creates enrollment if missing)."""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO student_subjects (student_id, subject_id) VALUES ($1, $2) ON CONFLICT (student_id, subject_id) DO NOTHING""",
                student_id, subject_id
            )

    async def _analyze_and_update_progress(self, student_id: str, concept_id: int,
                                           llm_response: Dict[str, Any]):
        """Analyze LLM response to derive engagement score and update concept progress."""
        engagement_score = 50

        if llm_response.get('flashcards'):
            engagement_score += len(llm_response['flashcards']) * 5
        if llm_response.get('quiz'):
            engagement_score += len(llm_response['quiz']) * 10
        if llm_response.get('summary'):
            engagement_score += min(
                len(llm_response['summary'].split()) // 10, 20)
        if llm_response.get('learning_objectives'):
            engagement_score += len(llm_response['learning_objectives']) * 3

        engagement_score = min(engagement_score, 100)

        # Treat as percentage out of 100
        await self.progress_tracker.update_concept_progress(
            student_id, concept_id, correct_answers=int(engagement_score), total_questions=100
        )


class RecommendationService:
    """Service for generating and managing personalized recommendations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.recommendation_engine = RecommendationEngine(db_manager)

    async def get_recommendations(self, student_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get student recommendations"""
        async with self.db.pool.acquire() as conn:
            query = """
                SELECT 
                    r.recommendation_id, r.recommendation_type, r.title, r.description,
                    r.priority_score, r.concept_id, r.is_active, r.is_completed,
                    r.created_at, r.expires_at,
                    c.concept_name, ch.chapter_name, s.subject_name
                FROM recommendations r
                LEFT JOIN concepts c ON r.concept_id = c.concept_id
                LEFT JOIN chapters ch ON c.chapter_id = ch.chapter_id
                LEFT JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE r.student_id = $1
            """

            params = [student_id]
            if active_only:
                query += " AND r.is_active = TRUE AND r.is_completed = FALSE"
                query += " AND (r.expires_at IS NULL OR r.expires_at > CURRENT_TIMESTAMP)"

            query += " ORDER BY r.priority_score DESC, r.created_at DESC"

            recommendations = await conn.fetch(query, *params)
            return [dict(rec) for rec in recommendations]

    async def generate_recommendations(self, student_id: str, force_regenerate: bool = False) -> int:
        """Generate new personalized recommendations"""
        if not force_regenerate:
            # Check if recent recommendations exist
            async with self.db.pool.acquire() as conn:
                recent_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM recommendations 
                    WHERE student_id = $1 AND is_active = TRUE 
                    AND created_at > CURRENT_TIMESTAMP - INTERVAL '1 day'
                """, student_id)

                if recent_count > 0:
                    return recent_count

        # Generate new recommendations
        recommendations = await self.recommendation_engine.generate_personalized_recommendations(student_id)
        await self.recommendation_engine.save_recommendations(student_id, recommendations)

        return len(recommendations)

    async def complete_recommendation(self, recommendation_id: int):
        """Mark recommendation as completed"""
        async with self.db.pool.acquire() as conn:
            await conn.execute("""
                UPDATE recommendations 
                SET is_completed = TRUE, updated_at = CURRENT_TIMESTAMP 
                WHERE recommendation_id = $1
            """, recommendation_id)
