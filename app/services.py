# progress_tracker/services.py
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math

from .db.models import (
    DatabaseManager, StudentManager, SubjectManager,
    ProgressTracker, RecommendationEngine, AnalyticsManager, StudySessionManager
)
from .models import *, StudentUpdate

logger = logging.getLogger("progress_tracker_services")


class LearningProgressService:
    """Service for managing student learning progress and LLM integration"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.student_manager = StudentManager(db_manager)
        self.subject_manager = SubjectManager(db_manager)
        self.progress_tracker = ProgressTracker(db_manager)

    async def create_or_get_student(self, student_id: str, username: str, email: str, full_name: str) -> str:
        """Create a new student or get existing one"""
        return await self.student_manager.create_or_get_student(student_id, username, email, full_name)

    async def update_learning_preferences(self, student_id: str, preferences: Dict[str, Any]):
        """Update student learning preferences"""
        await self.student_manager.update_learning_preferences(student_id, preferences)

    async def get_student_by_id(self, student_id: str) -> Optional[StudentResponse]:
        """Get student details by ID."""
        student_data = await self.student_manager.get_student_by_id(student_id)
        if student_data:
            return StudentResponse(
                student_id=student_data['student_id'],
                username=student_data['username'],
                email=student_data['email'],
                full_name=student_data['full_name'],
                message="Student retrieved successfully"
            )
        return None

    async def get_all_students(self) -> List[StudentResponse]:
        """Get all students."""
        students_data = await self.student_manager.get_all_students()
        return [
            StudentResponse(
                student_id=s['student_id'],
                username=s['username'],
                email=s['email'],
                full_name=s['full_name'],
                message="Student retrieved successfully"
            ) for s in students_data
        ]

    async def update_student(self, student_id: str, student_update: StudentUpdate) -> Optional[StudentResponse]:
        """Update student details."""
        updated_data = await self.student_manager.update_student(
            student_id,
            student_update.full_name,
            student_update.learning_preferences
        )
        if updated_data:
            return StudentResponse(
                student_id=updated_data['student_id'],
                username=updated_data['username'],
                email=updated_data['email'],
                full_name=updated_data['full_name'],
                message="Student updated successfully"
            )
        return None

    async def delete_student(self, student_id: str) -> bool:
        """Delete a student by ID."""
        return await self.student_manager.delete_student(student_id)

    async def process_llm_response(self, student_id: str, subject_name: str,
                                   chapter_name: str, concept_name: str,
                                   llm_response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        Main function to process LLM response and update student tracking
        This integrates with your RAG system output
        """
        try:
            # Create or get subject hierarchy
            subject_id = await self.subject_manager.create_or_get_subject(
                subject_name, f"Auto-created from LLM response"
            )
            chapter_id = await self.subject_manager.create_or_get_chapter(
                subject_id, chapter_name, f"Auto-created from LLM response"
            )
            concept_id = await self.subject_manager.create_or_get_concept(
                chapter_id, concept_name, "Medium", f"Auto-created from LLM response"
            )

            # Enroll student in subject if not already enrolled
            await self._ensure_student_enrollment(student_id, subject_id)

            # Record the study activity
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

            # Analyze response for automatic progress updates
            await self._analyze_and_update_progress(student_id, concept_id, llm_response)

            # Prepare enhanced response
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

    async def generate_and_save_recommendations(self, student_id: str):
        """Generate and save personalized recommendations for a student"""
        try:
            recommendation_engine = RecommendationEngine(self.db)
            recommendations = await recommendation_engine.generate_personalized_recommendations(student_id)
            await recommendation_engine.save_recommendations(student_id, recommendations)
            logger.info(
                f"Generated {len(recommendations)} recommendations for student {student_id}")
        except Exception as e:
            logger.error(
                f"Error generating recommendations for student {student_id}: {str(e)}")

    async def update_concept_progress(self, student_id: str, concept_id: int,
                                      correct_answers: int, total_questions: int,
                                      time_spent: Optional[int] = None):
        """Update student progress on a concept"""
        await self.progress_tracker.update_concept_progress(
            student_id, concept_id, correct_answers, total_questions
        )

        # Record the activity
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
        """Record a student weakness"""
        await self.progress_tracker.record_weakness(
            student_id, concept_id, weakness_type, error_pattern
        )

    async def get_student_progress(self, student_id: str, subject_id: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive student progress data"""
        async with self.db.pool.acquire() as conn:
            # Overall progress
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

            # Subject-wise progress
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
            # Add progress_percentage to each subject
            subject_progress_list = []
            for row in subject_progress:
                total = row['total_concepts'] or 0
                mastered = row['mastered_concepts'] or 0
                progress_percentage = (mastered / total) * \
                    100 if total > 0 else 0
                subject_dict = dict(row)
                subject_dict['progress_percentage'] = progress_percentage
                subject_progress_list.append(subject_dict)

            # Detailed concept progress
            concept_query = """
                SELECT 
                    scp.concept_id, c.concept_name, scp.status, scp.mastery_score,
                    scp.attempts_count, scp.correct_answers, scp.total_questions,
                    scp.last_practiced, scp.first_learned
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                WHERE scp.student_id = $1
                ORDER BY scp.last_practiced DESC NULLS LAST
                LIMIT 50
            """

            concept_progress = await conn.fetch(concept_query, student_id)

            # Recent activity
            activity_query = """
                SELECT 
                    la.activity_type, la.score, la.completed_at,
                    c.concept_name, ch.chapter_name, s.subject_name
                FROM learning_activities la
                JOIN concepts c ON la.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE la.student_id = $1
                ORDER BY la.completed_at DESC
                LIMIT 20
            """

            recent_activity = await conn.fetch(activity_query, student_id)

            return {
                'student_id': student_id,
                'overall_progress': dict(overall_stats) if overall_stats else {},
                'subject_progress': subject_progress_list,
                'concept_progress': [dict(row) for row in concept_progress],
                'recent_activity': [dict(row) for row in recent_activity],
                'total_concepts': overall_stats['total_concepts'] if overall_stats else 0,
                'mastered_concepts': overall_stats['mastered_concepts'] if overall_stats else 0,
                'weak_concepts': overall_stats['weak_concepts'] if overall_stats else 0
            }

    async def process_quiz_results(self, student_id: str, quiz_results: Dict[str, Any]) -> int:
        """Process batch quiz results and update progress"""
        processed_count = 0

        # Group results by concept
        concept_results = {}
        for result in quiz_results.get('quiz_results', []):
            concept_id = result.get('concept_id')
            if concept_id not in concept_results:
                concept_results[concept_id] = {'correct': 0, 'total': 0}

            concept_results[concept_id]['total'] += 1
            if result.get('is_correct', False):
                concept_results[concept_id]['correct'] += 1

        # Update progress for each concept
        for concept_id, results in concept_results.items():
            await self.update_concept_progress(
                student_id, concept_id,
                results['correct'], results['total']
            )
            processed_count += results['total']

        return processed_count

    async def get_all_subjects(self) -> List[Dict[str, Any]]:
        """Get all subjects with basic info"""
        async with self.db.pool.acquire() as conn:
            subjects = await conn.fetch("""
                SELECT 
                    s.subject_id, s.subject_name, s.description,
                    COUNT(DISTINCT ch.chapter_id) as chapter_count,
                    COUNT(c.concept_id) as concept_count
                FROM subjects s
                LEFT JOIN chapters ch ON s.subject_id = ch.subject_id
                LEFT JOIN concepts c ON ch.chapter_id = c.chapter_id
                GROUP BY s.subject_id, s.subject_name, s.description
                ORDER BY s.subject_name
            """)

            return [dict(subject) for subject in subjects]

    async def get_subject_structure(self, subject_id: int) -> Dict[str, Any]:
        """Get complete subject structure"""
        async with self.db.pool.acquire() as conn:
            # Get subject info
            subject = await conn.fetchrow("""
                SELECT subject_id, subject_name, description FROM subjects WHERE subject_id = $1
            """, subject_id)

            if not subject:
                raise ValueError(f"Subject {subject_id} not found")

            # Get chapters with concepts
            chapters = await conn.fetch("""
                SELECT 
                    ch.chapter_id, ch.chapter_name, ch.chapter_order, ch.description,
                    COUNT(c.concept_id) as concept_count
                FROM chapters ch
                LEFT JOIN concepts c ON ch.chapter_id = c.chapter_id
                WHERE ch.subject_id = $1
                GROUP BY ch.chapter_id, ch.chapter_name, ch.chapter_order, ch.description
                ORDER BY ch.chapter_order
            """, subject_id)

            # Get concepts for each chapter
            chapter_data = []
            for chapter in chapters:
                concepts = await conn.fetch("""
                    SELECT concept_id, concept_name, concept_order, difficulty_level, description
                    FROM concepts 
                    WHERE chapter_id = $1 
                    ORDER BY concept_order
                """, chapter['chapter_id'])

                chapter_info = {
                    'chapter_id': chapter['chapter_id'],
                    'chapter_name': chapter['chapter_name'],
                    'chapter_order': chapter['chapter_order'],
                    'description': chapter['description'],
                    'concepts': [dict(concept) for concept in concepts],
                    'concept_count': len(concepts)
                }
                chapter_data.append(chapter_info)

            return {
                'subject_info': dict(subject),
                'chapters': chapter_data
            }

    async def enroll_student_in_subject(self, student_id: str, subject_id: int):
        """Enroll a student in a subject"""
        await self._ensure_student_enrollment(student_id, subject_id)

    async def get_recent_activity(self, student_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent student activity"""
        async with self.db.pool.acquire() as conn:
            activities = await conn.fetch("""
                SELECT 
                    la.activity_type, la.score, la.completed_at, la.time_spent,
                    c.concept_name, ch.chapter_name, s.subject_name
                FROM learning_activities la
                JOIN concepts c ON la.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE la.student_id = $1
                ORDER BY la.completed_at DESC
                LIMIT $2
            """, student_id, limit)

            return [dict(activity) for activity in activities]

    # Private helper methods
    async def _ensure_student_enrollment(self, student_id: str, subject_id: int):
        """Ensure student is enrolled in subject"""
        async with self.db.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO student_subjects (student_id, subject_id) 
                VALUES ($1, $2) ON CONFLICT (student_id, subject_id) DO NOTHING
            """, student_id, subject_id)

    async def _analyze_and_update_progress(self, student_id: str, concept_id: int,
                                           llm_response: Dict[str, Any]):
        """Analyze LLM response and update progress accordingly"""
        # Simple heuristic: if content is generated, student is engaging with concept
        engagement_score = 50  # Base engagement

        # Increase score based on content richness
        if llm_response.get('flashcards'):
            engagement_score += len(llm_response['flashcards']) * 5
        if llm_response.get('quiz'):
            engagement_score += len(llm_response['quiz']) * 10
        if llm_response.get('summary'):
            engagement_score += min(
                len(llm_response['summary'].split()) // 10, 20)
        if llm_response.get('learning_objectives'):
            engagement_score += len(llm_response['learning_objectives']) * 3

        # Cap at 100
        engagement_score = min(engagement_score, 100)

        # Update progress with estimated engagement
        await self.progress_tracker.update_concept_progress(
            student_id, concept_id,
            correct_answers=engagement_score,
            total_questions=100  # Treat as percentage
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


# Utility Services
class ValidationService:
    """Service for validating data and business rules"""

    @staticmethod
    def validate_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM response structure and content"""
        errors = []
        warnings = []

        # Check flashcards
        flashcards = response.get('flashcards', {})
        for card_id, card in flashcards.items():
            if len(card.get('question', '')) < 5:
                errors.append(f"Flashcard {card_id}: Question too short")
            if len(card.get('answer', '')) < 3:
                errors.append(f"Flashcard {card_id}: Answer too short")

        # Check quiz questions
        quiz = response.get('quiz', {})
        for quiz_id, q in quiz.items():
            options = q.get('options', [])
            if len(options) < 2:
                errors.append(f"Quiz {quiz_id}: Need at least 2 options")
            if q.get('correct_answer') not in options:
                errors.append(f"Quiz {quiz_id}: Correct answer not in options")
            if len(q.get('question', '')) < 5:
                errors.append(f"Quiz {quiz_id}: Question too short")

        # Check summary
        if response.get('summary') and len(response['summary']) < 20:
            warnings.append("Summary seems too short")

        # Check learning objectives
        learning_objectives = response.get('learning_objectives', [])
        if not learning_objectives:
            warnings.append("No learning objectives provided")
        elif len(learning_objectives) > 10:
            warnings.append("Too many learning objectives (>10)")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class NotificationService:
    """Service for generating notifications and alerts"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def generate_study_reminders(self, student_id: str) -> List[Dict[str, Any]]:
        """Generate study reminders based on progress and patterns"""
        reminders = []

        async with self.db.pool.acquire() as conn:
            # Find concepts that haven't been practiced recently
            stale_concepts = await conn.fetch("""
                SELECT c.concept_name, ch.chapter_name, s.subject_name, scp.last_practiced
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE scp.student_id = $1 
                    AND scp.status = 'mastered'
                    AND scp.last_practiced < CURRENT_TIMESTAMP - INTERVAL '7 days'
                ORDER BY scp.last_practiced ASC
                LIMIT 5
            """, student_id)

            for concept in stale_concepts:
                reminders.append({
                    'type': 'review_reminder',
                    'title': f"Review {concept['concept_name']}",
                    'message': f"It's been a while since you practiced {concept['concept_name']} in {concept['subject_name']}",
                    'priority': 'medium',
                    'concept_name': concept['concept_name']
                })

            # Find struggling areas
            struggling_concepts = await conn.fetch("""
                SELECT c.concept_name, ch.chapter_name, s.subject_name, scp.mastery_score
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE scp.student_id = $1 
                    AND scp.status = 'needs_review'
                    AND scp.attempts_count >= 3
                ORDER BY scp.mastery_score ASC
                LIMIT 3
            """, student_id)

            for concept in struggling_concepts:
                reminders.append({
                    'type': 'focus_reminder',
                    'title': f"Focus on {concept['concept_name']}",
                    'message': f"You might benefit from extra practice with {concept['concept_name']}",
                    'priority': 'high',
                    'concept_name': concept['concept_name']
                })

        return reminders


# Integration helper functions
async def process_rag_integration(student_id: str, subject_name: str, chapter_name: str,
                                  concept_name: str, rag_response: Dict[str, Any],
                                  db_manager: DatabaseManager) -> Dict[str, Any]:
    """
    Main integration function for RAG system
    This should be called from your main RAG pipeline
    """
    try:
        # Process through our service
        service = LearningProgressService(db_manager)
        result = await service.process_llm_response(
            student_id, subject_name, chapter_name, concept_name,
            rag_response, rag_response.get('user_query', '')
        )

        # Generate background recommendations
        recommendation_service = RecommendationService(db_manager)
        await recommendation_service.generate_recommendations(student_id, force_regenerate=False)

        return result

    except Exception as e:
        logger.error(f"Error in RAG integration: {str(e)}")
        raise


# Export all services for easy import
__all__ = [
    'LearningProgressService',
    'RecommendationService',
    'ValidationService',
    'NotificationService',
    'process_rag_integration'
]
