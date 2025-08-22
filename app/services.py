# progress_tracker/services.py
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math

from .database import (
    DatabaseManager, StudentManager, SubjectManager,
    ProgressTracker, RecommendationEngine, AnalyticsManager, StudySessionManager
)
from .models import *

logger = logging.getLogger("progress_tracker_services")


class LearningProgressService:
    """Service for managing student learning progress and LLM integration"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.student_manager = StudentManager(db_manager)
        self.subject_manager = SubjectManager(db_manager)
        self.progress_tracker = ProgressTracker(db_manager)

    async def create_or_get_student(self, username: str, email: str, full_name: str) -> int:
        """Create a new student or get existing one"""
        return await self.student_manager.create_or_get_student(username, email, full_name)

    async def update_learning_preferences(self, student_id: int, preferences: Dict[str, Any]):
        """Update student learning preferences"""
        await self.student_manager.update_learning_preferences(student_id, preferences)

    async def process_llm_response(self, student_id: int, subject_name: str,
                                   chapter_name: str, concept_name: str,
                                   llm_response: LLMResponseContent, user_query: str) -> Dict[str, Any]:
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
                'llm_response': llm_response.dict(),
                'user_query': user_query,
                'timestamp': datetime.now().isoformat(),
                'response_type': 'structured_content'
            }

            await self.progress_tracker.record_learning_activity(
                student_id, concept_id, ActivityType.CONTENT_STUDY.value,
                activity_data, time_spent=None
            )

            # Analyze response for automatic progress updates
            await self._analyze_and_update_progress(student_id, concept_id, llm_response)

            # Prepare enhanced response
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
                subject_created=True,  # For simplicity, assuming creation
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
            student_id, concept_id, ActivityType.QUIZ_ATTEMPT.value,
            activity_data, score=(correct_answers / total_questions) * 100,
            time_spent=time_spent
        )

    async def record_weakness(self, student_id: int, concept_id: int,
                              weakness_type: str, error_pattern: Optional[str] = None,
                              severity: float = 0.3):
        """Record a student weakness"""
        await self.progress_tracker.record_weakness(
            student_id, concept_id, weakness_type, error_pattern
        )

    async def get_student_progress(self, student_id: int, subject_id: Optional[int] = None) -> Dict[str, Any]:
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

    async def process_quiz_results(self, student_id: int, quiz_results: QuizResultsBatch) -> int:
        """Process batch quiz results and update progress"""
        processed_count = 0

        # Group results by concept
        concept_results = {}
        for result in quiz_results.quiz_results:
            if result.concept_id not in concept_results:
                concept_results[result.concept_id] = {'correct': 0, 'total': 0}

            concept_results[result.concept_id]['total'] += 1
            if result.is_correct:
                concept_results[result.concept_id]['correct'] += 1

        # Update progress for each concept
        for concept_id, results in concept_results.items():
            await self.update_concept_progress(
                student_id, concept_id,
                results['correct'], results['total']
            )
            processed_count += results['total']

        return processed_count

    async def process_flashcard_session(self, student_id: int, session: FlashcardSessionResults):
        """Process flashcard session results"""
        # Group by concept
        concept_performance = {}
        for result in session.flashcard_results:
            if result.concept_id not in concept_performance:
                concept_performance[result.concept_id] = []
            concept_performance[result.concept_id].append(
                result.confidence_level)

        # Update progress based on confidence levels
        for concept_id, confidence_levels in concept_performance.items():
            avg_confidence = sum(confidence_levels) / len(confidence_levels)

            # Convert confidence to score (1-5 scale to 0-100)
            score = ((avg_confidence - 1) / 4) * 100

            # Record as practice activity
            activity_data = {
                'session_type': 'flashcard_practice',
                'cards_reviewed': len(confidence_levels),
                'average_confidence': avg_confidence,
                'session_time': session.total_session_time
            }

            await self.progress_tracker.record_learning_activity(
                student_id, concept_id, ActivityType.FLASHCARD_PRACTICE.value,
                activity_data, score=score, time_spent=session.total_session_time
            )

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

                chapter_info = ChapterInfo(
                    chapter_id=chapter['chapter_id'],
                    chapter_name=chapter['chapter_name'],
                    chapter_order=chapter['chapter_order'],
                    description=chapter['description'],
                    concepts=[ConceptInfo(**dict(concept))
                              for concept in concepts],
                    concept_count=len(concepts)
                )
                chapter_data.append(chapter_info)

            return {
                'subject_info': SubjectInfo(**dict(subject),
                                            chapter_count=len(chapters),
                                            concept_count=sum(ch.concept_count for ch in chapter_data)),
                'chapters': chapter_data
            }

    async def enroll_student_in_subject(self, student_id: int, subject_id: int):
        """Enroll a student in a subject"""
        await self._ensure_student_enrollment(student_id, subject_id)

    async def get_recent_activity(self, student_id: int, limit: int = 10) -> List[Dict[str, Any]]:
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
    async def _ensure_student_enrollment(self, student_id: int, subject_id: int):
        """Ensure student is enrolled in subject"""
        async with self.db.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO student_subjects (student_id, subject_id) 
                VALUES ($1, $2) ON CONFLICT (student_id, subject_id) DO NOTHING
            """, student_id, subject_id)

    async def _analyze_and_update_progress(self, student_id: int, concept_id: int,
                                           llm_response: LLMResponseContent):
        """Analyze LLM response and update progress accordingly"""
        # Simple heuristic: if content is generated, student is engaging with concept
        engagement_score = 50  # Base engagement

        # Increase score based on content richness
        if llm_response.flashcards:
            engagement_score += len(llm_response.flashcards) * 5
        if llm_response.quiz:
            engagement_score += len(llm_response.quiz) * 10
        if llm_response.summary:
            engagement_score += min(len(llm_response.summary.split()) // 10, 20)
        if llm_response.learning_objectives:
            engagement_score += len(llm_response.learning_objectives) * 3

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

    async def get_recommendations(self, student_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
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

    async def generate_recommendations(self, student_id: int, force_regenerate: bool = False) -> int:
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


class AnalyticsService:
    """Service for generating analytics and insights"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.analytics_manager = AnalyticsManager(db_manager)

    async def get_student_analytics(self, student_id: int, time_period: str = "30d") -> Dict[str, Any]:
        """Get comprehensive student analytics"""
        base_analytics = await self.analytics_manager.get_student_analytics(student_id)

        # Add time-based analytics
        time_filter = self._get_time_filter(time_period)

        async with self.db.pool.acquire() as conn:
            # Weekly progress data
            weekly_progress = await conn.fetch("""
                SELECT 
                    DATE_TRUNC('week', la.completed_at) as week,
                    COUNT(*) as activities,
                    AVG(la.score) as avg_score,
                    COUNT(DISTINCT la.concept_id) as concepts_practiced
                FROM learning_activities la
                WHERE la.student_id = $1 AND la.completed_at > $2
                GROUP BY DATE_TRUNC('week', la.completed_at)
                ORDER BY week DESC
                LIMIT 12
            """, student_id, time_filter)

            # Learning velocity (concepts mastered per week)
            velocity = await conn.fetchval("""
                SELECT COUNT(*)::float / GREATEST(EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - MIN(first_learned))) / 7, 1)
                FROM student_concept_progress 
                WHERE student_id = $1 AND status = 'mastered' AND first_learned > $2
            """, student_id, time_filter)

            # Focus areas (most practiced concepts)
            focus_areas = await conn.fetch("""
                SELECT c.concept_name, COUNT(*) as practice_count
                FROM learning_activities la
                JOIN concepts c ON la.concept_id = c.concept_id
                WHERE la.student_id = $1 AND la.completed_at > $2
                GROUP BY c.concept_id, c.concept_name
                ORDER BY practice_count DESC
                LIMIT 5
            """, student_id, time_filter)

        # Enhance base analytics
        enhanced_analytics = base_analytics.copy()
        enhanced_analytics.update({
            'weekly_progress': [dict(row) for row in weekly_progress],
            'learning_velocity': velocity or 0.0,
            'focus_areas': [row['concept_name'] for row in focus_areas],
            'achievements': await self._get_achievements(student_id),
            'time_period': time_period
        })

        return enhanced_analytics

    async def get_weakness_analysis(self, student_id: int, resolved: bool = False) -> List[Dict[str, Any]]:
        """Get detailed weakness analysis"""
        async with self.db.pool.acquire() as conn:
            weaknesses = await conn.fetch("""
                SELECT 
                    sw.weakness_id, sw.weakness_type, sw.error_pattern,
                    sw.frequency_count, sw.severity_score, sw.last_occurrence, sw.is_resolved,
                    c.concept_name, ch.chapter_name, s.subject_name
                FROM student_weaknesses sw
                JOIN concepts c ON sw.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE sw.student_id = $1 AND sw.is_resolved = $2
                ORDER BY sw.severity_score DESC, sw.frequency_count DESC
            """, student_id, resolved)

            # Add recommended actions for each weakness
            enhanced_weaknesses = []
            for weakness in weaknesses:
                weakness_dict = dict(weakness)
                weakness_dict['recommended_actions'] = await self._get_weakness_recommendations(
                    weakness['weakness_type'], weakness['severity_score']
                )
                enhanced_weaknesses.append(weakness_dict)

            return enhanced_weaknesses

    async def generate_learning_path(self, student_id: int, subject_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate suggested learning path"""
        async with self.db.pool.acquire() as conn:
            # Get concepts with prerequisites analysis
            concepts_query = """
                SELECT 
                    c.concept_id, c.concept_name, c.difficulty_level, c.concept_order,
                    ch.chapter_name, s.subject_name,
                    scp.status, scp.mastery_score,
                    CASE 
                        WHEN scp.status IS NULL THEN 0
                        WHEN scp.status = 'mastered' THEN 100
                        ELSE scp.mastery_score 
                    END as current_score
                FROM concepts c
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                LEFT JOIN student_concept_progress scp ON c.concept_id = scp.concept_id AND scp.student_id = $1
            """

            params = [student_id]
            if subject_id:
                concepts_query += " WHERE s.subject_id = $2"
                params.append(subject_id)

            concepts_query += " ORDER BY ch.chapter_id, c.concept_order"
            concepts = await conn.fetch(concepts_query, *params)

            # Generate learning path steps
            path_steps = []
            for idx, concept in enumerate(concepts):
                if concept['current_score'] < 80:  # Not mastered
                    estimated_time = self._estimate_learning_time(
                        concept['difficulty_level'], concept['current_score']
                    )

                    step = LearningPathStep(
                        step_order=len(path_steps) + 1,
                        concept_id=concept['concept_id'],
                        concept_name=concept['concept_name'],
                        chapter_name=concept['chapter_name'],
                        estimated_time=estimated_time,
                        prerequisites_met=self._check_prerequisites(
                            concepts[:idx], 70),
                        difficulty_level=DifficultyLevel(
                            concept['difficulty_level']),
                        recommended_activities=self._get_recommended_activities(
                            concept['difficulty_level'])
                    )
                    path_steps.append(step)

            # Calculate completion percentage
            total_concepts = len(concepts)
            mastered_concepts = sum(
                1 for c in concepts if c['current_score'] >= 80)
            completion_percentage = (
                mastered_concepts / total_concepts) * 100 if total_concepts > 0 else 0

            return {
                'student_id': student_id,
                'subject_name': concepts[0]['subject_name'] if concepts else None,
                'path_steps': path_steps[:20],  # Limit to next 20 steps
                'total_estimated_time': sum(step.estimated_time for step in path_steps[:20]),
                'completion_percentage': completion_percentage,
                'next_milestone': self._get_next_milestone(path_steps)
            }

    # Private helper methods
    def _get_time_filter(self, time_period: str) -> datetime:
        """Get datetime filter for time period"""
        now = datetime.now()
        if time_period == "7d":
            return now - timedelta(days=7)
        elif time_period == "30d":
            return now - timedelta(days=30)
        elif time_period == "90d":
            return now - timedelta(days=90)
        else:  # "all"
            return datetime(2020, 1, 1)

    async def _get_achievements(self, student_id: int) -> List[str]:
        """Get student achievements"""
        achievements = []

        async with self.db.pool.acquire() as conn:
            # Check various achievement conditions
            mastered_count = await conn.fetchval("""
                SELECT COUNT(*) FROM student_concept_progress 
                WHERE student_id = $1 AND status = 'mastered'
            """, student_id)

            if mastered_count >= 10:
                achievements.append("Concept Master - Mastered 10+ concepts")
            if mastered_count >= 50:
                achievements.append(
                    "Learning Champion - Mastered 50+ concepts")

            # Check for consistency
            recent_activity_days = await conn.fetchval("""
                SELECT COUNT(DISTINCT DATE(completed_at)) 
                FROM learning_activities 
                WHERE student_id = $1 AND completed_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """, student_id)

            if recent_activity_days >= 5:
                achievements.append(
                    "Consistent Learner - Active 5+ days this week")

        return achievements

    async def _get_weakness_recommendations(self, weakness_type: str, severity: float) -> List[str]:
        """Get recommendations for addressing specific weakness"""
        recommendations = [
            "Practice more exercises on this concept",
            "Review foundational material",
            "Try different learning approaches"
        ]

        if severity > 0.7:
            recommendations.insert(
                0, "Focus heavily on this area - schedule daily practice")

        return recommendations

    def _estimate_learning_time(self, difficulty: str, current_score: float) -> int:
        """Estimate time needed to master concept (in minutes)"""
        base_time = {
            "Easy": 30,
            "Medium": 60,
            "Hard": 120
        }

        time_needed = base_time.get(difficulty, 60)
        progress_factor = (100 - current_score) / 100

        return int(time_needed * progress_factor)

    def _check_prerequisites(self, previous_concepts: List[Dict], threshold: float) -> bool:
        """Check if prerequisites are met"""
        if not previous_concepts:
            return True

        # Simple heuristic: 80% of previous concepts should be above threshold
        prerequisite_concepts = [
            c for c in previous_concepts if c['current_score'] is not None]
        if not prerequisite_concepts:
            return True

        met_prerequisites = sum(
            1 for c in prerequisite_concepts if c['current_score'] >= threshold)
        return (met_prerequisites / len(prerequisite_concepts)) >= 0.8

    def _get_recommended_activities(self, difficulty: str) -> List[str]:
        """Get recommended activities based on difficulty"""
        activities = {
            "Easy": [
                "Review flashcards",
                "Take practice quiz",
                "Read concept summary"
            ],
            "Medium": [
                "Complete practice exercises",
                "Take comprehensive quiz",
                "Create concept map",
                "Review with flashcards"
            ],
            "Hard": [
                "Work through detailed examples",
                "Take multiple practice quizzes",
                "Create detailed notes",
                "Discuss with peers or instructor",
                "Apply concept to real problems"
            ]
        }
        return activities.get(difficulty, activities["Medium"])

    def _get_next_milestone(self, path_steps: List[LearningPathStep]) -> Optional[str]:
        """Get next learning milestone"""
        if not path_steps:
            return None

        # Group by chapter to find next chapter completion
        chapters = {}
        for step in path_steps:
            if step.chapter_name not in chapters:
                chapters[step.chapter_name] = []
            chapters[step.chapter_name].append(step)

        # Find first chapter that can be completed
        for chapter, steps in chapters.items():
            if len(steps) <= 5:  # Manageable number of concepts
                return f"Complete {chapter} chapter"

        return f"Master {path_steps[0].concept_name}"


class StudySessionService:
    """Service for managing study sessions"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.session_manager = StudySessionManager(db_manager)
        self.subject_manager = SubjectManager(db_manager)

    async def start_session(self, student_id: int, subject_name: str,
                            session_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Start a new study session"""
        # Get or create subject
        subject_id = await self.subject_manager.create_or_get_subject(subject_name)

        # Start session
        session_id = await self.session_manager.start_study_session(student_id, subject_id)

        # Record session start activity
        async with self.db.pool.acquire() as conn:
            await conn.execute("""
                UPDATE study_sessions 
                SET session_data = $1 
                WHERE session_id = $2
            """, json.dumps(session_metadata or {}), session_id)

        return session_id

    async def end_session(self, session_id: int, results: StudySessionEnd):
        """End a study session with results"""
        await self.session_manager.end_study_session(
            session_id,
            results.session_data,
            results.total_questions,
            results.correct_answers,
            results.duration
        )

        # Update concept progress for concepts covered
        if results.concepts_covered:
            accuracy = results.correct_answers / \
                results.total_questions if results.total_questions > 0 else 0

            async with self.db.pool.acquire() as conn:
                # Get student_id from session
                session_info = await conn.fetchrow("""
                    SELECT student_id FROM study_sessions WHERE session_id = $1
                """, session_id)

                if session_info:
                    student_id = session_info['student_id']

                    # Distribute questions across concepts
                    questions_per_concept = results.total_questions // len(
                        results.concepts_covered)
                    correct_per_concept = int(
                        results.correct_answers * accuracy / len(results.concepts_covered))

                    for concept_id in results.concepts_covered:
                        # Record activity for each concept
                        activity_data = {
                            'session_id': session_id,
                            'questions': questions_per_concept,
                            'correct': correct_per_concept,
                            'session_duration': results.duration
                        }

                        progress_tracker = ProgressTracker(self.db)
                        await progress_tracker.record_learning_activity(
                            student_id, concept_id, ActivityType.QUIZ_ATTEMPT.value,
                            activity_data, score=accuracy * 100,
                            time_spent=results.duration // len(
                                results.concepts_covered)
                        )

    async def get_session_history(self, student_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get study session history for a student"""
        async with self.db.pool.acquire() as conn:
            sessions = await conn.fetch("""
                SELECT 
                    ss.session_id, s.subject_name, ss.total_questions, ss.correct_answers,
                    CASE 
                        WHEN ss.total_questions > 0 THEN (ss.correct_answers::float / ss.total_questions * 100)
                        ELSE 0 
                    END as accuracy,
                    ss.session_duration as duration,
                    COALESCE(
                        (SELECT COUNT(DISTINCT concept_id) 
                         FROM learning_activities la 
                         WHERE la.student_id = ss.student_id 
                         AND la.completed_at BETWEEN ss.started_at AND ss.completed_at),
                        0
                    ) as concepts_covered,
                    ss.started_at, ss.completed_at
                FROM study_sessions ss
                JOIN subjects s ON ss.subject_id = s.subject_id
                WHERE ss.student_id = $1
                ORDER BY ss.started_at DESC
                LIMIT $2
            """, student_id, limit)

            return [dict(session) for session in sessions]

    async def get_session_analytics(self, student_id: int, time_period: str = "30d") -> Dict[str, Any]:
        """Get session-based analytics"""
        time_filter = self._get_time_filter(time_period)

        async with self.db.pool.acquire() as conn:
            # Overall session stats
            session_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_sessions,
                    AVG(session_duration) as avg_duration,
                    SUM(session_duration) as total_study_time,
                    AVG(CASE WHEN total_questions > 0 THEN correct_answers::float / total_questions * 100 ELSE 0 END) as avg_accuracy,
                    MAX(completed_at) as last_session
                FROM study_sessions
                WHERE student_id = $1 AND started_at > $2 AND completed_at IS NOT NULL
            """, student_id, time_filter)

            # Subject-wise session performance
            subject_performance = await conn.fetch("""
                SELECT 
                    s.subject_name,
                    COUNT(*) as session_count,
                    AVG(ss.session_duration) as avg_duration,
                    AVG(CASE WHEN ss.total_questions > 0 THEN ss.correct_answers::float / ss.total_questions * 100 ELSE 0 END) as avg_accuracy
                FROM study_sessions ss
                JOIN subjects s ON ss.subject_id = s.subject_id
                WHERE ss.student_id = $1 AND ss.started_at > $2 AND ss.completed_at IS NOT NULL
                GROUP BY s.subject_id, s.subject_name
                ORDER BY session_count DESC
            """, student_id, time_filter)

            # Daily study pattern
            daily_pattern = await conn.fetch("""
                SELECT 
                    DATE(started_at) as study_date,
                    COUNT(*) as sessions,
                    SUM(session_duration) as total_time,
                    AVG(CASE WHEN total_questions > 0 THEN correct_answers::float / total_questions * 100 ELSE 0 END) as avg_accuracy
                FROM study_sessions
                WHERE student_id = $1 AND started_at > $2 AND completed_at IS NOT NULL
                GROUP BY DATE(started_at)
                ORDER BY study_date DESC
                LIMIT 30
            """, student_id, time_filter)

            return {
                'session_stats': dict(session_stats) if session_stats else {},
                'subject_performance': [dict(row) for row in subject_performance],
                'daily_pattern': [dict(row) for row in daily_pattern],
                'time_period': time_period
            }

    def _get_time_filter(self, time_period: str) -> datetime:
        """Get datetime filter for time period"""
        now = datetime.now()
        if time_period == "7d":
            return now - timedelta(days=7)
        elif time_period == "30d":
            return now - timedelta(days=30)
        elif time_period == "90d":
            return now - timedelta(days=90)
        else:  # "all"
            return datetime(2020, 1, 1)


# Utility Services
class ValidationService:
    """Service for validating data and business rules"""

    @staticmethod
    def validate_llm_response(response: LLMResponseContent) -> ValidationResult:
        """Validate LLM response structure and content"""
        errors = []
        warnings = []

        # Check flashcards
        for card_id, card in response.flashcards.items():
            if len(card.question) < 5:
                errors.append(f"Flashcard {card_id}: Question too short")
            if len(card.answer) < 3:
                errors.append(f"Flashcard {card_id}: Answer too short")

        # Check quiz questions
        for quiz_id, quiz in response.quiz.items():
            if len(quiz.options) < 2:
                errors.append(f"Quiz {quiz_id}: Need at least 2 options")
            if quiz.correct_answer not in quiz.options:
                errors.append(f"Quiz {quiz_id}: Correct answer not in options")
            if len(quiz.question) < 5:
                errors.append(f"Quiz {quiz_id}: Question too short")

        # Check summary
        if response.summary and len(response.summary) < 20:
            warnings.append("Summary seems too short")

        # Check learning objectives
        if not response.learning_objectives:
            warnings.append("No learning objectives provided")
        elif len(response.learning_objectives) > 10:
            warnings.append("Too many learning objectives (>10)")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    @staticmethod
    def validate_progress_update(update: ConceptProgressUpdate) -> ValidationResult:
        """Validate progress update data"""
        errors = []

        if update.correct_answers > update.total_questions:
            errors.append("Correct answers cannot exceed total questions")

        if update.time_spent is not None and update.time_spent < 0:
            errors.append("Time spent cannot be negative")

        if update.total_questions <= 0:
            errors.append("Total questions must be positive")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


class NotificationService:
    """Service for generating notifications and alerts"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def generate_study_reminders(self, student_id: int) -> List[Dict[str, Any]]:
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

    async def check_learning_milestones(self, student_id: int) -> List[Dict[str, Any]]:
        """Check for learning milestones and achievements"""
        milestones = []

        async with self.db.pool.acquire() as conn:
            # Check for recent achievements
            recent_mastery = await conn.fetchval("""
                SELECT COUNT(*) FROM student_concept_progress 
                WHERE student_id = $1 AND status = 'mastered' 
                AND first_learned > CURRENT_TIMESTAMP - INTERVAL '1 day'
            """, student_id)

            if recent_mastery > 0:
                milestones.append({
                    'type': 'achievement',
                    'title': 'Concepts Mastered!',
                    'message': f'Congratulations! You mastered {recent_mastery} new concept(s) today!',
                    'count': recent_mastery
                })

            # Check for consistency streaks
            active_days = await conn.fetchval("""
                SELECT COUNT(DISTINCT DATE(completed_at)) 
                FROM learning_activities 
                WHERE student_id = $1 
                AND completed_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """, student_id)

            if active_days >= 7:
                milestones.append({
                    'type': 'streak',
                    'title': 'Weekly Streak!',
                    'message': 'Amazing! You\'ve been active every day this week!',
                    'streak_days': active_days
                })

        return milestones


# Integration helper functions
async def process_rag_integration(student_id: int, subject_name: str, chapter_name: str,
                                  concept_name: str, rag_response: Dict[str, Any],
                                  db_manager: DatabaseManager) -> Dict[str, Any]:
    """
    Main integration function for RAG system
    This should be called from your main RAG pipeline
    """
    try:
        # Convert RAG response to our model format
        llm_response = LLMResponseContent(
            flashcards={k: FlashcardData(
                **v) for k, v in rag_response.get('flashcards', {}).items()},
            quiz={k: QuizData(**v)
                  for k, v in rag_response.get('quiz', {}).items()},
            summary=rag_response.get('summary', ''),
            learning_objectives=rag_response.get('learning_objectives', [])
        )

        # Process through our service
        service = LearningProgressService(db_manager)
        result = await service.process_llm_response(
            student_id, subject_name, chapter_name, concept_name,
            llm_response, rag_response.get('user_query', '')
        )

        # Generate background recommendations
        recommendation_service = RecommendationService(db_manager)
        await recommendation_service.generate_recommendations(student_id, force_regenerate=False)

        return result

    except Exception as e:
        logger.error(f"Error in RAG integration: {str(e)}")
        raise


# Background task functions for periodic maintenance
async def cleanup_old_data(db_manager: DatabaseManager):
    """Clean up old data periodically"""
    async with db_manager.pool.acquire() as conn:
        # Archive old learning activities (older than 6 months)
        await conn.execute("""
            DELETE FROM learning_activities 
            WHERE completed_at < CURRENT_TIMESTAMP - INTERVAL '6 months'
        """)

        # Expire old recommendations
        await conn.execute("""
            UPDATE recommendations 
            SET is_active = FALSE 
            WHERE expires_at < CURRENT_TIMESTAMP AND expires_at IS NOT NULL
        """)


async def update_recommendation_priorities(db_manager: DatabaseManager):
    """Update recommendation priorities based on recent activity"""
    async with db_manager.pool.acquire() as conn:
        # Increase priority for concepts with recent poor performance
        await conn.execute("""
            UPDATE recommendations 
            SET priority_score = LEAST(priority_score + 2, 10)
            WHERE concept_id IN (
                SELECT DISTINCT concept_id 
                FROM learning_activities 
                WHERE score < 60 AND completed_at > CURRENT_TIMESTAMP - INTERVAL '3 days'
            ) AND is_active = TRUE
        """)

# Export all services for easy import
__all__ = [
    'LearningProgressService',
    'RecommendationService',
    'AnalyticsService',
    'StudySessionService',
    'ValidationService',
    'NotificationService',
    'process_rag_integration',
    'cleanup_old_data',
    'update_recommendation_priorities'
]
