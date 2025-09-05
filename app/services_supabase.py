# services_supabase.py
"""
Supabase-powered services for StudyGenie
This replaces the PostgreSQL-based services with Supabase integration
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import math

from .supabase_service import SupabaseLearningProgressService
from .models import *

logger = logging.getLogger("supabase_services")


class LearningProgressServiceSupabase:
    """Supabase-powered service for managing student learning progress and LLM integration"""

    def __init__(self):
        self.supabase_service = SupabaseLearningProgressService()

    async def create_or_get_student(self, student_id: str, username: str, email: str, full_name: str) -> str:
        """Create a new student or get existing one"""
        return await self.supabase_service.create_or_get_student(student_id, username, email, full_name)

    async def update_learning_preferences(self, student_id: str, preferences: Dict[str, Any]):
        """Update student learning preferences"""
        await self.supabase_service.student_manager.update_student(
            student_id, learning_preferences=preferences
        )

    async def get_student_by_id(self, student_id: str) -> Optional[StudentResponse]:
        """Get student details by ID."""
        return await self.supabase_service.get_student_by_id(student_id)

    async def get_all_students(self) -> List[StudentResponse]:
        """Get all students."""
        students_data = await self.supabase_service.student_manager.get_all_students()
        return [
            StudentResponse(
                student_id=student['student_id'],
                username=student['username'],
                email=student['email'],
                full_name=student['full_name'],
                message="Student retrieved successfully"
            )
            for student in students_data
        ]

    async def delete_student(self, student_id: str) -> bool:
        """Delete a student by ID."""
        return await self.supabase_service.student_manager.delete_student(student_id)

    async def get_or_create_subject(self, subject_name: str, description: str = "") -> int:
        """Get or create a subject"""
        return await self.supabase_service.subject_manager.create_or_get_subject(subject_name, description)

    async def get_or_create_chapter(self, subject_id: int, chapter_name: str, description: str = "") -> int:
        """Get or create a chapter"""
        return await self.supabase_service.subject_manager.create_or_get_chapter(subject_id, chapter_name, description)

    async def get_or_create_concept(self, chapter_id: int, concept_name: str,
                                    difficulty_level: str = "Medium", description: str = "") -> int:
        """Get or create a concept"""
        return await self.supabase_service.subject_manager.create_or_get_concept(
            chapter_id, concept_name, difficulty_level, description
        )

    async def process_llm_response(self, student_id: str, subject_name: str,
                                   chapter_name: str, concept_name: str,
                                   llm_response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Process LLM response with comprehensive tracking"""
        return await self.supabase_service.process_llm_response(
            student_id, subject_name, chapter_name, concept_name, llm_response, user_query
        )

    async def update_concept_progress(self, student_id: str, concept_id: int,
                                      correct_answers: int, total_questions: int,
                                      time_spent: Optional[int] = None):
        """Update student progress on a concept"""
        await self.supabase_service.update_concept_progress(
            student_id, concept_id, correct_answers, total_questions, time_spent
        )

    async def ensure_student_exists(self, student_id: str):
        """Ensure student exists in database, create if not found"""
        try:
            logger.info(f"🔍 Checking if student exists: {student_id}")
            existing_student = await self.supabase_service.get_student_by_id(student_id)
            if not existing_student:
                logger.info(f"👤 Student {student_id} not found, creating...")
                # Create student with email as username and a default name
                await self.supabase_service.create_or_get_student(
                    student_id=student_id,
                    username=student_id,
                    email=student_id,
                    full_name=f"User {student_id}"
                )
                logger.info(f"✅ Created student record for {student_id}")
            else:
                logger.info(f"✅ Student {student_id} already exists")
        except Exception as e:
            logger.error(
                f"❌ Error in ensure_student_exists for {student_id}: {str(e)}")
            # Continue anyway - might be a temporary issue

    async def save_concept_progress(self, student_id: str, subject_name: str, concept_name: str,
                                    mastery_level: float, correct_answers: int, total_questions: int,
                                    time_spent: int, difficulty_level: str, activity_type: str):
        """Save concept progress with comprehensive tracking"""
        try:
            logger.info(
                f"💾 Starting save_concept_progress for student: {student_id}, concept: {concept_name}")

            # Ensure student exists - create if not found
            await self.ensure_student_exists(student_id)

            # Get or create subject, chapter, and concept
            subject_id = await self.get_or_create_subject(subject_name)
            chapter_id = await self.get_or_create_chapter(subject_id, f"{subject_name} Concepts")
            concept_id = await self.get_or_create_concept(chapter_id, concept_name, difficulty_level)

            # Update progress
            await self.update_concept_progress(student_id, concept_id, correct_answers, total_questions, time_spent)

            # Record learning activity
            activity_data = {
                'activity_type': activity_type,
                'correct_answers': correct_answers,
                'total_questions': total_questions,
                'mastery_level': mastery_level,
                'difficulty_level': difficulty_level
            }

            await self.supabase_service.progress_tracker.record_learning_activity(
                student_id, concept_id, activity_type, activity_data, mastery_level, time_spent
            )

            logger.info(
                f"Saved concept progress for student {student_id}, concept {concept_name}")

        except Exception as e:
            logger.error(f"Error saving concept progress: {str(e)}")
            raise

    async def save_recommendations(self, student_id: str, recommendations: List[Dict[str, Any]]):
        """Save recommendations for a student"""
        try:
            await self.supabase_service.recommendation_engine.save_recommendations(
                student_id, recommendations
            )
            logger.info(
                f"Saved {len(recommendations)} recommendations for student {student_id}")

        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
            raise

    async def record_quiz_attempt(self, student_id: str, concept_id: int,
                                  quiz_data: Dict[str, Any], score: float,
                                  time_spent: int) -> Dict[str, Any]:
        """Record a quiz attempt"""
        try:
            # Extract quiz performance data
            questions = quiz_data.get('questions', [])
            answers = quiz_data.get('answers', [])

            correct_answers = sum(1 for i, answer in enumerate(answers)
                                  if i < len(questions) and answer == questions[i].get('correct_answer'))
            total_questions = len(questions)

            # Update concept progress
            await self.update_concept_progress(
                student_id, concept_id, correct_answers, total_questions, time_spent
            )

            # Record the quiz activity
            activity_data = {
                'quiz_data': quiz_data,
                'answers': answers,
                'score': score,
                'correct_answers': correct_answers,
                'total_questions': total_questions
            }

            await self.supabase_service.progress_tracker.record_learning_activity(
                student_id, concept_id, "quiz_attempt", activity_data, score, time_spent
            )

            return {
                "quiz_recorded": True,
                "correct_answers": correct_answers,
                "total_questions": total_questions,
                "accuracy": (correct_answers / total_questions) * 100 if total_questions > 0 else 0,
                "score": score
            }

        except Exception as e:
            logger.error(f"Error recording quiz attempt: {str(e)}")
            raise

    async def get_student_progress(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive student progress"""
        try:
            # Get student progress from Supabase
            progress_data = await self.supabase_service.student_manager.client.get_student_progress(student_id)

            # Organize progress by subjects and chapters
            subjects = {}
            total_concepts = 0
            mastered_concepts = 0

            for progress in progress_data:
                concept = progress.get('concepts', {})
                chapter = concept.get('chapters', {})
                subject = chapter.get('subjects', {})

                subject_name = subject.get('subject_name', 'Unknown')
                chapter_name = chapter.get('chapter_name', 'Unknown')
                concept_name = concept.get('concept_name', 'Unknown')

                if subject_name not in subjects:
                    subjects[subject_name] = {
                        'chapters': {}, 'total_concepts': 0, 'mastered_concepts': 0}

                if chapter_name not in subjects[subject_name]['chapters']:
                    subjects[subject_name]['chapters'][chapter_name] = {
                        'concepts': [], 'mastery_rate': 0}

                concept_progress = {
                    'concept_name': concept_name,
                    'status': progress.get('status', 'not_started'),
                    'mastery_score': progress.get('mastery_score', 0),
                    'attempts_count': progress.get('attempts_count', 0),
                    'last_practiced': progress.get('last_practiced'),
                    'first_learned': progress.get('first_learned')
                }

                subjects[subject_name]['chapters'][chapter_name]['concepts'].append(
                    concept_progress)
                subjects[subject_name]['total_concepts'] += 1
                total_concepts += 1

                if progress.get('status') == 'mastered':
                    subjects[subject_name]['mastered_concepts'] += 1
                    mastered_concepts += 1

            # Calculate mastery rates
            for subject in subjects.values():
                for chapter in subject['chapters'].values():
                    mastered_in_chapter = sum(
                        1 for c in chapter['concepts'] if c['status'] == 'mastered')
                    chapter['mastery_rate'] = (
                        mastered_in_chapter / len(chapter['concepts'])) * 100 if chapter['concepts'] else 0

            overall_mastery_rate = (
                mastered_concepts / total_concepts) * 100 if total_concepts > 0 else 0

            return {
                'student_id': student_id,
                'subjects': subjects,
                'overall_stats': {
                    'total_concepts': total_concepts,
                    'mastered_concepts': mastered_concepts,
                    'overall_mastery_rate': overall_mastery_rate
                }
            }

        except Exception as e:
            logger.error(f"Error getting student progress: {str(e)}")
            return {'student_id': student_id, 'subjects': {}, 'overall_stats': {}}

    async def get_recommendations(self, student_id: str) -> List[Dict[str, Any]]:
        """Get personalized recommendations for student"""
        try:
            logger.info(
                f"LearningProgressService: Getting recommendations for student {student_id}")

            # Get fresh recommendations
            recommendations = await self.supabase_service.recommendation_engine.get_recommendations(student_id)
            logger.info(
                f"Retrieved {len(recommendations)} recommendations for student {student_id}")

            return recommendations

        except Exception as e:
            logger.error(
                f"Error getting recommendations for student {student_id}: {str(e)}")
            # Return default recommendations as fallback
            return [
                {
                    'recommendation_id': 'default-1',
                    'recommendation_type': 'welcome',
                    'title': 'Welcome to StudyGenie!',
                    'description': 'Start by uploading your study materials to get personalized recommendations',
                    'priority_score': 10,
                    'is_active': True,
                    'is_completed': False,
                    'created_at': datetime.now().isoformat()
                }
            ]

    async def get_learning_analytics(self, student_id: str, days: int = 30) -> Dict[str, Any]:
        """Get learning analytics for student with subject-wise breakdown"""
        try:
            # Get basic metrics from the database tables
            client = self.supabase_service.progress_tracker.client

            # Get concept progress data with subject and chapter information
            progress_response = client.client.table("student_concept_progress")\
                .select("""
                    *,
                    concepts:concept_id (
                        concept_name,
                        chapters:chapter_id (
                            chapter_name,
                            subjects:subject_id (
                                subject_name
                            )
                        )
                    )
                """)\
                .eq("student_id", student_id)\
                .execute()

            progress_data = progress_response.data if progress_response.data else []

            # Get learning activities with concept information
            activities_data = []
            try:
                activities_response = client.client.table("learning_activities")\
                    .select("""
                        *,
                        concepts:concept_id (
                            concept_name,
                            chapters:chapter_id (
                                chapter_name,
                                subjects:subject_id (
                                    subject_name
                                )
                            )
                        )
                    """)\
                    .eq("student_id", student_id)\
                    .gte("completed_at", (datetime.now() - timedelta(days=days)).isoformat())\
                    .execute()
                activities_data = activities_response.data if activities_response.data else []
            except:
                # If learning_activities table doesn't exist, continue with empty list
                pass

            # Calculate overall metrics with robust null handling
            total_activities = len(activities_data)
            total_time_spent = sum((activity.get('time_spent') or 0)
                                   for activity in activities_data)

            # Calculate quiz metrics from activities (only count actual quiz attempts)
            quiz_activities = [a for a in activities_data if a.get(
                'activity_type') in ['quiz_attempt', 'quiz']]
            flashcard_activities = [a for a in activities_data if a.get(
                'activity_type') in ['flashcard_practice', 'flashcard']]

            # Quiz statistics
            quiz_count = len(quiz_activities)
            total_questions = sum(a.get('activity_data', {}).get(
                'total_questions', 0) for a in quiz_activities)
            total_correct = sum(a.get('activity_data', {}).get(
                'correct_answers', 0) for a in quiz_activities)

            # Also include flashcard statistics in overall questions/answers
            flashcard_questions = sum(a.get('activity_data', {}).get(
                'total_questions', 0) for a in flashcard_activities)
            flashcard_correct = sum(a.get('activity_data', {}).get(
                'correct_answers', 0) for a in flashcard_activities)

            # Combined statistics
            total_questions += flashcard_questions
            total_correct += flashcard_correct

            # Calculate concepts learned (mastered)
            concepts_learned = len(
                [p for p in progress_data if p.get('mastery_score', 0) >= 80])

            # Study streak calculation - count consecutive days with activities
            study_streak = 0
            if activities_data:
                # Get dates with activities
                activity_dates = sorted(set(
                    datetime.fromisoformat(a['completed_at']).date()
                    for a in activities_data if a.get('completed_at')
                ))

                if activity_dates:
                    # Calculate streak from most recent date backwards
                    current_date = datetime.now().date()
                    streak_count = 0

                    # Check if there's activity today or yesterday
                    if current_date in activity_dates:
                        streak_count = 1
                        check_date = current_date - timedelta(days=1)
                    elif (current_date - timedelta(days=1)) in activity_dates:
                        streak_count = 1
                        check_date = current_date - timedelta(days=2)
                    else:
                        # No recent activity, streak is 0
                        study_streak = 0
                        check_date = None

                    # Count consecutive days backwards
                    while check_date and check_date in activity_dates:
                        streak_count += 1
                        check_date -= timedelta(days=1)

                    study_streak = streak_count

            # Calculate average accuracy
            quiz_accuracy = (total_correct / total_questions *
                             100) if total_questions > 0 else 0

            # **NEW: Calculate Subject-wise Analytics**
            subjects_analytics = {}

            # Process progress data by subject
            for progress in progress_data:
                concept_info = progress.get('concepts')
                if not concept_info:
                    continue

                chapter_info = concept_info.get('chapters')
                if not chapter_info:
                    continue

                subject_info = chapter_info.get('subjects')
                if not subject_info:
                    continue

                subject_name = subject_info.get('subject_name', 'Unknown')
                chapter_name = chapter_info.get('chapter_name', 'Unknown')
                concept_name = concept_info.get('concept_name', 'Unknown')

                # Initialize subject if not exists
                if subject_name not in subjects_analytics:
                    subjects_analytics[subject_name] = {
                        'subject_name': subject_name,
                        'total_concepts': 0,
                        'mastered_concepts': 0,
                        'time_spent': 0,
                        'activities_count': 0,
                        'quiz_accuracy': 0,
                        'chapters': {},
                        'concepts': []
                    }

                # Add concept progress
                subjects_analytics[subject_name]['total_concepts'] += 1
                mastery_score = progress.get('mastery_score', 0)
                if mastery_score >= 80:
                    subjects_analytics[subject_name]['mastered_concepts'] += 1

                # Add concept details
                subjects_analytics[subject_name]['concepts'].append({
                    'concept_name': concept_name,
                    'chapter_name': chapter_name,
                    'mastery_score': mastery_score,
                    'total_attempts': progress.get('total_attempts', 0),
                    'correct_answers': progress.get('correct_answers', 0),
                    'total_questions': progress.get('total_questions', 0),
                    'last_updated': progress.get('last_updated')
                })

                # Initialize chapter if not exists
                if chapter_name not in subjects_analytics[subject_name]['chapters']:
                    subjects_analytics[subject_name]['chapters'][chapter_name] = {
                        'chapter_name': chapter_name,
                        'concepts_count': 0,
                        'mastered_count': 0,
                        'average_mastery': 0,
                        'concepts': []
                    }

                # Add concept to chapter
                subjects_analytics[subject_name]['chapters'][chapter_name]['concepts'].append({
                    'concept_name': concept_name,
                    'mastery_score': mastery_score,
                    # Frontend expects attempts_count
                    'attempts_count': progress.get('total_attempts', 0),
                    'correct_answers': progress.get('correct_answers', 0),
                    'total_questions': progress.get('total_questions', 0),
                    # Frontend expects last_practiced
                    'last_practiced': progress.get('last_updated'),
                    'status': 'mastered' if mastery_score >= 80 else 'needs_review' if mastery_score < 60 else 'in_progress'
                })

                subjects_analytics[subject_name]['chapters'][chapter_name]['concepts_count'] += 1
                if mastery_score >= 80:
                    subjects_analytics[subject_name]['chapters'][chapter_name]['mastered_count'] += 1

            # Process activities data by subject
            for activity in activities_data:
                concept_info = activity.get('concepts')
                if not concept_info:
                    continue

                chapter_info = concept_info.get('chapters')
                if not chapter_info:
                    continue

                subject_info = chapter_info.get('subjects')
                if not subject_info:
                    continue

                subject_name = subject_info.get('subject_name', 'Unknown')

                if subject_name in subjects_analytics:
                    time_spent = activity.get('time_spent') or 0
                    subjects_analytics[subject_name]['time_spent'] += time_spent
                    subjects_analytics[subject_name]['activities_count'] += 1

            # Calculate subject-wise quiz accuracy and chapter mastery rates
            for subject_name, subject_data in subjects_analytics.items():
                # Calculate subject quiz accuracy
                subject_quiz_activities = [a for a in activities_data
                                           if a.get('concepts', {}).get('chapters', {}).get('subjects', {}).get('subject_name') == subject_name
                                           and a.get('activity_type') == 'quiz']

                subject_total_questions = sum(a.get('activity_data', {}).get(
                    'total_questions', 0) for a in subject_quiz_activities)
                subject_correct_answers = sum(a.get('activity_data', {}).get(
                    'correct_answers', 0) for a in subject_quiz_activities)

                subject_data['quiz_accuracy'] = (
                    subject_correct_answers / subject_total_questions * 100) if subject_total_questions > 0 else 0

                # Ensure these values are never None
                mastered_count = subject_data.get('mastered_concepts') or 0
                total_count = subject_data.get('total_concepts') or 0

                subject_data['mastery_percentage'] = (
                    mastered_count / total_count * 100) if total_count > 0 else 0

                # Calculate chapter mastery rates
                for chapter_name, chapter_data in subject_data['chapters'].items():
                    mastered_count = chapter_data.get('mastered_count') or 0
                    concepts_count = chapter_data.get('concepts_count') or 0
                    chapter_data['mastery_rate'] = (
                        mastered_count / concepts_count * 100) if concepts_count > 0 else 0

                    # Calculate average mastery score for chapter
                    concepts = chapter_data.get('concepts', [])
                    if concepts:
                        total_mastery = sum(concept.get(
                            'mastery_score', 0) for concept in concepts)
                        chapter_data['average_mastery'] = total_mastery / \
                            len(concepts)
                    else:
                        chapter_data['average_mastery'] = 0

            return {
                'student_id': student_id,
                'period_days': days,
                'study_streak': study_streak,
                'concepts_learned': concepts_learned,
                'time_spent': total_time_spent,
                'quiz_accuracy': quiz_accuracy,
                'total_activities': total_activities,
                'quiz_count': quiz_count,
                'flashcard_count': len(flashcard_activities),
                'total_questions': total_questions,
                'correct_answers': total_correct,
                'concepts_mastered': concepts_learned,
                'session_count': total_activities,
                'average_score': quiz_accuracy / 100 if quiz_accuracy > 0 else 0,
                'activity_timeline': activities_data[-10:] if activities_data else [],
                # **NEW: Subject-wise analytics**
                'subjects_analytics': subjects_analytics,
                'subjects_summary': [
                    {
                        'subject_name': subject_name,
                        'mastery_percentage': subject_data['mastery_percentage'],
                        'time_spent': subject_data['time_spent'],
                        'concepts_total': subject_data['total_concepts'],
                        'concepts_mastered': subject_data['mastered_concepts'],
                        'quiz_accuracy': subject_data['quiz_accuracy'],
                        'activities_count': subject_data['activities_count']
                    }
                    for subject_name, subject_data in subjects_analytics.items()
                ]
            }

        except Exception as e:
            logger.error(f"Error getting learning analytics: {str(e)}")
            # Return basic analytics structure if error
            return {
                'student_id': student_id,
                'period_days': days,
                'study_streak': 0,
                'concepts_learned': 0,
                'time_spent': 0,
                'quiz_accuracy': 0,
                'total_activities': 0,
                'total_questions': 0,
                'correct_answers': 0,
                'concepts_mastered': 0,
                'session_count': 0,
                'average_score': 0,
                'activity_timeline': []
            }

    # API Key management methods
    async def store_student_api_key(self, student_id: str, encrypted_api_key: str) -> bool:
        """Store encrypted API key for student"""
        return await self.supabase_service.student_manager.store_api_key(student_id, encrypted_api_key)

    async def get_student_api_key(self, student_id: str) -> Optional[str]:
        """Get encrypted API key for student"""
        return await self.supabase_service.student_manager.get_api_key(student_id)

    async def delete_student_api_key(self, student_id: str) -> bool:
        """Delete API key for student"""
        return await self.supabase_service.student_manager.delete_api_key(student_id)


# Create a global instance that can be imported
def get_learning_progress_service() -> LearningProgressServiceSupabase:
    """Get the Supabase-powered learning progress service"""
    return LearningProgressServiceSupabase()
