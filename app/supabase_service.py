"""
Supabase integration service for StudyGenie
This service layer bridges your existing database models with Supabase
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from app.supabase_client import get_supabase_client
from app.models import *

logger = logging.getLogger("supabase_service")


class SupabaseStudentManager:
    """Student management using Supabase"""

    def __init__(self):
        self.client = get_supabase_client()

    async def create_or_get_student(self, student_id: str, username: str, email: str, full_name: str) -> str:
        """Create a new student or get existing student ID"""
        try:
            # Try to get existing student
            existing_student = await self.client.get_student(student_id)
            if existing_student:
                return existing_student['student_id']

            # Create new student
            student_data = {
                'student_id': student_id,
                'username': username,
                'email': email,
                'full_name': full_name,
                'learning_preferences': {},
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            created_student = await self.client.create_student(student_data)
            logger.info(f"Created new student with ID: {student_id}")
            return created_student['student_id']

        except Exception as e:
            logger.error(f"Error creating/getting student {student_id}: {e}")
            raise

    async def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student details by ID"""
        try:
            return await self.client.get_student(student_id)
        except Exception as e:
            logger.error(f"Error fetching student {student_id}: {e}")
            return None

    async def update_student(self, student_id: str, full_name: Optional[str] = None,
                             learning_preferences: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Update student details"""
        try:
            update_data = {'updated_at': datetime.now().isoformat()}

            if full_name is not None:
                update_data['full_name'] = full_name
            if learning_preferences is not None:
                update_data['learning_preferences'] = json.dumps(
                    learning_preferences)

            if len(update_data) == 1:  # Only timestamp
                return await self.get_student_by_id(student_id)

            return await self.client.update_student(student_id, update_data)
        except Exception as e:
            logger.error(f"Error updating student {student_id}: {e}")
            return None

    async def delete_student(self, student_id: str) -> bool:
        """Delete a student by ID"""
        try:
            return await self.client.delete_student(student_id)
        except Exception as e:
            logger.error(f"Error deleting student {student_id}: {e}")
            return False

    async def get_all_students(self) -> List[Dict[str, Any]]:
        """Get all students"""
        try:
            response = self.client.client.table(
                "students").select("*").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching all students: {e}")
            return []

    # API Key management
    async def store_api_key(self, student_id: str, encrypted_api_key: str) -> bool:
        """Store encrypted API key for student"""
        try:
            return await self.client.store_api_key(student_id, encrypted_api_key)
        except Exception as e:
            logger.error(
                f"Error storing API key for student {student_id}: {e}")
            return False

    async def get_api_key(self, student_id: str) -> Optional[str]:
        """Get encrypted API key for student"""
        try:
            return await self.client.get_api_key(student_id)
        except Exception as e:
            logger.error(
                f"Error fetching API key for student {student_id}: {e}")
            return None

    async def delete_api_key(self, student_id: str) -> bool:
        """Delete API key for student"""
        try:
            return await self.client.delete_api_key(student_id)
        except Exception as e:
            logger.error(
                f"Error deleting API key for student {student_id}: {e}")
            return False


class SupabaseProgressTracker:
    """Progress tracking using Supabase"""

    def __init__(self):
        self.client = get_supabase_client()

    async def update_concept_progress(self, student_id: str, concept_id: int,
                                      correct_answers: int, total_questions: int):
        """Update student's progress on a specific concept"""
        try:
            # Calculate scores
            accuracy = (correct_answers / total_questions) * \
                100 if total_questions > 0 else 0

            # Determine status based on performance
            if accuracy >= 90:
                status = "mastered"
            elif accuracy >= 70:
                status = "in_progress"
            else:
                status = "needs_review"

            # Prepare progress data
            progress_data = {
                'student_id': student_id,
                'concept_id': concept_id,
                'status': status,
                'mastery_score': accuracy,
                'attempts_count': 1,  # This would be incremented in real implementation
                'correct_answers': correct_answers,
                'total_questions': total_questions,
                'last_practiced': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            # Check if progress record exists
            existing = self.client.client.table("student_concept_progress")\
                .select("*")\
                .eq("student_id", student_id)\
                .eq("concept_id", concept_id)\
                .execute()

            if existing.data:
                # Update existing record
                existing_record = existing.data[0]
                progress_data['attempts_count'] = existing_record['attempts_count'] + 1
                progress_data['correct_answers'] = existing_record['correct_answers'] + \
                    correct_answers
                progress_data['total_questions'] = existing_record['total_questions'] + \
                    total_questions

                # Recalculate mastery score
                total_accuracy = (
                    progress_data['correct_answers'] / progress_data['total_questions']) * 100
                progress_data['mastery_score'] = (
                    existing_record['mastery_score'] + total_accuracy) / 2

                if not existing_record.get('first_learned'):
                    progress_data['first_learned'] = datetime.now().isoformat()
                else:
                    progress_data['first_learned'] = existing_record['first_learned']
            else:
                # New record
                progress_data['first_learned'] = datetime.now().isoformat()

            await self.client.update_concept_progress(progress_data)
            logger.info(
                f"Updated progress for student {student_id}, concept {concept_id}")

        except Exception as e:
            logger.error(f"Error updating concept progress: {e}")
            raise

    async def record_learning_activity(self, student_id: str, concept_id: int,
                                       activity_type: str, activity_data: Dict[str, Any],
                                       score: Optional[float] = None, time_spent: Optional[int] = None):
        """Record a learning activity"""
        try:
            activity_record = {
                'student_id': student_id,
                'concept_id': concept_id,
                'activity_type': activity_type,
                'activity_data': activity_data,
                'score': score,
                'time_spent': time_spent,
                'completed_at': datetime.now().isoformat()
            }

            await self.client.record_learning_activity(activity_record)
            logger.info(f"Recorded learning activity for student {student_id}")

        except Exception as e:
            logger.error(f"Error recording learning activity: {e}")
            raise


class SupabaseSubjectManager:
    """Subject management using Supabase"""

    def __init__(self):
        self.client = get_supabase_client()

    async def create_or_get_subject(self, subject_name: str, description: str = "") -> int:
        """Create a new subject or get existing subject ID"""
        try:
            return await self.client.create_or_get_subject(subject_name, description)
        except Exception as e:
            logger.error(f"Error creating/getting subject {subject_name}: {e}")
            raise

    async def create_or_get_chapter(self, subject_id: int, chapter_name: str, description: str = "") -> int:
        """Create a new chapter or get existing chapter ID"""
        try:
            return await self.client.create_or_get_chapter(subject_id, chapter_name, description)
        except Exception as e:
            logger.error(f"Error creating/getting chapter {chapter_name}: {e}")
            raise

    async def create_or_get_concept(self, chapter_id: int, concept_name: str,
                                    difficulty_level: str = "Medium", description: str = "") -> int:
        """Create a new concept or get existing concept ID"""
        try:
            return await self.client.create_or_get_concept(chapter_id, concept_name, difficulty_level, description)
        except Exception as e:
            logger.error(f"Error creating/getting concept {concept_name}: {e}")
            raise


class SupabaseRecommendationEngine:
    """Recommendation engine using Supabase"""

    def __init__(self):
        self.client = get_supabase_client()

    async def get_recommendations(self, student_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get recommendations for student"""
        try:
            logger.info(
                f"Fetching recommendations for student: {student_id}, active_only: {active_only}")
            recommendations = await self.client.get_recommendations(student_id, active_only)
            logger.info(
                f"Found {len(recommendations)} existing recommendations for student {student_id}")

            # Always generate fresh recommendations to ensure they're based on latest activity
            logger.info(f"Generating fresh recommendations for {student_id}")
            generated_recommendations = await self.generate_personalized_recommendations(student_id)

            # Save the generated recommendations (this will deactivate old ones)
            if generated_recommendations:
                save_success = await self.save_recommendations(student_id, generated_recommendations)
                if save_success:
                    logger.info(
                        f"Successfully saved {len(generated_recommendations)} new recommendations for {student_id}")
                    return generated_recommendations
                else:
                    logger.warning(
                        f"Failed to save recommendations for {student_id}, returning generated ones anyway")
                    return generated_recommendations

            # If generation failed, return existing recommendations
            return recommendations

        except Exception as e:
            logger.error(
                f"Error fetching recommendations for student {student_id}: {e}")
            # Fallback: try to generate recommendations
            try:
                logger.info(
                    f"Attempting to generate fallback recommendations for {student_id}")
                return await self.generate_personalized_recommendations(student_id)
            except Exception as fallback_error:
                logger.error(
                    f"Failed to generate fallback recommendations: {fallback_error}")
                return []

    async def save_recommendations(self, student_id: str, recommendations: List[Dict[str, Any]]) -> bool:
        """Save recommendations for student"""
        try:
            return await self.client.save_recommendations(student_id, recommendations)
        except Exception as e:
            logger.error(
                f"Error saving recommendations for student {student_id}: {e}")
            return False

    async def generate_personalized_recommendations(self, student_id: str) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on student progress"""
        try:
            recommendations = []
            logger.info(
                f"Generating recommendations for student: {student_id}")

            # Get student progress to analyze weaknesses
            try:
                progress_data = await self.client.get_student_progress(student_id)
                logger.info(
                    f"Retrieved {len(progress_data)} progress records for student {student_id}")
            except Exception as e:
                logger.warning(
                    f"Could not get student progress for {student_id}: {e}")
                progress_data = []

            # Also get learning activities to understand what they've done
            try:
                activities_response = self.client.client.table("learning_activities")\
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
                    .order("completed_at", desc=True)\
                    .execute()

                activities_data = activities_response.data if activities_response.data else []
                logger.info(
                    f"Retrieved {len(activities_data)} learning activities for student {student_id}")
            except Exception as e:
                logger.warning(
                    f"Could not get learning activities for {student_id}: {e}")
                activities_data = []

            # If we have activities but no progress data, something is wrong with progress tracking
            if activities_data and not progress_data:
                logger.warning(
                    f"Student {student_id} has {len(activities_data)} activities but no progress records")

                # Generate recommendations based on recent activities
                recent_activities = activities_data[:10]  # Last 10 activities

                # Analyze performance from activities
                low_performing_concepts = []
                high_performing_concepts = []

                for activity in recent_activities:
                    activity_data = activity.get('activity_data', {})
                    correct = activity_data.get('correct_answers', 0)
                    total = activity_data.get('total_questions', 1)
                    accuracy = (correct / total) * 100 if total > 0 else 0

                    concept_info = activity.get('concepts', {})
                    concept_name = concept_info.get(
                        'concept_name', 'Unknown Concept')

                    if accuracy < 70:
                        low_performing_concepts.append({
                            'concept_name': concept_name,
                            'accuracy': accuracy,
                            'concept_id': activity.get('concept_id'),
                            'activity_type': activity.get('activity_type')
                        })
                    elif accuracy > 85:
                        high_performing_concepts.append({
                            'concept_name': concept_name,
                            'accuracy': accuracy,
                            'concept_id': activity.get('concept_id'),
                            'activity_type': activity.get('activity_type')
                        })

                # Create recommendations for low-performing concepts
                for concept in low_performing_concepts[:5]:
                    recommendations.append({
                        'recommendation_type': 'concept_review',
                        'concept_id': concept.get('concept_id'),
                        'title': f"Improve {concept['concept_name']}",
                        'description': f"Your accuracy is {concept['accuracy']:.1f}%. Practice more to master this concept.",
                        'priority_score': 10 if concept['accuracy'] < 50 else 7,
                        'is_active': True,
                        'is_completed': False
                    })

                # Create maintenance recommendations for high-performing concepts
                for concept in high_performing_concepts[:3]:
                    recommendations.append({
                        'recommendation_type': 'maintenance_practice',
                        'concept_id': concept.get('concept_id'),
                        'title': f"Maintain {concept['concept_name']}",
                        'description': f"Great job! Keep practicing to maintain your {concept['accuracy']:.1f}% accuracy.",
                        'priority_score': 4,
                        'is_active': True,
                        'is_completed': False
                    })

                # Add general study recommendations
                if len(recommendations) < 3:
                    recommendations.append({
                        'recommendation_type': 'continue_learning',
                        'concept_id': None,
                        'title': 'Continue Your Learning Journey',
                        'description': 'Keep practicing with quizzes and flashcards to improve your understanding.',
                        'priority_score': 6,
                        'is_active': True,
                        'is_completed': False
                    })

                logger.info(
                    f"Generated {len(recommendations)} activity-based recommendations for student {student_id}")
                return recommendations

            if not progress_data and not activities_data:
                # Generate default recommendations for new students
                logger.info(
                    f"No progress or activity data found for {student_id}, generating default recommendations")
                default_recommendations = [
                    {
                        'recommendation_type': 'onboarding',
                        'concept_id': None,
                        'title': 'Welcome to StudyGenie!',
                        'description': 'Start by uploading your study materials to get personalized recommendations',
                        'priority_score': 10,
                        'is_active': True,
                        'is_completed': False
                    },
                    {
                        'recommendation_type': 'first_quiz',
                        'concept_id': None,
                        'title': 'Take Your First Quiz',
                        'description': 'Complete a quiz to help us understand your learning level',
                        'priority_score': 8,
                        'is_active': True,
                        'is_completed': False
                    },
                    {
                        'recommendation_type': 'explore_features',
                        'concept_id': None,
                        'title': 'Explore Learning Features',
                        'description': 'Try out flashcards, quizzes, and interactive exercises',
                        'priority_score': 6,
                        'is_active': True,
                        'is_completed': False
                    }
                ]
                return default_recommendations

            # Analyze weak concepts from progress data
            weak_concepts = [
                p for p in progress_data
                if p.get('status') == 'needs_review' or p.get('mastery_score', 0) < 70
            ]

            for concept in weak_concepts[:5]:  # Top 5 weak concepts
                priority = 10 if concept.get('mastery_score', 0) < 50 else 7
                concept_name = concept.get('concepts', {}).get(
                    'concept_name', 'Unknown Concept')
                recommendations.append({
                    'recommendation_type': 'concept_review',
                    'concept_id': concept.get('concept_id'),
                    'title': f"Review {concept_name}",
                    'description': f"Focus on improving your understanding of this concept",
                    'priority_score': priority,
                    'is_active': True,
                    'is_completed': False
                })

            # Get concepts ready for maintenance practice
            strong_concepts = []
            for p in progress_data:
                if p.get('status') == 'mastered':
                    try:
                        last_practiced = p.get(
                            'last_practiced', datetime.now().isoformat())
                        if isinstance(last_practiced, str):
                            last_practiced_date = datetime.fromisoformat(
                                last_practiced.replace('Z', '+00:00'))
                        else:
                            last_practiced_date = last_practiced

                        days_since_practice = (
                            datetime.now() - last_practiced_date).days
                        if days_since_practice > 7:
                            strong_concepts.append(p)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Could not parse date for concept {p.get('concept_id')}: {e}")
                        continue

            for concept in strong_concepts[:3]:  # Top 3 for maintenance
                concept_name = concept.get('concepts', {}).get(
                    'concept_name', 'Unknown Concept')
                recommendations.append({
                    'recommendation_type': 'maintenance_practice',
                    'concept_id': concept.get('concept_id'),
                    'title': f"Practice {concept_name}",
                    'description': f"Keep your skills sharp with regular practice",
                    'priority_score': 4,
                    'is_active': True,
                    'is_completed': False
                })

            logger.info(
                f"Generated {len(recommendations)} recommendations for student {student_id}")
            return recommendations

        except Exception as e:
            logger.error(
                f"Error generating recommendations for student {student_id}: {e}")
            # Return default recommendations as fallback
            return [
                {
                    'recommendation_type': 'system_error',
                    'concept_id': None,
                    'title': 'Continue Learning',
                    'description': 'Keep practicing with your study materials to improve your knowledge',
                    'priority_score': 5,
                    'is_active': True,
                    'is_completed': False
                }
            ]

# Integration with existing services


class SupabaseLearningProgressService:
    """Learning progress service using Supabase backend"""

    def __init__(self):
        self.student_manager = SupabaseStudentManager()
        self.subject_manager = SupabaseSubjectManager()
        self.progress_tracker = SupabaseProgressTracker()
        self.recommendation_engine = SupabaseRecommendationEngine()

    async def create_or_get_student(self, student_id: str, username: str, email: str, full_name: str) -> str:
        """Create a new student or get existing one"""
        return await self.student_manager.create_or_get_student(student_id, username, email, full_name)

    async def get_student_by_id(self, student_id: str) -> Optional[StudentResponse]:
        """Get student details by ID"""
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

    async def process_llm_response(self, student_id: str, subject_name: str,
                                   chapter_name: str, concept_name: str,
                                   llm_response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Process LLM response and update student tracking"""
        try:
            # Extract additional metadata from LLM response
            metadata = llm_response.get('metadata', {})
            difficulty_level = metadata.get('difficulty_level', 'Medium')
            estimated_study_time = metadata.get('estimated_study_time', '')
            learning_objectives = llm_response.get('learning_objectives', [])
            summary = llm_response.get('summary', '')

            # Create enhanced description with LLM-generated content
            concept_description = f"Auto-created from LLM response. {summary}"
            if learning_objectives:
                concept_description += f"\nLearning Objectives: {'; '.join(learning_objectives)}"
            if estimated_study_time:
                concept_description += f"\nEstimated Study Time: {estimated_study_time}"

            # Create or get subject hierarchy with enhanced metadata
            subject_id = await self.subject_manager.create_or_get_subject(
                subject_name, f"Auto-created from LLM response"
            )
            chapter_id = await self.subject_manager.create_or_get_chapter(
                subject_id, chapter_name, f"Auto-created from LLM response"
            )
            concept_id = await self.subject_manager.create_or_get_concept(
                chapter_id, concept_name, difficulty_level, concept_description
            )

            # Record the study activity with enhanced data
            activity_data = {
                'llm_response': llm_response,
                'user_query': user_query,
                'timestamp': datetime.now().isoformat(),
                'response_type': 'structured_content',
                'metadata': metadata,
                'difficulty_level': difficulty_level,
                'estimated_study_time': estimated_study_time,
                'learning_objectives_count': len(learning_objectives),
                'flashcards_count': len(llm_response.get('flashcards', [])),
                'quiz_questions_count': len(llm_response.get('quiz', []))
            }

            await self.progress_tracker.record_learning_activity(
                student_id, concept_id, "content_study", activity_data, time_spent=None
            )

            # Prepare enhanced response with all LLM metadata preserved
            enhanced_response = llm_response.copy()

            tracking_metadata = {
                'student_id': student_id,
                'subject_id': subject_id,
                'chapter_id': chapter_id,
                'concept_id': concept_id,
                'subject_name': subject_name,
                'chapter_name': chapter_name,
                'concept_name': concept_name,
                'difficulty_level': difficulty_level,
                'estimated_study_time': estimated_study_time,
                'learning_objectives': learning_objectives,
                'content_summary': summary,
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
