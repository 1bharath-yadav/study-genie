"""
Pure Functional Analytics Service
Provides analytics data through functional composition
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import reduce
import logging
from dataclasses import dataclass

from app.services.db import (
    get_student_by_id
)
from app.db.db_client import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticsTimeRange:
    """Immutable time range for analytics queries"""
    start_date: datetime
    end_date: datetime
    days: int


@dataclass(frozen=True)
class SubjectAnalytics:
    """Immutable subject analytics data"""
    subject_id: int
    subject_name: str
    mastery_percentage: float
    quiz_accuracy: float
    time_spent: int
    concepts_total: int
    concepts_mastered: int
    last_activity: Optional[datetime]


@dataclass(frozen=True)
class ConceptProgress:
    """Immutable concept progress data"""
    concept_id: int
    concept_name: str
    chapter_name: str
    subject_name: str
    mastery_score: float
    attempts_count: int
    correct_answers: int
    total_questions: int
    accuracy: float


@dataclass(frozen=True)
class WeeklyTrend:
    """Immutable weekly trend data"""
    week: str
    average_score: float
    concepts_learned: int
    time_spent: int
    quiz_accuracy: float


@dataclass(frozen=True)
class StudyPattern:
    """Immutable study pattern analysis"""
    daily_average_time: int
    peak_hours: List[int]
    consistency_score: float
    study_streak: int
    session_patterns: Dict[str, Any]


# Pure Functions for Data Transformation

def create_time_range(days: int) -> AnalyticsTimeRange:
    """Create immutable time range for analytics"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return AnalyticsTimeRange(start_date, end_date, days)


def calculate_mastery_percentage(correct: int, total: int) -> float:
    """Pure function to calculate mastery percentage"""
    return (correct / max(1, total)) * 100


def calculate_accuracy(correct: int, total: int) -> float:
    """Pure function to calculate accuracy"""
    return (correct / max(1, total)) * 100


def group_by_subject(concept_list: List[Dict]) -> Dict[str, List[Dict]]:
    """Pure function to group concepts by subject"""
    return reduce(
        lambda acc, concept: {
            **acc,
            concept.get('subject_name', 'Unknown'): 
                acc.get(concept.get('subject_name', 'Unknown'), []) + [concept]
        },
        concept_list,
        {}
    )


def calculate_trend(scores: List[float]) -> str:
    """Pure function to calculate trend from scores"""
    if len(scores) < 2:
        return "stable"
    
    first_half = scores[:len(scores)//2]
    second_half = scores[len(scores)//2:]
    
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    
    if second_avg > first_avg * 1.05:
        return "improving"
    elif second_avg < first_avg * 0.95:
        return "declining"
    else:
        return "stable"


# Database Query Functions

async def get_student_progress_data(student_id: str, time_range: AnalyticsTimeRange) -> Dict[str, Any]:
    """Get raw student progress data from database"""
    try:
        supabase = get_supabase_client()
        
        # Get concept progress with related data
        concept_query = """
        SELECT 
            cp.concept_id,
            cp.mastery_score,
            cp.attempts_count,
            cp.correct_answers,
            cp.total_questions,
            cp.last_practiced,
            c.concept_name,
            ch.chapter_name,
            s.subject_name
        FROM concept_progress cp
        JOIN concepts c ON cp.concept_id = c.concept_id
        JOIN chapters ch ON c.chapter_id = ch.chapter_id
        JOIN subjects s ON ch.subject_id = s.subject_id
        WHERE cp.student_id = %s
        AND cp.updated_at >= %s
        ORDER BY cp.mastery_score DESC
        """
        
        concept_response = supabase.rpc(
            'execute_sql',
            {
                'query': concept_query,
                'params': [student_id, time_range.start_date.isoformat()]
            }
        ).execute()
        
        return {
            'concepts': concept_response.data if concept_response.data else [],
            'time_range': time_range
        }
    except Exception as e:
        logger.error(f"Error fetching student progress: {e}")
        return {'concepts': [], 'time_range': time_range}


async def get_learning_activities_data(student_id: str, time_range: AnalyticsTimeRange) -> Dict[str, Any]:
    """Get learning activities data"""
    try:
        supabase = get_supabase_client()
        
        activities_query = """
        SELECT 
            la.activity_type,
            la.score,
            la.time_spent,
            la.completed_at,
            c.concept_name,
            s.subject_name
        FROM learning_activities la
        JOIN concepts c ON la.concept_id = c.concept_id
        JOIN chapters ch ON c.chapter_id = ch.chapter_id
        JOIN subjects s ON ch.subject_id = s.subject_id
        WHERE la.student_id = %s
        AND la.completed_at >= %s
        ORDER BY la.completed_at DESC
        """
        
        activities_response = supabase.rpc(
            'execute_sql',
            {
                'query': activities_query,
                'params': [student_id, time_range.start_date.isoformat()]
            }
        ).execute()
        
        return {
            'activities': activities_response.data if activities_response.data else [],
            'time_range': time_range
        }
    except Exception as e:
        logger.error(f"Error fetching learning activities: {e}")
        return {'activities': [], 'time_range': time_range}


# Analytics Computation Functions

def compute_subject_analytics(concepts_data: List[Dict]) -> List[SubjectAnalytics]:
    """Pure function to compute subject analytics from concepts data"""
    subjects_grouped = group_by_subject(concepts_data)
    
    subject_analytics = []
    for subject_name, concepts in subjects_grouped.items():
        if not concepts:
            continue
            
        total_concepts = len(concepts)
        mastered_concepts = len([c for c in concepts if c.get('mastery_score', 0) >= 70])
        
        total_correct = sum(c.get('correct_answers', 0) for c in concepts)
        total_questions = sum(c.get('total_questions', 0) for c in concepts)
        
        mastery_percentage = (mastered_concepts / total_concepts) * 100
        quiz_accuracy = calculate_accuracy(total_correct, total_questions)
        
        # Get the first concept to extract subject_id (assuming all concepts have same subject_id)
        subject_id = concepts[0].get('subject_id', 0)
        
        last_activities = []
        for c in concepts:
            last_practiced = c.get('last_practiced')
            if last_practiced and isinstance(last_practiced, str):
                try:
                    last_activities.append(datetime.fromisoformat(last_practiced))
                except ValueError:
                    continue
            elif isinstance(last_practiced, datetime):
                last_activities.append(last_practiced)
        
        last_activity = max(last_activities) if last_activities else None
        
        subject_analytics.append(SubjectAnalytics(
            subject_id=subject_id,
            subject_name=subject_name,
            mastery_percentage=mastery_percentage,
            quiz_accuracy=quiz_accuracy,
            time_spent=0,  # To be calculated from activities data
            concepts_total=total_concepts,
            concepts_mastered=mastered_concepts,
            last_activity=last_activity
        ))
    
    return subject_analytics


def compute_concept_progress(concepts_data: List[Dict]) -> List[ConceptProgress]:
    """Pure function to compute concept progress"""
    return [
        ConceptProgress(
            concept_id=concept.get('concept_id', 0),
            concept_name=concept.get('concept_name', 'Unknown'),
            chapter_name=concept.get('chapter_name', 'Unknown'),
            subject_name=concept.get('subject_name', 'Unknown'),
            mastery_score=concept.get('mastery_score', 0.0),
            attempts_count=concept.get('attempts_count', 0),
            correct_answers=concept.get('correct_answers', 0),
            total_questions=concept.get('total_questions', 0),
            accuracy=calculate_accuracy(
                concept.get('correct_answers', 0),
                concept.get('total_questions', 0)
            )
        )
        for concept in concepts_data
    ]


def compute_weekly_trends(activities_data: List[Dict], weeks: int = 4) -> Tuple[List[WeeklyTrend], str]:
    """Pure function to compute weekly trends"""
    weekly_data = []
    
    for i in range(weeks):
        week_start = datetime.now() - timedelta(weeks=weeks-i-1)
        week_end = week_start + timedelta(days=7)
        
        # Filter activities for this week
        week_activities = [
            activity for activity in activities_data
            if week_start <= datetime.fromisoformat(activity.get('completed_at', '')) < week_end
        ]
        
        if week_activities:
            scores = [activity.get('score', 0) for activity in week_activities if activity.get('score')]
            time_spent = sum(activity.get('time_spent', 0) for activity in week_activities)
            concepts_count = len(set(activity.get('concept_name') for activity in week_activities))
            
            avg_score = sum(scores) / len(scores) if scores else 0
            
            weekly_data.append(WeeklyTrend(
                week=week_start.strftime("%Y-W%U"),
                average_score=avg_score,
                concepts_learned=concepts_count,
                time_spent=time_spent,
                quiz_accuracy=avg_score  # Using same as average score for simplicity
            ))
        else:
            # Default data for weeks with no activity
            weekly_data.append(WeeklyTrend(
                week=week_start.strftime("%Y-W%U"),
                average_score=0,
                concepts_learned=0,
                time_spent=0,
                quiz_accuracy=0
            ))
    
    # Calculate trend
    scores = [week.average_score for week in weekly_data]
    trend = calculate_trend(scores)
    
    return weekly_data, trend


# High-level Analytics Functions

async def resolve_student_id(student_identifier: str) -> Optional[str]:
    """Resolve student identifier (email or ID) to actual student_id"""
    try:
        # Try direct lookup first
        student = get_student_by_id(student_identifier)
        if student:
            return student_identifier
        
        logger.warning(f"Could not resolve student identifier: {student_identifier}")
        return None
    except Exception as e:
        logger.error(f"Error resolving student ID: {e}")
        return None


async def get_dashboard_analytics(student_id: str, days: int = 30) -> Dict[str, Any]:
    """Get comprehensive dashboard analytics"""
    try:
        time_range = create_time_range(days)
        
        # Get raw data
        progress_data = await get_student_progress_data(student_id, time_range)
        activities_data = await get_learning_activities_data(student_id, time_range)
        
        # Compute analytics
        subject_analytics = compute_subject_analytics(progress_data['concepts'])
        concept_progress = compute_concept_progress(progress_data['concepts'])
        
        # Calculate overall metrics
        total_concepts = len(concept_progress)
        mastered_concepts = len([c for c in concept_progress if c.mastery_score >= 70])
        overall_mastery = (mastered_concepts / max(1, total_concepts)) * 100
        
        total_time_spent = sum(activity.get('time_spent', 0) for activity in activities_data['activities'])
        
        return {
            'student_id': student_id,
            'time_range_days': days,
            'overall_mastery_percentage': overall_mastery,
            'total_concepts': total_concepts,
            'mastered_concepts': mastered_concepts,
            'total_time_spent': total_time_spent,
            'subjects_analytics': {
                subject.subject_name: {
                    'mastery_percentage': subject.mastery_percentage,
                    'quiz_accuracy': subject.quiz_accuracy,
                    'concepts_total': subject.concepts_total,
                    'concepts_mastered': subject.concepts_mastered,
                    'last_activity': subject.last_activity.isoformat() if subject.last_activity else None
                }
                for subject in subject_analytics
            },
            'subjects_summary': [
                {
                    'subject_name': subject.subject_name,
                    'mastery_percentage': subject.mastery_percentage,
                    'concepts_total': subject.concepts_total,
                    'concepts_mastered': subject.concepts_mastered,
                    'time_spent': subject.time_spent
                }
                for subject in subject_analytics
            ],
            'concept_progress': [
                {
                    'concept_name': concept.concept_name,
                    'subject_name': concept.subject_name,
                    'chapter_name': concept.chapter_name,
                    'mastery_score': concept.mastery_score,
                    'accuracy': concept.accuracy
                }
                for concept in concept_progress
            ]
        }
    except Exception as e:
        logger.error(f"Error computing dashboard analytics: {e}")
        return {
            'student_id': student_id,
            'error': str(e),
            'subjects_analytics': {},
            'subjects_summary': [],
            'concept_progress': []
        }


async def get_weekly_trends_analytics(student_id: str, weeks: int = 4) -> Dict[str, Any]:
    """Get weekly trends analytics"""
    try:
        time_range = create_time_range(weeks * 7)
        activities_data = await get_learning_activities_data(student_id, time_range)
        
        weekly_trends, trend = compute_weekly_trends(activities_data['activities'], weeks)
        
        # Calculate improvement rate
        if len(weekly_trends) >= 2:
            first_score = weekly_trends[0].average_score
            last_score = weekly_trends[-1].average_score
            improvement_rate = ((last_score - first_score) / max(1, first_score)) * 100
        else:
            improvement_rate = 0
        
        return {
            'weekly_progress': [
                {
                    'week': week.week,
                    'average_score': week.average_score,
                    'concepts_learned': week.concepts_learned,
                    'time_spent': week.time_spent,
                    'quiz_accuracy': week.quiz_accuracy
                }
                for week in weekly_trends
            ],
            'trend': trend,
            'improvement_rate': round(improvement_rate, 1)
        }
    except Exception as e:
        logger.error(f"Error computing weekly trends: {e}")
        return {
            'weekly_progress': [],
            'trend': 'stable',
            'improvement_rate': 0
        }


async def get_weakness_analysis(student_id: str) -> Dict[str, Any]:
    """Get weakness analysis for student"""
    try:
        time_range = create_time_range(30)  # Last 30 days
        progress_data = await get_student_progress_data(student_id, time_range)
        
        concept_progress = compute_concept_progress(progress_data['concepts'])
        subject_analytics = compute_subject_analytics(progress_data['concepts'])
        
        # Find weak concepts (mastery < 60%)
        weak_concepts = [
            {
                'concept_name': concept.concept_name,
                'subject_name': concept.subject_name,
                'chapter_name': concept.chapter_name,
                'mastery_score': concept.mastery_score,
                'accuracy': concept.accuracy,
                'total_attempts': concept.attempts_count
            }
            for concept in concept_progress
            if concept.mastery_score < 60
        ]
        
        # Find weak subjects (mastery < 60%)
        weak_subjects = [
            {
                'subject_name': subject.subject_name,
                'mastery_percentage': subject.mastery_percentage,
                'quiz_accuracy': subject.quiz_accuracy,
                'concepts_total': subject.concepts_total,
                'concepts_mastered': subject.concepts_mastered
            }
            for subject in subject_analytics
            if subject.mastery_percentage < 60
        ]
        
        # Generate improvement suggestions
        improvement_areas = []
        for subject in weak_subjects[:3]:  # Top 3 weak subjects
            priority = "high" if subject['mastery_percentage'] < 30 else "medium"
            suggestion = f"Focus on {subject['subject_name']} - current mastery: {subject['mastery_percentage']:.1f}%"
            
            improvement_areas.append({
                'type': 'subject',
                'subject': subject['subject_name'],
                'priority': priority,
                'suggestion': suggestion,
                'metrics': {
                    'mastery_percentage': subject['mastery_percentage'],
                    'quiz_accuracy': subject['quiz_accuracy']
                }
            })
        
        return {
            'weak_concepts': weak_concepts,
            'weak_subjects': weak_subjects,
            'improvement_areas': improvement_areas,
            'needs_attention': len(weak_concepts) + len(weak_subjects),
            'priority_actions': improvement_areas[:3],
            'overall_weakness_score': len(weak_concepts) + (len(weak_subjects) * 2)
        }
    except Exception as e:
        logger.error(f"Error computing weakness analysis: {e}")
        return {
            'weak_concepts': [],
            'weak_subjects': [],
            'improvement_areas': [],
            'needs_attention': 0,
            'priority_actions': [],
            'overall_weakness_score': 0
        }
