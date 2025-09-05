# database_models.py
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import asyncpg
from asyncpg import Pool
import json
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
logger = logging.getLogger("learning_tracker")


class DifficultyLevel(Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class ConceptStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    MASTERED = "mastered"
    NEEDS_REVIEW = "needs_review"


class ActivityType(Enum):
    FLASHCARD_PRACTICE = "flashcard_practice"
    QUIZ_ATTEMPT = "quiz_attempt"
    CONTENT_STUDY = "content_study"
    CONCEPT_REVIEW = "concept_review"


@dataclass
class DatabaseManager:
    pool: Optional[Pool] = None

    async def initialize_pool(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
        logger.info("Database connection pool initialized")

    async def close_pool(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def create_tables(self):
        """Create all required tables with support for large Google IDs"""
        async with self.pool.acquire() as conn:
            # Students table - changed student_id to VARCHAR for Google IDs
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id VARCHAR(100) PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    full_name VARCHAR(100) NOT NULL,
                    learning_preferences JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Subjects table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS subjects (
                    subject_id BIGSERIAL PRIMARY KEY,
                    subject_name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Chapters table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id BIGSERIAL PRIMARY KEY,
                    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
                    chapter_name VARCHAR(200) NOT NULL,
                    chapter_order BIGINT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(subject_id, chapter_name),
                    UNIQUE(subject_id, chapter_order)
                );
            """)

            # Concepts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    concept_id BIGSERIAL PRIMARY KEY,
                    chapter_id BIGINT REFERENCES chapters(chapter_id) ON DELETE CASCADE,
                    concept_name VARCHAR(200) NOT NULL,
                    concept_order INTEGER NOT NULL,
                    description TEXT,
                    difficulty_level VARCHAR(20) DEFAULT 'Medium',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chapter_id, concept_name),
                    UNIQUE(chapter_id, concept_order)
                );
            """)

            # Student Subject Enrollment - changed student_id to VARCHAR
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_subjects (
                    enrollment_id BIGSERIAL PRIMARY KEY,
                    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
                    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(student_id, subject_id)
                );
            """)

            # Student Concept Progress - changed student_id to VARCHAR
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_concept_progress (
                    progress_id BIGSERIAL PRIMARY KEY,
                    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id BIGINT REFERENCES concepts(concept_id) ON DELETE CASCADE,
                    status VARCHAR(20) DEFAULT 'not_started',
                    mastery_score DECIMAL(5,2) DEFAULT 0.00,
                    attempts_count INTEGER DEFAULT 0,
                    correct_answers INTEGER DEFAULT 0,
                    total_questions INTEGER DEFAULT 0,
                    last_practiced TIMESTAMP,
                    first_learned TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(student_id, concept_id)
                );
            """)

            # Student Weaknesses - changed student_id to VARCHAR
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_weaknesses (
                    weakness_id BIGSERIAL PRIMARY KEY,
                    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id BIGINT REFERENCES concepts(concept_id) ON DELETE CASCADE,
                    weakness_type VARCHAR(50) NOT NULL,
                    error_pattern TEXT,
                    frequency_count INTEGER DEFAULT 1,
                    severity_score DECIMAL(3,2) DEFAULT 0.00,
                    last_occurrence TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Learning Activities - changed student_id to VARCHAR
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_activities (
                    activity_id BIGSERIAL PRIMARY KEY,
                    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id BIGINT REFERENCES concepts(concept_id) ON DELETE CASCADE,
                    activity_type VARCHAR(50) NOT NULL,
                    activity_data JSONB NOT NULL,
                    score DECIMAL(5,2),
                    time_spent INTEGER, -- in seconds
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Personalized Recommendations - changed student_id to VARCHAR
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id BIGSERIAL PRIMARY KEY,
                    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id BIGINT REFERENCES concepts(concept_id),
                    recommendation_type VARCHAR(50) NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    description TEXT NOT NULL,
                    priority_score INTEGER DEFAULT 5,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_completed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                );
            """)

            # Study Sessions - changed student_id to VARCHAR
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    session_id SERIAL PRIMARY KEY,
                    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
                    session_data JSONB NOT NULL,
                    total_questions INTEGER DEFAULT 0,
                    correct_answers INTEGER DEFAULT 0,
                    session_duration INTEGER, -- in seconds
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Student Recommendations table (new addition for the service)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_recommendations (
                    id BIGSERIAL PRIMARY KEY,
                    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                    recommendation_type VARCHAR(100) NOT NULL,
                    subject VARCHAR(255),
                    chapter VARCHAR(255),
                    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
                    message TEXT NOT NULL,
                    suggested_actions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    order_index INTEGER DEFAULT 0,
                    is_read BOOLEAN DEFAULT FALSE,
                    is_dismissed BOOLEAN DEFAULT FALSE
                );
            """)

            # Create indexes for better performance
            await self._create_indexes(conn)
            logger.info(
                "All tables created successfully with Google ID support")

    async def _create_indexes(self, conn):
        """Create indexes for optimized queries"""
        indexes = [
            # Original indexes
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_student ON student_concept_progress(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_concept ON student_concept_progress(concept_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_weaknesses_student ON student_weaknesses(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_weaknesses_concept ON student_weaknesses(concept_id);",
            "CREATE INDEX IF NOT EXISTS idx_learning_activities_student ON learning_activities(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_learning_activities_concept ON learning_activities(concept_id);",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_student ON recommendations(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_study_sessions_student ON study_sessions(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_subject ON chapters(subject_id);",
            "CREATE INDEX IF NOT EXISTS idx_students_external_id ON students(student_id);",

            # New indexes for student_recommendations
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_student_id ON student_recommendations(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_priority ON student_recommendations(priority);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_created_at ON student_recommendations(created_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_type ON student_recommendations(recommendation_type);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_active ON student_recommendations(student_id, is_dismissed) WHERE is_dismissed = FALSE;",

            # Performance indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_learning_activities_completed_at ON learning_activities(completed_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_status ON student_concept_progress(status);",
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_last_practiced ON student_concept_progress(last_practiced DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_weaknesses_severity ON student_weaknesses(severity_score DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_subjects_active ON student_subjects(student_id) WHERE is_active = TRUE;"
        ]

        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")


# Core database operations
class StudentManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def create_or_get_student(self, student_id: str, username: str, email: str, full_name: str) -> str:
        """Create a new student or get existing student ID"""
        async with self.db.pool.acquire() as conn:
            # Try to get existing student
            student = await conn.fetchrow(
                "SELECT student_id FROM students WHERE student_id = $1 OR email = $2",
                student_id, email
            )

            if student:
                return student['student_id']

            # Create new student with provided Google ID
            try:
                await conn.execute(
                    """INSERT INTO students (student_id, username, email, full_name) 
                       VALUES ($1, $2, $3, $4)""",
                    student_id, username, email, full_name
                )
                logger.info(
                    f"Created new student with Google ID: {student_id}")
                return student_id
            except asyncpg.UniqueViolationError as e:
                # Handle race conditions or duplicate emails/usernames
                logger.warning(f"Duplicate student creation attempt: {e}")
                # Try to get the existing student
                existing = await conn.fetchrow(
                    "SELECT student_id FROM students WHERE student_id = $1 OR email = $2",
                    student_id, email
                )
                if existing:
                    return existing['student_id']
                raise

    async def update_learning_preferences(self, student_id: str, preferences: Dict[str, Any]):
        """Update student learning preferences"""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                "UPDATE students SET learning_preferences = $1, updated_at = CURRENT_TIMESTAMP WHERE student_id = $2",
                json.dumps(preferences), student_id
            )

    async def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student details by ID"""
        async with self.db.pool.acquire() as conn:
            student = await conn.fetchrow(
                "SELECT * FROM students WHERE student_id = $1",
                student_id
            )
            return dict(student) if student else None

    async def get_student_stats(self, student_id: str) -> Dict[str, Any]:
        """Get basic student statistics"""
        async with self.db.pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT ss.subject_id) as enrolled_subjects,
                    COUNT(DISTINCT scp.concept_id) as total_concepts_attempted,
                    COUNT(CASE WHEN scp.status = 'mastered' THEN 1 END) as mastered_concepts,
                    AVG(scp.mastery_score) as avg_mastery_score,
                    COUNT(DISTINCT la.activity_id) as total_activities
                FROM students s
                LEFT JOIN student_subjects ss ON s.student_id = ss.student_id AND ss.is_active = TRUE
                LEFT JOIN student_concept_progress scp ON s.student_id = scp.student_id
                LEFT JOIN learning_activities la ON s.student_id = la.student_id
                WHERE s.student_id = $1
                GROUP BY s.student_id
            """, student_id)

            return dict(stats) if stats else {}

    async def get_all_students(self) -> List[Dict[str, Any]]:
        """Get all students."""
        async with self.db.pool.acquire() as conn:
            students = await conn.fetch("SELECT student_id, username, email, full_name, learning_preferences, created_at, updated_at FROM students")
            return [dict(student) for student in students]

    async def update_student(self, student_id: str, full_name: Optional[str] = None, learning_preferences: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Update student details."""
        async with self.db.pool.acquire() as conn:
            updates = {}
            if full_name is not None:
                updates["full_name"] = full_name
            if learning_preferences is not None:
                updates["learning_preferences"] = json.dumps(
                    learning_preferences)

            if not updates:
                # No updates, return current student
                return await self.get_student_by_id(student_id)

            set_clauses = [f"{key} = ${i+1}" for i,
                           key in enumerate(updates.keys())]
            params = list(updates.values())
            params.append(student_id)  # student_id is the last parameter

            query = f"UPDATE students SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE student_id = ${len(params)} RETURNING *"

            updated_student = await conn.fetchrow(query, *params)
            return dict(updated_student) if updated_student else None

    async def delete_student(self, student_id: str) -> bool:
        """Delete a student by ID."""
        async with self.db.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM students WHERE student_id = $1", student_id)
            return result == "DELETE 1"


class SubjectManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def create_or_get_subject(self, subject_name: str, description: str = None) -> int:
        """Create a new subject or get existing subject ID"""
        async with self.db.pool.acquire() as conn:
            # Try to get existing subject
            subject = await conn.fetchrow(
                "SELECT subject_id FROM subjects WHERE subject_name = $1",
                subject_name
            )

            if subject:
                return subject['subject_id']

            # Create new subject
            subject_id = await conn.fetchval(
                "INSERT INTO subjects (subject_name, description) VALUES ($1, $2) RETURNING subject_id",
                subject_name, description
            )
            logger.info(f"Created new subject: {subject_name}")
            return subject_id

    async def create_or_get_chapter(self, subject_id: int, chapter_name: str, description: str = None) -> int:
        """Create a new chapter or get existing chapter ID"""
        async with self.db.pool.acquire() as conn:
            # Try to get existing chapter
            chapter = await conn.fetchrow(
                "SELECT chapter_id FROM chapters WHERE subject_id = $1 AND chapter_name = $2",
                subject_id, chapter_name
            )

            if chapter:
                return chapter['chapter_id']

            # Get next chapter order
            max_order = await conn.fetchval(
                "SELECT COALESCE(MAX(chapter_order), 0) FROM chapters WHERE subject_id = $1",
                subject_id
            )

            # Create new chapter
            chapter_id = await conn.fetchval(
                """INSERT INTO chapters (subject_id, chapter_name, chapter_order, description) 
                   VALUES ($1, $2, $3, $4) RETURNING chapter_id""",
                subject_id, chapter_name, max_order + 1, description
            )
            logger.info(
                f"Created new chapter: {chapter_name} in subject {subject_id}")
            return chapter_id

    async def create_or_get_concept(self, chapter_id: int, concept_name: str,
                                    difficulty_level: str = "Medium", description: str = None) -> int:
        """Create a new concept or get existing concept ID"""
        async with self.db.pool.acquire() as conn:
            # Try to get existing concept
            concept = await conn.fetchrow(
                "SELECT concept_id FROM concepts WHERE chapter_id = $1 AND concept_name = $2",
                chapter_id, concept_name
            )

            if concept:
                return concept['concept_id']

            # Get next concept order
            max_order = await conn.fetchval(
                "SELECT COALESCE(MAX(concept_order), 0) FROM concepts WHERE chapter_id = $1",
                chapter_id
            )

            # Create new concept
            concept_id = await conn.fetchval(
                """INSERT INTO concepts (chapter_id, concept_name, concept_order, difficulty_level, description) 
                   VALUES ($1, $2, $3, $4, $5) RETURNING concept_id""",
                chapter_id, concept_name, max_order + 1, difficulty_level, description
            )
            logger.info(
                f"Created new concept: {concept_name} in chapter {chapter_id}")
            return concept_id

    async def get_subject_hierarchy(self, subject_id: int) -> Dict[str, Any]:
        """Get complete subject hierarchy with chapters and concepts"""
        async with self.db.pool.acquire() as conn:
            # Get subject info
            subject = await conn.fetchrow(
                "SELECT * FROM subjects WHERE subject_id = $1", subject_id
            )
            if not subject:
                return {}

            # Get chapters
            chapters = await conn.fetch(
                "SELECT * FROM chapters WHERE subject_id = $1 ORDER BY chapter_order",
                subject_id
            )

            subject_data = dict(subject)
            subject_data['chapters'] = []

            for chapter in chapters:
                # Get concepts for each chapter
                concepts = await conn.fetch(
                    "SELECT * FROM concepts WHERE chapter_id = $1 ORDER BY concept_order",
                    chapter['chapter_id']
                )

                chapter_data = dict(chapter)
                chapter_data['concepts'] = [
                    dict(concept) for concept in concepts]
                subject_data['chapters'].append(chapter_data)

            return subject_data


class ProgressTracker:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def update_concept_progress(self, student_id: str, concept_id: int,
                                      correct_answers: int, total_questions: int):
        """Update student's progress on a specific concept"""
        async with self.db.pool.acquire() as conn:
            # Calculate scores
            accuracy = (correct_answers / total_questions) * \
                100 if total_questions > 0 else 0

            # Determine status based on performance
            if accuracy >= 90:
                status = ConceptStatus.MASTERED.value
            elif accuracy >= 70:
                status = ConceptStatus.IN_PROGRESS.value
            else:
                status = ConceptStatus.NEEDS_REVIEW.value

            # Update or insert progress
            await conn.execute("""
                INSERT INTO student_concept_progress 
                (student_id, concept_id, status, mastery_score, attempts_count, correct_answers, total_questions, last_practiced, first_learned)
                VALUES ($1, $2, $3, $4, 1, $5, $6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (student_id, concept_id) 
                DO UPDATE SET 
                    status = $3,
                    mastery_score = (student_concept_progress.mastery_score + $4) / 2,
                    attempts_count = student_concept_progress.attempts_count + 1,
                    correct_answers = student_concept_progress.correct_answers + $5,
                    total_questions = student_concept_progress.total_questions + $6,
                    last_practiced = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
            """, student_id, concept_id, status, accuracy, correct_answers, total_questions)

    async def record_weakness(self, student_id: str, concept_id: int,
                              weakness_type: str, error_pattern: str = None):
        """Record a student weakness for a specific concept"""
        async with self.db.pool.acquire() as conn:
            # Check if weakness already exists
            existing = await conn.fetchrow(
                """SELECT weakness_id, frequency_count FROM student_weaknesses 
                   WHERE student_id = $1 AND concept_id = $2 AND weakness_type = $3 AND is_resolved = FALSE""",
                student_id, concept_id, weakness_type
            )

            if existing:
                # Update frequency
                await conn.execute(
                    """UPDATE student_weaknesses 
                       SET frequency_count = frequency_count + 1, 
                           last_occurrence = CURRENT_TIMESTAMP,
                           severity_score = LEAST(severity_score + 0.1, 1.0),
                           updated_at = CURRENT_TIMESTAMP
                       WHERE weakness_id = $1""",
                    existing['weakness_id']
                )
            else:
                # Create new weakness record
                await conn.execute(
                    """INSERT INTO student_weaknesses 
                       (student_id, concept_id, weakness_type, error_pattern, severity_score)
                       VALUES ($1, $2, $3, $4, 0.3)""",
                    student_id, concept_id, weakness_type, error_pattern
                )

    async def record_learning_activity(self, student_id: str, concept_id: int,
                                       activity_type: str, activity_data: Dict[str, Any],
                                       score: float = None, time_spent: int = None):
        """Record a learning activity"""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO learning_activities 
                   (student_id, concept_id, activity_type, activity_data, score, time_spent)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                student_id, concept_id, activity_type, json.dumps(
                    activity_data), score, time_spent
            )

    async def get_concept_progress_history(self, student_id: str, concept_id: int) -> List[Dict[str, Any]]:
        """Get historical progress for a specific concept"""
        async with self.db.pool.acquire() as conn:
            activities = await conn.fetch("""
                SELECT activity_type, activity_data, score, time_spent, completed_at
                FROM learning_activities
                WHERE student_id = $1 AND concept_id = $2
                ORDER BY completed_at DESC
                LIMIT 50
            """, student_id, concept_id)

            return [dict(activity) for activity in activities]


class RecommendationEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def generate_personalized_recommendations(self, student_id: str) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on student progress and weaknesses"""
        async with self.db.pool.acquire() as conn:
            recommendations = []

            # Get concepts that need review
            weak_concepts = await conn.fetch("""
                SELECT c.concept_id, c.concept_name, ch.chapter_name, s.subject_name,
                       scp.mastery_score, sw.severity_score
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                LEFT JOIN student_weaknesses sw ON scp.student_id = sw.student_id 
                    AND scp.concept_id = sw.concept_id AND sw.is_resolved = FALSE
                WHERE scp.student_id = $1 
                    AND (scp.status = 'needs_review' OR scp.mastery_score < 70)
                ORDER BY COALESCE(sw.severity_score, 0) DESC, scp.mastery_score ASC
                LIMIT 10
            """, student_id)

            for concept in weak_concepts:
                priority = 10 if concept['severity_score'] and concept['severity_score'] > 0.7 else 7
                recommendations.append({
                    'recommendation_type': 'concept_review',
                    'concept_id': concept['concept_id'],
                    'title': f"Review {concept['concept_name']}",
                    'description': f"Focus on {concept['concept_name']} in {concept['chapter_name']} ({concept['subject_name']})",
                    'priority_score': priority
                })

            # Get concepts ready for advanced practice
            strong_concepts = await conn.fetch("""
                SELECT c.concept_id, c.concept_name, ch.chapter_name, s.subject_name
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

            for concept in strong_concepts:
                recommendations.append({
                    'recommendation_type': 'maintenance_practice',
                    'concept_id': concept['concept_id'],
                    'title': f"Practice {concept['concept_name']}",
                    'description': f"Keep your skills sharp with {concept['concept_name']} practice",
                    'priority_score': 4
                })

            return recommendations

    async def save_recommendations(self, student_id: str, recommendations: List[Dict[str, Any]]):
        """Save generated recommendations to database"""
        async with self.db.pool.acquire() as conn:
            # Clear old recommendations
            await conn.execute(
                "UPDATE recommendations SET is_active = FALSE WHERE student_id = $1",
                student_id
            )

            # Insert new recommendations
            for rec in recommendations:
                await conn.execute(
                    """INSERT INTO recommendations 
                       (student_id, concept_id, recommendation_type, title, description, priority_score)
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    student_id, rec.get(
                        'concept_id'), rec['recommendation_type'],
                    rec['title'], rec['description'], rec['priority_score']
                )

    async def get_active_recommendations(self, student_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get active recommendations for a student"""
        async with self.db.pool.acquire() as conn:
            recommendations = await conn.fetch("""
                SELECT r.*, c.concept_name, ch.chapter_name, s.subject_name
                FROM recommendations r
                LEFT JOIN concepts c ON r.concept_id = c.concept_id
                LEFT JOIN chapters ch ON c.chapter_id = ch.chapter_id
                LEFT JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE r.student_id = $1 AND r.is_active = TRUE AND r.is_completed = FALSE
                ORDER BY r.priority_score DESC, r.created_at DESC
                LIMIT $2
            """, student_id, limit)

            return [dict(rec) for rec in recommendations]


class AnalyticsManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def get_student_analytics(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive student analytics"""
        async with self.db.pool.acquire() as conn:
            # Overall progress
            overall_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_concepts,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_concepts,
                    COUNT(CASE WHEN status = 'needs_review' THEN 1 END) as weak_concepts,
                    AVG(mastery_score) as avg_mastery_score,
                    SUM(attempts_count) as total_attempts
                FROM student_concept_progress 
                WHERE student_id = $1
            """, student_id)

            # Subject-wise progress
            subject_progress = await conn.fetch("""
                SELECT 
                    s.subject_name,
                    COUNT(*) as total_concepts,
                    COUNT(CASE WHEN scp.status = 'mastered' THEN 1 END) as mastered_concepts,
                    AVG(scp.mastery_score) as avg_score
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE scp.student_id = $1
                GROUP BY s.subject_id, s.subject_name
                ORDER BY s.subject_name
            """, student_id)

            # Recent activity
            recent_activity = await conn.fetch("""
                SELECT activity_type, COUNT(*) as count,
                       AVG(score) as avg_score
                FROM learning_activities 
                WHERE student_id = $1 
                    AND completed_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
                GROUP BY activity_type
                ORDER BY count DESC
            """, student_id)

            # Active weaknesses
            active_weaknesses = await conn.fetch("""
                SELECT c.concept_name, ch.chapter_name, s.subject_name,
                       sw.weakness_type, sw.frequency_count, sw.severity_score
                FROM student_weaknesses sw
                JOIN concepts c ON sw.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE sw.student_id = $1 AND sw.is_resolved = FALSE
                ORDER BY sw.severity_score DESC
                LIMIT 20
            """, student_id)

            # Learning streak and consistency
            learning_streak = await conn.fetchrow("""
                WITH daily_activity AS (
                    SELECT DATE(completed_at) as activity_date,
                           COUNT(*) as activities_count
                    FROM learning_activities
                    WHERE student_id = $1
                    GROUP BY DATE(completed_at)
                    ORDER BY activity_date DESC
                ),
                streak_data AS (
                    SELECT activity_date,
                           ROW_NUMBER() OVER (ORDER BY activity_date DESC) as row_num,
                           activity_date - INTERVAL '1 day' * ROW_NUMBER() OVER (ORDER BY activity_date DESC) as streak_group
                    FROM daily_activity
                ),
                current_streak AS (
                    SELECT COUNT(*) as streak_days
                    FROM streak_data
                    WHERE streak_group = (
                        SELECT streak_group 
                        FROM streak_data 
                        WHERE row_num = 1
                    )
                )
                SELECT 
                    COALESCE((SELECT streak_days FROM current_streak), 0) as current_streak,
                    COUNT(DISTINCT DATE(la.completed_at)) as total_active_days,
                    ROUND(AVG(daily_counts.daily_activities), 2) as avg_daily_activities
                FROM learning_activities la
                LEFT JOIN (
                    SELECT DATE(completed_at) as date, COUNT(*) as daily_activities
                    FROM learning_activities
                    WHERE student_id = $1
                    GROUP BY DATE(completed_at)
                ) daily_counts ON DATE(la.completed_at) = daily_counts.date
                WHERE la.student_id = $1
            """, student_id)

            return {
                'overall_stats': dict(overall_stats) if overall_stats else {},
                'subject_progress': [dict(row) for row in subject_progress],
                'recent_activity': [dict(row) for row in recent_activity],
                'active_weaknesses': [dict(row) for row in active_weaknesses],
                'learning_streak': dict(learning_streak) if learning_streak else {}
            }

    async def get_performance_trends(self, student_id: str, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time"""
        async with self.db.pool.acquire() as conn:
            # Daily performance trends
            daily_trends = await conn.fetch("""
                SELECT 
                    DATE(completed_at) as date,
                    COUNT(*) as total_activities,
                    AVG(score) as avg_score,
                    COUNT(DISTINCT concept_id) as concepts_practiced
                FROM learning_activities
                WHERE student_id = $1 
                    AND completed_at >= CURRENT_DATE - INTERVAL '%s days'
                    AND score IS NOT NULL
                GROUP BY DATE(completed_at)
                ORDER BY date DESC
            """ % days, student_id)

            # Subject performance trends
            subject_trends = await conn.fetch("""
                SELECT 
                    s.subject_name,
                    DATE(la.completed_at) as date,
                    AVG(la.score) as avg_score,
                    COUNT(*) as activity_count
                FROM learning_activities la
                JOIN concepts c ON la.concept_id = c.concept_id
                JOIN chapters ch ON c.chapter_id = ch.chapter_id
                JOIN subjects s ON ch.subject_id = s.subject_id
                WHERE la.student_id = $1 
                    AND la.completed_at >= CURRENT_DATE - INTERVAL '%s days'
                    AND la.score IS NOT NULL
                GROUP BY s.subject_name, DATE(la.completed_at)
                ORDER BY s.subject_name, date DESC
            """ % days, student_id)

            return {
                'daily_trends': [dict(row) for row in daily_trends],
                'subject_trends': [dict(row) for row in subject_trends],
                'period_days': days
            }


# Session management for RAG integration
class StudySessionManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def start_study_session(self, student_id: str, subject_id: int, session_metadata: Dict[str, Any] = None) -> int:
        """Start a new study session"""
        async with self.db.pool.acquire() as conn:
            session_data = session_metadata or {}
            session_id = await conn.fetchval(
                """INSERT INTO study_sessions (student_id, subject_id, session_data, started_at)
                   VALUES ($1, $2, $3, CURRENT_TIMESTAMP) RETURNING session_id""",
                student_id, subject_id, json.dumps(session_data)
            )
            return session_id

    async def end_study_session(self, session_id: int, session_data: Dict[str, Any],
                                total_questions: int, correct_answers: int, duration: int):
        """End a study session with results"""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """UPDATE study_sessions 
                   SET session_data = $1, total_questions = $2, correct_answers = $3, 
                       session_duration = $4, completed_at = CURRENT_TIMESTAMP
                   WHERE session_id = $5""",
                json.dumps(
                    session_data), total_questions, correct_answers, duration, session_id
            )

    async def get_session_history(self, student_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get study session history for a student"""
        async with self.db.pool.acquire() as conn:
            sessions = await conn.fetch("""
                SELECT ss.*, s.subject_name
                FROM study_sessions ss
                JOIN subjects s ON ss.subject_id = s.subject_id
                WHERE ss.student_id = $1
                ORDER BY ss.started_at DESC
                LIMIT $2
            """, student_id, limit)

            return [dict(session) for session in sessions]

    async def get_session_statistics(self, student_id: str) -> Dict[str, Any]:
        """Get overall session statistics for a student"""
        async with self.db.pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(session_duration) as total_study_time,
                    AVG(session_duration) as avg_session_duration,
                    SUM(total_questions) as total_questions_attempted,
                    SUM(correct_answers) as total_correct_answers,
                    AVG(CASE WHEN total_questions > 0 THEN (correct_answers::float / total_questions) * 100 ELSE 0 END) as avg_accuracy
                FROM study_sessions
                WHERE student_id = $1 AND completed_at IS NOT NULL
            """, student_id)

            return dict(stats) if stats else {}


# Advanced Analytics and Reporting
class AdvancedAnalytics:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def get_learning_velocity(self, student_id: str, days: int = 30) -> Dict[str, Any]:
        """Calculate learning velocity - concepts mastered per time period"""
        async with self.db.pool.acquire() as conn:
            velocity_data = await conn.fetchrow("""
                WITH mastery_timeline AS (
                    SELECT 
                        DATE(updated_at) as mastery_date,
                        COUNT(*) as concepts_mastered
                    FROM student_concept_progress
                    WHERE student_id = $1 
                        AND status = 'mastered'
                        AND updated_at >= CURRENT_DATE - INTERVAL '%s days'
                    GROUP BY DATE(updated_at)
                )
                SELECT 
                    COUNT(*) as total_concepts_mastered,
                    ROUND(COUNT(*)::numeric / %s, 2) as concepts_per_day,
                    MAX(concepts_mastered) as best_day_count,
                    ROUND(AVG(concepts_mastered), 2) as avg_concepts_per_active_day
                FROM mastery_timeline
            """ % (days, days), student_id)

            return dict(velocity_data) if velocity_data else {}

    async def get_difficulty_progression(self, student_id: str) -> Dict[str, Any]:
        """Analyze student's progression through difficulty levels"""
        async with self.db.pool.acquire() as conn:
            difficulty_stats = await conn.fetch("""
                SELECT 
                    c.difficulty_level,
                    COUNT(*) as total_concepts,
                    COUNT(CASE WHEN scp.status = 'mastered' THEN 1 END) as mastered_concepts,
                    AVG(scp.mastery_score) as avg_mastery_score,
                    AVG(scp.attempts_count) as avg_attempts
                FROM student_concept_progress scp
                JOIN concepts c ON scp.concept_id = c.concept_id
                WHERE scp.student_id = $1
                GROUP BY c.difficulty_level
                ORDER BY 
                    CASE c.difficulty_level 
                        WHEN 'Easy' THEN 1 
                        WHEN 'Medium' THEN 2 
                        WHEN 'Hard' THEN 3 
                        ELSE 4 
                    END
            """, student_id)

            return {
                'difficulty_breakdown': [dict(row) for row in difficulty_stats],
                'analysis': {
                    'ready_for_advanced': any(
                        row['mastery_score'] > 85 and row['difficulty_level'] == 'Medium'
                        for row in difficulty_stats
                    ),
                    'needs_foundation_work': any(
                        row['mastery_score'] < 60 and row['difficulty_level'] == 'Easy'
                        for row in difficulty_stats
                    )
                }
            }

    async def identify_knowledge_gaps(self, student_id: str) -> List[Dict[str, Any]]:
        """Identify knowledge gaps and prerequisites"""
        async with self.db.pool.acquire() as conn:
            gaps = await conn.fetch("""
                WITH concept_hierarchy AS (
                    SELECT 
                        c1.concept_id as current_concept,
                        c1.concept_name as current_name,
                        c1.concept_order as current_order,
                        c2.concept_id as prerequisite_concept,
                        c2.concept_name as prerequisite_name,
                        c2.concept_order as prerequisite_order,
                        ch.chapter_name,
                        s.subject_name
                    FROM concepts c1
                    JOIN concepts c2 ON c1.chapter_id = c2.chapter_id AND c2.concept_order < c1.concept_order
                    JOIN chapters ch ON c1.chapter_id = ch.chapter_id
                    JOIN subjects s ON ch.subject_id = s.subject_id
                ),
                student_mastery AS (
                    SELECT concept_id, status, mastery_score
                    FROM student_concept_progress
                    WHERE student_id = $1
                )
                SELECT DISTINCT
                    ch.current_name,
                    ch.prerequisite_name,
                    ch.chapter_name,
                    ch.subject_name,
                    COALESCE(sm1.mastery_score, 0) as current_mastery,
                    COALESCE(sm2.mastery_score, 0) as prerequisite_mastery,
                    COALESCE(sm1.status, 'not_started') as current_status,
                    COALESCE(sm2.status, 'not_started') as prerequisite_status
                FROM concept_hierarchy ch
                LEFT JOIN student_mastery sm1 ON ch.current_concept = sm1.concept_id
                LEFT JOIN student_mastery sm2 ON ch.prerequisite_concept = sm2.concept_id
                WHERE (sm1.mastery_score > 0 OR sm1.status != 'not_started')
                    AND (sm2.mastery_score < 70 OR sm2.status = 'needs_review' OR sm2.status = 'not_started')
                ORDER BY ch.subject_name, ch.chapter_name, ch.current_order
            """, student_id)

            return [dict(gap) for gap in gaps]


# Integration function for RAG system
async def process_rag_response(student_id: str, subject_name: str, chapter_name: str,
                               concept_name: str, rag_response: Dict[str, Any],
                               db_manager: DatabaseManager) -> Dict[str, Any]:
    """Process RAG response and update student progress tracking"""

    subject_manager = SubjectManager(db_manager)
    progress_tracker = ProgressTracker(db_manager)

    # Create/get subject, chapter, concept hierarchy
    subject_id = await subject_manager.create_or_get_subject(subject_name)
    chapter_id = await subject_manager.create_or_get_chapter(subject_id, chapter_name)
    concept_id = await subject_manager.create_or_get_concept(chapter_id, concept_name)

    # Enroll student in subject if not already enrolled
    async with db_manager.pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO student_subjects (student_id, subject_id) 
               VALUES ($1, $2) ON CONFLICT (student_id, subject_id) DO NOTHING""",
            student_id, subject_id
        )

    # Record study activity
    await progress_tracker.record_learning_activity(
        student_id, concept_id, ActivityType.CONTENT_STUDY.value,
        {'rag_response': rag_response, 'timestamp': datetime.now().isoformat()}
    )

    # Add concept metadata to response
    enhanced_response = rag_response.copy()
    enhanced_response['tracking_metadata'] = {
        'student_id': student_id,
        'subject_id': subject_id,
        'chapter_id': chapter_id,
        'concept_id': concept_id,
        'subject_name': subject_name,
        'chapter_name': chapter_name,
        'concept_name': concept_name
    }

    return enhanced_response


# Database initialization and utility functions
async def initialize_database():
    """Initialize database with all tables and indexes"""
    db_manager = DatabaseManager()
    await db_manager.initialize_pool()
    await db_manager.create_tables()
    logger.info("Database initialized successfully with Google ID support")
    return db_manager


async def cleanup_database(db_manager: DatabaseManager):
    """Cleanup old data and optimize database"""
    async with db_manager.pool.acquire() as conn:
        # Clean up old learning activities (older than 6 months)
        deleted_activities = await conn.fetchval("""
            DELETE FROM learning_activities 
            WHERE completed_at < CURRENT_TIMESTAMP - INTERVAL '6 months'
            RETURNING COUNT(*)
        """)

        # Clean up resolved weaknesses (older than 3 months)
        deleted_weaknesses = await conn.fetchval("""
            DELETE FROM student_weaknesses 
            WHERE is_resolved = TRUE 
                AND updated_at < CURRENT_TIMESTAMP - INTERVAL '3 months'
            RETURNING COUNT(*)
        """)

        # Clean up expired recommendations
        deleted_recommendations = await conn.fetchval("""
            DELETE FROM recommendations 
            WHERE expires_at < CURRENT_TIMESTAMP
            RETURNING COUNT(*)
        """)

        logger.info(
            f"Cleanup completed: {deleted_activities} activities, {deleted_weaknesses} weaknesses, {deleted_recommendations} recommendations deleted")


async def migrate_existing_data(db_manager: DatabaseManager):
    """Migrate existing data if schema changes are needed"""
    async with db_manager.pool.acquire() as conn:
        # Example: Add missing columns if they don't exist
        try:
            await conn.execute("""
                ALTER TABLE students 
                ADD COLUMN IF NOT EXISTS last_login TIMESTAMP,
                ADD COLUMN IF NOT EXISTS total_study_time INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS preferred_difficulty VARCHAR(20) DEFAULT 'Medium'
            """)
            logger.info("Schema migration completed successfully")
        except Exception as e:
            logger.warning(f"Schema migration warning: {e}")


# Example usage and testing functions
async def example_usage():
    """Example of how to use the database system"""
    db_manager = await initialize_database()

    try:
        # Create managers
        student_manager = StudentManager(db_manager)
        subject_manager = SubjectManager(db_manager)
        progress_tracker = ProgressTracker(db_manager)
        recommendation_engine = RecommendationEngine(db_manager)
        analytics_manager = AnalyticsManager(db_manager)
        session_manager = StudySessionManager(db_manager)
        advanced_analytics = AdvancedAnalytics(db_manager)

        # Create a student with Google ID
        google_id = "117555238368884389018"
        student_id = await student_manager.create_or_get_student(
            google_id, "john_doe", "john@example.com", "John Doe"
        )

        # Process a RAG response (example)
        enhanced_response = await process_rag_response(
            student_id, "Mathematics", "Algebra", "Linear Equations",
            {"flashcards": {"card1": {"question": "What is a linear equation?",
                                      "answer": "An equation of the first degree",
                                      "difficulty": "Easy"}}},
            db_manager
        )

        # Update progress (simulate quiz results)
        concept_id = enhanced_response['tracking_metadata']['concept_id']
        await progress_tracker.update_concept_progress(student_id, concept_id, 7, 10)

        # Record a weakness
        await progress_tracker.record_weakness(
            student_id, concept_id, "algebraic_manipulation",
            "Difficulty with variable isolation"
        )

        # Start and end a study session
        subject_id = enhanced_response['tracking_metadata']['subject_id']
        session_id = await session_manager.start_study_session(
            student_id, subject_id, {"session_type": "practice"}
        )
        await session_manager.end_study_session(session_id, {"completed": True}, 10, 7, 1800)

        # Generate recommendations
        recommendations = await recommendation_engine.generate_personalized_recommendations(student_id)
        await recommendation_engine.save_recommendations(student_id, recommendations)

        # Get comprehensive analytics
        analytics = await analytics_manager.get_student_analytics(student_id)
        performance_trends = await analytics_manager.get_performance_trends(student_id, 30)
        learning_velocity = await advanced_analytics.get_learning_velocity(student_id, 30)
        knowledge_gaps = await advanced_analytics.identify_knowledge_gaps(student_id)

        # Print results
        print("Student Analytics:", json.dumps(
            analytics, indent=2, default=str))
        print("\nPerformance Trends:", json.dumps(
            performance_trends, indent=2, default=str))
        print("\nLearning Velocity:", json.dumps(
            learning_velocity, indent=2, default=str))
        print("\nKnowledge Gaps:", json.dumps(
            knowledge_gaps, indent=2, default=str))

    finally:
        await db_manager.close_pool()


# Database health check
async def health_check(db_manager: DatabaseManager) -> Dict[str, Any]:
    """Perform database health check"""
    try:
        async with db_manager.pool.acquire() as conn:
            # Test basic connectivity
            result = await conn.fetchval("SELECT 1")

            # Get table counts
            tables = ['students', 'subjects', 'chapters', 'concepts',
                      'student_concept_progress', 'learning_activities',
                      'recommendations', 'study_sessions']

            table_stats = {}
            for table in tables:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                table_stats[table] = count

            # Get database size
            db_size = await conn.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)

            return {
                'status': 'healthy',
                'connection': 'ok' if result == 1 else 'failed',
                'table_stats': table_stats,
                'database_size': db_size,
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    asyncio.run(example_usage())
