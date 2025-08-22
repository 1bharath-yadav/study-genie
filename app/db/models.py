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
        """Create all required tables with 4NF compliance"""
        async with self.pool.acquire() as conn:
            # Students table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id SERIAL PRIMARY KEY,
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
                    subject_id SERIAL PRIMARY KEY,
                    subject_name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Chapters table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chapters (
                    chapter_id SERIAL PRIMARY KEY,
                    subject_id INTEGER REFERENCES subjects(subject_id) ON DELETE CASCADE,
                    chapter_name VARCHAR(200) NOT NULL,
                    chapter_order INTEGER NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(subject_id, chapter_name),
                    UNIQUE(subject_id, chapter_order)
                );
            """)

            # Concepts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    concept_id SERIAL PRIMARY KEY,
                    chapter_id INTEGER REFERENCES chapters(chapter_id) ON DELETE CASCADE,
                    concept_name VARCHAR(200) NOT NULL,
                    concept_order INTEGER NOT NULL,
                    description TEXT,
                    difficulty_level VARCHAR(20) DEFAULT 'Medium',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chapter_id, concept_name),
                    UNIQUE(chapter_id, concept_order)
                );
            """)

            # Student Subject Enrollment
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_subjects (
                    enrollment_id SERIAL PRIMARY KEY,
                    student_id INTEGER REFERENCES students(student_id) ON DELETE CASCADE,
                    subject_id INTEGER REFERENCES subjects(subject_id) ON DELETE CASCADE,
                    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(student_id, subject_id)
                );
            """)

            # Student Concept Progress
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_concept_progress (
                    progress_id SERIAL PRIMARY KEY,
                    student_id INTEGER REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id INTEGER REFERENCES concepts(concept_id) ON DELETE CASCADE,
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

            # Student Weaknesses
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS student_weaknesses (
                    weakness_id SERIAL PRIMARY KEY,
                    student_id INTEGER REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id INTEGER REFERENCES concepts(concept_id) ON DELETE CASCADE,
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

            # Learning Activities
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_activities (
                    activity_id SERIAL PRIMARY KEY,
                    student_id INTEGER REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id INTEGER REFERENCES concepts(concept_id) ON DELETE CASCADE,
                    activity_type VARCHAR(50) NOT NULL,
                    activity_data JSONB NOT NULL,
                    score DECIMAL(5,2),
                    time_spent INTEGER, -- in seconds
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Personalized Recommendations
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id SERIAL PRIMARY KEY,
                    student_id INTEGER REFERENCES students(student_id) ON DELETE CASCADE,
                    concept_id INTEGER REFERENCES concepts(concept_id),
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

            # Study Sessions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    session_id SERIAL PRIMARY KEY,
                    student_id INTEGER REFERENCES students(student_id) ON DELETE CASCADE,
                    subject_id INTEGER REFERENCES subjects(subject_id) ON DELETE CASCADE,
                    session_data JSONB NOT NULL,
                    total_questions INTEGER DEFAULT 0,
                    correct_answers INTEGER DEFAULT 0,
                    session_duration INTEGER, -- in seconds
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create indexes for better performance
            await self._create_indexes(conn)
            logger.info("All tables created successfully")

    async def _create_indexes(self, conn):
        """Create indexes for optimized queries"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_student ON student_concept_progress(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_concept ON student_concept_progress(concept_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_weaknesses_student ON student_weaknesses(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_weaknesses_concept ON student_weaknesses(concept_id);",
            "CREATE INDEX IF NOT EXISTS idx_learning_activities_student ON learning_activities(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_learning_activities_concept ON learning_activities(concept_id);",
            "CREATE INDEX IF NOT EXISTS idx_recommendations_student ON recommendations(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_study_sessions_student ON study_sessions(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);",
            "CREATE INDEX IF NOT EXISTS idx_chapters_subject ON chapters(subject_id);"
        ]

        for index_sql in indexes:
            await conn.execute(index_sql)

# Core database operations


class StudentManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def create_or_get_student(self, username: str, email: str, full_name: str) -> int:
        """Create a new student or get existing student ID"""
        async with self.db.pool.acquire() as conn:
            # Try to get existing student
            student = await conn.fetchrow(
                "SELECT student_id FROM students WHERE username = $1 OR email = $2",
                username, email
            )

            if student:
                return student['student_id']

            # Create new student
            student_id = await conn.fetchval(
                """INSERT INTO students (username, email, full_name) 
                   VALUES ($1, $2, $3) RETURNING student_id""",
                username, email, full_name
            )
            logger.info(f"Created new student: {username}")
            return student_id

    async def update_learning_preferences(self, student_id: int, preferences: Dict[str, Any]):
        """Update student learning preferences"""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                "UPDATE students SET learning_preferences = $1, updated_at = CURRENT_TIMESTAMP WHERE student_id = $2",
                json.dumps(preferences), student_id
            )


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


class ProgressTracker:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def update_concept_progress(self, student_id: int, concept_id: int,
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
                (student_id, concept_id, status, mastery_score, attempts_count, correct_answers, total_questions, last_practiced)
                VALUES ($1, $2, $3, $4, 1, $5, $6, CURRENT_TIMESTAMP)
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

    async def record_weakness(self, student_id: int, concept_id: int,
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

    async def record_learning_activity(self, student_id: int, concept_id: int,
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


class RecommendationEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def generate_personalized_recommendations(self, student_id: int) -> List[Dict[str, Any]]:
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
                    'type': 'concept_review',
                    'concept_id': concept['concept_id'],
                    'title': f"Review {concept['concept_name']}",
                    'description': f"Focus on {concept['concept_name']} in {concept['chapter_name']} ({concept['subject_name']})",
                    'priority': priority
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
                    'type': 'maintenance_practice',
                    'concept_id': concept['concept_id'],
                    'title': f"Practice {concept['concept_name']}",
                    'description': f"Keep your skills sharp with {concept['concept_name']} practice",
                    'priority': 4
                })

            return recommendations

    async def save_recommendations(self, student_id: int, recommendations: List[Dict[str, Any]]):
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
                    student_id, rec.get('concept_id'), rec['type'],
                    rec['title'], rec['description'], rec['priority']
                )


class AnalyticsManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def get_student_analytics(self, student_id: int) -> Dict[str, Any]:
        """Get comprehensive student analytics"""
        async with self.db.pool.acquire() as conn:
            # Overall progress
            overall_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_concepts,
                    COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_concepts,
                    COUNT(CASE WHEN status = 'needs_review' THEN 1 END) as weak_concepts,
                    AVG(mastery_score) as avg_mastery_score
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
            """, student_id)

            # Recent activity
            recent_activity = await conn.fetch("""
                SELECT activity_type, COUNT(*) as count,
                       AVG(score) as avg_score
                FROM learning_activities 
                WHERE student_id = $1 
                    AND completed_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
                GROUP BY activity_type
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
            """, student_id)

            return {
                'overall_stats': dict(overall_stats) if overall_stats else {},
                'subject_progress': [dict(row) for row in subject_progress],
                'recent_activity': [dict(row) for row in recent_activity],
                'active_weaknesses': [dict(row) for row in active_weaknesses]
            }

# Session management for RAG integration


class StudySessionManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def start_study_session(self, student_id: int, subject_id: int) -> int:
        """Start a new study session"""
        async with self.db.pool.acquire() as conn:
            session_id = await conn.fetchval(
                """INSERT INTO study_sessions (student_id, subject_id, session_data, started_at)
                   VALUES ($1, $2, '{}', CURRENT_TIMESTAMP) RETURNING session_id""",
                student_id, subject_id
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

# Integration function for RAG system


async def process_rag_response(student_id: int, subject_name: str, chapter_name: str,
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

# Database initialization function


async def initialize_database():
    """Initialize database with all tables and indexes"""
    db_manager = DatabaseManager()
    await db_manager.initialize_pool()
    await db_manager.create_tables()
    logger.info("Database initialized successfully")
    return db_manager

# Example usage and testing functions


async def example_usage():
    """Example of how to use the database system"""
    db_manager = await initialize_database()

    try:
        # Create managers
        student_manager = StudentManager(db_manager)
        progress_tracker = ProgressTracker(db_manager)
        recommendation_engine = RecommendationEngine(db_manager)
        analytics_manager = AnalyticsManager(db_manager)

        # Create a student
        student_id = await student_manager.create_or_get_student(
            "john_doe", "john@example.com", "John Doe"
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

        # Generate recommendations
        recommendations = await recommendation_engine.generate_personalized_recommendations(student_id)
        await recommendation_engine.save_recommendations(student_id, recommendations)

        # Get analytics
        analytics = await analytics_manager.get_student_analytics(student_id)
        print("Student Analytics:", json.dumps(
            analytics, indent=2, default=str))

    finally:
        await db_manager.close_pool()

if __name__ == "__main__":
    asyncio.run(example_usage())
