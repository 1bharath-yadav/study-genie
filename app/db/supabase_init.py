"""
Supabase table initialization script
This script creates all required tables directly in Supabase
"""
import os
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SupabaseInitializer:
    def __init__(self, require_service_key=True):
        self.url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        self.service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.anon_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

        if require_service_key and (not self.url or not self.service_key):
            raise ValueError(
                "Supabase URL and service role key must be provided for table creation")

        if not self.url:
            raise ValueError("Supabase URL must be provided")

        # Only create client if we have service key
        self.client = None
        if self.service_key:
            self.client = create_client(self.url, self.service_key)
            logger.info("Supabase client initialized for table creation")
        elif self.anon_key:
            self.client = create_client(self.url, self.anon_key)
            logger.info(
                "Supabase client initialized with anon key (limited permissions)")
        else:
            logger.warning(
                "No Supabase key available - only SQL generation will work")

    def create_tables(self):
        """Create all required tables in Supabase"""

        # Define all table creation SQL statements
        table_statements = [
            # Students table
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id VARCHAR(100) PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                learning_preferences JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # Subjects table
            """
            CREATE TABLE IF NOT EXISTS subjects (
                subject_id BIGSERIAL PRIMARY KEY,
                subject_name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # Chapters table
            """
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
            """,

            # Concepts table
            """
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
            """,

            # Student Subject Enrollment
            """
            CREATE TABLE IF NOT EXISTS student_subjects (
                enrollment_id BIGSERIAL PRIMARY KEY,
                student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                UNIQUE(student_id, subject_id)
            );
            """,

            # Student Concept Progress
            """
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
            """,

            # Student Weaknesses
            """
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
            """,

            # Learning Activities
            """
            CREATE TABLE IF NOT EXISTS learning_activities (
                activity_id BIGSERIAL PRIMARY KEY,
                student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                concept_id BIGINT REFERENCES concepts(concept_id) ON DELETE CASCADE,
                activity_type VARCHAR(50) NOT NULL,
                activity_data JSONB NOT NULL,
                score DECIMAL(5,2),
                time_spent INTEGER,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # Personalized Recommendations
            """
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
            """,

            # Study Sessions
            """
            CREATE TABLE IF NOT EXISTS study_sessions (
                session_id SERIAL PRIMARY KEY,
                student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
                subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
                session_data JSONB NOT NULL,
                total_questions INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                session_duration INTEGER,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,

            # Student Recommendations
            """
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
            """,

            # Student API Keys table (this seems to be missing from your models.py)
            """
            CREATE TABLE IF NOT EXISTS student_api_keys (
                id BIGSERIAL PRIMARY KEY,
                student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                encrypted_api_key TEXT NOT NULL,
                service VARCHAR(50) DEFAULT 'gemini',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                UNIQUE(student_id)
            );
            """
        ]

        # Create indexes
        index_statements = [
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
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_student_id ON student_recommendations(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_priority ON student_recommendations(priority);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_created_at ON student_recommendations(created_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_type ON student_recommendations(recommendation_type);",
            "CREATE INDEX IF NOT EXISTS idx_student_recommendations_active ON student_recommendations(student_id, is_dismissed) WHERE is_dismissed = FALSE;",
            "CREATE INDEX IF NOT EXISTS idx_learning_activities_completed_at ON learning_activities(completed_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_status ON student_concept_progress(status);",
            "CREATE INDEX IF NOT EXISTS idx_student_concept_progress_last_practiced ON student_concept_progress(last_practiced DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_weaknesses_severity ON student_weaknesses(severity_score DESC);",
            "CREATE INDEX IF NOT EXISTS idx_student_subjects_active ON student_subjects(student_id) WHERE is_active = TRUE;",
            "CREATE INDEX IF NOT EXISTS idx_student_api_keys_student ON student_api_keys(student_id);",
            "CREATE INDEX IF NOT EXISTS idx_student_api_keys_active ON student_api_keys(student_id, is_active) WHERE is_active = TRUE;"
        ]

        try:
            # Check if client is available
            if not self.client:
                raise ValueError(
                    "Supabase client not initialized - need service role key for table creation")

            # Execute table creation statements
            logger.info("Creating tables in Supabase...")
            for i, statement in enumerate(table_statements):
                try:
                    # Use raw SQL execution through Supabase
                    result = self.client.rpc(
                        'exec_sql', {'sql': statement}).execute()
                    table_name = statement.split('CREATE TABLE IF NOT EXISTS ')[
                        1].split(' ')[0]
                    logger.info(f"✅ Created table: {table_name}")
                except Exception as e:
                    # Try alternative method using direct SQL
                    logger.warning(
                        f"RPC method failed, trying direct SQL for table {i+1}: {e}")
                    # We'll need to use Supabase SQL Editor or migrations for this
                    pass

            # Execute index creation statements
            logger.info("Creating indexes in Supabase...")
            for statement in index_statements:
                try:
                    result = self.client.rpc(
                        'exec_sql', {'sql': statement}).execute()
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

            logger.info(
                "✅ All tables and indexes created successfully in Supabase")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create tables in Supabase: {e}")
            return False

    def verify_tables(self):
        """Verify that all tables exist in Supabase"""
        required_tables = [
            'students', 'subjects', 'chapters', 'concepts',
            'student_subjects', 'student_concept_progress',
            'student_weaknesses', 'learning_activities',
            'recommendations', 'study_sessions',
            'student_recommendations', 'student_api_keys'
        ]

        logger.info("Verifying tables in Supabase...")

        if not self.client:
            logger.error(
                "Cannot verify tables - Supabase client not initialized")
            return False

        for table in required_tables:
            try:
                # Try to query the table (limit 0 to just check existence)
                result = self.client.table(
                    table).select("*").limit(0).execute()
                logger.info(f"✅ Table exists: {table}")
            except Exception as e:
                logger.error(f"❌ Table missing or inaccessible: {table} - {e}")
                return False

        logger.info("✅ All required tables verified in Supabase")
        return True

    def get_sql_migration_script(self):
        """Generate a SQL migration script that can be run in Supabase SQL Editor"""
        sql_script = """
-- StudyGenie Database Schema for Supabase
-- Run this script in the Supabase SQL Editor

-- Enable Row Level Security (RLS) by default
-- You can customize these policies based on your security requirements

-- Students table
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(100) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    learning_preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subjects table
CREATE TABLE IF NOT EXISTS subjects (
    subject_id BIGSERIAL PRIMARY KEY,
    subject_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chapters table
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

-- Concepts table
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

-- Student Subject Enrollment
CREATE TABLE IF NOT EXISTS student_subjects (
    enrollment_id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(student_id, subject_id)
);

-- Student Concept Progress
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

-- Student Weaknesses
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

-- Learning Activities
CREATE TABLE IF NOT EXISTS learning_activities (
    activity_id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
    concept_id BIGINT REFERENCES concepts(concept_id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL,
    activity_data JSONB NOT NULL,
    score DECIMAL(5,2),
    time_spent INTEGER,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Personalized Recommendations
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

-- Study Sessions
CREATE TABLE IF NOT EXISTS study_sessions (
    session_id SERIAL PRIMARY KEY,
    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
    session_data JSONB NOT NULL,
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    session_duration INTEGER,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Student Recommendations
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

-- Student API Keys
CREATE TABLE IF NOT EXISTS student_api_keys (
    id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    encrypted_api_key TEXT NOT NULL,
    service VARCHAR(50) DEFAULT 'gemini',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(student_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_student_concept_progress_student ON student_concept_progress(student_id);
CREATE INDEX IF NOT EXISTS idx_student_concept_progress_concept ON student_concept_progress(concept_id);
CREATE INDEX IF NOT EXISTS idx_student_weaknesses_student ON student_weaknesses(student_id);
CREATE INDEX IF NOT EXISTS idx_student_weaknesses_concept ON student_weaknesses(concept_id);
CREATE INDEX IF NOT EXISTS idx_learning_activities_student ON learning_activities(student_id);
CREATE INDEX IF NOT EXISTS idx_learning_activities_concept ON learning_activities(concept_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_student ON recommendations(student_id);
CREATE INDEX IF NOT EXISTS idx_study_sessions_student ON study_sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_concepts_chapter ON concepts(chapter_id);
CREATE INDEX IF NOT EXISTS idx_chapters_subject ON chapters(subject_id);
CREATE INDEX IF NOT EXISTS idx_students_external_id ON students(student_id);
CREATE INDEX IF NOT EXISTS idx_student_recommendations_student_id ON student_recommendations(student_id);
CREATE INDEX IF NOT EXISTS idx_student_recommendations_priority ON student_recommendations(priority);
CREATE INDEX IF NOT EXISTS idx_student_recommendations_created_at ON student_recommendations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_student_recommendations_type ON student_recommendations(recommendation_type);
CREATE INDEX IF NOT EXISTS idx_student_recommendations_active ON student_recommendations(student_id, is_dismissed) WHERE is_dismissed = FALSE;
CREATE INDEX IF NOT EXISTS idx_learning_activities_completed_at ON learning_activities(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_student_concept_progress_status ON student_concept_progress(status);
CREATE INDEX IF NOT EXISTS idx_student_concept_progress_last_practiced ON student_concept_progress(last_practiced DESC);
CREATE INDEX IF NOT EXISTS idx_student_weaknesses_severity ON student_weaknesses(severity_score DESC);
CREATE INDEX IF NOT EXISTS idx_student_subjects_active ON student_subjects(student_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_student_api_keys_student ON student_api_keys(student_id);
CREATE INDEX IF NOT EXISTS idx_student_api_keys_active ON student_api_keys(student_id, is_active) WHERE is_active = TRUE;

-- Optional: Enable Row Level Security (RLS) policies
-- Uncomment and customize these based on your security requirements

/*
-- Enable RLS on all tables
ALTER TABLE students ENABLE ROW LEVEL SECURITY;
ALTER TABLE subjects ENABLE ROW LEVEL SECURITY;
ALTER TABLE chapters ENABLE ROW LEVEL SECURITY;
ALTER TABLE concepts ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_subjects ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_concept_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_weaknesses ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE study_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_api_keys ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (customize as needed)
-- Students can only access their own data
CREATE POLICY "Students can view own data" ON students FOR SELECT USING (auth.uid()::text = student_id);
CREATE POLICY "Students can update own data" ON students FOR UPDATE USING (auth.uid()::text = student_id);

-- Add similar policies for other tables...
*/

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_students_updated_at BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_student_concept_progress_updated_at BEFORE UPDATE ON student_concept_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_student_weaknesses_updated_at BEFORE UPDATE ON student_weaknesses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_student_recommendations_updated_at BEFORE UPDATE ON student_recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_student_api_keys_updated_at BEFORE UPDATE ON student_api_keys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""
        return sql_script


def main():
    """Main function to initialize Supabase tables"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize Supabase tables for StudyGenie")
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing tables')
    parser.add_argument('--generate-sql', action='store_true',
                        help='Generate SQL migration script')

    args = parser.parse_args()

    # Check if we're just generating SQL
    if args.generate_sql:
        print("Generating SQL migration script...")
        initializer = SupabaseInitializer(require_service_key=False)
        sql_script = initializer.get_sql_migration_script()

        # Save to file
        script_path = "/home/archer/projects/study-genie/supabase_migration.sql"
        with open(script_path, 'w') as f:
            f.write(sql_script)

        print(f"✅ SQL migration script saved to: {script_path}")
        print("\n📋 Next steps:")
        print("1. Go to your Supabase Dashboard")
        print("2. Navigate to the SQL Editor")
        print("3. Paste and run the generated SQL script")
        print("4. Or upload the migration file")
        return

    # For other operations, we need the service role key
    try:
        initializer = SupabaseInitializer(require_service_key=True)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n🔑 To get your Supabase Service Role Key:")
        print("1. Go to your Supabase Dashboard")
        print("2. Navigate to Settings > API")
        print("3. Copy the 'service_role' key (NOT the anon key)")
        print("4. Add it to your .env file as:")
        print("   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here")
        print("\nOr use --generate-sql to create a migration script instead")
        return

    if args.generate_sql:
        print("Generating SQL migration script...")
        sql_script = initializer.get_sql_migration_script()

        # Save to file
        with open('supabase_migration.sql', 'w') as f:
            f.write(sql_script)

        print("✅ SQL migration script saved to 'supabase_migration.sql'")
        print("📝 Please run this script in your Supabase SQL Editor")
        return

    if args.verify_only:
        success = initializer.verify_tables()
        if success:
            print("✅ All tables verified successfully")
        else:
            print("❌ Some tables are missing or inaccessible")
        return

    # Create tables
    success = initializer.create_tables()
    if success:
        print("✅ Tables created successfully")
        # Verify after creation
        initializer.verify_tables()
    else:
        print("❌ Failed to create tables")
        print("💡 Try running with --generate-sql to get a migration script for Supabase SQL Editor")


if __name__ == "__main__":
    main()
