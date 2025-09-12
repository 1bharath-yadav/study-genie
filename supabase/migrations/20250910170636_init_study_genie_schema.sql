-- StudyGenie Database Schema (Supabase)
-- Run this once in the Supabase SQL Editor

CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(100) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    learning_preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subjects table - created dynamically when students upload content and LLM analyzes it
CREATE TABLE IF NOT EXISTS subjects (
    subject_id BIGSERIAL PRIMARY KEY,
    subject_name VARCHAR(200) NOT NULL UNIQUE,
    description TEXT,
    difficulty_level VARCHAR(20) DEFAULT 'Medium',
    created_from_content BOOLEAN DEFAULT TRUE, -- indicates this was created from uploaded content
    llm_generated BOOLEAN DEFAULT TRUE, -- indicates this was suggested by LLM
    content_source TEXT, -- reference to the original content that created this subject
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS student_enrollments (
    enrollment_id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(student_id, subject_id)
);

-- Chapters table - created dynamically from LLM content analysis
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id BIGSERIAL PRIMARY KEY,
    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
    chapter_name VARCHAR(200) NOT NULL,
    chapter_order BIGINT NOT NULL,
    description TEXT,
    llm_generated BOOLEAN DEFAULT TRUE, -- indicates this was suggested by LLM from content
    content_source TEXT, -- reference to the original content that created this chapter
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, chapter_name),
    UNIQUE(subject_id, chapter_order)
);

-- Concepts table - created dynamically from LLM content analysis
CREATE TABLE IF NOT EXISTS concepts (
    concept_id BIGSERIAL PRIMARY KEY,
    chapter_id BIGINT REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    concept_name VARCHAR(200) NOT NULL,
    concept_order INTEGER NOT NULL,
    description TEXT,
    difficulty_level VARCHAR(20) DEFAULT 'Medium',
    llm_generated BOOLEAN DEFAULT TRUE, -- indicates this was suggested by LLM from content
    content_source TEXT, -- reference to the original content that created this concept
    learning_objectives TEXT[], -- specific learning objectives for this concept
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chapter_id, concept_name),
    UNIQUE(chapter_id, concept_order)
);

-- Uploaded Content table - tracks student uploads and LLM analysis
CREATE TABLE IF NOT EXISTS uploaded_content (
    content_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
    original_filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL, -- 'pdf', 'image', 'text', 'docx', etc.
    file_size_bytes BIGINT,
    file_path TEXT, -- where the file is stored
    content_text TEXT, -- extracted text content
    llm_analysis JSONB, -- structured analysis from LLM (subjects, chapters, concepts)
    processing_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    processing_error TEXT, -- error message if processing failed
    subjects_created INTEGER DEFAULT 0, -- count of subjects created from this content
    chapters_created INTEGER DEFAULT 0, -- count of chapters created from this content
    concepts_created INTEGER DEFAULT 0, -- count of concepts created from this content
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Content Learning Mapping - tracks which content created which learning structure
CREATE TABLE IF NOT EXISTS content_learning_mapping (
    mapping_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content_id UUID REFERENCES uploaded_content(content_id) ON DELETE CASCADE,
    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
    chapter_id BIGINT REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    concept_id BIGINT REFERENCES concepts(concept_id) ON DELETE CASCADE,
    confidence_score DECIMAL(5,2) DEFAULT 0.0, -- LLM confidence in this mapping
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS subject_progress (
    progress_id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT NOT NULL,
    status VARCHAR(20) DEFAULT 'not_started',
    mastery_score DECIMAL(5,2) DEFAULT 0.00,
    attempts_count INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    total_questions INTEGER DEFAULT 0,
    last_practiced TIMESTAMP,
    first_learned TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(student_id, subject_id)
);

CREATE TABLE IF NOT EXISTS concept_progress (
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

CREATE TABLE IF NOT EXISTS weaknesses (
    weakness_id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(100) REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT REFERENCES subjects(subject_id) ON DELETE CASCADE,
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

-- Providers table
CREATE TABLE IF NOT EXISTS llm_providers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,  -- e.g., 'google', 'openai', 'anthropic'
    display_name VARCHAR(100) NOT NULL,  -- e.g., 'Google Gemini', 'OpenAI GPT', 'Anthropic Claude'
    base_url VARCHAR(255),              -- For custom providers
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Available models table
CREATE TABLE IF NOT EXISTS available_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    provider_id UUID REFERENCES llm_providers(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'chat', 'completion', 'embeddings'
    context_length INTEGER,
    supports_system_prompt BOOLEAN DEFAULT TRUE,
    supports_function_calling BOOLEAN DEFAULT FALSE,
    max_tokens INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    features JSONB DEFAULT '{}',  -- Store model capabilities
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(provider_id, model_name)
);

-- User API Keys table 
CREATE TABLE IF NOT EXISTS user_api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    provider_id UUID REFERENCES llm_providers(id) ON DELETE CASCADE,
    encrypted_api_key TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, provider_id)
);

-- User Model Preferences table
CREATE TABLE IF NOT EXISTS user_model_preferences (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    model_id UUID REFERENCES available_models(id) ON DELETE CASCADE,
    use_for_chat BOOLEAN DEFAULT FALSE,
    use_for_embedding BOOLEAN DEFAULT FALSE,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, model_id)
);

-- Insert default providers
INSERT INTO llm_providers (name, display_name, is_active) VALUES 
    ('google', 'Google Gemini', true),
    ('openai', 'OpenAI', true),
    ('anthropic', 'Anthropic Claude', true)
ON CONFLICT (name) DO NOTHING;

-- Insert default models
INSERT INTO available_models (provider_id, model_name, display_name, model_type, context_length, supports_function_calling, max_tokens) 
SELECT 
    p.id,
    model_data.model_name,
    model_data.display_name,
    model_data.model_type,
    model_data.context_length,
    model_data.supports_function_calling,
    model_data.max_tokens
FROM llm_providers p
CROSS JOIN (VALUES
    -- Google models
    ('google', 'gemini-2.0-flash', 'Gemini 2.0 Flash', 'chat', 1000000, true, 8192),
    ('google', 'gemini-1.5-pro', 'Gemini 1.5 Pro', 'chat', 2000000, true, 8192),
    ('google', 'text-embedding-004', 'Text Embedding 004', 'embeddings', 2048, false, 768),
    -- OpenAI models
    ('openai', 'gpt-4o', 'GPT-4o', 'chat', 128000, true, 4096),
    ('openai', 'gpt-4o-mini', 'GPT-4o Mini', 'chat', 128000, true, 16384),
    ('openai', 'text-embedding-3-small', 'Text Embedding 3 Small', 'embeddings', 8191, false, 1536),
    ('openai', 'text-embedding-3-large', 'Text Embedding 3 Large', 'embeddings', 8191, false, 3072),
    -- Anthropic models
    ('anthropic', 'claude-3-5-sonnet-20241022', 'Claude 3.5 Sonnet', 'chat', 200000, true, 8192),
    ('anthropic', 'claude-3-5-haiku-20241022', 'Claude 3.5 Haiku', 'chat', 200000, true, 8192)
) AS model_data(provider_name, model_name, display_name, model_type, context_length, supports_function_calling, max_tokens)
WHERE p.name = model_data.provider_name
ON CONFLICT (provider_id, model_name) DO NOTHING;

-- ===================================================================
-- ANALYTICS TABLES FOR COMPREHENSIVE STUDENT PERFORMANCE TRACKING
-- ===================================================================

-- Study Sessions table for detailed session tracking
CREATE TABLE IF NOT EXISTS detailed_study_sessions (
    session_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT,
    chapter_id BIGINT REFERENCES chapters(chapter_id) ON DELETE SET NULL,
    session_type VARCHAR(50) NOT NULL DEFAULT 'study', -- 'study', 'quiz', 'review', 'practice'
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER DEFAULT 0,
    concepts_covered TEXT[], -- Array of concept names/IDs covered
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    accuracy_percentage DECIMAL(5,2) DEFAULT 0.0,
    difficulty_level VARCHAR(20) DEFAULT 'medium',
    session_metadata JSONB DEFAULT '{}', -- Store additional session data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Concept Mastery Tracking
CREATE TABLE IF NOT EXISTS concept_mastery (
    mastery_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    concept_id BIGINT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    subject_id BIGINT NOT NULL,
    chapter_id BIGINT NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    mastery_score DECIMAL(5,2) NOT NULL DEFAULT 0.0, -- 0-100 scale
    confidence_level VARCHAR(20) DEFAULT 'low', -- 'low', 'medium', 'high'
    total_attempts INTEGER DEFAULT 0,
    correct_attempts INTEGER DEFAULT 0,
    last_practiced_at TIMESTAMP WITH TIME ZONE,
    mastery_achieved_at TIMESTAMP WITH TIME ZONE,
    time_to_mastery_seconds INTEGER,
    learning_velocity DECIMAL(8,2) DEFAULT 0.0, -- concepts per hour
    retention_rate DECIMAL(5,2) DEFAULT 0.0, -- percentage retained over time
    difficulty_rating INTEGER DEFAULT 3, -- 1-5 scale (student's perceived difficulty)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, concept_id)
);

-- Weekly Performance Summaries
CREATE TABLE IF NOT EXISTS weekly_performance (
    performance_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    week_start_date DATE NOT NULL,
    week_end_date DATE NOT NULL,
    total_study_time_seconds INTEGER DEFAULT 0,
    total_sessions INTEGER DEFAULT 0,
    total_questions_attempted INTEGER DEFAULT 0,
    total_correct_answers INTEGER DEFAULT 0,
    overall_accuracy DECIMAL(5,2) DEFAULT 0.0,
    concepts_learned INTEGER DEFAULT 0,
    concepts_reviewed INTEGER DEFAULT 0,
    mastery_gained DECIMAL(5,2) DEFAULT 0.0, -- Total mastery points gained
    subjects_studied TEXT[], -- Array of subject names studied
    peak_performance_day VARCHAR(20), -- Day of week with best performance
    study_consistency_score DECIMAL(5,2) DEFAULT 0.0, -- How consistent was the study pattern
    improvement_rate DECIMAL(5,2) DEFAULT 0.0, -- Week-over-week improvement
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, week_start_date)
);

-- Subject Performance Analytics
CREATE TABLE IF NOT EXISTS subject_analytics (
    analytics_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT NOT NULL,
    subject_name VARCHAR(200) NOT NULL,
    total_concepts INTEGER DEFAULT 0,
    mastered_concepts INTEGER DEFAULT 0,
    mastery_percentage DECIMAL(5,2) DEFAULT 0.0,
    average_mastery_score DECIMAL(5,2) DEFAULT 0.0,
    total_study_time_seconds INTEGER DEFAULT 0,
    total_sessions INTEGER DEFAULT 0,
    quiz_accuracy DECIMAL(5,2) DEFAULT 0.0,
    learning_velocity DECIMAL(8,2) DEFAULT 0.0, -- concepts per hour
    retention_rate DECIMAL(5,2) DEFAULT 0.0,
    difficulty_perception DECIMAL(3,1) DEFAULT 3.0, -- 1-5 scale
    last_studied_at TIMESTAMP WITH TIME ZONE,
    first_studied_at TIMESTAMP WITH TIME ZONE,
    performance_trend VARCHAR(20) DEFAULT 'stable', -- 'improving', 'declining', 'stable'
    weak_concepts TEXT[], -- Array of concept names where student struggles
    strong_concepts TEXT[], -- Array of concept names where student excels
    recommended_focus_areas TEXT[], -- AI-generated recommendations
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, subject_id)
);

-- Learning Path Analytics
CREATE TABLE IF NOT EXISTS learning_path_analytics (
    path_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id BIGINT NOT NULL,
    chapter_id BIGINT REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    optimal_sequence TEXT[], -- Recommended order of concepts
    actual_sequence TEXT[], -- Actual order student followed
    sequence_efficiency DECIMAL(5,2) DEFAULT 0.0, -- How close to optimal
    prerequisite_gaps TEXT[], -- Missing prerequisite concepts
    knowledge_gaps TEXT[], -- Identified gaps in understanding
    suggested_remediation JSONB DEFAULT '{}', -- Structured remediation plan
    learning_style_indicators JSONB DEFAULT '{}', -- Visual, auditory, kinesthetic preferences
    pacing_analysis JSONB DEFAULT '{}', -- Fast/slow learner indicators
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Study Behavior Patterns
CREATE TABLE IF NOT EXISTS study_patterns (
    pattern_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    analysis_period_start DATE NOT NULL,
    analysis_period_end DATE NOT NULL,
    daily_study_time_avg INTEGER DEFAULT 0, -- Average minutes per day
    peak_study_hours INTEGER[], -- Hours of day when most active
    study_days_per_week DECIMAL(3,1) DEFAULT 0.0,
    session_duration_avg INTEGER DEFAULT 0, -- Average session length in minutes
    break_frequency_minutes INTEGER DEFAULT 0, -- How often they take breaks
    consistency_score DECIMAL(5,2) DEFAULT 0.0, -- How consistent is their schedule
    procrastination_indicators JSONB DEFAULT '{}', -- Patterns indicating procrastination
    engagement_metrics JSONB DEFAULT '{}', -- Time on task, focus indicators
    preferred_difficulty_level VARCHAR(20) DEFAULT 'medium',
    multitasking_tendency DECIMAL(5,2) DEFAULT 0.0, -- How often they switch topics
    optimal_study_duration INTEGER DEFAULT 45, -- Optimal session length for this student
    fatigue_patterns JSONB DEFAULT '{}', -- When performance drops due to fatigue
    motivation_indicators JSONB DEFAULT '{}', -- What motivates this student
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, analysis_period_start)
);

-- Achievements and Milestones
CREATE TABLE IF NOT EXISTS student_achievements (
    achievement_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    achievement_type VARCHAR(50) NOT NULL, -- 'mastery', 'streak', 'improvement', 'milestone'
    achievement_name VARCHAR(200) NOT NULL,
    achievement_description TEXT,
    points_earned INTEGER DEFAULT 0,
    badge_icon VARCHAR(100), -- Icon identifier for frontend
    badge_color VARCHAR(20), -- Color for badge display
    criteria_met JSONB NOT NULL, -- What criteria were met for this achievement
    achieved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE, -- For time-limited achievements
    is_visible BOOLEAN DEFAULT TRUE,
    celebration_shown BOOLEAN DEFAULT FALSE, -- Has user seen the celebration
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Real-time Performance Metrics (for dashboard)
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    metric_date DATE NOT NULL DEFAULT CURRENT_DATE,
    daily_study_minutes INTEGER DEFAULT 0,
    daily_concepts_learned INTEGER DEFAULT 0,
    daily_accuracy DECIMAL(5,2) DEFAULT 0.0,
    daily_mastery_gained DECIMAL(5,2) DEFAULT 0.0,
    streak_days INTEGER DEFAULT 0,
    weekly_improvement DECIMAL(5,2) DEFAULT 0.0,
    monthly_improvement DECIMAL(5,2) DEFAULT 0.0,
    overall_mastery_level DECIMAL(5,2) DEFAULT 0.0,
    learning_momentum DECIMAL(5,2) DEFAULT 0.0, -- Trend indicator
    focus_score DECIMAL(5,2) DEFAULT 0.0, -- How focused the student was
    challenge_level DECIMAL(5,2) DEFAULT 0.0, -- How challenging the material was
    satisfaction_score DECIMAL(5,2) DEFAULT 0.0, -- Self-reported satisfaction
    energy_level DECIMAL(5,2) DEFAULT 0.0, -- Self-reported energy
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(student_id, metric_date)
);

-- ===================================================================
-- INDEXES FOR ANALYTICS PERFORMANCE OPTIMIZATION
-- ===================================================================

-- Study Sessions Indexes
CREATE INDEX IF NOT EXISTS idx_detailed_study_sessions_student_date 
    ON detailed_study_sessions(student_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_detailed_study_sessions_subject 
    ON detailed_study_sessions(subject_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_detailed_study_sessions_type 
    ON detailed_study_sessions(session_type, started_at DESC);

-- Concept Mastery Indexes
CREATE INDEX IF NOT EXISTS idx_concept_mastery_student_subject 
    ON concept_mastery(student_id, subject_id, mastery_score DESC);
CREATE INDEX IF NOT EXISTS idx_concept_mastery_concept 
    ON concept_mastery(concept_id, mastery_score DESC);
CREATE INDEX IF NOT EXISTS idx_concept_mastery_updated 
    ON concept_mastery(updated_at DESC);

-- Performance Metrics Indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_student_date 
    ON performance_metrics(student_id, metric_date DESC);
CREATE INDEX IF NOT EXISTS idx_weekly_performance_student_week 
    ON weekly_performance(student_id, week_start_date DESC);
CREATE INDEX IF NOT EXISTS idx_subject_analytics_student 
    ON subject_analytics(student_id, mastery_percentage DESC);

-- Achievement Indexes
CREATE INDEX IF NOT EXISTS idx_student_achievements_student_date 
    ON student_achievements(student_id, achieved_at DESC);
CREATE INDEX IF NOT EXISTS idx_student_achievements_type 
    ON student_achievements(achievement_type, achieved_at DESC);

-- ===================================================================
-- ANALYTICS HELPER FUNCTIONS
-- ===================================================================

-- Function to calculate mastery percentage for a student
CREATE OR REPLACE FUNCTION calculate_student_mastery_percentage(p_student_id VARCHAR(100))
RETURNS DECIMAL(5,2) AS $$
DECLARE
    total_concepts INTEGER;
    mastered_concepts INTEGER;
    mastery_percentage DECIMAL(5,2);
BEGIN
    -- Count total concepts the student has attempted
    SELECT COUNT(*) INTO total_concepts
    FROM concept_mastery 
    WHERE student_id = p_student_id;
    
    -- Count mastered concepts (mastery_score >= 80)
    SELECT COUNT(*) INTO mastered_concepts
    FROM concept_mastery 
    WHERE student_id = p_student_id AND mastery_score >= 80.0;
    
    -- Calculate percentage
    IF total_concepts > 0 THEN
        mastery_percentage := (mastered_concepts::DECIMAL / total_concepts::DECIMAL) * 100.0;
    ELSE
        mastery_percentage := 0.0;
    END IF;
    
    RETURN mastery_percentage;
END;
$$ LANGUAGE plpgsql;

-- Function to update weekly performance summary
CREATE OR REPLACE FUNCTION update_weekly_performance_summary(p_student_id VARCHAR(100), p_week_start DATE)
RETURNS VOID AS $$
DECLARE
    week_end DATE := p_week_start + INTERVAL '6 days';
    total_time INTEGER;
    total_sessions INTEGER;
    total_questions INTEGER;
    total_correct INTEGER;
    concepts_count INTEGER;
BEGIN
    -- Calculate weekly aggregates
    SELECT 
        COALESCE(SUM(duration_seconds), 0),
        COUNT(*),
        COALESCE(SUM(total_questions), 0),
        COALESCE(SUM(correct_answers), 0),
        COALESCE(COUNT(DISTINCT UNNEST(concepts_covered)), 0)
    INTO total_time, total_sessions, total_questions, total_correct, concepts_count
    FROM detailed_study_sessions 
    WHERE student_id = p_student_id 
    AND DATE(started_at) BETWEEN p_week_start AND week_end;
    
    -- Insert or update weekly summary
    INSERT INTO weekly_performance (
        student_id, week_start_date, week_end_date,
        total_study_time_seconds, total_sessions,
        total_questions_attempted, total_correct_answers,
        overall_accuracy, concepts_learned
    ) VALUES (
        p_student_id, p_week_start, week_end,
        total_time, total_sessions,
        total_questions, total_correct,
        CASE WHEN total_questions > 0 THEN (total_correct::DECIMAL / total_questions::DECIMAL) * 100.0 ELSE 0.0 END,
        concepts_count
    )
    ON CONFLICT (student_id, week_start_date) 
    DO UPDATE SET
        total_study_time_seconds = EXCLUDED.total_study_time_seconds,
        total_sessions = EXCLUDED.total_sessions,
        total_questions_attempted = EXCLUDED.total_questions_attempted,
        total_correct_answers = EXCLUDED.total_correct_answers,
        overall_accuracy = EXCLUDED.overall_accuracy,
        concepts_learned = EXCLUDED.concepts_learned,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- ===================================================================
-- ROW LEVEL SECURITY POLICIES
-- ===================================================================

-- Enable RLS on analytics tables
ALTER TABLE detailed_study_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE concept_mastery ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE subject_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_path_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE study_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_achievements ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;

-- Create policies for student data access
CREATE POLICY "Students can view own analytics" ON detailed_study_sessions
    FOR SELECT USING (auth.jwt() ->> 'sub' = student_id);

CREATE POLICY "Students can insert own study sessions" ON detailed_study_sessions
    FOR INSERT WITH CHECK (auth.jwt() ->> 'sub' = student_id);

CREATE POLICY "Students can view own concept mastery" ON concept_mastery
    FOR SELECT USING (auth.jwt() ->> 'sub' = student_id);

CREATE POLICY "Students can update own concept mastery" ON concept_mastery
    FOR ALL USING (auth.jwt() ->> 'sub' = student_id);

CREATE POLICY "Students can view own performance" ON weekly_performance
    FOR SELECT USING (auth.jwt() ->> 'sub' = student_id);

CREATE POLICY "Students can view own subject analytics" ON subject_analytics
    FOR SELECT USING (auth.jwt() ->> 'sub' = student_id);

CREATE POLICY "Students can view own achievements" ON student_achievements
    FOR SELECT USING (auth.jwt() ->> 'sub' = student_id);

CREATE POLICY "Students can view own metrics" ON performance_metrics
    FOR SELECT USING (auth.jwt() ->> 'sub' = student_id);

-- ===================================================================
-- SAMPLE DATA FOR TESTING (Optional - remove in production)
-- ===================================================================

-- This would typically be removed in production but useful for development
-- INSERT INTO detailed_study_sessions (student_id, subject_id, session_type, duration_seconds, total_questions, correct_answers)
-- VALUES 
--     ('sample-student-1', 1, 'study', 1800, 10, 8),
--     ('sample-student-1', 1, 'quiz', 900, 15, 12);
