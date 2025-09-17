-- Migration: add student_activity table for storing user learning activity
CREATE TABLE IF NOT EXISTS student_activity (
    activity_id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(100) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    activity_type VARCHAR(100) NOT NULL, -- e.g. 'flashcard_review', 'quiz_attempt', 'study_session'
    related_subject_id BIGINT, -- optional FK to subjects
    related_chapter_id BIGINT, -- optional FK to chapters
    related_concept_id BIGINT, -- optional FK to concepts
    payload JSONB NOT NULL, -- arbitrary activity payload (questions answered, flashcard ids, etc.)
    score DECIMAL(5,2) DEFAULT NULL,
    time_spent_seconds INTEGER DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_student_activity_student_date ON student_activity(student_id, created_at DESC);

-- Row level security: allow users to insert/select their own activity when authenticated
ALTER TABLE student_activity ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Students can insert own activity" ON student_activity
    FOR INSERT WITH CHECK (auth.jwt() ->> 'sub' = student_id);
CREATE POLICY "Students can view own activity" ON student_activity
    FOR SELECT USING (auth.jwt() ->> 'sub' = student_id);
