-- Simple migration to add learning_sessions and migrate minimal session data
-- This file is intentionally small and idempotent to avoid failures in complex environments.

BEGIN;

-- 1) Ensure session_id exists on learning_history (add if missing)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='learning_history' AND column_name='session_id'
    ) THEN
        ALTER TABLE learning_history ADD COLUMN session_id UUID DEFAULT gen_random_uuid();
    END IF;
END$$;

-- 2) Backfill NULL session_id values with a new UUID (so every row belongs to a session)
UPDATE learning_history SET session_id = gen_random_uuid() WHERE session_id IS NULL;

-- 3) Create a minimal sessions table to list sessions and hold aggregated LLM history
CREATE TABLE IF NOT EXISTS learning_sessions (
    session_id UUID PRIMARY KEY,
    student_id VARCHAR(100),
    session_name VARCHAR(500),
    llm_response_history JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- 4) Populate learning_sessions by aggregating per-session llm_response_history from learning_history
--    This aggregates arrays across rows for the same session_id into one array.
INSERT INTO learning_sessions (session_id, student_id, session_name, llm_response_history, created_at, updated_at)
SELECT
    lh.session_id,
    lh.student_id,
    -- prefer session_name if present, else NULL
    MAX(CASE WHEN lh.session_name IS NOT NULL THEN lh.session_name END) AS session_name,
    COALESCE(jsonb_agg(elem) FILTER (WHERE elem IS NOT NULL), '[]'::jsonb) AS llm_response_history,
    MIN(lh.created_at) AS created_at,
    MAX(lh.updated_at) AS updated_at
FROM learning_history lh,
    LATERAL (
        SELECT jsonb_array_elements(coalesce(lh.llm_response_history, '[]'::jsonb)) AS elem
    ) elems
GROUP BY lh.session_id, lh.student_id
ON CONFLICT (session_id) DO NOTHING;

COMMIT;

-- Notes:
-- - This migration avoids triggers, foreign keys, and RLS to stay simple and robust.
-- - After this runs you can later add more columns/indexes/policies in a follow-up migration.
-- - To append new responses at runtime, update learning_sessions.llm_response_history with:
--     UPDATE learning_sessions SET llm_response_history = coalesce(llm_response_history, '[]'::jsonb) || to_jsonb($1::jsonb), updated_at = NOW() WHERE session_id = $2;
