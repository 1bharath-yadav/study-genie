-- -- Create a lightweight chat_messages table to store per-message chat history
-- BEGIN;

-- CREATE TABLE IF NOT EXISTS chat_messages (
--     id BIGSERIAL PRIMARY KEY,
--     session_id UUID,
--     student_id VARCHAR(200),
--     role VARCHAR(50), -- e.g. user, assistant, system, tool
--     content TEXT,
--     message_json JSONB DEFAULT '{}'::jsonb,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
-- );

-- CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
-- CREATE INDEX IF NOT EXISTS idx_chat_messages_student_id ON chat_messages(student_id);

-- COMMIT;
 -----   DELETED FROM SUPABASE because of using learnig_history 