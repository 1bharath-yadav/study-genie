


ALTER TABLE study_sessions ALTER COLUMN student_id TYPE UUID USING student_id::uuid;
ALTER TABLE study_sessions ALTER COLUMN subject_id_id TYPE UUID USING subject_id::uuid;

