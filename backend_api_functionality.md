# Backend API Functionality Documentation

## Overview
Study Genie Backend API provides comprehensive learning progress tracking, student management, and LLM response processing functionality built with FastAPI.

**Base URL:** `/api`  
**Framework:** FastAPI  
**Database:** PostgreSQL with async support  

---

## API Endpoints

### 1. Health Check
- **Endpoint:** `GET /health`
- **Description:** Check API health status
- **Response:**
  ```json
  {
    "status": "healthy",
    "message": "Progress tracker API is running"
  }
  ```

---

## Student Management

### 2. Create Student
- **Endpoint:** `POST /api/students`
- **Description:** Create a new student or return existing one
- **Request Body:** `StudentCreate`
  ```json
  {
    "username": "string (3-50 chars)",
    "email": "string (valid email format)",
    "full_name": "string (2-100 chars)",
    "learning_preferences": {
      "optional": "object"
    }
  }
  ```
- **Response:** `StudentResponse`
  ```json
  {
    "student_id": "integer",
    "username": "string",
    "email": "string",
    "full_name": "string",
    "message": "string"
  }
  ```

### 3. Update Student Progress
- **Endpoint:** `POST /api/students/{student_id}/progress`
- **Description:** Update concept progress for a student
- **Path Parameters:**
  - `student_id`: integer
- **Request Body:** `ConceptProgressUpdate`
  ```json
  {
    "concept_id": "integer",
    "correct_answers": "integer (>=0)",
    "total_questions": "integer (>0)",
    "time_spent": "integer (>=0, optional, in seconds)",
    "activity_type": "enum (flashcard_practice, quiz_attempt, content_study, concept_review)"
  }
  ```
- **Response:**
  ```json
  {
    "message": "Progress updated successfully"
  }
  ```

### 4. Get Student Progress
- **Endpoint:** `GET /api/students/{student_id}/progress`
- **Description:** Retrieve student progress data
- **Path Parameters:**
  - `student_id`: integer
- **Query Parameters:**
  - `subject_id`: integer (optional)
- **Response:** `StudentProgressResponse`
  ```json
  {
    "student_id": "integer",
    "overall_progress": "object",
    "subject_progress": ["array of objects"],
    "concept_progress": [
      {
        "concept_id": "integer",
        "concept_name": "string",
        "status": "enum (not_started, in_progress, mastered, needs_review)",
        "mastery_score": "float (0-100)",
        "attempts_count": "integer",
        "correct_answers": "integer",
        "total_questions": "integer",
        "last_practiced": "datetime (optional)",
        "first_learned": "datetime (optional)"
      }
    ],
    "recent_activity": ["array of objects"],
    "total_concepts": "integer",
    "mastered_concepts": "integer",
    "weak_concepts": "integer"
  }
  ```

---

## LLM Processing

### 5. Process LLM Response
- **Endpoint:** `POST /api/process-llm-response`
- **Description:** Process and track LLM responses for learning content
- **Request Body:** `LLMResponseRequest`
  ```json
  {
    "student_id": "integer",
    "subject_name": "string",
    "chapter_name": "string",
    "concept_name": "string",
    "llm_response": {
      "flashcards": {
        "card_id": {
          "question": "string",
          "answer": "string",
          "difficulty": "enum (Easy, Medium, Hard)"
        }
      },
      "quiz": {
        "question_id": {
          "question": "string",
          "options": ["array of 2-6 strings"],
          "correct_answer": "string",
          "explanation": "string"
        }
      },
      "summary": "string",
      "learning_objectives": ["array of strings"]
    },
    "user_query": "string",
    "difficulty_level": "enum (Easy, Medium, Hard, optional, default: Medium)"
  }
  ```
- **Response:** `ProcessedLLMResponse`
  ```json
  {
    "enhanced_response": {
      "flashcards": "object",
      "quiz": "object",
      "summary": "string",
      "learning_objectives": ["array"]
    },
    "tracking_metadata": {
      "student_id": "integer",
      "subject_id": "integer",
      "chapter_id": "integer",
      "concept_id": "integer",
      "subject_name": "string",
      "chapter_name": "string",
      "concept_name": "string",
      "created_at": "datetime"
    },
    "created_entities": {
      "subject_created": "boolean",
      "chapter_created": "boolean",
      "concept_created": "boolean"
    },
    "message": "string"
  }
  ```

---

## Data Models and Schemas

### Enums

#### DifficultyLevel
- `EASY`
- `MEDIUM`
- `HARD`

#### ConceptStatus
- `NOT_STARTED`
- `IN_PROGRESS`
- `MASTERED`
- `NEEDS_REVIEW`

#### ActivityType
- `FLASHCARD_PRACTICE`
- `QUIZ_ATTEMPT`
- `CONTENT_STUDY`
- `CONCEPT_REVIEW`

#### RecommendationType
- `CONCEPT_REVIEW`
- `PRACTICE_MORE`
- `ADVANCE_TOPIC`
- `MAINTENANCE_PRACTICE`
- `WEAKNESS_FOCUS`

### Extended Models (Available but not yet exposed via API)

#### WeaknessAnalysis
```json
{
  "weakness_id": "integer",
  "concept_name": "string",
  "chapter_name": "string",
  "subject_name": "string",
  "weakness_type": "string",
  "error_pattern": "string (optional)",
  "frequency_count": "integer",
  "severity_score": "float",
  "last_occurrence": "datetime",
  "is_resolved": "boolean",
  "recommended_actions": ["array of strings"]
}
```

#### StudySessionHistory
```json
{
  "session_id": "integer",
  "subject_name": "string",
  "total_questions": "integer",
  "correct_answers": "integer",
  "accuracy": "float",
  "duration": "integer (seconds)",
  "concepts_covered": "integer",
  "started_at": "datetime",
  "completed_at": "datetime (optional)"
}
```

#### RecommendationResponse
```json
{
  "recommendation_id": "integer",
  "recommendation_type": "enum",
  "title": "string",
  "description": "string",
  "priority_score": "integer (1-10)",
  "concept_id": "integer (optional)",
  "concept_name": "string (optional)",
  "subject_name": "string (optional)",
  "is_active": "boolean",
  "is_completed": "boolean",
  "created_at": "datetime",
  "expires_at": "datetime (optional)"
}
```

#### StudentAnalyticsResponse
```json
{
  "student_id": "integer",
  "overall_stats": {
    "total_concepts": "integer",
    "mastered_concepts": "integer",
    "in_progress_concepts": "integer",
    "weak_concepts": "integer",
    "average_mastery_score": "float",
    "total_study_time": "integer (minutes)",
    "streak_days": "integer",
    "last_active": "datetime (optional)"
  },
  "subject_progress": [
    {
      "subject_id": "integer",
      "subject_name": "string",
      "total_concepts": "integer",
      "mastered_concepts": "integer",
      "average_score": "float",
      "time_spent": "integer (minutes)",
      "last_activity": "datetime (optional)",
      "progress_percentage": "float"
    }
  ],
  "activity_stats": [
    {
      "activity_type": "enum",
      "count": "integer",
      "average_score": "float (optional)",
      "total_time": "integer (minutes)",
      "last_activity": "datetime (optional)"
    }
  ],
  "weekly_progress": ["array of objects"],
  "learning_velocity": "float",
  "focus_areas": ["array of strings"],
  "achievements": ["array of strings"]
}
```

---

## Error Handling

All endpoints return standardized error responses:

```json
{
  "error": "string",
  "message": "string",
  "details": "object (optional)"
}
```

**Common HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (validation errors)
- `404`: Not Found
- `500`: Internal Server Error

---

## Authentication & Authorization

Currently, the API does not implement authentication. All endpoints are publicly accessible.

---

## Background Tasks

The API uses FastAPI's background tasks for:
- Recommendation generation after LLM response processing
- Analytics computation
- Progress optimization

---

## Database Integration

- **ORM:** Custom database manager with async support
- **Database:** PostgreSQL
- **Connection Management:** Dependency injection via `get_db_manager()`

---

## Technology Stack

- **Backend Framework:** FastAPI
- **Language:** Python 3.12+
- **Database:** PostgreSQL
- **ORM:** Custom async database manager
- **Validation:** Pydantic models
- **LLM Integration:** LangChain, OpenAI API
- **Additional:** Hugging Face Transformers, Tesseract OCR/Google Vision API

---

## Future Endpoints (Models Available)

The codebase includes models for additional functionality that could be exposed:

1. **Study Session Management**
2. **Weakness Analysis & Tracking**
3. **Recommendation System**
4. **Analytics Dashboard**
5. **Learning Path Generation**
6. **Batch Quiz/Flashcard Processing**
7. **Subject/Chapter/Concept Management**

These endpoints are not yet implemented but have complete data models ready for implementation.
