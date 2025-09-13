# Project Functionality

This document outlines the project structure and provides a detailed explanation of the backend API endpoints.

## Project Structure

The project is a full-stack application with a Python backend using FastAPI and a React frontend.

### Backend (`app/`)

The backend is responsible for handling business logic, data storage, and communication with external services like LLMs.

-   `main.py`: The main entry point for the FastAPI application.
-   `config.py`: Handles application configuration and settings.
-   `models.py`: Contains Pydantic models for data validation and serialization.
-   `api/`: Contains the API routes.
    -   `v1/`: Version 1 of the API.
        -   `main.py`: Combines all the v1 routers.
        -   `auth.py`: Handles user authentication via Google OAuth.
        -   `student_routes.py`: CRUD operations for student data.
        -   `api_key_routes.py`: CRUD operations for user's LLM provider API keys.
        -   `provider_routes.py`: Routes to get available LLM providers and models.
        -   `llm_routes.py`: Routes for interacting with LLMs for text generation, study content creation, etc.
        -   `analytics_routes.py`: Routes for fetching analytics data.
-   `core/`: Core application logic.
    -   `auth_service.py`: Service for handling user sign-in logic.
    -   `dependencies.py`: FastAPI dependencies.
    -   `encryption.py`: Utilities for encrypting and decrypting API keys.
    -   `security.py`: Handles JWT generation and user authentication.
-   `db/`: Database interaction layer.
    -   `db_client.py`: Supabase client setup.
-   `llm/`: LLM-related logic.
    -   `langchain.py`: LangChain specific implementations.
    -   `llm_orchestrator.py`: Orchestrates requests to different LLM providers.
    -   `provider_service.py`: Service for managing LLM providers and models.
    -   `providers.py`: Concrete implementations for different LLM providers (e.g., OpenAI, Google).
    -   `types.py`: Pydantic models for LLM requests and responses.
    -   `process_files/file_process.py`: Logic for processing uploaded files.
-   `services/`: Business logic services.
    -   `analytics_service.py`: Service for analytics-related logic.
    -   `api_key_service.py`: Service for managing API keys.
    -   `db.py`: General database service functions.
    -   `student_service.py`: Service for student-related logic.

### Frontend (`frontend/`)

The frontend is a React application built with Vite, providing the user interface for the application.

-   `src/`: The main source code for the React application.
    -   `main.tsx`: The entry point of the React app.
    -   `App.tsx`: The root component of the application.
    -   `components/`: Reusable UI components.
        -   `charts/`: Components for displaying analytics charts.
        -   `chat/`: Chat interface components.
        -   `dashboard/`: Dashboard components.
        -   `layout/`: Layout components like the main app layout.
        -   `study/`: Components for study activities like flashcards and quizzes.
        -   `ui/`: Generic UI components (buttons, cards, etc.).
        -   `upload/`: File upload components.
    -   `contexts/`: React contexts (e.g., `AuthContext`).
    -   `hooks/`: Custom React hooks.
    -   `lib/`: Utility functions and libraries.
    -   `pages/`: Top-level page components for different routes.
    -   `types/`: TypeScript type definitions.
    -   `utils/`: Utility functions.

## Backend API Routes

The backend exposes a RESTful API for the frontend to consume.

### Authentication (`/api/v1/auth`)

Handles user authentication.

-   **GET /auth/login**
    -   **Description:** Initiates the Google OAuth2 login flow by redirecting the user to the Google consent screen.
    -   **Request:** None.
    -   **Response:** A `307 Temporary Redirect` to Google's authentication URL.

-   **GET /auth/callback**
    -   **Description:** Handles the callback from Google after the user has authenticated. It exchanges the authorization code for an access token, fetches user information, signs the user in (creating an account if necessary), and returns a custom JWT to the frontend.
    -   **Request Query Parameters:**
        -   `code`: The authorization code from Google.
        -   `error` (optional): An error message if authentication failed.
    -   **Response:** A `307 Temporary Redirect` to the frontend URL (`/`) with a `token` in the query parameters on success, or an `error` on failure.

### Students (`/api/v1/students`)

CRUD operations for student data. Requires authentication.

-   **POST /**
    -   **Description:** Creates a new student profile associated with the authenticated user.
    -   **Request Body:** `StudentCreate` model.
        ```json
        {
          "name": "string",
          "email": "string"
        }
        ```
    -   **Response:** `StudentResponse` model.
        ```json
        {
          "id": "string",
          "name": "string",
          "email": "string",
          "student_id": "string",
          "created_at": "string"
        }
        ```

-   **GET /**
    -   **Description:** Retrieves all student profiles associated with the authenticated user.
    -   **Request:** None.
    -   **Response:** A list of `StudentResponse` models.

-   **GET /{student_id}**
    -   **Description:** Retrieves a specific student profile by its ID.
    -   **Request Path Parameter:**
        -   `student_id`: The ID of the student to retrieve.
    -   **Response:** `StudentResponse` model.

-   **PUT /{student_id}**
    -   **Description:** Updates a student's profile.
    -   **Request Path Parameter:**
        -   `student_id`: The ID of the student to update.
    -   **Request Body:** `StudentUpdate` model.
        ```json
        {
          "name": "string"
        }
        ```
    -   **Response:** `StudentResponse` model.

-   **DELETE /{student_id}**
    -   **Description:** Deletes a student profile.
    -   **Request Path Parameter:**
        -   `student_id`: The ID of the student to delete.
    -   **Response:**
        ```json
        {
          "message": "Student deleted successfully"
        }
        ```

### API Keys (`/api/v1/api-keys`)

Manages API keys for different LLM providers. Requires authentication.

-   **POST /**
    -   **Description:** Creates and stores a new encrypted API key for a specific provider for the authenticated user.
    -   **Request Body:** `APIKeyCreate` model.
        ```json
        {
          "provider_id": "string", // e.g., "openai", "google"
          "api_key": "string"
        }
        ```
    -   **Response:** `APIKeyResponse` model.
        ```json
        {
          "id": "string",
          "provider_id": "string",
          "provider_name": "string",
          "provider_display_name": "string",
          "is_active": true,
          "is_default": false,
          "student_id": "string",
          "created_at": "string"
        }
        ```

-   **GET /**
    -   **Description:** Retrieves all API keys for the authenticated user.
    -   **Request:** None.
    -   **Response:** A list of `APIKeyResponse` models.

-   **DELETE /{api_key_id}**
    -   **Description:** Deactivates an API key.
    -   **Request Path Parameter:**
        -   `api_key_id`: The ID of the API key to deactivate.
    -   **Response:**
        ```json
        {
          "message": "API key deactivated successfully"
        }
        ```

### Providers (`/api/v1/providers`)

Provides information about available LLM providers and models.

-   **GET /**
    -   **Description:** Gets a list of all available LLM providers.
    -   **Request:** None.
    -   **Response:** A list of `ProviderResponse` models.
        ```json
        [
          {
            "id": "string",
            "name": "string",
            "display_name": "string"
          }
        ]
        ```

-   **GET /{provider}/models**
    -   **Description:** Gets all models for a specific provider.
    -   **Request Path Parameter:**
        -   `provider`: The name of the provider (e.g., "openai").
    -   **Response:** A list of `ModelResponse` models.
        ```json
        [
          {
            "id": "string",
            "name": "string",
            "provider_id": "string",
            "model_type": "string" // "CHAT" or "EMBEDDING"
          }
        ]
        ```

-   **GET /models/chat**
    -   **Description:** Gets all chat models, optionally filtered by provider.
    -   **Request Query Parameter:**
        -   `provider` (optional): The name of the provider.
    -   **Response:** A list of `ModelResponse` models.

-   **GET /models/embedding**
    -   **Description:** Gets all embedding models, optionally filtered by provider.
    -   **Request Query Parameter:**
        -   `provider` (optional): The name of the provider.
    -   **Response:** A list of `ModelResponse` models.

-   **GET /models/available/chat**
    -   **Description:** Gets chat models available to the current user based on their saved API keys. Requires authentication.
    -   **Request:** None.
    -   **Response:** A list of `ModelResponse` models.

-   **GET /models/available/embedding**
    -   **Description:** Gets embedding models available to the current user based on their saved API keys. Requires authentication.
    -   **Request:** None.
    -   **Response:** A list of `ModelResponse` models.

### LLM (`/api/v1/llm`)

Endpoints for interacting with Large Language Models. Requires authentication.

-   **POST /generate**
    -   **Description:** Generates text using an LLM based on a prompt and model ID.
    -   **Request Body:** `GenerateTextRequest` model.
        ```json
        {
          "prompt": "string",
          "model_id": "string",
          "max_tokens": 1000,
          "temperature": 0.7,
          "system_prompt": "string"
        }
        ```
    -   **Response:** `LLMResponse` model.
        ```json
        {
          "text": "string"
        }
        ```

-   **POST /study-content**
    -   **Description:** Generates structured study content from a prompt.
    -   **Request Body:** `StudyContentRequest` model.
        ```json
        {
          "prompt": "string",
          "model_id": "string"
        }
        ```
    -   **Response:** `StudyContent` model.
        ```json
        {
          "key_concepts": ["string"],
          "summary": "string",
          "important_figures": ["string"]
        }
        ```

-   **POST /qa**
    -   **Description:** Generates question-answer pairs from a prompt.
    -   **Request Body:** `QARequest` model.
        ```json
        {
          "prompt": "string",
          "model_id": "string"
        }
        ```
    -   **Response:** `QuestionAnswer` model.
        ```json
        {
          "questions": [
            {
              "question": "string",
              "answer": "string"
            }
          ]
        }
        ```

-   **GET /providers**
    -   **Description:** Gets the LLM providers available to the current user based on their saved API keys.
    -   **Request:** None.
    -   **Response:** A list of provider dictionaries.

### Analytics (`/api/v1/analytics`)

Endpoints for retrieving analytics data. Requires authentication.

-   **GET /dashboard**
    -   **Description:** Gets dashboard analytics data for the currently authenticated user.
    -   **Request Query Parameter:**
        -   `days` (optional, default: 30): Number of days to analyze.
    -   **Response:**
        ```json
        {
          "success": true,
          "data": { ... } // Analytics data object
        }
        ```

-   **GET /{student_identifier}/dashboard**
    -   **Description:** Gets dashboard analytics for a specific student (for admin/teacher roles).
    -   **Request Path Parameter:**
        -   `student_identifier`: The ID or unique identifier of the student.
    -   **Request Query Parameter:**
        -   `days` (optional, default: 30): Number of days to analyze.
    -   **Response:**
        ```json
        {
          "success": true,
          "data": { ... } // Analytics data object
        }
        ```

-   **GET /{student_identifier}/subjects**
    -   **Description:** Gets detailed subject-wise analytics for a student.
    -   **Request Path Parameter:**
        -   `student_identifier`: The ID or unique identifier of the student.
    -   **Request Query Parameter:**
        -   `days` (optional, default: 30): Number of days to analyze.
    -   **Response:**
        ```json
        {
          "success": true,
          "data": { ... } // Subject-wise analytics data
        }
        ```
