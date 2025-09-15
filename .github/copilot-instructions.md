This repository implements StudyGenie: a FastAPI backend + React frontend that orchestrates RAG/LLM workflows (LangChain / pydantic-ai / Google Gemini). The objective of these instructions is to help automated coding agents be immediately productive by surfacing the project's architecture, conventions, and important integration touchpoints.

Keep guidance short and concrete. When editing code, prefer small, well-tested changes and follow the explicit patterns below.

Key facts (read before editing)
- Backend entry: `app/main.py` (FastAPI app, lifespan handles startup DB checks). Include routes under `/api/v1` via `app/api/v1/routers.py`.
- Settings/config: `app/config.py` uses Pydantic Settings; environment variables are authoritative (.env used in dev).
- LLM layer: `app/llm/langchain.py` is the central orchestration for providers (pydantic-ai wrappers). It:
  - Expects per-student/per-user API keys (do not hardcode server-wide keys).
  - Uses PROVIDER_CAPABILITIES and PROVIDER_CONFIGS in `app/llm/providers.py` (models are activated by users).
  - Produces structured learning content validated by Pydantic models defined in the same file (e.g., LearningContent, Flashcard, QuizQuestion).
- Data models: `app/models.py` contains the canonical Pydantic models used across routers/services. Update it when API schemas change.
- DB & external: project expects Supabase/Postgres (env: SUPABASE_URL, SUPABASE_API_KEY) and a vector store (FAISS). See `pyproject.toml` / `requirements.txt` for dependencies.

Project conventions and patterns
- Per-user API keys: APIs require storing/retrieving per-user provider keys. See `app/llm/langchain.py` and `app/models.py` for API key models and checks. Never read/write global secrets in code changes—use `app/services/api_key_service` hooks.
- Structured LLM outputs: LLM responses are expected to conform to Pydantic schemas (validate with parse_obj). When adding new output shapes, add/update models in `app/llm/langchain.py` and `app/models.py` and include schema validation in the generation pipeline.
- Provider abstraction: The repo uses provider-agnostic wrappers (pydantic-ai Agent + per-provider Provider classes). When adding providers:
  - Register capabilities in `app/llm/providers.py`
  - Provide an embeddings handler in PROVIDER_CONFIGS if embeddings are supported
  - Follow the pattern in `create_llm_agent()` inside `app/llm/langchain.py`.
- File processing flow: Document ingestion -> chunking -> embeddings -> vector store -> retriever -> LLM. The helpers live under `app/llm/process_files/` (use these helpers rather than reimplementing chunking/embedding logic).

Developer workflows (commands & quick checks)
- Start backend dev server (reload):
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
- Frontend dev server (in `frontend/`):
  cd frontend && npm install && npm run dev
- Run tests (pytest is configured in dependencies):
  pytest -q
- Lint/format (project uses black / pre-commit optionally):
  black .

Important integration points & files to inspect before changes
- `app/main.py` — app lifecycle and CORS; running health checks calls `app.services.db.test_connection`.
- `app/config.py` — authoritative configuration via Pydantic Settings; prefer adding env vars here when needed.
- `app/llm/langchain.py` — LLM orchestration, schemas, provider creation, and structured response formatting. Most LLM-related edits belong here.
- `app/llm/providers.py` — provider capability registry. Add models/capabilities here for new providers.
- `app/llm/process_files/` — document loaders, chunking, embeddings, and vector store wiring.
- `app/api/v1/` — individual route modules (auth, student, api_key, provider, llm, analytics). Add API endpoints here; follow existing route structure and Pydantic request/response models.
- `app/models.py` — canonical Pydantic models used by routers and services. Keep it consistent with API responses.
- `services/*` — business logic (DB interaction, API key storage, analytics). Prefer small service-level unit changes over large router modifications.

Examples and gotchas
- Per-user keys: code in `app/llm/langchain.py` explicitly states "Do NOT use server-level provider API keys. Always require per-user keys." Ensure any new LLM call path fetches keys through `app.services.api_key_service`.
- Validation-first: The project prefers validating LLM outputs with Pydantic. If you alter prompts or output format, update the Pydantic models and the `format_learning_content_response()` validator.
- Default provider: There is intentionally no default provider (DEFAULT_PROVIDER = ""). Code should error clearly if no provider/model is selected.
- Named entrypoints: API routes are namespaced under `/api/v1` (see `app/api/v1/routers.py`). Add new routes there, and include them in the router pattern used in that module.

When you can't find something
- Look for the service under `app/services/` (e.g., `api_key_service`, `student_service`, `db`). Business logic lives there.
- Search for Pydantic models in `app/models.py` before inventing new shapes; reuse shared models for consistency.

If you modify runtime behavior
- Update `pyproject.toml` or `requirements.txt` if a new dependency is needed and mention it in the change description.
- Add small unit tests (pytest/pytest-asyncio) for new logic under a tests/ folder; run `pytest` locally.

Feedback
If anything in this instruction file is unclear or missing (for example: expected file paths under `app/services` weren't found or there's a newer provider pattern), tell me which area to expand and I will iterate.
 
Service summaries (quick reference)
- `app/services/api_key_service.py` — High-level helpers for per-user API keys. Use `get_api_key_for_provider(student_id, provider_name)` to obtain a decrypted key at runtime, and `store_api_key(...)` to persist encrypted keys. Never store or read global provider secrets; always use these helpers.
- `app/services/db.py` — Supabase wrapper: `create_supabase_client()`, `test_connection()` and safe extract helpers. Most DB operations go through this module.
- `app/services/student_service.py` — Business logic for students and API key CRUD. Important functions: `create_api_key()`, `get_api_keys_by_user_id()`, `get_active_api_key_by_provider()`, `set_default_api_key()`.

LLM & document processing (quick reference)
- `app/llm/providers.py` — Provider capability registry and model/provider factory helpers. See `PROVIDER_CAPABILITIES` and `_create_model()` / `create_learning_agent()` for the pattern when adding providers.
- `app/llm/langchain.py` — Central orchestration: LLM agent creation (follow `create_llm_agent()` pattern), structured Pydantic schemas (LearningContent, Flashcard, QuizQuestion), and `format_learning_content_response()` validator.
- `app/llm/process_files/file_process.py` — Document ingestion and processing pipeline. Key functions/constants:
  - `DEFAULT_CONFIG` / `ProcessingConfig` — chunk size, overlap, model and embedding defaults.
  - `load_documents_from_files(file_paths, api_key, config)` — loads/extracts text from various file types.
  - `create_document_chunks(documents, config)` — chunking via LangChain text splitter.
  - Vision / LLM extraction fallbacks: `extract_text_with_unstructured()` and `extract_text_with_llm_vision()`.

Quick examples
- Fetch a user's provider key (decrypted):
  - `api_key = await app.services.api_key_service.get_api_key_for_provider(student_id, "google")`
- Load and chunk files for RAG:
  - `docs = await app.llm.process_files.file_process.load_documents_from_files(["/tmp/file.pdf"], api_key=api_key)`
  - `chunks = app.llm.process_files.file_process.create_document_chunks(docs)`

- write concise,pure functional programming code

- remve the redundent files immediately, keep code concise,modularise
- modify the supabase migration that is suitable for our code,      this is development phase so we can delete and modify any tables.
 - the providers and models should come from   PROVIDERS_JSON  all providers related functions should in app/llm/providers.py
 - Never add fallbacks and extra clutter , keep code simple,concise.
If you want, I can expand these entries into a short checklist of functions to call when implementing a new end-to-end flow (ingest -> embed -> store -> retrieve -> generate) and add code snippets/tests for them.
