## StudyGenie — Copilot instructions (short, practical)

This file captures the essential knowledge an AI coding agent needs to be productive in this repository.

1) Big-picture architecture
- FastAPI backend: entrypoint is `app/main.py` (lifespan startup/teardown, CORS, `/health`).
- Services layer: pure-function style helpers live under `app/services/*` (DB, student, api_key, model preferences).
- Core domain helpers: `app/core/*` (auth flows, encryption helpers). Example: `app/core/auth_service.py` exposes `handle_user_signin` and `get_user_api_key_for_llm`.
- LLM stack: under `app/llm/` — provider catalog and factories (`providers.py`), streaming chat (`chat.py`), RAG/langchain pipeline (`langchain.py`), and orchestration (`orchestrator.py`).
- API surface: routes under `app/api/v1` (router mounted in `app/main.py` as `/api/v1`).
- Persistence and infra: Supabase client in `app/services/db.py` (uses `SUPABASE_URL`/`SUPABASE_API_KEY`), optional Redis in `app/deps/redis_client.py`.

2) Key integration & data-flow patterns
- Model preferences are stored and queried via `app/services/model_preference_service.py` and consumed by `app/llm/orchestrator.py` (see `resolve_model_and_key_for_user`).
- Model identifier convention: stored `model_id` is formatted as `"{provider}-{model_name}"`. Many places split on `-` to infer provider (see `orchestrator.resolve_model_and_key_for_user`).
- Provider definitions and metadata live in `app/llm/providers.py` (`PROVIDERS_JSON`, `get_models_for_provider`). To add a provider: update `PROVIDERS_JSON` and implement provider/model factory functions.
- API keys: user API keys live in `user_api_keys` Supabase table and are managed by `app/services/student_service.py` / `app/services/api_key_service.py`. Auth helpers decrypt keys in `app/core/auth_service.py`.
- Streaming: NDJSON streaming is used for long-running LLM flows. `app/llm/orchestrator.py` returns `StreamingResponse(..., media_type="application/x-ndjson")` and `langchain.py` yields async dicts. Error streams use small JSON objects with `status: error` and `error_type`.
- RAG/embeddings: `app/llm/langchain.py` expects an embedding factory exposed by `app/llm/providers` (search for `get_embeddings` usage). Vector store is FAISS via LangChain wrappers. Persistence path is `PERSIST_DIR/faiss_{session_id}` (see `save_vector_store_and_chunks` and `_faiss_session_dir`).

3) Concurrency & I/O patterns
- Startup uses `lifespan` in `app/main.py` to initialize Supabase check and Redis (`create_redis_client`). Close Redis with `close_redis()` on shutdown.
- Mixed sync/async: Supabase client calls are synchronous (plain functions in `app/services/db.py`), whereas orchestrator and LLM pipelines use `async` and async generators. When calling sync DB helpers from async routes, the project currently does direct calls (no threadpool wrapper) — keep changes minimal and follow existing sync/async patterns.

4) Project-specific conventions and pitfalls
- Prefer pure, testable functions in `app/services/*` and `app/core/*` (they return domain types or None/error). Side effects (I/O) are centralized in service modules.
- Model/provider format: the UI and orchestration code expect `model_id` values like `google-gemini-2.5-pro` or `openai-gpt-4o`.
- Structured vs chat flows: `orchestrate_prompt_stream` uses `selected_content_types` (JSON list or non-empty string) to decide the structured RAG pipeline vs plain chat. Uploaded files alone do not force RAG — explicit `selected_content_types` must be present.
- Embedding factory required: `langchain.setup_vector_store_and_retriever` will raise if `app.llm.providers` does not expose a `get_embeddings(provider, model, key)` factory. Implement or wire one when adding a new embedding provider.
- FAISS persistence: indexes are stored per-session under `PERSIST_DIR` (see config) — be careful not to delete the persistent uploads dir when cleaning up temp dirs.

5) Developer workflows (commands you can run)
- This project uses `uv` (astral) to manage the per-project virtual environment and dependency sync. The included Dockerfile also uses `uv sync`.
- Install / sync dependencies (creates `.venv` and installs from `pyproject.toml` / `requirements.txt`):

  uv sync

- Add a new dependency (installs into the `uv` environment and updates lock):

  uv add <package>

- Run the dev server inside the `uv` environment (ensure `.env` or env vars set; see `app/config.py`):

  uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

- Run tests inside the `uv` environment:

  uv run pytest -q

- Docker: `docker build -t study-genie .` and run with required envs (the included Dockerfile installs `uv` and uses `uv sync`).

6) Important env vars / config
- See `app/config.py` for the canonical list. Most important in CI/dev:
  - SUPABASE_URL, SUPABASE_API_KEY, SUPABASE_JWT_SECRET
  - REDIS_URL (optional)
  - GEMINI_API_KEY / GEMINI_MODEL (if using Google provider)
  - PERSIST_DIR and TEMP_DIR influence where uploads and FAISS indices are stored

7) Where to look for examples
- Streaming NDJSON pattern: `app/llm/orchestrator.py` (error streams, structured_gen wrapper).
- Provider/model resolution: `app/llm/orchestrator.py::resolve_model_and_key_for_user`.
- Provider catalog & agent factory: `app/llm/providers.py` (`PROVIDERS_JSON`, `_create_model`, `create_learning_agent`, `create_chat_agent`).
- FAISS persistence/loads: `app/llm/langchain.py` (`save_vector_store_and_chunks`, `load_vector_store_and_chunks`).
- Supabase usage: `app/services/db.py` and `app/services/student_service.py` (table names: `students`, `user_api_keys`).

8) Small, actionable rules for automated edits
- When adding a new provider: update `PROVIDERS_JSON`, add model/provider factory in `app/llm/providers.py`, and (if embeddings) expose `get_embeddings(provider, model, key)` from that module.
- When changing the model preference schema, update `orchestrator.resolve_model_and_key_for_user` and ensure `model_id` parsing stays compatible.
- Preserve NDJSON streaming shape: all streaming steps yield dict-like objects that are JSON-serialized line-by-line. Keep `status` fields for observability.

If anything above is unclear or you'd like me to expand a section (examples, tests, or a quick-start script), tell me which part and I will iterate.
