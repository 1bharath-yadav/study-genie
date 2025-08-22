import asyncio
import json
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


class FakeConn:
    def __init__(self, state):
        self.state = state

    async def fetchrow(self, query, *args):
        # Very small heuristic: if query contains "SELECT student_id FROM students" return None
        return None

    async def fetchval(self, query, *args):
        # When inserts return an id
        # Use table hints in query to choose a counter
        if "INSERT INTO students" in query or "RETURNING student_id" in query:
            self.state['student_id'] += 1
            return self.state['student_id']
        if "INSERT INTO subjects" in query or "RETURNING subject_id" in query:
            self.state['subject_id'] += 1
            return self.state['subject_id']
        if "INSERT INTO chapters" in query or "RETURNING chapter_id" in query:
            self.state['chapter_id'] += 1
            return self.state['chapter_id']
        if "INSERT INTO concepts" in query or "RETURNING concept_id" in query:
            self.state['concept_id'] += 1
            return self.state['concept_id']
        # Return defaults for MAX(...) queries used to compute ordering
        if "MAX(chapter_order" in query or "COALESCE(MAX(chapter_order)" in query:
            return 0
        if "MAX(concept_order" in query or "COALESCE(MAX(concept_order)" in query:
            return 0
        # default
        return None

    async def fetch(self, query, *args):
        return []

    async def execute(self, query, *args):
        return None


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, state):
        self.state = state

    def acquire(self):
        return FakeAcquire(FakeConn(self.state))


class FakeDBManager:
    def __init__(self):
        # counters
        self._state = {
            'student_id': 0,
            'subject_id': 0,
            'chapter_id': 0,
            'concept_id': 0,
        }
        self.pool = FakePool(self._state)

    async def initialize_pool(self):
        return None

    async def create_tables(self):
        return None

    async def close_pool(self):
        return None


@pytest.fixture(autouse=True)
def override_db(monkeypatch):
    """Override the app dependency get_db_manager to return a fake DB manager."""
    from app.main import get_db_manager, app as fastapi_app

    fake = FakeDBManager()

    async def _get_db():
        return fake

    # Use FastAPI's dependency_overrides so the route dependencies use our fake
    fastapi_app.dependency_overrides[get_db_manager] = _get_db

    try:
        yield fake
    finally:
        # Cleanup override
        fastapi_app.dependency_overrides.pop(get_db_manager, None)


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'healthy'


@pytest.mark.asyncio
async def test_create_student(override_db):
    payload = {
        "username": "alice",
        "email": "alice@example.com",
        "full_name": "Alice Example"
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post('/api/students', json=payload)

    assert r.status_code == 200
    data = r.json()
    assert data['username'] == payload['username']
    assert data['email'] == payload['email']
    assert 'student_id' in data


@pytest.mark.asyncio
async def test_process_llm_response_minimal(override_db):
    # Build a minimal valid LLMResponseRequest payload
    payload = {
        "student_id": 1,
        "subject_name": "Math",
        "chapter_name": "Algebra",
        "concept_name": "Linear Equations",
        "llm_response": {
            "flashcards": {
                "c1": {"question": "Q?", "answer": "A", "difficulty": "Easy"}
            },
            "quiz": {
                "q1": {"question": "Q?", "options": ["A", "B"], "correct_answer": "A", "explanation": "e"}
            },
            "summary": "This is a summary with enough length to avoid warnings.",
            "learning_objectives": ["Understand linear eqs"]
        },
        "user_query": "Explain",
        "difficulty_level": "Medium"
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post('/api/process-llm-response', json=payload)

    assert r.status_code == 200
    data = r.json()
    # The response_model in the route includes keys: enhanced_response, tracking_metadata, created_entities, message
    assert 'enhanced_response' in data
    assert 'tracking_metadata' in data
    assert 'created_entities' in data


@pytest.mark.asyncio
async def test_update_progress_and_get(override_db):
    # First, create student to get an id
    student_payload = {"username": "bob",
                       "email": "bob@example.com", "full_name": "Bob"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post('/api/students', json=student_payload)
        sid = r.json()['student_id']

        # Update concept progress
        progress_payload = {"concept_id": 1, "correct_answers": 8,
                            "total_questions": 10, "time_spent": 120}
        r2 = await ac.post(f'/api/students/{sid}/progress', json=progress_payload)
        assert r2.status_code == 200

        # Get student progress
        r3 = await ac.get(f'/api/students/{sid}/progress')
        assert r3.status_code == 200
        pdata = r3.json()
        assert pdata['student_id'] == sid
