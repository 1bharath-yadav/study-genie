# Ensure project root is importable so tests can `import app`
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optionally configure pytest-asyncio default
try:
    import asyncio
    import pytest

    @pytest.fixture(scope='session')
    def event_loop():
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
except Exception:
    # keep file safe if pytest isn't present when this file is read
    pass
