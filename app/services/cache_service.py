"""Simple Redis-backed cache helpers for StudyGenie.

This provides small helpers for cache-aside patterns and TTL management.
"""
import json
import hashlib
from typing import Any, Optional
from app.deps.redis_client import get_redis


def _key(ns: str, *parts: object) -> str:
    base = ":".join(map(str, parts))
    return f"{ns}:{hashlib.sha1(base.encode()).hexdigest()}"


async def get_cached(ns: str, *parts: object) -> Optional[Any]:
    r = get_redis()
    key = _key(ns, *parts)
    raw = await r.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


async def set_cached(value: Any, ttl: int, ns: str, *parts: object) -> None:
    r = get_redis()
    key = _key(ns, *parts)
    raw = json.dumps(value)
    await r.setex(key, ttl, raw)


async def del_cached(ns: str, *parts: object) -> None:
    r = get_redis()
    key = _key(ns, *parts)
    await r.delete(key)
