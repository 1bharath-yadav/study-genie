"""Redis client dependency (async) for StudyGenie"""
import os
from typing import Optional
import redis.asyncio as redis

_client: Optional[redis.Redis] = None


def get_redis_url() -> str:
    return os.getenv('REDIS_URL', 'redis://localhost:6379/0')


def create_redis_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(get_redis_url(), decode_responses=True)
    return _client


def get_redis() -> redis.Redis:
    # Return the shared client (startup should have created it)
    return create_redis_client()


async def close_redis():
    global _client
    if _client is not None:
        try:
            await _client.close()
        except Exception:
            pass
        _client = None
