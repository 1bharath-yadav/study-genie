from fastapi import APIRouter
from app.services.cache_service import get_cached, set_cached

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get('/profile/{user_id}')
async def cached_profile(user_id: str):
    """Example cache-aside: return cached profile if present, otherwise simulate DB fetch and cache it."""
    key_ns = 'profile'
    cached = await get_cached(key_ns, user_id)
    if cached:
        return {"from_cache": True, "profile": cached}

    # Simulate DB fetch (replace with actual DB call)
    profile = {"id": user_id, "name": f"User {user_id}", "picture": None}

    # cache for 5 minutes
    await set_cached(profile, 300, key_ns, user_id)
    return {"from_cache": False, "profile": profile}
