from fastapi import APIRouter

from .llm import router as llm_router
from .auth import router as auth_router
from .api_keys_supabase import router as api_keys_router
from .users_supabase import router as users_router
from .students import router as students_router  # Now using modular students
from .analytics import router as analytics_router

router = APIRouter()

# Include modular routers
router.include_router(llm_router)
router.include_router(auth_router)
router.include_router(api_keys_router)
router.include_router(users_router)
router.include_router(students_router)
router.include_router(analytics_router)
