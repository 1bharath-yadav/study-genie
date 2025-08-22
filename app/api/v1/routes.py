from fastapi import APIRouter

from .students import router as students_router
from .llm import router as llm_router
from .generation import router as generation_router


router = APIRouter()

# Include modular routers
router.include_router(students_router)
router.include_router(llm_router)
router.include_router(generation_router)
