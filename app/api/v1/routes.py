from fastapi import APIRouter

from .students import router as students_router
from .llm import router as llm_router


router = APIRouter()

# Include modular routers
router.include_router(students_router)
router.include_router(llm_router)
