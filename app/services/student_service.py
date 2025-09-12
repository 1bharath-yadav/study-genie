# Pure functional student service
from typing import Optional, List
from app.models import StudentCreate, StudentUpdate, StudentResponse
from app.services.db import (
    create_student as db_create_student,
    get_student_by_id as db_get_student_by_id,
    get_students_by_user_id as db_get_students_by_user_id,
    update_student as db_update_student,
    delete_student as db_delete_student,
)
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


def create_student(student_data: StudentCreate, user_id: str) -> Optional[StudentResponse]:
    """Create a new student for a user."""
    try:
        # Prepare student data with user_id and timestamps
        db_data = {
            "id": str(uuid.uuid4()),
            "name": student_data.name,
            "email": student_data.email,
            "grade_level": student_data.grade_level,
            "subjects": student_data.subjects or [],
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        result = db_create_student(db_data)
        
        if result:
            return StudentResponse(**result)
        return None
        
    except Exception as e:
        logger.error(f"Failed to create student: {str(e)}")
        return None


def get_student_by_id(student_id: str) -> Optional[StudentResponse]:
    """Get a student by ID."""
    try:
        result = db_get_student_by_id(student_id)
        if result:
            return StudentResponse(**result)
        return None
    except Exception as e:
        logger.error(f"Failed to get student {student_id}: {str(e)}")
        return None


def get_students_by_user(user_id: str) -> List[StudentResponse]:
    """Get all students for a user."""
    try:
        results = db_get_students_by_user_id(user_id)
        return [StudentResponse(**student) for student in results]
    except Exception as e:
        logger.error(f"Failed to get students for user {user_id}: {str(e)}")
        return []


def update_student_data(student_id: str, student_data: StudentUpdate) -> Optional[StudentResponse]:
    """Update student data."""
    try:
        # Only include non-None values in update
        update_data = {}
        if student_data.name is not None:
            update_data["name"] = student_data.name
        if student_data.email is not None:
            update_data["email"] = student_data.email
        if student_data.grade_level is not None:
            update_data["grade_level"] = student_data.grade_level
        if student_data.subjects is not None:
            update_data["subjects"] = student_data.subjects
        
        if not update_data:
            return get_student_by_id(student_id)
        
        result = db_update_student(student_id, update_data)
        if result:
            return StudentResponse(**result)
        return None
        
    except Exception as e:
        logger.error(f"Failed to update student {student_id}: {str(e)}")
        return None


def delete_student_by_id(student_id: str) -> bool:
    """Delete a student by ID."""
    try:
        return db_delete_student(student_id)
    except Exception as e:
        logger.error(f"Failed to delete student {student_id}: {str(e)}")
        return False


def check_student_ownership(student_id: str, user_id: str) -> bool:
    """Check if a student belongs to a user."""
    try:
        student = db_get_student_by_id(student_id)
        return student is not None and student.get("user_id") == user_id
    except Exception as e:
        logger.error(f"Failed to check student ownership: {str(e)}")
        return False
