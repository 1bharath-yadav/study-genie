"""
Authentication service that handles user sign-in and API key retrieval
"""
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class AuthService:
    """Service for handling authentication and API key management during sign-in"""

    def __init__(self):
        self.supabase = get_supabase_client()

    async def handle_user_signin(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle user sign-in process including:
        1. Create/update user in database
        2. Retrieve API keys if available
        3. Return user info with API key status
        """
        try:
            student_id = user_data.get("sub")
            if not student_id:
                raise ValueError("User ID (sub) is required")

            # Create or update student record
            student_info = await self._ensure_student_exists(user_data)

            # Check for existing API keys
            api_key_info = await self._get_user_api_key_info(student_id)

            # Prepare response with user info and API key status
            signin_response = {
                "user": student_info,
                "api_key_status": {
                    "has_api_key": api_key_info is not None,
                    "service": api_key_info.get("service") if api_key_info else None,
                    "created_at": api_key_info.get("created_at") if api_key_info else None,
                    "updated_at": api_key_info.get("updated_at") if api_key_info else None
                },
                "signin_timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"User {student_id} signed in successfully. API key status: {signin_response['api_key_status']['has_api_key']}")
            return signin_response

        except Exception as e:
            logger.error(f"Error handling user sign-in: {e}")
            raise

    async def _ensure_student_exists(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update student record"""
        try:
            student_id = user_data.get("sub")
            if not student_id:
                raise ValueError("Student ID is required")

            # Try to get existing student
            existing_student = await self.supabase.get_student(str(student_id))

            if existing_student:
                # Update existing student with latest info
                update_data = {
                    "email": user_data.get("email", existing_student.get("email", "")),
                    "full_name": user_data.get("name", existing_student.get("full_name", "")),
                    "updated_at": datetime.now().isoformat()
                }
                return await self.supabase.update_student(str(student_id), update_data)
            else:
                # Create new student
                student_data = {
                    "student_id": str(student_id),
                    # Use email prefix as username
                    "username": user_data.get("email", "").split("@")[0],
                    "email": user_data.get("email", ""),
                    "full_name": user_data.get("name", "")
                }
                return await self.supabase.create_student(student_data)

        except Exception as e:
            logger.error(f"Error ensuring student exists: {e}")
            raise

    async def _get_user_api_key_info(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get user's API key information without the actual key"""
        try:
            return await self.supabase.get_api_key_info(student_id)
        except Exception as e:
            logger.error(f"Error getting API key info for {student_id}: {e}")
            return None

    async def get_user_api_key_for_llm(self, student_id: str) -> Optional[str]:
        """
        Retrieve the actual API key for LLM services
        This should only be called by internal services
        """
        try:
            logger.info(f"Retrieving API key for student_id: {student_id}")
            api_key = await self.supabase.get_api_key(student_id)
            if api_key:
                logger.info(
                    f"Retrieved API key for student {student_id}: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else '***'}")
                return api_key
            else:
                logger.warning(f"No API key found for student {student_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving API key for LLM service: {e}")
            return None


# Global auth service instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get or create global auth service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
