"""
Functional Authentication Service
Pure functions for handling user authentication and API key management
"""
# app/core/auth_service.py
from typing import Optional, Dict, Any
import logging
from functools import wraps

from app.core.encryption import decrypt_api_key
from app.services.student_service import create_new_student, create_signin_response, create_update_data, extract_student_fields, get_api_key_info, get_encrypted_api_key, get_existing_student, update_existing_student, validate_student_data

logger = logging.getLogger(__name__)

# Type definitions
UserData = Dict[str, Any]
StudentInfo = Dict[str, Any]
APIKeyInfo = Optional[Dict[str, Any]]
SigninResponse = Dict[str, Any]

# Error handling decorator
def handle_auth_errors(func):
    """Decorator for handling authentication errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Auth error in {func.__name__}: {e}")
            raise
    return wrapper

# Core authentication functions
@handle_auth_errors
async def ensure_student_exists(student_data: UserData) -> StudentInfo:
    """Ensure student exists in database, create or update as needed"""
    student_id = validate_student_data(student_data)
    existing_student = await get_existing_student(student_id)
    
    if existing_student:
        update_data = create_update_data(student_data, existing_student)
        return await update_existing_student(student_id, update_data)
    else:
        student_data = extract_student_fields(student_data)
        return await create_new_student(student_data)

@handle_auth_errors
async def handle_user_signin(student_data: UserData) -> SigninResponse:
    """
    Handle complete user sign-in process:
    1. Validate user data
    2. Create/update student record
    3. Retrieve API key status
    4. Return complete signin response
    """
    student_id = validate_student_data(student_data)
    
    # Create or update student record
    student_info = await ensure_student_exists(student_data)
    
    # Get API key information
    api_key_info = await get_api_key_info(student_id)
    
    # Create and return complete response
    signin_response = create_signin_response(student_info, api_key_info)
    
    logger.info(
        f"User {student_id} signed in successfully. "
        f"API key status: {signin_response['api_key_status']['has_api_key']}"
    )
    
    return signin_response

@handle_auth_errors
async def get_user_api_key_for_llm(student_id: str) -> Optional[str]:
    """
    Retrieve and decrypt the actual API key for LLM services
    This should only be called by internal services
    """
    logger.info(f"Retrieving API key for student_id: {student_id}")
    
    encrypted_key = await get_encrypted_api_key(student_id)
    if not encrypted_key:
        logger.warning(f"No API key found for student {student_id}")
        return None
    
    try:
        # Decrypt the API key
        decrypted_key = decrypt_api_key(encrypted_key)
        if decrypted_key:
            logger.info(
                f"Retrieved API key for student {student_id}: "
                f"{decrypted_key[:10]}...{decrypted_key[-5:] if len(decrypted_key) > 15 else '***'}"
            )
            return decrypted_key
        else:
            logger.error(f"Failed to decrypt API key for student {student_id}")
            return None
    except Exception as e:
        logger.error(f"Error decrypting API key for student {student_id}: {e}")
        return None

# Utility functions for backward compatibility
async def get_auth_service_instance():
    """Backward compatibility helper - returns auth functions as dict"""
    return {
        "handle_user_signin": handle_user_signin,
        "get_user_api_key_for_llm": get_user_api_key_for_llm,
        "ensure_student_exists": ensure_student_exists
    }