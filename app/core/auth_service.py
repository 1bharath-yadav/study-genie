"""
Functional Authentication Service
Pure functions for handling user authentication and API key management
"""
from typing import Optional, Dict, Any
import logging
from datetime import datetime
from functools import wraps

from app.services.db import (
    create_supabase_client,
    safe_extract_single
)
from app.core.encryption import decrypt_api_key

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

# Pure functions for data validation
def validate_user_data(user_data: UserData) -> str:
    """Validate user data and extract student ID"""
    student_id = user_data.get("sub")
    if not student_id:
        raise ValueError("User ID (sub) is required")
    return str(student_id)

def extract_student_fields(user_data: UserData) -> Dict[str, str]:
    """Extract and validate student fields from user data"""
    return {
        "student_id": str(user_data.get("sub", "")),
        "username": user_data.get("email", "").split("@")[0],
        "email": user_data.get("email", ""),
        "full_name": user_data.get("name", "")
    }

def create_update_data(user_data: UserData, existing_student: StudentInfo) -> Dict[str, Any]:
    """Create update data for existing student"""
    return {
        "email": user_data.get("email", existing_student.get("email", "")),
        "full_name": user_data.get("name", existing_student.get("full_name", "")),
        "updated_at": datetime.now().isoformat()
    }

def create_api_key_status(api_key_info: APIKeyInfo) -> Dict[str, Any]:
    """Create API key status information"""
    return {
        "has_api_key": api_key_info is not None,
        "service": api_key_info.get("service") if api_key_info else None,
        "created_at": api_key_info.get("created_at") if api_key_info else None,
        "updated_at": api_key_info.get("updated_at") if api_key_info else None
    }

def create_signin_response(
    student_info: StudentInfo,
    api_key_info: APIKeyInfo
) -> SigninResponse:
    """Create complete signin response"""
    return {
        "user": student_info,
        "api_key_status": create_api_key_status(api_key_info),
        "signin_timestamp": datetime.now().isoformat()
    }

# Database operations
async def get_existing_student(student_id: str) -> Optional[StudentInfo]:
    """Get existing student from database"""
    try:
        client = create_supabase_client()
        if not client:
            return None
            
        response = client.table("students").select("*").eq("student_id", student_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Error getting student {student_id}: {e}")
        return None

async def create_new_student(student_data: Dict[str, str]) -> StudentInfo:
    """Create new student in database"""
    client = create_supabase_client()
    if not client:
        raise ValueError("Database connection failed")
        
    response = client.table("students").insert(student_data).execute()
    result = safe_extract_single(response)
    if not result:
        raise ValueError("Failed to create student")
    return result

async def update_existing_student(student_id: str, update_data: Dict[str, Any]) -> StudentInfo:
    """Update existing student in database"""
    client = create_supabase_client()
    if not client:
        raise ValueError("Database connection failed")
        
    response = client.table("students").update(update_data).eq("student_id", student_id).execute()
    result = safe_extract_single(response)
    if not result:
        raise ValueError("Failed to update student")
    return result

async def get_api_key_info(student_id: str) -> APIKeyInfo:
    """Get API key information without the actual key"""
    try:
        client = create_supabase_client()
        if not client:
            return None
            
        response = client.table("user_api_keys").select(
            "*, llm_providers(name, display_name)"
        ).eq("student_id", student_id).execute()
        return safe_extract_single(response)
    except Exception as e:
        logger.error(f"Error getting API key info for {student_id}: {e}")
        return None

async def get_encrypted_api_key(student_id: str) -> Optional[str]:
    """Get encrypted API key from database"""
    try:
        client = create_supabase_client()
        if not client:
            return None
            
        response = client.table("user_api_keys").select("encrypted_api_key").eq("student_id", student_id).execute()
        result = safe_extract_single(response)
        return result.get("encrypted_api_key") if result else None
    except Exception as e:
        logger.error(f"Error getting encrypted API key for {student_id}: {e}")
        return None

# Core authentication functions
@handle_auth_errors
async def ensure_student_exists(user_data: UserData) -> StudentInfo:
    """Ensure student exists in database, create or update as needed"""
    student_id = validate_user_data(user_data)
    existing_student = await get_existing_student(student_id)
    
    if existing_student:
        update_data = create_update_data(user_data, existing_student)
        return await update_existing_student(student_id, update_data)
    else:
        student_data = extract_student_fields(user_data)
        return await create_new_student(student_data)

@handle_auth_errors
async def handle_user_signin(user_data: UserData) -> SigninResponse:
    """
    Handle complete user sign-in process:
    1. Validate user data
    2. Create/update student record
    3. Retrieve API key status
    4. Return complete signin response
    """
    student_id = validate_user_data(user_data)
    
    # Create or update student record
    student_info = await ensure_student_exists(user_data)
    
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