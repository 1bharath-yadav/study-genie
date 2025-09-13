"""
Security module for authentication and authorization
Functional programming approach with modern JWT handling
"""

# app/core/security.py

from fastapi import HTTPException, status, Depends, Header
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, List
import logging
import hashlib

from app.config import settings

logger = logging.getLogger(__name__)

# Type definitions
JWTPayload = Dict[str, Any]
UserData = Dict[str, Any]

# Models for request/response
class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []

# OAuth2 scheme for OpenAPI documentation
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    description="JWT token for authentication"
)

# Pure functions for JWT operations
def create_token_hash(token: str) -> str:
    """Create a hash of the token for logging purposes"""
    return hashlib.md5(token.encode()).hexdigest()[:8]

def decode_jwt_token(token: str) -> JWTPayload:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET,
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False}  # Disable audience verification
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def extract_student_id(payload: JWTPayload) -> str:
    """Extract user ID from JWT payload"""
    student_id = payload.get("sub")
    if not student_id:
        logger.error("No 'sub' field in JWT payload")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return str(student_id)

def extract_student_data(payload: JWTPayload) -> UserData:
    """Extract user data from JWT payload"""
    student_id = payload.get("sub") # use sub id as unique id
    
    return {
        "student_id": student_id,
        "email": payload.get("email"),
        "name": payload.get("name"),
        "picture": payload.get("picture"),
        "api_key_status": payload.get("api_key_status"),
        "created_at": payload.get("iat"),
        "updated_at": payload.get("iat"),
        "exp": payload.get("exp"),
        "iss": payload.get("iss"),
        "aud": payload.get("aud")
    }

def log_user_validation(student_data: UserData, token_hash: str) -> None:
    """Log user validation for debugging"""
    student_id = student_data.get("student_id")
    email = student_data.get("email")
    
    logger.info(f"Token validated (hash: {token_hash}) - Student ID: {student_id}, Email: {email}")
    logger.debug(f"JWT payload keys: {list(student_data.keys())}")

def validate_token_expiry(payload: JWTPayload) -> None:
    """Validate that the token hasn't expired"""
    exp = payload.get("exp")
    if exp:
        exp_datetime = datetime.fromtimestamp(exp)
        if datetime.now() > exp_datetime:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

def create_credentials_exception(detail: str = "Could not validate credentials") -> HTTPException:
    """Create a standardized credentials exception"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )

# Main authentication function
def get_current_user(authorization: str = Header(...)) -> dict:
    """
    Extract and verify JWT token from Authorization header
    """
    try:
        # Log the received authorization header for debugging
        logger.info(f"Received authorization header: {authorization[:50] if len(authorization) > 50 else authorization}...")
        
        # Remove 'Bearer ' prefix if present
        token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
        
        logger.info(f"Extracted token (first 50 chars): {token[:50]}...")
        
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET, 
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False}  # Disable audience verification
        )
        
        logger.info(f"JWT decoded successfully for user: {payload.get('email', 'unknown')}")
        return payload
        
    except JWTError as e:
        logger.error(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def validate_bearer_token(token: str) -> dict:
    """
    Validate Bearer token and return user data
    Used for functional authentication
    """
    try:
        # Remove 'Bearer ' prefix if present
        clean_token = token.replace("Bearer ", "") if token.startswith("Bearer ") else token
        
        # Decode JWT token
        payload = jwt.decode(
            clean_token, 
            settings.JWT_SECRET, 
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False}  # Disable audience verification
        )
        
        return payload
        
    except JWTError as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


def get_current_student_id(current_user: dict = Depends(get_current_user)) -> str:
    """
    Extract user ID from current user token
    Returns the user ID (sub or student_id) as a string
    """
    student_id = current_user.get("student_id")
    if not student_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    return str(student_id)


# Admin/special role validation
def validate_admin_user(student_data: UserData) -> UserData:
    """
    Validate that the user has admin privileges
    This can be extended to check specific roles/permissions
    """
    # For now, we'll check if the user has certain email domains or explicit admin status
    email = student_data.get("email", "")
    
    # Add your admin validation logic here
    # Example: check for admin email domains or explicit admin flag
    admin_domains = []
    is_admin = any(email.endswith(domain) for domain in admin_domains)
    
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return student_data

async def get_admin_user(current_user: UserData = Depends(get_current_user)) -> UserData:
    """Dependency for admin-only endpoints"""
    return validate_admin_user(current_user)

# Utility functions for token creation (if needed for testing)
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET, 
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt

# Export for backward compatibility
__all__ = [
    "get_current_user",
    "get_admin_user",
    "create_access_token",
    "oauth2_scheme"
]