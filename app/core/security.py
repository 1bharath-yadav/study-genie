from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

from app.config import settings
from app.db.models import DatabaseManager, StudentManager
from app.core.dependencies import get_db


# This is a placeholder for the actual token URL.
# In our case, the token is issued via the /auth/callback redirect.
# This is primarily for OpenAPI documentation.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: str = Depends(oauth2_scheme), db_manager: DatabaseManager = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Debug: Log the token hash for unique identification
        import hashlib
        token_hash = hashlib.md5(token.encode()).hexdigest()[:8]
        print(f"Debug: Validating token (hash: {token_hash})")

        payload = jwt.decode(token, settings.JWT_SECRET,
                             algorithms=[settings.ALGORITHM])

        user_id = payload.get("sub")
        if user_id is None:
            print("Debug: No 'sub' field in JWT payload")
            raise credentials_exception

        # Debug: Log more detailed payload info including user identification
        student_id = payload.get("student_id") or payload.get("sub")
        email = payload.get("email")
        print(f"Debug: JWT payload - student_id: {student_id}, email: {email}")
        print(f"Debug: JWT payload keys: {list(payload.keys())}")

        # Return the user data directly from JWT payload
        # Include all fields from the JWT for compatibility
        user_data = {
            "id": payload.get("sub"),
            "sub": payload.get("sub"),
            "student_id": payload.get("student_id"),  # Add student_id field
            "email": payload.get("email"),
            "name": payload.get("name"),
            "picture": payload.get("picture"),
            "api_key_status": payload.get("api_key_status"),
            "created_at": payload.get("iat"),
            "updated_at": payload.get("iat")
        }

        print(f"Debug: Validated user - ID: {student_id}, Email: {email}")
        return user_data

    except JWTError as e:
        print(f"JWT Error: {e}")  # Debug log
        raise credentials_exception
