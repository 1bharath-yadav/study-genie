from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.security import get_current_user
from app.supabase_client import get_supabase_client

router = APIRouter()


class UserProfileResponse(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    grade_level: Optional[str] = None
    learning_preferences: Optional[list] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    grade_level: Optional[str] = None
    learning_preferences: Optional[list] = None


@router.get("/users/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user)
):
    """Get current user's profile information"""
    try:
        supabase = get_supabase_client()
        # Fix: Use student_id or sub from JWT token
        user_id = current_user.get("student_id") or current_user.get(
            "sub") or current_user.get("id")

        print(f"Debug - current_user keys: {current_user.keys()}")  # Debug
        print(f"Debug - user_id: {user_id}")  # Debug

        if not user_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Try to get student data from Supabase
        try:
            student_data = await supabase.get_student(str(user_id))
            print(f"Debug - student_data: {student_data}")  # Debug
        except Exception as e:
            print(f"Error getting student data: {e}")  # Debug log
            student_data = None

        # Merge JWT data with student data if available
        profile_data = {
            "id": user_id,
            "email": current_user.get("email"),
            "name": current_user.get("name"),
            "picture": current_user.get("picture"),
            "created_at": str(current_user.get("iat", "")),
            "updated_at": str(current_user.get("iat", "")),
            "grade_level": None,
            "learning_preferences": []
        }

        if student_data:
            # Handle learning_preferences - convert dict to list if needed
            learning_prefs = student_data.get("learning_preferences", [])
            if isinstance(learning_prefs, dict):
                # Convert dict to list of values or empty list
                learning_prefs = list(
                    learning_prefs.values()) if learning_prefs else []

            profile_data.update({
                "grade_level": student_data.get("grade_level"),
                "learning_preferences": learning_prefs
            })

        # Debug
        print(f"Debug - profile_data before validation: {profile_data}")

        try:
            response = UserProfileResponse(**profile_data)
            print(f"Debug - UserProfileResponse created successfully")  # Debug
            return response
        except Exception as validation_error:
            print(f"Debug - Validation error: {validation_error}")  # Debug
            raise HTTPException(
                status_code=500, detail=f"Profile validation error: {str(validation_error)}")

    except Exception as e:
        print(f"Debug - General error: {e}")  # Debug
        raise HTTPException(
            status_code=500, detail=f"Error fetching user profile: {str(e)}")


@router.put("/users/me", response_model=UserProfileResponse)
async def update_current_user_profile(
    profile_update: UserProfileUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update current user's profile information"""
    try:
        supabase = get_supabase_client()
        # Fix: Use student_id or sub from JWT token
        user_id = current_user.get("student_id") or current_user.get(
            "sub") or current_user.get("id")

        if not user_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Check if student exists
        try:
            existing_student = await supabase.get_student(str(user_id))
        except Exception:
            existing_student = None

        student_data = {
            "student_id": str(user_id),
            "username": current_user.get("name", ""),
            "email": current_user.get("email", ""),
            "full_name": profile_update.name or current_user.get("name", ""),
            "grade_level": profile_update.grade_level,
            "learning_preferences": profile_update.learning_preferences or []
        }

        if not existing_student:
            # Create new student record
            await supabase.create_student(student_data)
        else:
            # Update existing student
            await supabase.update_student(str(user_id), {
                "full_name": profile_update.name or current_user.get("name", ""),
                "learning_preferences": profile_update.learning_preferences or [],
                "updated_at": datetime.now().isoformat()
            })

        # Return updated profile
        return await get_current_user_profile(current_user)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating user profile: {str(e)}")


@router.delete("/users/me")
async def delete_current_user_profile(
    current_user: dict = Depends(get_current_user)
):
    """Delete current user's profile and all associated data"""
    try:
        supabase = get_supabase_client()
        user_id = current_user.get("id") or current_user.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        # Delete student data
        await supabase.delete_student(str(user_id))

        return {"message": "User profile deleted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting user profile: {str(e)}")
