from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
import httpx
import jwt
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode

from app.config import settings
from app.core.auth_service import handle_user_signin
from app.core.security import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


def create_custom_jwt_token(user_data: dict) -> str:
    """
    Create a custom JWT token for users (primarily for Google OAuth flow)
    """
    jwt_payload = {
        "sub": user_data.get("sub") or user_data.get("id"),
        "student_id": user_data.get("sub") or user_data.get("id"),
        "email": user_data.get("email"),
        "name": user_data.get("name"),
        "picture": user_data.get("picture"),
        "api_key_status": user_data.get("api_key_status"),
        "iat": int(datetime.utcnow().timestamp()),
        "exp": int((datetime.utcnow() + timedelta(hours=24)).timestamp()),
        "iss": f"{settings.SUPABASE_URL}/auth/v1" if settings.SUPABASE_URL else "study-genie",
        "aud": "authenticated",
        "role": "authenticated"
    }
    return jwt.encode(jwt_payload, settings.JWT_SECRET, algorithm=settings.ALGORITHM)


@router.get("/auth/login")
def login():
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    url = f"{settings.GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url)


@router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        code, error = request.query_params.get("code"), request.query_params.get("error")
        if error:
            return RedirectResponse(f"{settings.FRONTEND_URL}?{urlencode({'error': error})}")
        if not code:
            return RedirectResponse(f"{settings.FRONTEND_URL}?{urlencode({'error': 'missing_code'})}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            token_resp = await client.post(
                settings.GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if token_resp.status_code != 200:
                return RedirectResponse(f"{settings.FRONTEND_URL}?{urlencode({'error': 'token_failed'})}")

            access_token = token_resp.json().get("access_token")
            
            userinfo_resp = await client.get(
                settings.GOOGLE_USERINFO_URL, headers={
                    "Authorization": f"Bearer {access_token}"}
            )
            if userinfo_resp.status_code != 200:
                return RedirectResponse(f"{settings.FRONTEND_URL}?{urlencode({'error': 'userinfo_failed'})}")

            userinfo = userinfo_resp.json()

        signin_result = await handle_user_signin(userinfo)

        # Create custom JWT token with proper structure
        user_data = {
            "sub": userinfo.get("sub"),
            "email": userinfo.get("email"),
            "name": userinfo.get("name"),
            "picture": userinfo.get("picture"),
            "api_key_status": signin_result["api_key_status"]
        }
        jwt_token = create_custom_jwt_token(user_data)

        redirect_url = f"{settings.FRONTEND_URL}?{urlencode({'token': jwt_token})}"
        logger.info(f"OAuth callback successful, redirecting to: {redirect_url[:100]}...")
        return RedirectResponse(redirect_url)

    except Exception as e:
        logger.error(f"Callback error: {e}", exc_info=True)
        return RedirectResponse(f"{settings.FRONTEND_URL}?{urlencode({'error': 'internal_error'})}")


@router.post("/auth/verify")
async def verify_token(request: Request):
    body = await request.json()
    token = body.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Token required")
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return {"valid": True, "user": payload}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/auth/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    student_id = current_user.get("student_id") or current_user.get("sub")
    if not student_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

    signin_result = await handle_user_signin(current_user)

    return {
        "user": {
            "id": student_id,
            "email": current_user.get("email"),
            "name": current_user.get("name"),
            "picture": current_user.get("picture"),
        },
        "api_key_status": signin_result["api_key_status"],
        "last_updated": datetime.now().isoformat(),
    }

