from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
import httpx
import jwt
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote
from app.config import settings
from app.core.auth_service import get_auth_service
from app.core.security import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/auth/login")
def login():
    google_auth_url = (
        f"{settings.GOOGLE_AUTH_URL}?"
        f"client_id={settings.GOOGLE_CLIENT_ID}&"
        "response_type=code&"
        f"redirect_uri={settings.GOOGLE_REDIRECT_URI}&"
        "scope=openid%20email%20profile&"
        "access_type=offline&"
        "prompt=consent"
    )
    logger.info(f"Generated Google Auth URL: {google_auth_url}")
    return RedirectResponse(google_auth_url)


@router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        code = request.query_params.get("code")
        error = request.query_params.get("error")

        # Handle OAuth errors
        if error:
            logger.error(f"OAuth error: {error}")
            error_params = urlencode(
                {"error": "oauth_error", "message": error})
            return RedirectResponse(f"{settings.FRONTEND_URL}?{error_params}")

        if not code:
            logger.error("Missing authorization code in callback")
            error_params = urlencode({"error": "missing_code"})
            return RedirectResponse(f"{settings.FRONTEND_URL}?{error_params}")

        logger.info(f"Received authorization code: {code[:10]}...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Exchange code for tokens
            token_data = {
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            }

            logger.info("Exchanging code for access token...")
            token_resp = await client.post(
                settings.GOOGLE_TOKEN_URL,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            logger.info(f"Token response status: {token_resp.status_code}")

            if token_resp.status_code != 200:
                logger.error(f"Token exchange failed: {token_resp.text}")
                error_params = urlencode({"error": "token_exchange_failed"})
                return RedirectResponse(f"{settings.FRONTEND_URL}?{error_params}")

            tokens = token_resp.json()
            access_token = tokens.get("access_token")

            if not access_token:
                logger.error("No access token received")
                error_params = urlencode({"error": "no_access_token"})
                return RedirectResponse(f"{settings.FRONTEND_URL}?{error_params}")

            # Get user info from Google
            logger.info("Fetching user info from Google...")
            userinfo_resp = await client.get(
                settings.GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )

            logger.info(
                f"User info response status: {userinfo_resp.status_code}")

            if userinfo_resp.status_code != 200:
                logger.error(f"Failed to get user info: {userinfo_resp.text}")
                error_params = urlencode({"error": "userinfo_failed"})
                return RedirectResponse(f"{settings.FRONTEND_URL}?{error_params}")

            userinfo = userinfo_resp.json()
            logger.info(f"User info received for: {userinfo.get('email')}")

            # Handle user sign-in and API key retrieval
            auth_service = get_auth_service()
            signin_result = await auth_service.handle_user_signin(userinfo)

            # Create JWT token with user data and API key status
            jwt_payload = {
                "sub": userinfo.get("sub"),
                "email": userinfo.get("email"),
                "name": userinfo.get("name"),
                "picture": userinfo.get("picture"),
                # Add student_id for consistency
                "student_id": userinfo.get("sub"),
                "api_key_status": signin_result["api_key_status"],
                "iat": int(datetime.utcnow().timestamp()),
                "exp": int((datetime.utcnow() + timedelta(hours=24)).timestamp())
            }

            # You'll need to add JWT_SECRET to your settings
            jwt_token = jwt.encode(
                jwt_payload, settings.JWT_SECRET, algorithm="HS256")

            # Redirect to frontend with just the JWT token
            success_params = urlencode({"token": jwt_token})
            frontend_redirect = f"{settings.FRONTEND_URL}?{success_params}"

            logger.info("Redirecting to frontend with JWT token")
            return RedirectResponse(frontend_redirect)

    except Exception as e:
        logger.error(
            f"Unexpected error in auth callback: {str(e)}", exc_info=True)
        error_params = urlencode({"error": "internal_error"})
        return RedirectResponse(f"{settings.FRONTEND_URL}?{error_params}")

# Helper endpoint to decode JWT on frontend


@router.post("/auth/verify")
async def verify_token(request: Request):
    try:
        body = await request.json()
        token = body.get("token")

        if not token:
            raise HTTPException(status_code=400, detail="Token required")

        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return {"valid": True, "user": payload}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/auth/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get complete user profile including API key status"""
    try:
        student_id = current_user.get("student_id") or current_user.get("sub")
        if not student_id:
            raise HTTPException(
                status_code=401, detail="User not authenticated")

        auth_service = get_auth_service()

        # Get fresh API key status
        signin_result = await auth_service.handle_user_signin(current_user)

        return {
            "user": {
                "id": student_id,
                "email": current_user.get("email"),
                "name": current_user.get("name"),
                "picture": current_user.get("picture")
            },
            "api_key_status": signin_result["api_key_status"],
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=500, detail="Error retrieving user profile")
