from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
import httpx
from app.config import settings

router = APIRouter()


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
    return RedirectResponse(google_auth_url)


@router.get("/auth/callback")
async def auth_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code in callback")
    async with httpx.AsyncClient() as client:
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
            raise HTTPException(
                status_code=400, detail="Failed to get token from Google")
        tokens = token_resp.json()
        access_token = tokens.get("access_token")
        # Get user info
        userinfo_resp = await client.get(
            settings.GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(
                status_code=400, detail="Failed to get user info from Google")
        userinfo = userinfo_resp.json()
        # Redirect to frontend with user info as query params
        frontend_url = settings.FRONTEND_URL
        params = f"?sub={userinfo.get('sub')}&name={userinfo.get('name')}&email={userinfo.get('email')}&picture={userinfo.get('picture')}"
        return RedirectResponse(frontend_url + params)
