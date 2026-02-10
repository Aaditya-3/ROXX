"""
Google authentication endpoint.

Frontend should obtain a Google ID token (e.g. via @react-oauth/google)
and POST it to /auth/google. Backend verifies it with Google, upserts the
user in MongoDB, and returns a JWT for authenticated access.
"""

import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.core.db.mongo import get_db
from backend.app.core.auth.jwt_auth import create_access_token


router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


class GoogleLoginRequest(BaseModel):
    id_token: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


async def verify_google_id_token(id_token: str) -> Dict[str, Any]:
    """
    Verify Google ID token via Google's tokeninfo endpoint.
    """
    url = "https://oauth2.googleapis.com/tokeninfo"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params={"id_token": id_token})
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token",
        )
    data = resp.json()
    # Verify audience (client_id)
    aud = data.get("aud")
    if GOOGLE_CLIENT_ID and aud != GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google client_id",
        )
    return data


@router.post("/google", response_model=AuthResponse)
async def login_with_google(payload: GoogleLoginRequest):
    """
    Login endpoint for Google ID token.

    - Verifies Google ID token with Google
    - Upserts user in MongoDB using sub/email
    - Returns JWT (access_token) for use with /chat and other endpoints
    """
    token_info = await verify_google_id_token(payload.id_token)

    email = token_info.get("email")
    sub = token_info.get("sub")
    if not email or not sub:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google token missing email or sub",
        )

    db = get_db()
    users = db["users"]

    # Upsert user by Google sub
    normalized_email = str(email).strip().lower()
    existing = users.find_one({"google_id": sub})
    if existing:
        user_id = existing["_id"]
    else:
        doc = {
            "_id": sub,  # use Google sub as primary id
            "email": normalized_email,
            "google_id": sub,
        }
        users.insert_one(doc)
        user_id = doc["_id"]

    access_token = create_access_token({"sub": str(user_id), "email": normalized_email})
    return AuthResponse(access_token=access_token)

