"""
Email/password authentication endpoints.
"""

import hashlib
import hmac
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from pymongo.errors import PyMongoError

from backend.app.core.auth.jwt_auth import create_access_token
from backend.app.core.db.mongo import get_db


router = APIRouter(prefix="/auth", tags=["auth"])

PBKDF2_ITERATIONS = 120_000
PBKDF2_ALGORITHM = "sha256"
SALT_BYTES = 16


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algo, iterations, salt_hex, digest_hex = password_hash.split("$", 3)
        if algo != f"pbkdf2_{PBKDF2_ALGORITHM}":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        computed = hashlib.pbkdf2_hmac(
            PBKDF2_ALGORITHM,
            password.encode("utf-8"),
            salt,
            int(iterations),
        )
        return hmac.compare_digest(computed, expected)
    except Exception:
        return False


def validate_auth_payload(email: str, password: str):
    if not email or "@" not in email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid email is required",
        )
    if not password or len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters",
        )


@router.post("/signup", response_model=AuthResponse)
async def signup(payload: SignupRequest):
    email = normalize_email(payload.email)
    validate_auth_payload(email, payload.password)

    try:
        db = get_db()
        users = db["users"]
        existing = users.find_one({"email": email})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Account already exists for this email",
            )

        user_id = str(uuid.uuid4())
        users.insert_one(
            {
                "_id": user_id,
                "email": email,
                "name": (payload.name or "").strip(),
                "auth_provider": "local",
                "password_hash": hash_password(payload.password),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )
    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable. Please ensure MongoDB is running and try again.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signup failed due to an internal error.",
        )

    token = create_access_token({"sub": user_id, "email": email})
    return AuthResponse(access_token=token)


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    email = normalize_email(payload.email)
    validate_auth_payload(email, payload.password)

    try:
        db = get_db()
        user = db["users"].find_one({"email": email})
        if not user or not user.get("password_hash"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        if not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable. Please ensure MongoDB is running and try again.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to an internal error.",
        )

    token = create_access_token({"sub": str(user["_id"]), "email": email})
    return AuthResponse(access_token=token)
