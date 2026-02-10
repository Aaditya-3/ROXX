"""
Simple username/password auth with signup + login.
"""

import os
import json
import hashlib
import hmac
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel


router = APIRouter(prefix="/auth", tags=["auth"])

APP_USERNAME = os.getenv("APP_USERNAME", "aaditya")
APP_PASSWORD = os.getenv("APP_PASSWORD", "aaditya123")
PBKDF2_ITERATIONS = 120_000
PBKDF2_ALGORITHM = "sha256"
SALT_BYTES = 16


class SimpleSignupRequest(BaseModel):
    username: str
    password: str


class SimpleLoginRequest(BaseModel):
    username: str
    password: str


class SimpleLoginResponse(BaseModel):
    ok: bool = True
    user_id: str


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
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


def _validate_payload(username: str, password: str):
    if not username or len(username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters",
        )
    if not password or len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters",
        )


# In-memory account store for this app session.
_users: dict[str, str] = {_normalize_username(APP_USERNAME): _hash_password(APP_PASSWORD)}
_project_root = Path(__file__).resolve().parents[4]
_users_store_path = _project_root / "memory" / "users.json"


def _load_users():
    if not _users_store_path.exists():
        return
    try:
        with open(_users_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for username, password_hash in data.items():
                if isinstance(username, str) and isinstance(password_hash, str):
                    _users[_normalize_username(username)] = password_hash
    except Exception:
        pass


def _save_users():
    try:
        _users_store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_users_store_path, "w", encoding="utf-8") as f:
            json.dump(_users, f, indent=2)
    except Exception:
        pass


_load_users()


@router.post("/signup", response_model=SimpleLoginResponse)
async def signup(payload: SimpleSignupRequest):
    username = _normalize_username(payload.username)
    password = payload.password or ""
    _validate_payload(username, password)

    if username in _users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    _users[username] = _hash_password(password)
    _save_users()
    return SimpleLoginResponse(user_id=username)


@router.post("/login", response_model=SimpleLoginResponse)
async def login(payload: SimpleLoginRequest):
    username = _normalize_username(payload.username)
    password = payload.password or ""
    _validate_payload(username, password)

    stored_hash = _users.get(username)
    if not stored_hash or not _verify_password(password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return SimpleLoginResponse(user_id=username)
