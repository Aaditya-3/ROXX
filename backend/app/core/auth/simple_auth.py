"""
Simple username/password auth with signup + login.
"""

import os
import json
import hashlib
import hmac
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.core.db.relational import DBUser, get_relational_session, init_relational_db


router = APIRouter(prefix="/auth", tags=["auth"])

APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")
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


_project_root = Path(__file__).resolve().parents[4]
_users_store_path = _project_root / "memory" / "users.json"
init_relational_db()


def _get_user_by_username(username: str) -> DBUser | None:
    with get_relational_session() as db:
        return db.query(DBUser).filter(DBUser.username == username).first()


def _seed_env_user_if_needed():
    if not (APP_USERNAME and APP_PASSWORD):
        return
    username = _normalize_username(APP_USERNAME)
    if not username:
        return
    with get_relational_session() as db:
        existing = db.query(DBUser).filter(DBUser.username == username).first()
        if existing:
            return
        db.add(
            DBUser(
                id=str(uuid.uuid4()),
                username=username,
                email=None,
                password_hash=_hash_password(APP_PASSWORD),
                name=username,
                plan_type="free",
                is_active=True,
            )
        )


def _migrate_legacy_users_json():
    if not _users_store_path.exists():
        return
    try:
        with open(_users_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        with get_relational_session() as db:
            for username, password_hash in data.items():
                uname = _normalize_username(str(username))
                if not uname or not isinstance(password_hash, str):
                    continue
                existing = db.query(DBUser).filter(DBUser.username == uname).first()
                if existing:
                    continue
                db.add(
                    DBUser(
                        id=str(uuid.uuid4()),
                        username=uname,
                        email=None,
                        password_hash=password_hash,
                        name=uname,
                        plan_type="free",
                        is_active=True,
                    )
                )
    except Exception:
        return


_seed_env_user_if_needed()
_migrate_legacy_users_json()


@router.post("/signup", response_model=SimpleLoginResponse)
async def signup(payload: SimpleSignupRequest):
    username = _normalize_username(payload.username)
    password = payload.password or ""
    _validate_payload(username, password)

    with get_relational_session() as db:
        existing = db.query(DBUser).filter(DBUser.username == username).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )
        user = DBUser(
            id=str(uuid.uuid4()),
            username=username,
            email=None,
            password_hash=_hash_password(password),
            name=username,
            plan_type="free",
            is_active=True,
        )
        db.add(user)
        return SimpleLoginResponse(user_id=user.id)


@router.post("/login", response_model=SimpleLoginResponse)
async def login(payload: SimpleLoginRequest):
    username = _normalize_username(payload.username)
    password = payload.password or ""
    _validate_payload(username, password)

    user = _get_user_by_username(username)
    if not user or not _verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return SimpleLoginResponse(user_id=user.id)
