"""User + admin auth on top of core.db (Postgres). Password/JWT logic matches backend/core/auth.py,
so existing password hashes and token formats stay valid."""
import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr

from . import db

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


def _secret() -> str:
    key = os.environ.get("JWT_SECRET_KEY")
    if not key:
        # A per-instance random key would silently break logins across serverless instances.
        raise RuntimeError("JWT_SECRET_KEY is not set.")
    return key


class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: str  # accepts email or username
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime


# ── Passwords ─────────────────────────────────────────────────────────────────
_ITERATIONS = 260_000
_HASH_NAME = "sha256"


def get_password_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(_HASH_NAME, password.encode(), salt.encode(), _ITERATIONS)
    return f"pbkdf2:{_HASH_NAME}:{_ITERATIONS}${salt}${dk.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    if hashed.startswith("pbkdf2:"):
        try:
            _, hash_name, rest = hashed.split(":", 2)
            iterations_str, salt, stored_hex = rest.split("$", 2)
            dk = hashlib.pbkdf2_hmac(hash_name, plain.encode(), salt.encode(), int(iterations_str))
            return secrets.compare_digest(dk.hex(), stored_hex)
        except Exception:
            return False
    if hashed.startswith("$2"):
        try:
            from passlib.hash import bcrypt as _bcrypt
            return _bcrypt.verify(plain, hashed)
        except Exception:
            return False
    return secrets.compare_digest(hashlib.sha256(plain.encode()).hexdigest(), hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, _secret(), algorithm=ALGORITHM)


def get_current_user(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
    except JWTError:
        return None
    user_id = payload.get("sub")
    return db.get_user_by_id(user_id) if user_id else None


# ── Users ─────────────────────────────────────────────────────────────────────
def register_user(username: str, email: str, password: str) -> dict:
    user = db.create_user(username, email, get_password_hash(password))
    logger.info("Registered new user: %s", username)
    return user


def authenticate_user(identifier: str, password: str) -> Optional[dict]:
    user = db.find_user(identifier)
    if user is None or not verify_password(password, user["password_hash"]):
        return None
    if not user["password_hash"].startswith("pbkdf2:"):  # upgrade legacy hashes
        db.set_password_hash(user["id"], get_password_hash(password))
    return user


# ── Admin (single account from env vars; sessions in Postgres) ───────────────
_CONFIG_PATH = Path(__file__).parent.parent / "data" / "config.json"


def _admin_cfg() -> dict:
    try:
        cfg = json.loads(_CONFIG_PATH.read_text()).get("admin", {})
    except Exception:
        cfg = {}
    cfg["email"] = os.environ.get("ADMIN_EMAIL", cfg.get("email", "admin"))
    cfg["password"] = os.environ.get("ADMIN_PASSWORD", cfg.get("password", ""))
    return cfg


def admin_record() -> dict:
    cfg = _admin_cfg()
    return {"id": "admin", "name": cfg.get("name", "Admin"), "email": cfg["email"]}


def authenticate_admin(email: str, password: str) -> Optional[dict]:
    cfg = _admin_cfg()
    email_ok = hmac.compare_digest(email.strip().lower(), cfg["email"].lower())
    password_ok = hmac.compare_digest(password, cfg["password"])
    return admin_record() if (email_ok and password_ok and cfg["password"]) else None
