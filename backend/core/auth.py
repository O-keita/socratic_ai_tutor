import hashlib
import json
import logging
import os
import secrets
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JWT configuration — must be set via JWT_SECRET_KEY environment variable
# ---------------------------------------------------------------------------
_jwt_secret = os.environ.get("JWT_SECRET_KEY")
if not _jwt_secret:
    # Generate a random key so dev still works, but tokens are invalidated on restart.
    # Set JWT_SECRET_KEY env var for stable tokens across restarts.
    _jwt_secret = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET_KEY is not set — using a random key. "
        "All tokens will be invalidated on server restart. "
        "Set JWT_SECRET_KEY for persistent authentication."
    )
SECRET_KEY = _jwt_secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# File-based user store
# ---------------------------------------------------------------------------
DB_FILE = Path(__file__).parent.parent / "data" / "users.json"
users_db: dict = {}
_lock = threading.RLock()  # Reentrant lock — protects users_db reads/writes


def load_users() -> None:
    """Load users from the JSON file into the in-memory dict."""
    with _lock:
        if DB_FILE.exists():
            try:
                with open(DB_FILE, "r") as f:
                    data = json.load(f)
                users_db.clear()
                users_db.update(data)
                logger.info("Loaded %d user(s) from %s", len(users_db), DB_FILE)
            except Exception as e:
                logger.error("Error loading users: %s", e)


def save_users() -> None:
    """Persist the in-memory user dict to the JSON file."""
    with _lock:
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            serializable = {}
            for key, user in users_db.items():
                user_copy = user.copy()
                if isinstance(user_copy.get("created_at"), datetime):
                    user_copy["created_at"] = user_copy["created_at"].isoformat()
                serializable[key] = user_copy

            with open(DB_FILE, "w") as f:
                json.dump(serializable, f, indent=4)
        except Exception as e:
            logger.error("Error saving users: %s", e)


# Initial load on import
load_users()


# ---------------------------------------------------------------------------
# Password helpers  (no external deps — uses stdlib hashlib PBKDF2)
# ---------------------------------------------------------------------------
_ITERATIONS = 260_000
_HASH_NAME = "sha256"


def get_password_hash(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random salt."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(_HASH_NAME, password.encode(), salt.encode(), _ITERATIONS)
    return f"pbkdf2:{_HASH_NAME}:{_ITERATIONS}${salt}${dk.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify plain-text password against a stored hash.

    Supports:
      - pbkdf2:sha256:... (new format, created by get_password_hash above)
      - $2... (legacy bcrypt — kept so any pre-existing accounts still work)
      - bare hex (legacy SHA-256)
    """
    if hashed.startswith("pbkdf2:"):
        try:
            _, hash_name, rest = hashed.split(":", 2)
            iterations_str, salt, stored_hex = rest.split("$", 2)
            dk = hashlib.pbkdf2_hmac(
                hash_name, plain.encode(), salt.encode(), int(iterations_str)
            )
            return secrets.compare_digest(dk.hex(), stored_hex)
        except Exception:
            return False

    if hashed.startswith("$2"):
        # Legacy bcrypt — attempt via passlib if available, else reject
        try:
            from passlib.hash import bcrypt as _bcrypt
            return _bcrypt.verify(plain, hashed)
        except Exception:
            return False

    # Legacy bare SHA-256
    return secrets.compare_digest(
        hashlib.sha256(plain.encode()).hexdigest(), hashed
    )


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT. Returns the payload dict or None if invalid."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def get_current_user(token: str) -> Optional[dict]:
    """Verify token and return the user dict."""
    payload = verify_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None

    with _lock:
        for user in users_db.values():
            if user["id"] == user_id:
                return user
    return None


def update_user_progress(user_id: str, progress_data: dict) -> bool:
    """Update progress for a user."""
    with _lock:
        for user in users_db.values():
            if user["id"] == user_id:
                if "progress" not in user:
                    user["progress"] = {}
                user["progress"].update(progress_data)
                save_users()
                return True
    return False


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------
def register_user(username: str, email: str, password: str) -> dict:
    """Register a new user. Raises ValueError on duplicate email/username."""
    with _lock:
        email_lower = email.lower()

        if email_lower in users_db:
            raise ValueError("An account with this email already exists.")

        if any(u["username"].lower() == username.lower() for u in users_db.values()):
            raise ValueError("This username is already taken.")

        user_id = str(uuid.uuid4())
        users_db[email_lower] = {
            "id": user_id,
            "username": username,
            "email": email_lower,
            "password_hash": get_password_hash(password),
            "created_at": datetime.utcnow().isoformat(),
        }
        save_users()
        logger.info("Registered new user: %s (%s)", username, email_lower)
        return users_db[email_lower]


def authenticate_user(identifier: str, password: str) -> Optional[dict]:
    """
    Find user by email or username and verify password.
    Returns the user dict on success, None on failure.
    """
    with _lock:
        identifier_lower = identifier.strip().lower()

        # Look up by email first
        user = users_db.get(identifier_lower)

        # Fall back to username search
        if user is None:
            user = next(
                (u for u in users_db.values() if u["username"].lower() == identifier_lower),
                None,
            )

        if user is None:
            return None

        if not verify_password(password, user["password_hash"]):
            return None

        # Transparently upgrade legacy hashes to PBKDF2 on successful login
        if not user["password_hash"].startswith("pbkdf2:"):
            user["password_hash"] = get_password_hash(password)
            save_users()
            logger.info("Upgraded password hash for user %s to pbkdf2", user["username"])

        return user
