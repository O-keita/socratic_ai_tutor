"""Admin auth — credentials from environment variables, sessions in data/admin_sessions.json."""
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path

_SESSION_TTL_HOURS = 24

_DATA_DIR = Path(__file__).parent.parent / "data"
_CONFIG_PATH = _DATA_DIR / "config.json"
_SESSIONS_FILE = _DATA_DIR / "admin_sessions.json"


def _load_config() -> dict:
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
    except Exception:
        cfg = {}

    # Credentials come from environment variables; config only provides the display name.
    admin_cfg = cfg.get("admin", {})
    admin_cfg["email"] = os.environ.get("ADMIN_EMAIL", admin_cfg.get("email", "admin"))
    admin_cfg["password"] = os.environ.get("ADMIN_PASSWORD", admin_cfg.get("password", ""))
    cfg["admin"] = admin_cfg

    return cfg


def _admin_record() -> dict:
    """Return the single admin record from config."""
    cfg = _load_config().get("admin", {})
    return {
        "id": "admin",
        "name": cfg.get("name", "Admin"),
        "email": cfg.get("email", "admin"),
    }


def authenticate_admin(email: str, password: str):
    cfg = _load_config().get("admin", {})
    stored_email = cfg.get("email", "")
    stored_password = cfg.get("password", "")
    # Use timing-safe comparison to prevent timing attacks
    email_ok = hmac.compare_digest(email.strip().lower(), stored_email.lower())
    password_ok = hmac.compare_digest(password, stored_password)
    if email_ok and password_ok and stored_password:
        return _admin_record()
    return None


# ── Sessions ───────────────────────────────────────────────────────────────────

def _load_sessions() -> dict:
    if _SESSIONS_FILE.exists():
        return json.loads(_SESSIONS_FILE.read_text())
    return {}


def _save_sessions(sessions: dict):
    _SESSIONS_FILE.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))


def create_session(admin_id: str) -> str:
    sessions = _load_sessions()
    token = secrets.token_urlsafe(32)
    sessions[token] = {"admin_id": admin_id, "created_at": datetime.utcnow().isoformat()}
    _save_sessions(sessions)
    return token


def verify_session(token: str | None):
    if not token:
        return None
    sessions = _load_sessions()
    session = sessions.get(token)
    if not session:
        return None
    # Enforce session TTL
    try:
        created_at = datetime.fromisoformat(session["created_at"])
        if datetime.utcnow() - created_at > timedelta(hours=_SESSION_TTL_HOURS):
            # Expired — clean up and reject
            sessions.pop(token, None)
            _save_sessions(sessions)
            return None
    except (KeyError, ValueError):
        # Malformed session — reject
        sessions.pop(token, None)
        _save_sessions(sessions)
        return None
    return _admin_record()


def delete_session(token: str):
    sessions = _load_sessions()
    sessions.pop(token, None)
    _save_sessions(sessions)
