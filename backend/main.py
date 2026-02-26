import asyncio
import concurrent.futures
import json
import logging
import shutil
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Thread pool for running synchronous inference without blocking the event loop
_inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

from ml.inference_engine import inference_engine
from core.auth import (
    UserRegister,
    UserLogin,
    UserResponse,
    register_user,
    authenticate_user,
    create_access_token,
    get_current_user,
    update_user_progress,
)
from core.admin_auth import (
    authenticate_admin,
    create_session,
    verify_session,
    delete_session,
)

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "data" / "config.json"


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


_cfg = _load_config()
_server_cfg = _cfg.get("server", {})

# CORS origins — set CORS_ORIGINS env var in production (comma-separated)
_cors_origins: List[str] = os.environ.get("CORS_ORIGINS", "").split(",")
if not _cors_origins or _cors_origins == [""]:
    _cors_origins = _server_cfg.get("cors_origins", ["*"])


def _safe_path(base: Path, *parts: str) -> Path:
    """Resolve a path from user-supplied parts and ensure it stays inside base.

    Raises HTTP 400 if the resolved path escapes the base directory
    (path traversal guard).
    """
    try:
        resolved = (base / Path(*parts)).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not str(resolved).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path.")
    return resolved

# Content directory — runtime data, not source code
_CONTENT_DIR = Path(__file__).parent / "content"
_CONTENT_DIR.mkdir(exist_ok=True)

# Paths for static assets served to the admin panel
_ADMIN_HTML = Path(__file__).parent / "admin.html"
_LOGO_PATH = Path(__file__).parent / "logo.png"  # Fallback for containerized deployment
if not _LOGO_PATH.exists():
    # Attempt to find it relative to workspace if running locally
    _WS_LOGO = Path(__file__).parent.parent / "socratic_app" / "assets" / "images" / "logo.png"
    if _WS_LOGO.exists():
        _LOGO_PATH = _WS_LOGO

# ---------------------------------------------------------------------------
# App lifespan — startup checks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Warn about insecure configuration; crash in production (ENVIRONMENT=production)."""
    is_prod = os.environ.get("ENVIRONMENT", "").lower() == "production"
    required = {
        "JWT_SECRET_KEY": "Signs JWT tokens — must be a long random string",
        "CORS_ORIGINS": "Comma-separated list of allowed browser origins",
        "ADMIN_EMAIL": "Admin panel login email",
        "ADMIN_PASSWORD": "Admin panel login password",
    }
    missing = {k: v for k, v in required.items() if not os.environ.get(k)}
    if missing:
        msg = "Missing environment variables:\n" + "\n".join(
            f"  {k}: {v}" for k, v in missing.items()
        )
        if is_prod:
            raise RuntimeError(msg)
        logger.warning(msg)
    yield  # application runs here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Socratic AI Tutor", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if "*" in _cors_origins:
    logger.warning(
        "CORS is configured with wildcard origin. "
        "Set CORS_ORIGINS env var to restrict origins in production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,  # Auth uses header tokens, not cookies — no need for credentials
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[Dict[str, str]]] = None
    max_tokens: Optional[int] = Field(None, ge=1, le=512)


class ChatResponse(BaseModel):
    response: str
    socratic_index: float
    scaffolding_level: str
    sentiment: str


# ---------------------------------------------------------------------------
# Helper — auth dependencies
# ---------------------------------------------------------------------------
async def _get_auth_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token.")
    
    token = authorization.replace("Bearer ", "")
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")
    return user


def _require_admin(x_admin_token: Optional[str]) -> dict:
    admin = verify_session(x_admin_token)
    if not admin:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return admin


# ---------------------------------------------------------------------------
# App routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Socratic AI Tutor API is running"}


@app.post("/register", response_model=UserResponse, status_code=201)
@limiter.limit("5/minute")
async def register(payload: UserRegister, request: Request):
    logger.info("Register attempt: username=%s email=%s", payload.username, payload.email)
    try:
        user = register_user(payload.username, payload.email, payload.password)
        return UserResponse(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            created_at=user["created_at"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Registration error: %s", e)
        raise HTTPException(status_code=500, detail="Registration failed.")


@app.post("/login")
@limiter.limit("10/minute")
async def login(payload: UserLogin, request: Request):
    logger.info("Login attempt: identifier=%s", payload.email)
    user = authenticate_user(payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_access_token({"sub": user["id"], "username": user["username"]})
    logger.info("Login successful: %s", user["username"])
    return {"id": user["id"], "username": user["username"], "token": token}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": inference_engine.model is not None}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(payload: ChatRequest, request: Request, user: dict = Depends(_get_auth_user)):
    logger.info("Chat request from %s: %.80s...", user.get("username", "?"), payload.message)
    try:
        loop = asyncio.get_event_loop()
        data = await asyncio.wait_for(
            loop.run_in_executor(
                _inference_executor,
                lambda: inference_engine.generate_response(
                    user_message=payload.message,
                    history=payload.history,
                    max_tokens=payload.max_tokens,
                ),
            ),
            timeout=60.0,
        )
        logger.info("Chat response: %.80s...", data["response"])
        return ChatResponse(**data)
    except asyncio.TimeoutError:
        logger.error("Chat timeout for user %s", user.get("username", "?"))
        raise HTTPException(status_code=503, detail="The AI is taking too long. Please try again.")
    except Exception:
        logger.exception("Chat error for user %s", user.get("username", "?"))
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")


@app.get("/user/progress")
async def get_progress(user: dict = Depends(_get_auth_user)):
    return user.get("progress", {})


@app.post("/user/progress")
async def sync_progress(data: dict, user: dict = Depends(_get_auth_user)):
    success = update_user_progress(user["id"], data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to sync progress.")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin web UI
# ---------------------------------------------------------------------------
@app.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    if not _ADMIN_HTML.exists():
        raise HTTPException(status_code=503, detail="Admin panel not available.")
    return HTMLResponse(_ADMIN_HTML.read_text())


@app.get("/admin/logo")
async def admin_logo():
    if _LOGO_PATH.exists():
        return FileResponse(str(_LOGO_PATH), media_type="image/png")
    raise HTTPException(status_code=404)


# ---------------------------------------------------------------------------
# Admin auth endpoints
# ---------------------------------------------------------------------------
@app.post("/admin/api/login")
async def admin_login(request: Request):
    body = await request.json()
    admin = authenticate_admin(body.get("email", ""), body.get("password", ""))
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_session(admin["id"])
    logger.info("Admin login: %s", admin["email"])
    return {"token": token, "name": admin["name"], "email": admin["email"]}


@app.post("/admin/api/logout")
async def admin_logout(x_admin_token: Optional[str] = Header(default=None)):
    if x_admin_token:
        delete_session(x_admin_token)
    return {"status": "ok"}


@app.get("/admin/api/me")
async def admin_me(x_admin_token: Optional[str] = Header(default=None)):
    admin = _require_admin(x_admin_token)
    return {"id": admin["id"], "name": admin["name"], "email": admin["email"]}


# ---------------------------------------------------------------------------
# Public content routes  (used by the Flutter app)
# ---------------------------------------------------------------------------
@app.get("/content/manifest")
async def content_manifest():
    courses = []
    for course_dir in sorted(_CONTENT_DIR.iterdir()):
        course_json = course_dir / "course.json"
        if course_dir.is_dir() and course_json.exists():
            try:
                data = json.loads(course_json.read_text())
                courses.append({
                    "id": data.get("id", course_dir.name),
                    "title": data.get("title", ""),
                    "description": data.get("description", ""),
                    "thumbnail": data.get("thumbnail", ""),
                    "difficulty": data.get("difficulty", ""),
                    "duration": data.get("duration", ""),
                    "totalLessons": data.get("totalLessons", 0),
                })
            except Exception:
                pass
    return {"courses": courses}


@app.get("/content/{course_id}/course.json")
async def content_course(course_id: str):
    course_json = _safe_path(_CONTENT_DIR, course_id, "course.json")
    if not course_json.exists():
        raise HTTPException(status_code=404, detail="Course not found.")
    return json.loads(course_json.read_text())


@app.get("/content/{course_id}/lessons/{filename}")
async def content_lesson(course_id: str, filename: str):
    lesson_file = _safe_path(_CONTENT_DIR, course_id, "lessons", filename)
    if not lesson_file.exists():
        raise HTTPException(status_code=404, detail="Lesson not found.")
    return PlainTextResponse(lesson_file.read_text())


# ---------------------------------------------------------------------------
# Admin content CRUD  (require session token)
# ---------------------------------------------------------------------------
@app.get("/admin/api/courses")
async def admin_list_courses(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    courses = []
    for course_dir in sorted(_CONTENT_DIR.iterdir()):
        course_json = course_dir / "course.json"
        if course_dir.is_dir() and course_json.exists():
            try:
                courses.append(json.loads(course_json.read_text()))
            except Exception:
                pass
    return {"courses": courses}


@app.post("/admin/api/courses/{course_id}", status_code=201)
async def admin_save_course(
    course_id: str,
    request: Request,
    x_admin_token: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_token)
    course_dir = _safe_path(_CONTENT_DIR, course_id)
    course_dir.mkdir(parents=True, exist_ok=True)
    (course_dir / "lessons").mkdir(exist_ok=True)
    body = await request.json()
    body["id"] = course_id
    (course_dir / "course.json").write_text(json.dumps(body, indent=2, ensure_ascii=False))
    logger.info("Admin: saved course %s", course_id)
    return {"status": "ok", "id": course_id}


@app.delete("/admin/api/courses/{course_id}")
async def admin_delete_course(
    course_id: str,
    x_admin_token: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_token)
    course_dir = _safe_path(_CONTENT_DIR, course_id)
    if not course_dir.exists():
        raise HTTPException(status_code=404, detail="Course not found.")
    shutil.rmtree(course_dir)
    logger.info("Admin: deleted course %s", course_id)
    return {"status": "ok"}


@app.post("/admin/api/courses/{course_id}/lessons/{filename}", status_code=201)
async def admin_save_lesson(
    course_id: str,
    filename: str,
    request: Request,
    x_admin_token: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_token)
    lesson_file = _safe_path(_CONTENT_DIR, course_id, "lessons", filename)
    lesson_file.parent.mkdir(parents=True, exist_ok=True)
    content = (await request.body()).decode("utf-8")
    lesson_file.write_text(content)
    logger.info("Admin: saved lesson %s/%s", course_id, filename)
    return {"status": "ok"}


@app.delete("/admin/api/courses/{course_id}/lessons/{filename}")
async def admin_delete_lesson(
    course_id: str,
    filename: str,
    x_admin_token: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_token)
    lesson_file = _safe_path(_CONTENT_DIR, course_id, "lessons", filename)
    if not lesson_file.exists():
        raise HTTPException(status_code=404, detail="Lesson not found.")
    lesson_file.unlink()
    logger.info("Admin: deleted lesson %s/%s", course_id, filename)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    host = _server_cfg.get("host", "0.0.0.0")
    port = _server_cfg.get("port", 8000)
    uvicorn.run(app, host=host, port=port)
