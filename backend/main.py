import asyncio
import concurrent.futures
import json
import logging
import shutil
import os
from contextlib import asynccontextmanager
from datetime import datetime as _dt
from pathlib import Path
from typing import Optional, List, Dict

# ── Server-wide counters (reset on restart) ──────────────────────────────────
_server_start: _dt = _dt.utcnow()

# ── Chat performance logs (persisted to data/chat_logs.json) ─────────────────
_MAX_CHAT_LOGS = 500
_CHAT_LOGS_PATH = Path(__file__).parent / "data" / "chat_logs.json"


def _load_chat_logs() -> List[Dict]:
    """Load persisted chat logs from disk."""
    try:
        if _CHAT_LOGS_PATH.exists():
            with open(_CHAT_LOGS_PATH, "r") as f:
                logs = json.load(f)
            # Cap to max size
            return logs[-_MAX_CHAT_LOGS:] if len(logs) > _MAX_CHAT_LOGS else logs
    except Exception as e:
        logging.getLogger(__name__).warning("Could not load chat_logs.json: %s", e)
    return []


def _save_chat_logs() -> None:
    """Persist chat logs to disk (best-effort, non-blocking)."""
    try:
        _CHAT_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CHAT_LOGS_PATH, "w") as f:
            json.dump(_chat_logs, f, ensure_ascii=False)
    except Exception as e:
        logging.getLogger(__name__).warning("Could not save chat_logs.json: %s", e)


_chat_logs: List[Dict] = _load_chat_logs()
_chat_request_count = len(_chat_logs)  # Initialize count from persisted logs

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
from ml.model_loader import model_loader
from core.auth import (
    UserRegister,
    UserLogin,
    UserResponse,
    register_user,
    authenticate_user,
    create_access_token,
    get_current_user,
    update_user_progress,
    users_db,
    save_users,
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
app = FastAPI(title="Bantaba AI", lifespan=_lifespan)
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
    response_time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ends_with_question: bool = False


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
_INDEX_HTML = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    if _INDEX_HTML.exists():
        return HTMLResponse(_INDEX_HTML.read_text())
    return HTMLResponse("<h1>Socratic AI Tutor API</h1>")


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


@app.get("/model/version")
async def model_version():
    """Return the latest mobile model version info for the Flutter app."""
    mobile = _cfg.get("mobile_model", {})
    return {
        "version": mobile.get("version", "1.0"),
        "display_name": mobile.get("display_name", "Qwen3-0.6B"),
        "download_url": mobile.get("download_url", ""),
        "release_notes": mobile.get("release_notes", ""),
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(payload: ChatRequest, request: Request, user: dict = Depends(_get_auth_user)):
    global _chat_request_count
    _chat_request_count += 1
    username = user.get("username", "?")
    logger.info("Chat request from %s: %.80s...", username, payload.message)
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

        # ── Store performance log ──
        _chat_logs.append({
            "timestamp": _dt.utcnow().isoformat(),
            "username": username,
            "message_preview": payload.message[:120],
            "response_preview": data["response"][:120],
            "socratic_index": data["socratic_index"],
            "scaffolding_level": data["scaffolding_level"],
            "sentiment": data["sentiment"],
            "response_time_ms": data.get("response_time_ms", 0),
            "prompt_tokens": data.get("prompt_tokens", 0),
            "completion_tokens": data.get("completion_tokens", 0),
            "ends_with_question": data.get("ends_with_question", False),
        })
        # Cap the ring buffer and persist
        if len(_chat_logs) > _MAX_CHAT_LOGS:
            del _chat_logs[: len(_chat_logs) - _MAX_CHAT_LOGS]
        _save_chat_logs()

        return ChatResponse(**data)
    except asyncio.TimeoutError:
        logger.error("Chat timeout for user %s", username)
        raise HTTPException(status_code=503, detail="The AI is taking too long. Please try again.")
    except Exception:
        logger.exception("Chat error for user %s", username)
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
@app.get("/admin/api/stats")
async def admin_stats(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    uptime_secs = int((_dt.utcnow() - _server_start).total_seconds())
    total_courses, total_lessons = 0, 0
    for course_dir in sorted(_CONTENT_DIR.iterdir()):
        if course_dir.is_dir() and (course_dir / "course.json").exists():
            total_courses += 1
            lessons_dir = course_dir / "lessons"
            if lessons_dir.exists():
                total_lessons += len(list(lessons_dir.glob("*.md")))
    all_users = sorted(
        [
            {
                "id": u.get("id", ""),
                "username": u.get("username", ""),
                "email": email,
                "created_at": u.get("created_at", ""),
            }
            for email, u in users_db.items()
        ],
        key=lambda u: u.get("created_at", ""),
        reverse=True,
    )
    return {
        "total_users": len(users_db),
        "chats_since_restart": _chat_request_count,
        "model_loaded": inference_engine.model is not None,
        "model_name": model_loader.filename,
        "model_path": str(model_loader.model_path),
        "model_hf_url": f"https://huggingface.co/{model_loader.repo_id}",
        "n_ctx": model_loader.n_ctx,
        "n_threads": model_loader.n_threads,
        "n_gpu_layers": model_loader.n_gpu_layers,
        "temperature": inference_engine.temperature,
        "max_tokens": inference_engine.default_max_tokens,
        "top_p": inference_engine.top_p,
        "top_k": inference_engine.top_k,
        "repeat_penalty": inference_engine.repeat_penalty,
        "uptime_seconds": uptime_secs,
        "total_courses": total_courses,
        "total_lessons": total_lessons,
        "recent_users": all_users[:5],
    }


@app.get("/admin/api/chat-logs")
async def admin_chat_logs(
    x_admin_token: Optional[str] = Header(default=None),
    limit: int = 200,
):
    """Return stored chat performance logs (most recent first) and aggregate stats."""
    _require_admin(x_admin_token)
    logs = list(reversed(_chat_logs[-limit:]))

    # Compute aggregates
    total = len(_chat_logs)
    if total == 0:
        return {
            "logs": [],
            "total": 0,
            "aggregates": {
                "avg_response_time_ms": 0,
                "avg_socratic_index": 0,
                "question_rate": 0,
                "avg_prompt_tokens": 0,
                "avg_completion_tokens": 0,
                "scaffolding_distribution": {},
                "sentiment_distribution": {},
            },
        }

    avg_rt = round(sum(l["response_time_ms"] for l in _chat_logs) / total)
    avg_si = round(sum(l["socratic_index"] for l in _chat_logs) / total, 3)
    q_count = sum(1 for l in _chat_logs if l.get("ends_with_question"))
    q_rate = round(q_count / total, 3)
    avg_pt = round(sum(l["prompt_tokens"] for l in _chat_logs) / total)
    avg_ct = round(sum(l["completion_tokens"] for l in _chat_logs) / total)

    scaffolding_dist: Dict[str, int] = {}
    sentiment_dist: Dict[str, int] = {}
    for l in _chat_logs:
        scaffolding_dist[l["scaffolding_level"]] = scaffolding_dist.get(l["scaffolding_level"], 0) + 1
        sentiment_dist[l["sentiment"]] = sentiment_dist.get(l["sentiment"], 0) + 1

    return {
        "logs": logs,
        "total": total,
        "aggregates": {
            "avg_response_time_ms": avg_rt,
            "avg_socratic_index": avg_si,
            "question_rate": q_rate,
            "avg_prompt_tokens": avg_pt,
            "avg_completion_tokens": avg_ct,
            "scaffolding_distribution": scaffolding_dist,
            "sentiment_distribution": sentiment_dist,
        },
    }


@app.get("/admin/api/users")
async def admin_list_users(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    users = [
        {
            "id": u.get("id", ""),
            "username": u.get("username", ""),
            "email": u.get("email", email),
            "created_at": u.get("created_at", ""),
        }
        for email, u in users_db.items()
    ]
    users.sort(key=lambda u: u.get("created_at", ""), reverse=True)
    return {"users": users, "total": len(users)}


@app.delete("/admin/api/users/{user_id}", status_code=200)
async def admin_delete_user(
    user_id: str,
    x_admin_token: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_token)
    key_to_delete = next(
        (k for k, u in users_db.items() if u.get("id") == user_id), None
    )
    if key_to_delete is None:
        raise HTTPException(status_code=404, detail="User not found.")
    del users_db[key_to_delete]
    save_users()
    logger.info("Admin: deleted user %s", user_id)
    return {"status": "ok"}


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


@app.get("/admin/api/courses/{course_id}/lessons")
async def admin_list_lessons(
    course_id: str,
    x_admin_token: Optional[str] = Header(default=None),
):
    _require_admin(x_admin_token)
    lessons_dir = _safe_path(_CONTENT_DIR, course_id, "lessons")
    if not lessons_dir.exists():
        return {"lessons": []}
    files = sorted(f.name for f in lessons_dir.iterdir() if f.is_file() and f.suffix == ".md")
    return {"lessons": files}


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
