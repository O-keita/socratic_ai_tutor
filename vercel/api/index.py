"""Vercel entrypoint: the full Bantaba API (landing, chat, auth, progress, courses, admin).

Same routes as backend/main.py, but state lives in Neon Postgres (core/db.py)
instead of JSON files, because serverless has no persistent disk.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import groq
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from core import auth, db
from ml.inference_engine import inference_engine
from ml.model_loader import model_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAGES = ROOT / "pages"
# Pages are static; let Vercel's CDN serve them so the function rarely runs.
PAGE_CACHE = {"Cache-Control": "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400"}

MAX_SESSION_QUESTIONS = 5
MAX_HISTORY_MESSAGES = 8
MAX_FIELD_CHARS = 2000
MAX_TOKENS_CAP = 400
SESSION_LIMIT_MESSAGE = (
    "You've reached the limit of {limit} questions for this session. "
    "Start a new conversation to keep exploring!"
)
UNAVAILABLE_MESSAGE = (
    "The tutor is taking a short break and can't respond right now. "
    "Please try again in a little while."
)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


# Per-instance counter only (serverless instances don't share memory) — best-effort
# abuse brake; the per-conversation cap and token caps are the real cost controls.
limiter = Limiter(key_func=_client_ip)

app = FastAPI(title="Bantaba AI", docs_url=None, redoc_url=None, openapi_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_FIELD_CHARS)
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


def _clean_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Keep only recent, bounded user/assistant turns so a client can't inflate token use."""
    cleaned = [
        {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:MAX_FIELD_CHARS]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant")
    ]
    return cleaned[-MAX_HISTORY_MESSAGES:]


def _get_auth_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token.")
    user = auth.get_current_user(authorization.replace("Bearer ", "", 1))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session.")
    return user


def _require_admin(x_admin_token: Optional[str]) -> dict:
    if not db.admin_session_valid(x_admin_token):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return auth.admin_record()


def _checked(*names: str) -> None:
    if not all(db.safe_name(n) for n in names):
        raise HTTPException(status_code=400, detail="Invalid path.")


def _page(name: str) -> HTMLResponse:
    path = PAGES / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return HTMLResponse(path.read_text(encoding="utf-8"), headers=PAGE_CACHE)


@app.get("/", response_class=HTMLResponse)
def root():
    return _page("index.html")


@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return _page("chat.html")


@app.get("/logo.png")
def logo():
    return FileResponse(PAGES / "logo.png", media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400, s-maxage=604800"})


@app.get("/health")
def health():
    configured = inference_engine.model is not None
    return {"status": "ok", "model_loaded": configured, "groq_configured": configured}


def _run_chat(payload: ChatRequest, username: str) -> ChatResponse:
    history = _clean_history(payload.history)
    if sum(1 for m in history if m["role"] == "assistant") >= MAX_SESSION_QUESTIONS:
        raise HTTPException(status_code=403,
                            detail=SESSION_LIMIT_MESSAGE.format(limit=MAX_SESSION_QUESTIONS))
    try:
        data = inference_engine.generate_response(
            user_message=payload.message,
            history=history,
            max_tokens=min(payload.max_tokens or MAX_TOKENS_CAP, MAX_TOKENS_CAP),
        )
        # Vercel captures stdout — usage numbers are visible in the project logs.
        logger.info("chat ok ms=%s prompt_tokens=%s completion_tokens=%s",
                    data.get("response_time_ms"), data.get("prompt_tokens"),
                    data.get("completion_tokens"))
        try:
            db.add_chat_log(username, payload.message, data)
        except Exception:  # never fail a tutor reply because the log write failed
            logger.exception("Could not store chat log")
        return ChatResponse(**data)
    except groq.GroqError as e:
        logger.error("Groq unavailable: %s", e)
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
    except RuntimeError as e:  # GROQ_API_KEY missing
        logger.error("%s", e)
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
    except Exception:
        logger.exception("Public chat error")
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")


@app.post("/chat/public", response_model=ChatResponse)
@limiter.limit("15/minute")
def chat_public(payload: ChatRequest, request: Request):
    """Public chat endpoint — no auth required. Used by the web chat UI."""
    return _run_chat(payload, "guest")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
def chat(payload: ChatRequest, request: Request, user: dict = Depends(_get_auth_user)):
    return _run_chat(payload, user.get("username", "?"))


@app.exception_handler(404)
async def not_found(_request: Request, _exc):
    return JSONResponse({"detail": "Not found."}, status_code=404)


@app.exception_handler(RuntimeError)
async def config_error(_request: Request, exc: RuntimeError):
    logger.error("Server misconfigured: %s", exc)
    return JSONResponse({"detail": "The service is not configured correctly."}, status_code=503)


# ── Accounts ──────────────────────────────────────────────────────────────────
@app.post("/register", response_model=auth.UserResponse, status_code=201)
@limiter.limit("5/minute")
def register(payload: auth.UserRegister, request: Request):
    try:
        user = auth.register_user(payload.username, payload.email, payload.password)
        return auth.UserResponse(id=user["id"], username=user["username"],
                                 email=user["email"], created_at=user["created_at"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/login")
@limiter.limit("10/minute")
def login(payload: auth.UserLogin, request: Request):
    user = auth.authenticate_user(payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = auth.create_access_token({"sub": user["id"], "username": user["username"]})
    return {"id": user["id"], "username": user["username"], "token": token}


@app.get("/user/progress")
def get_progress(user: dict = Depends(_get_auth_user)):
    return user.get("progress") or {}


@app.post("/user/progress")
def sync_progress(data: dict, user: dict = Depends(_get_auth_user)):
    if not db.update_progress(user["id"], data):
        raise HTTPException(status_code=500, detail="Failed to sync progress.")
    return {"status": "ok"}


@app.get("/model/version")
def model_version():
    """Latest mobile model version info for the Flutter app."""
    mobile = model_loader_cfg().get("mobile_model", {})
    return {
        "version": mobile.get("version", "1.0"),
        "display_name": mobile.get("display_name", "Qwen3-0.6B"),
        "download_url": mobile.get("download_url", ""),
        "release_notes": mobile.get("release_notes", ""),
    }


def model_loader_cfg() -> dict:
    from ml.model_loader import load_config
    return load_config()


# ── Public course content (Flutter app) ───────────────────────────────────────
@app.get("/content/manifest")
def content_manifest():
    keys = ("title", "description", "thumbnail", "difficulty", "duration")
    courses = []
    for data in db.list_courses():
        courses.append({"id": data.get("id", ""), **{k: data.get(k, "") for k in keys},
                        "totalLessons": data.get("totalLessons", 0)})
    return {"courses": courses}


@app.get("/content/{course_id}/course.json")
def content_course(course_id: str):
    _checked(course_id)
    course = db.get_course(course_id)
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found.")
    return course


@app.get("/content/{course_id}/lessons/{filename}")
def content_lesson(course_id: str, filename: str):
    _checked(course_id, filename)
    content = db.get_lesson(course_id, filename)
    if content is None:
        raise HTTPException(status_code=404, detail="Lesson not found.")
    return PlainTextResponse(content)


# ── Admin ─────────────────────────────────────────────────────────────────────
@app.get("/admin", response_class=HTMLResponse)
def admin_ui():
    return _page("admin.html")


@app.get("/admin/logo")
def admin_logo():
    return logo()


@app.post("/admin/api/login")
@limiter.limit("10/minute")
async def admin_login(request: Request):
    body = await request.json()
    admin = auth.authenticate_admin(body.get("email", ""), body.get("password", ""))
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = db.create_admin_session(admin["id"])
    return {"token": token, "name": admin["name"], "email": admin["email"]}


@app.post("/admin/api/logout")
def admin_logout(x_admin_token: Optional[str] = Header(default=None)):
    if x_admin_token:
        db.delete_admin_session(x_admin_token)
    return {"status": "ok"}


@app.get("/admin/api/me")
def admin_me(x_admin_token: Optional[str] = Header(default=None)):
    return _require_admin(x_admin_token)


@app.get("/admin/api/stats")
def admin_stats(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    users = db.list_users()
    total_courses, total_lessons = db.course_counts()
    return {
        "total_users": len(users),
        "chats_since_restart": db.chat_count(),
        "model_loaded": inference_engine.model is not None,
        "model_name": model_loader.model_name,
        "n_ctx": model_loader.n_ctx,
        "temperature": inference_engine.temperature,
        "max_tokens": inference_engine.default_max_tokens,
        "top_p": inference_engine.top_p,
        "uptime_seconds": 0,  # serverless: no long-lived process to measure
        "total_courses": total_courses,
        "total_lessons": total_lessons,
        "recent_users": [{"id": u["id"], "username": u["username"], "email": u["email"],
                          "created_at": u["created_at"]} for u in users[:5]],
    }


@app.get("/admin/api/chat-logs")
def admin_chat_logs(x_admin_token: Optional[str] = Header(default=None), limit: int = 200):
    _require_admin(x_admin_token)
    return db.chat_logs_report(max(1, min(limit, 500)))


@app.get("/admin/api/users")
def admin_list_users(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    users = [{"id": u["id"], "username": u["username"], "email": u["email"],
              "created_at": u["created_at"]} for u in db.list_users()]
    return {"users": users, "total": len(users)}


@app.delete("/admin/api/users/{user_id}")
def admin_delete_user(user_id: str, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    if not db.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "ok"}


@app.get("/admin/api/courses")
def admin_list_courses(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    return {"courses": db.list_courses()}


@app.post("/admin/api/courses/{course_id}", status_code=201)
async def admin_save_course(course_id: str, request: Request,
                            x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    _checked(course_id)
    body = await request.json()
    body["id"] = course_id
    db.save_course(course_id, body)
    return {"status": "ok", "id": course_id}


@app.delete("/admin/api/courses/{course_id}")
def admin_delete_course(course_id: str, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    _checked(course_id)
    if not db.delete_course(course_id):
        raise HTTPException(status_code=404, detail="Course not found.")
    return {"status": "ok"}


@app.post("/admin/api/courses/{course_id}/lessons/{filename}", status_code=201)
async def admin_save_lesson(course_id: str, filename: str, request: Request,
                            x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    _checked(course_id, filename)
    db.save_lesson(course_id, filename, (await request.body()).decode("utf-8"))
    return {"status": "ok"}


@app.get("/admin/api/courses/{course_id}/lessons")
def admin_list_lessons(course_id: str, x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    _checked(course_id)
    return {"lessons": db.list_lessons(course_id)}


@app.delete("/admin/api/courses/{course_id}/lessons/{filename}")
def admin_delete_lesson(course_id: str, filename: str,
                        x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    _checked(course_id, filename)
    if not db.delete_lesson(course_id, filename):
        raise HTTPException(status_code=404, detail="Lesson not found.")
    return {"status": "ok"}
