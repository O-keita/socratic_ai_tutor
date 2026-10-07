"""Lean Vercel entrypoint: landing page, chat UI, and the public Groq-backed chat API.

Everything else from backend/main.py (auth, admin, courses) is intentionally
left out — it needs a persistent disk, which serverless doesn't have.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ml.inference_engine import inference_engine

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
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
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
    return {"status": "ok", "groq_configured": inference_engine.model is not None}


@app.post("/chat/public", response_model=ChatResponse)
@limiter.limit("15/minute")
def chat_public(payload: ChatRequest, request: Request):
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


@app.exception_handler(404)
async def not_found(_request: Request, _exc):
    return JSONResponse({"detail": "Not found."}, status_code=404)
