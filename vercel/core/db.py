"""Neon Postgres storage — replaces the JSON files the Docker backend keeps on disk.

Serverless instances have no persistent disk, so users, progress, admin sessions,
chat logs and course content all live in Postgres. DATABASE_URL should be Neon's
*pooled* connection string (host contains "-pooler").
"""
import json
import logging
import os
import re
import secrets
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

ADMIN_SESSION_TTL_HOURS = 24
MAX_CHAT_LOGS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    progress      JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_idx ON users (lower(username));

CREATE TABLE IF NOT EXISTS admin_sessions (
    token      TEXT PRIMARY KEY,
    admin_id   TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_logs (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    username          TEXT,
    message_preview   TEXT,
    response_preview  TEXT,
    socratic_index    DOUBLE PRECISION,
    scaffolding_level TEXT,
    sentiment         TEXT,
    response_time_ms  INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    ends_with_question BOOLEAN
);

CREATE TABLE IF NOT EXISTS courses (
    id   TEXT PRIMARY KEY,
    data JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    filename  TEXT NOT NULL,
    content   TEXT NOT NULL,
    PRIMARY KEY (course_id, filename)
);
"""

_schema_ready = False
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def safe_name(value: str) -> bool:
    """Course ids / lesson filenames: no separators, no leading dot, no '..'."""
    return bool(_SAFE_NAME.match(value)) and ".." not in value


@contextmanager
def connect():
    """Open a short-lived connection (Neon's pooler makes this cheap) and commit on success."""
    global _schema_ready
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    with psycopg.connect(url, row_factory=dict_row, connect_timeout=10) as conn:
        if not _schema_ready:
            conn.execute(_SCHEMA)
            _schema_ready = True
        yield conn


def _iso(dt) -> str:
    return dt.replace(tzinfo=None).isoformat() if isinstance(dt, datetime) else str(dt or "")


def _user(row: Optional[dict]) -> Optional[dict]:
    if row is None:
        return None
    return {**row, "created_at": _iso(row["created_at"])}


# ── Users ─────────────────────────────────────────────────────────────────────
def create_user(username: str, email: str, password_hash: str) -> dict:
    """Insert a user. Raises ValueError on duplicate email/username."""
    email = email.lower()
    with connect() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email = %s", (email,)).fetchone():
            raise ValueError("An account with this email already exists.")
        if conn.execute("SELECT 1 FROM users WHERE lower(username) = lower(%s)", (username,)).fetchone():
            raise ValueError("This username is already taken.")
        try:
            row = conn.execute(
                "INSERT INTO users (id, username, email, password_hash) VALUES (%s, %s, %s, %s) RETURNING *",
                (str(uuid.uuid4()), username, email, password_hash),
            ).fetchone()
        except psycopg.errors.UniqueViolation:  # lost a race with a concurrent signup
            raise ValueError("An account with this email or username already exists.")
    return _user(row)


def find_user(identifier: str) -> Optional[dict]:
    ident = identifier.strip().lower()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = %s OR lower(username) = %s "
            "ORDER BY (email = %s) DESC LIMIT 1",
            (ident, ident, ident),
        ).fetchone()
    return _user(row)


def get_user_by_id(user_id: str) -> Optional[dict]:
    with connect() as conn:
        return _user(conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone())


def set_password_hash(user_id: str, password_hash: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = %s WHERE id = %s", (password_hash, user_id))


def update_progress(user_id: str, data: dict) -> bool:
    """Shallow-merge into the user's progress (same semantics as dict.update)."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE users SET progress = progress || %s WHERE id = %s", (Jsonb(data), user_id)
        )
        return cur.rowcount > 0


def list_users() -> list:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return [_user(r) for r in rows]


def delete_user(user_id: str) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM users WHERE id = %s", (user_id,)).rowcount > 0


# ── Admin sessions ────────────────────────────────────────────────────────────
def create_admin_session(admin_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with connect() as conn:
        conn.execute("DELETE FROM admin_sessions WHERE created_at < now() - %s::interval",
                     (f"{ADMIN_SESSION_TTL_HOURS} hours",))
        conn.execute("INSERT INTO admin_sessions (token, admin_id) VALUES (%s, %s)", (token, admin_id))
    return token


def admin_session_valid(token: Optional[str]) -> bool:
    if not token:
        return False
    with connect() as conn:
        row = conn.execute("SELECT created_at FROM admin_sessions WHERE token = %s", (token,)).fetchone()
        if not row:
            return False
        if datetime.now(timezone.utc) - row["created_at"] > timedelta(hours=ADMIN_SESSION_TTL_HOURS):
            conn.execute("DELETE FROM admin_sessions WHERE token = %s", (token,))
            return False
    return True


def delete_admin_session(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM admin_sessions WHERE token = %s", (token,))


# ── Chat logs ─────────────────────────────────────────────────────────────────
def add_chat_log(username: str, message: str, data: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO chat_logs (username, message_preview, response_preview, socratic_index, "
            "scaffolding_level, sentiment, response_time_ms, prompt_tokens, completion_tokens, "
            "ends_with_question) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (username, message[:120], data["response"][:120], data["socratic_index"],
             data["scaffolding_level"], data["sentiment"], data.get("response_time_ms", 0),
             data.get("prompt_tokens", 0), data.get("completion_tokens", 0),
             data.get("ends_with_question", False)),
        )
        # Ring buffer, like the file-based version: keep only the newest MAX_CHAT_LOGS rows.
        conn.execute(
            "DELETE FROM chat_logs WHERE id <= (SELECT id FROM chat_logs ORDER BY id DESC "
            "OFFSET %s LIMIT 1)", (MAX_CHAT_LOGS,),
        )


def chat_logs_report(limit: int) -> dict:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM chat_logs ORDER BY id DESC LIMIT %s", (limit,)).fetchall()
        agg = conn.execute(
            "SELECT count(*) AS total, avg(response_time_ms) AS rt, avg(socratic_index) AS si, "
            "avg(ends_with_question::int) AS qr, avg(prompt_tokens) AS pt, "
            "avg(completion_tokens) AS ct FROM chat_logs"
        ).fetchone()
        scaff = conn.execute("SELECT scaffolding_level AS k, count(*) AS n FROM chat_logs GROUP BY 1").fetchall()
        sent = conn.execute("SELECT sentiment AS k, count(*) AS n FROM chat_logs GROUP BY 1").fetchall()
    logs = [
        {"timestamp": _iso(r["ts"]), "username": r["username"],
         "message_preview": r["message_preview"], "response_preview": r["response_preview"],
         "socratic_index": r["socratic_index"], "scaffolding_level": r["scaffolding_level"],
         "sentiment": r["sentiment"], "response_time_ms": r["response_time_ms"],
         "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
         "ends_with_question": r["ends_with_question"]}
        for r in rows
    ]
    total = agg["total"]
    if not total:
        return {"logs": [], "total": 0, "aggregates": {
            "avg_response_time_ms": 0, "avg_socratic_index": 0, "question_rate": 0,
            "avg_prompt_tokens": 0, "avg_completion_tokens": 0,
            "scaffolding_distribution": {}, "sentiment_distribution": {}}}
    return {"logs": logs, "total": total, "aggregates": {
        "avg_response_time_ms": round(agg["rt"] or 0),
        "avg_socratic_index": round(agg["si"] or 0, 3),
        "question_rate": round(agg["qr"] or 0, 3),
        "avg_prompt_tokens": round(agg["pt"] or 0),
        "avg_completion_tokens": round(agg["ct"] or 0),
        "scaffolding_distribution": {r["k"]: r["n"] for r in scaff},
        "sentiment_distribution": {r["k"]: r["n"] for r in sent}}}


def chat_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT count(*) AS n FROM chat_logs").fetchone()["n"]


# ── Courses & lessons ─────────────────────────────────────────────────────────
def list_courses() -> list:
    with connect() as conn:
        return [r["data"] for r in conn.execute("SELECT data FROM courses ORDER BY id").fetchall()]


def get_course(course_id: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT data FROM courses WHERE id = %s", (course_id,)).fetchone()
    return row["data"] if row else None


def save_course(course_id: str, data: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO courses (id, data) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data", (course_id, Jsonb(data)))


def delete_course(course_id: str) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM courses WHERE id = %s", (course_id,)).rowcount > 0


def course_counts() -> tuple:
    with connect() as conn:
        c = conn.execute("SELECT count(*) AS n FROM courses").fetchone()["n"]
        l = conn.execute("SELECT count(*) AS n FROM lessons").fetchone()["n"]
    return c, l


def get_lesson(course_id: str, filename: str) -> Optional[str]:
    with connect() as conn:
        row = conn.execute("SELECT content FROM lessons WHERE course_id = %s AND filename = %s",
                           (course_id, filename)).fetchone()
    return row["content"] if row else None


def save_lesson(course_id: str, filename: str, content: str) -> None:
    """Upsert a lesson, creating a stub course row if the course was never saved
    (the file-based version created the directory implicitly)."""
    with connect() as conn:
        conn.execute("INSERT INTO courses (id, data) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                     (course_id, Jsonb({"id": course_id})))
        conn.execute(
            "INSERT INTO lessons (course_id, filename, content) VALUES (%s, %s, %s) "
            "ON CONFLICT (course_id, filename) DO UPDATE SET content = EXCLUDED.content",
            (course_id, filename, content))


def list_lessons(course_id: str) -> list:
    with connect() as conn:
        rows = conn.execute("SELECT filename FROM lessons WHERE course_id = %s AND filename LIKE '%%.md' "
                            "ORDER BY filename", (course_id,)).fetchall()
    return [r["filename"] for r in rows]


def delete_lesson(course_id: str, filename: str) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM lessons WHERE course_id = %s AND filename = %s",
                            (course_id, filename)).rowcount > 0
