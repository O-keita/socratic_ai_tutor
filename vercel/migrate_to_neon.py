"""One-off import of the file-based data (backend/data/users.json, backend/content/, chat_logs.json)
into Neon. Safe to re-run: existing rows are left alone.

    DATABASE_URL=postgres://... python migrate_to_neon.py [path/to/backend]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from psycopg.types.json import Jsonb

from core import db

backend = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "backend")

with db.connect() as conn:
    users = json.loads((backend / "data" / "users.json").read_text() or "{}")
    n = 0
    for email, u in users.items():
        h = u.get("password_hash") or u.get("password") or ""
        cur = conn.execute(
            "INSERT INTO users (id, username, email, password_hash, created_at, progress) "
            "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (u["id"], u["username"], u.get("email", email).lower(), h,
             datetime.fromisoformat(u["created_at"]) if u.get("created_at") else datetime.utcnow(),
             Jsonb(u.get("progress", {}))))
        n += cur.rowcount
    print(f"users: {n} imported of {len(users)}")

    courses = lessons = 0
    for cdir in sorted((backend / "content").iterdir()):
        cj = cdir / "course.json"
        if not cdir.is_dir() or not cj.exists():
            continue
        data = json.loads(cj.read_text())
        data["id"] = cdir.name
        courses += conn.execute("INSERT INTO courses (id, data) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                                (cdir.name, Jsonb(data))).rowcount
        for f in sorted((cdir / "lessons").glob("*.md")):
            lessons += conn.execute(
                "INSERT INTO lessons (course_id, filename, content) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (cdir.name, f.name, f.read_text())).rowcount
    print(f"courses: {courses}, lessons: {lessons}")
