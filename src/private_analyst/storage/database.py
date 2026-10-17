from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    username TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    UNIQUE(org_id, username)
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    username TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS saved_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


class ObjectStore:
    """Manages saved artifacts and objects with strict organization isolation."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def save(self, org_id: str, kind: str, name: str, path: str) -> int:
        if not org_id.strip():
            from private_analyst.exceptions import IsolationError
            raise IsolationError("Organization ID must not be empty.")
        from datetime import datetime, timezone
        with connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO saved_objects(org_id, kind, name, path, created_at) VALUES (?, ?, ?, ?, ?)",
                (org_id.strip(), kind.strip(), name.strip(), str(path), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_for_org(self, org_id: str, kind: str | None = None) -> list[dict]:
        with connect(self.db_path) as conn:
            if kind:
                rows = conn.execute(
                    "SELECT id, org_id, kind, name, path, created_at FROM saved_objects "
                    "WHERE org_id = ? AND kind = ? ORDER BY id DESC",
                    (org_id.strip(), kind.strip()),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, org_id, kind, name, path, created_at FROM saved_objects "
                    "WHERE org_id = ? ORDER BY id DESC",
                    (org_id.strip(),),
                ).fetchall()
        return [dict(row) for row in rows]

    def get(self, org_id: str, object_id: int) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, org_id, kind, name, path, created_at FROM saved_objects WHERE id = ?",
                (object_id,),
            ).fetchone()
        if row is None:
            return None
        if row["org_id"] != org_id.strip():
            from private_analyst.exceptions import IsolationError
            raise IsolationError("Cross-organization access is not allowed.")
        return dict(row)

    def delete(self, org_id: str, object_id: int) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT org_id FROM saved_objects WHERE id = ?",
                (object_id,),
            ).fetchone()
            if row is None:
                return False
            if row["org_id"] != org_id.strip():
                from private_analyst.exceptions import IsolationError
                raise IsolationError("Cross-organization access is not allowed.")
            conn.execute("DELETE FROM saved_objects WHERE id = ? AND org_id = ?", (object_id, org_id.strip()))
            conn.commit()
            return True

