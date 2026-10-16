from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from private_analyst.storage.database import connect

SENSITIVE_KEYS = ("password", "secret", "token", "csv", "payload", "prompt")


def _sanitize(detail: str | None) -> str | None:
    if detail is None:
        return None
    lowered = detail.lower()
    if any(key in lowered for key in SENSITIVE_KEYS):
        return "[redacted]"
    return detail[:300]


class AuditLog:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def record(self, org_id: str, username: str | None, action: str, detail: str | None = None) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_events(org_id, username, action, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    org_id,
                    username,
                    action,
                    _sanitize(detail),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def list_for_org(self, org_id: str, limit: int = 100) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, org_id, username, action, detail, created_at FROM audit_events "
                "WHERE org_id = ? ORDER BY id DESC LIMIT ?",
                (org_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
