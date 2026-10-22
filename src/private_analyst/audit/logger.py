from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from private_analyst.exceptions import AuditLogError
from private_analyst.storage.database import connect

SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\b(password|secret|token|api[_-]?key|bearer|credential|auth|private[_-]?key|prompt|payload|ssn|csv)\b"),
    re.compile(r"(?i)(?:key|pass|token|secret|auth|credential)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]+"),
    re.compile(r"-----BEGIN [A-Z ]+KEY-----"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),  # Credit card
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN format
)


def _sanitize(detail: str | None) -> str | None:
    if detail is None:
        return None
    # Strip multiline control characters to prevent log injection
    flattened = " ".join(detail.split())
    if any(p.search(flattened) for p in SENSITIVE_PATTERNS):
        return "[redacted]"
    return flattened[:300]


class AuditLog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def record(self, org_id: str, username: str | None, action: str, detail: str | None = None) -> None:
        clean_org = (org_id or "").strip()
        if not clean_org:
            clean_org = "unattributed"
        clean_user = username.strip() if username else None
        clean_action = (action or "").strip()
        if not clean_action:
            raise AuditLogError("Audit action must not be empty.")

        try:
            with connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO audit_events(org_id, username, action, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        clean_org,
                        clean_user,
                        clean_action,
                        _sanitize(detail),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
        except Exception:
            raise AuditLogError("Audit log write failure: audit database unavailable or unwritable.") from None

    def list_for_org(self, org_id: str, limit: int = 100) -> list[dict]:
        clean_org = (org_id or "").strip()
        if not clean_org:
            return []
        try:
            with connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, org_id, username, action, detail, created_at FROM audit_events "
                    "WHERE org_id = ? ORDER BY id DESC LIMIT ?",
                    (clean_org, limit),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            raise AuditLogError("Audit log read failure: audit database unavailable or unreadable.") from None
