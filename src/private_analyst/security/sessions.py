from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
import threading


@dataclass
class SessionInfo:
    """Session details for an authenticated user."""

    user_id: int
    org_id: str
    created_at: datetime
    last_activity: datetime
    token: str


class SessionManager:
    """Manages active user sessions with idle timeout and thread-safe operations."""

    def __init__(self, idle_timeout_minutes: int = 30) -> None:
        self.idle_timeout_minutes = idle_timeout_minutes
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = threading.RLock()

    def _is_expired(self, session: SessionInfo, now: datetime) -> bool:
        """Check if a session has exceeded the idle timeout."""
        if self.idle_timeout_minutes <= 0:
            return True
        return (now - session.last_activity) >= timedelta(minutes=self.idle_timeout_minutes)

    def create_session(self, user_id: int, org_id: str) -> str:
        """Create a new session and return the secure token.

        Args:
            user_id: Identifier of the user.
            org_id: Identifier of the organization.

        Returns:
            Secure URL-safe session token string.
        """
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        session = SessionInfo(
            user_id=user_id,
            org_id=org_id,
            created_at=now,
            last_activity=now,
            token=token,
        )
        with self._lock:
            self._sessions[token] = session
        return token

    def validate_session(self, token: str) -> SessionInfo | None:
        """Validate a session token.

        Args:
            token: Session token to validate.

        Returns:
            SessionInfo if valid and not expired; otherwise None.
        """
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if not secrets.compare_digest(session.token, token):
                return None
            now = datetime.now(timezone.utc)
            if self._is_expired(session, now):
                return None
            return session

    def touch_session(self, token: str) -> None:
        """Update last activity timestamp for an active, unexpired session.

        Args:
            token: Session token to refresh.
        """
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return
            now = datetime.now(timezone.utc)
            if self._is_expired(session, now):
                return
            session.last_activity = now

    def destroy_session(self, token: str) -> None:
        """Invalidate and remove a session.

        Args:
            token: Session token to destroy.
        """
        with self._lock:
            self._sessions.pop(token, None)

    def cleanup_expired(self) -> int:
        """Remove all expired sessions from memory.

        Returns:
            Number of sessions removed.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            expired_tokens = [
                tok for tok, sess in self._sessions.items()
                if self._is_expired(sess, now)
            ]
            for tok in expired_tokens:
                del self._sessions[tok]
            return len(expired_tokens)
