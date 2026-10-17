from __future__ import annotations

from private_analyst.auth.users import User
from private_analyst.exceptions import AuthorizationError

PERMISSIONS = {
    "admin": {"ingest", "analyze", "export", "documents", "manage_users", "backup", "settings"},
    "analyst": {"ingest", "analyze", "export", "documents"},
    "viewer": {"analyze"},
}


def require(user: User, permission: str) -> None:
    allowed = PERMISSIONS.get(user.role, set())
    if permission not in allowed:
        raise AuthorizationError("You do not have permission for this action.")


def can(user: User, permission: str) -> bool:
    return permission in PERMISSIONS.get(user.role, set())
