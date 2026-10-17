from __future__ import annotations

from dataclasses import dataclass

import bcrypt

from private_analyst.exceptions import AuthenticationError, AuthorizationError
from private_analyst.storage.database import connect
from pathlib import Path

ROLES = ("admin", "analyst", "viewer")


@dataclass
class User:
    id: int
    org_id: str
    username: str
    role: str


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise AuthenticationError("Passwords must be at least 12 characters.")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


class UserStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_user(self, org_id: str, username: str, password: str, role: str) -> User:
        if role not in ROLES:
            raise AuthorizationError("Unknown role.")
        if not username.strip() or any(ch in username for ch in " /\\"):
            raise AuthenticationError("Invalid username.")
        digest = hash_password(password)
        with connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO users(org_id, username, password_hash, role) VALUES (?, ?, ?, ?)",
                (org_id, username.strip(), digest, role),
            )
            conn.commit()
            return User(int(cur.lastrowid), org_id, username.strip(), role)

    def authenticate(self, org_id: str, username: str, password: str) -> User:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, org_id, username, password_hash, role FROM users WHERE org_id = ? AND username = ?",
                (org_id, username),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            raise AuthenticationError("Invalid credentials.")
        return User(int(row["id"]), row["org_id"], row["username"], row["role"])

    def get(self, org_id: str, username: str) -> User | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, org_id, username, role FROM users WHERE org_id = ? AND username = ?",
                (org_id, username),
            ).fetchone()
        if row is None:
            return None
        return User(int(row["id"]), row["org_id"], row["username"], row["role"])
