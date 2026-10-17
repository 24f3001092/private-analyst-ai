from __future__ import annotations

import pytest

from private_analyst.auth.rbac import can, require
from private_analyst.auth.users import User, UserStore
from private_analyst.exceptions import AuthenticationError, AuthorizationError, IsolationError
from private_analyst.security.isolation import assert_same_org
from private_analyst.audit.logger import AuditLog


def test_password_hash_and_auth(tmp_path) -> None:
    store = UserStore(tmp_path / "app.db")
    store.create_user("acme", "ada", "correct-horse-battery", "admin")
    user = store.authenticate("acme", "ada", "correct-horse-battery")
    assert user.role == "admin"
    with pytest.raises(AuthenticationError):
        store.authenticate("acme", "ada", "wrong-password-xx")
    with pytest.raises(AuthenticationError):
        store.authenticate("other-org", "ada", "correct-horse-battery")


def test_cross_customer_isolation(tmp_path) -> None:
    store = UserStore(tmp_path / "app.db")
    store.create_user("acme", "ada", "correct-horse-battery", "analyst")
    store.create_user("beta", "ada", "correct-horse-battery", "analyst")
    acme = store.authenticate("acme", "ada", "correct-horse-battery")
    beta = store.authenticate("beta", "ada", "correct-horse-battery")
    assert acme.org_id != beta.org_id
    with pytest.raises(IsolationError):
        assert_same_org(acme.org_id, beta.org_id)


def test_viewer_cannot_ingest() -> None:
    viewer = User(1, "acme", "vie", "viewer")
    assert can(viewer, "analyze")
    assert not can(viewer, "ingest")
    with pytest.raises(AuthorizationError):
        require(viewer, "export")


def test_audit_redacts_payloads(tmp_path) -> None:
    log = AuditLog(tmp_path / "app.db")
    log.record("acme", "ada", "login", "password=super-secret-value")
    events = log.list_for_org("acme")
    assert events[0]["detail"] == "[redacted]"
    log.record("beta", "bob", "login", "ok")
    assert all(e["org_id"] == "acme" for e in log.list_for_org("acme"))
