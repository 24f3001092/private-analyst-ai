"""Regression tests for multi-tenant organization isolation and scoping."""

from __future__ import annotations

from pathlib import Path
import pytest

from private_analyst.audit.logger import AuditLog
from private_analyst.auth.users import UserStore
from private_analyst.exceptions import AuthenticationError, IsolationError
from private_analyst.security.isolation import assert_same_org, scoped_query_params
from private_analyst.security.sessions import SessionManager
from private_analyst.storage.database import ObjectStore


def test_user_authentication_cross_tenant_isolation(tmp_path: Path):
    """Users with the same username in different orgs are isolated and cannot cross-authenticate."""
    db_path = tmp_path / "app.db"
    store = UserStore(db_path)

    # Register user 'analyst_alice' in both orgs with distinct passwords
    u_acme = store.create_user("acme", "alice", "AlicePasswordAcme123!", "analyst")
    u_beta = store.create_user("beta", "alice", "AlicePasswordBeta456!", "analyst")

    assert u_acme.org_id == "acme"
    assert u_beta.org_id == "beta"
    assert u_acme.id != u_beta.id

    # Alice can authenticate in acme with acme password
    auth_acme = store.authenticate("acme", "alice", "AlicePasswordAcme123!")
    assert auth_acme.id == u_acme.id
    assert auth_acme.org_id == "acme"

    # Alice cannot authenticate in beta with acme password
    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        store.authenticate("beta", "alice", "AlicePasswordAcme123!")

    # Alice can authenticate in beta with beta password
    auth_beta = store.authenticate("beta", "alice", "AlicePasswordBeta456!")
    assert auth_beta.id == u_beta.id
    assert auth_beta.org_id == "beta"

    # Lookup by org strictly scopes
    assert store.get("acme", "alice").id == u_acme.id
    assert store.get("beta", "alice").id == u_beta.id
    assert store.get("gamma", "alice") is None


def test_audit_log_tenant_isolation(tmp_path: Path):
    """Audit logs for one organization are completely invisible to other organizations."""
    db_path = tmp_path / "app.db"
    log = AuditLog(db_path)

    # Record events for Org A and Org B
    log.record("acme", "alice", "export_report", "sales_q1.csv")
    log.record("acme", "bob", "login", "successful")
    log.record("beta", "charlie", "sql_query", "SELECT 1")

    # Org A sees only Org A events
    acme_events = log.list_for_org("acme")
    assert len(acme_events) == 2
    assert all(e["org_id"] == "acme" for e in acme_events)
    actions_acme = {e["action"] for e in acme_events}
    assert actions_acme == {"export_report", "login"}

    # Org B sees only Org B events
    beta_events = log.list_for_org("beta")
    assert len(beta_events) == 1
    assert beta_events[0]["org_id"] == "beta"
    assert beta_events[0]["action"] == "sql_query"

    # Unrelated Org C sees zero events
    assert log.list_for_org("gamma") == []


def test_audit_log_sensitive_redaction(tmp_path: Path):
    """Audit log sanitizes sensitive terms such as password, token, or secret."""
    db_path = tmp_path / "app.db"
    log = AuditLog(db_path)

    log.record("acme", "admin", "change_password", "new password was Secret12345678!")
    log.record("acme", "admin", "api_token", "created token abcdef123456")

    events = log.list_for_org("acme")
    assert len(events) == 2
    assert events[0]["detail"] == "[redacted]"
    assert events[1]["detail"] == "[redacted]"


def test_object_store_cross_tenant_isolation(tmp_path: Path):
    """ObjectStore enforces strict org boundary; cross-tenant gets and deletes raise IsolationError."""
    db_path = tmp_path / "app.db"
    store = ObjectStore(db_path)

    # Org A saves an object
    acme_obj_id = store.save("acme", "report", "Acme Q1 Revenue", "/storage/acme_q1.pdf")
    # Org B saves an object
    beta_obj_id = store.save("beta", "report", "Beta Product Plan", "/storage/beta_plan.pdf")

    # Scoped listing
    acme_list = store.list_for_org("acme")
    assert len(acme_list) == 1
    assert acme_list[0]["id"] == acme_obj_id
    assert acme_list[0]["name"] == "Acme Q1 Revenue"

    beta_list = store.list_for_org("beta")
    assert len(beta_list) == 1
    assert beta_list[0]["id"] == beta_obj_id
    assert beta_list[0]["name"] == "Beta Product Plan"

    # Org A can get their own object
    obj = store.get("acme", acme_obj_id)
    assert obj is not None
    assert obj["name"] == "Acme Q1 Revenue"

    # Org A attempting to access Org B's object raises IsolationError
    with pytest.raises(IsolationError, match="Cross-organization access is not allowed"):
        store.get("acme", beta_obj_id)

    # Org A attempting to delete Org B's object raises IsolationError
    with pytest.raises(IsolationError, match="Cross-organization access is not allowed"):
        store.delete("acme", beta_obj_id)

    # Org B's object remains intact
    assert store.get("beta", beta_obj_id)["name"] == "Beta Product Plan"

    # Org A can delete their own object
    assert store.delete("acme", acme_obj_id) is True
    assert store.get("acme", acme_obj_id) is None


def test_assert_same_org_guard():
    """assert_same_org raises IsolationError on mismatch."""
    assert_same_org("tenant_1", "tenant_1")  # No error

    with pytest.raises(IsolationError, match="Cross-organization access is not allowed"):
        assert_same_org("tenant_1", "tenant_2")


def test_scoped_query_params():
    """scoped_query_params returns correct SQL fragment and bound parameters."""
    clause, params = scoped_query_params("org_abc")
    assert clause == "org_id = ?"
    assert params == ("org_abc",)


def test_session_cross_tenant_validation():
    """Session created for one organization cannot be claimed by another organization."""
    mgr = SessionManager()
    token = mgr.create_session(user_id=10, org_id="tenant_alpha")

    sess = mgr.validate_session(token)
    assert sess is not None
    assert sess.org_id == "tenant_alpha"

    # Cross-tenant check
    with pytest.raises(IsolationError):
        assert_same_org(sess.org_id, "tenant_bravo")
