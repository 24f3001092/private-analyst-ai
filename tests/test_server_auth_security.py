"""Regression tests for server authentication hardening and configuration precedence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from private_analyst.auth.users import User, UserStore
from private_analyst.config import AppConfig, load_config, save_config
from private_analyst.exceptions import AuthenticationError, ValidationError
from private_analyst.security.sessions import SessionManager
from private_analyst.ui.streamlit_app import _check_session_timeout, _desktop_user, _init_state


def test_server_mode_forces_require_auth_from_config(tmp_path: Path):
    """Config file setting edition to server with require_auth=False must force require_auth=True."""
    config_file = tmp_path / "config.json"
    payload = {
        "server": {
            "edition": "server",
            "require_auth": False,
        }
    }
    config_file.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_config(config_file)
    assert cfg.server.edition == "server"
    assert cfg.server.require_auth is True


def test_server_mode_forces_require_auth_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Environment variable PRIVATE_ANALYST_EDITION=server forces require_auth=True."""
    monkeypatch.setenv("PRIVATE_ANALYST_EDITION", "server")
    monkeypatch.setenv("PRIVATE_ANALYST_DATA_DIR", str(tmp_path))

    # Even if config.json says require_auth = false
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"server": {"require_auth": False}}), encoding="utf-8")

    cfg = load_config(config_file)
    assert cfg.server.edition == "server"
    assert cfg.server.require_auth is True


def test_save_config_preserves_server_auth_invariant(tmp_path: Path):
    """Saving configuration in server mode must always persist require_auth=True."""
    cfg = AppConfig()
    cfg.data_dir = str(tmp_path)
    cfg.server.edition = "server"
    cfg.server.require_auth = False  # Try to set False

    saved_path = save_config(cfg)
    raw = json.loads(saved_path.read_text(encoding="utf-8"))
    assert raw["server"]["edition"] == "server"
    assert raw["server"]["require_auth"] is True


def test_invalid_edition_rejected():
    """Any edition other than 'desktop' or 'server' must be rejected with ValidationError."""
    with patch.dict("os.environ", {"PRIVATE_ANALYST_EDITION": "cloud_hosted"}):
        with pytest.raises(ValidationError, match="Invalid edition"):
            load_config()


def test_ai_non_loopback_host_rejected():
    """Non-loopback AI hosts must be rejected to prevent remote data exfiltration."""
    with patch.dict("os.environ", {"PRIVATE_ANALYST_AI_HOST": "192.168.1.100"}):
        with pytest.raises(ValidationError, match="AI inference host.*not allowed"):
            load_config()


def test_desktop_edition_defaults():
    """Desktop edition allows require_auth=False and returns local desktop user."""
    with patch.dict("os.environ", {"PRIVATE_ANALYST_EDITION": "desktop"}):
        cfg = load_config()
        assert cfg.server.edition == "desktop"
        user = _desktop_user("desktop")
        assert user.id == 0
        assert user.username == "local-owner"
        assert user.role == "admin"


def test_desktop_user_rejected_in_server_mode():
    """Calling _desktop_user in server mode must fail closed with AuthenticationError."""
    with pytest.raises(AuthenticationError, match="Desktop auto-login is not allowed in server mode"):
        _desktop_user("server")


class MockSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


def test_init_state_server_mode_leaves_user_unauthenticated(tmp_path: Path):
    """In server mode, _init_state must set st.session_state.user to None."""
    cfg = AppConfig()
    cfg.data_dir = str(tmp_path)
    cfg.server.edition = "server"
    cfg.server.require_auth = True

    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        _init_state(cfg)
        assert mock_state["user"] is None
        assert mock_state["session_token"] is None


def test_session_manager_lifecycle():
    """Verify session creation, token validation, touch, expiration, and destruction."""
    mgr = SessionManager(idle_timeout_minutes=15)
    token = mgr.create_session(user_id=42, org_id="acme")
    assert isinstance(token, str) and len(token) > 20

    # Validation succeeds
    sess = mgr.validate_session(token)
    assert sess is not None
    assert sess.user_id == 42
    assert sess.org_id == "acme"

    # Unknown token returns None
    assert mgr.validate_session("invalid-token-123") is None

    # Touch updates activity
    mgr.touch_session(token)

    # Destruction removes session
    mgr.destroy_session(token)
    assert mgr.validate_session(token) is None


def test_session_manager_idle_expiration():
    """Sessions exceeding idle timeout must be considered expired and rejected."""
    mgr = SessionManager(idle_timeout_minutes=10)
    token = mgr.create_session(user_id=1, org_id="tenant-x")

    # Fast-forward session's last activity by 11 minutes
    sess = mgr._sessions[token]
    sess.last_activity = datetime.now(timezone.utc) - timedelta(minutes=11)

    assert mgr.validate_session(token) is None
    # Cleanup should remove the expired session
    removed = mgr.cleanup_expired()
    assert removed == 1
    assert token not in mgr._sessions


def test_check_session_timeout_active_session():
    """An active session with valid token must not be expired."""
    cfg = AppConfig()
    cfg.server.edition = "server"
    cfg.server.require_auth = True
    cfg.server.session_idle_minutes = 30

    mgr = SessionManager(idle_timeout_minutes=30)
    token = mgr.create_session(user_id=1, org_id="org1")

    user = User(id=1, org_id="org1", username="analyst1", role="analyst")
    mock_state = MockSessionState({
        "user": user,
        "session_token": token,
        "last_activity": datetime.now(timezone.utc),
    })

    with patch("streamlit.session_state", mock_state), \
         patch("private_analyst.ui.streamlit_app._get_session_manager", return_value=mgr):
        expired = _check_session_timeout(cfg)
        assert expired is False
        assert mock_state["user"] is not None
        assert mock_state["session_token"] == token


def test_check_session_timeout_expired_session():
    """An expired session must be cleared from state and return True."""
    cfg = AppConfig()
    cfg.server.edition = "server"
    cfg.server.require_auth = True
    cfg.server.session_idle_minutes = 15

    mgr = SessionManager(idle_timeout_minutes=15)
    token = mgr.create_session(user_id=1, org_id="org1")

    # Fast-forward activity beyond timeout
    mgr._sessions[token].last_activity = datetime.now(timezone.utc) - timedelta(minutes=20)

    user = User(id=1, org_id="org1", username="analyst1", role="analyst")
    mock_state = MockSessionState({
        "user": user,
        "session_token": token,
        "ingest": "some_data",
        "duckdb_console": "console",
        "last_activity": datetime.now(timezone.utc) - timedelta(minutes=20),
    })

    with patch("streamlit.session_state", mock_state), \
         patch("streamlit.warning") as mock_warning, \
         patch("private_analyst.ui.streamlit_app._get_session_manager", return_value=mgr):
        expired = _check_session_timeout(cfg)
        assert expired is True
        assert mock_state["user"] is None
        assert mock_state["session_token"] is None
        assert mock_state["ingest"] is None
        assert mock_state["duckdb_console"] is None
        mock_warning.assert_called_once()

