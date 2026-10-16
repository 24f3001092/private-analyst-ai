from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from private_analyst.analytics.operations import run_operation
from private_analyst.analytics.sql_console import DuckDBConsole, QueryResult
from private_analyst.audit.logger import AuditLog, _sanitize
from private_analyst.auth.users import User, UserStore
from private_analyst.config import AppConfig
from private_analyst.documents.index import LocalTextIndex
from private_analyst.exceptions import AuditLogError, IsolationError
from private_analyst.ingestion.loader import IngestResult, load_tabular
from private_analyst.security.sessions import SessionManager
from private_analyst.ui.streamlit_app import (
    _ask_tab,
    _check_session_timeout,
    _documents_tab,
    _import_tab,
    _login,
    _ml_tab,
    _reports_tab,
    _safe_record_export_requested,
    _settings_tab,
    _sidebar,
    _sql_tab,
)


class MockSessionState(dict):
    """Dictionary supporting attribute-style access for Streamlit session_state simulation."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


class MockUploadedFile:
    """Mock file upload object with name and getvalue()."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self.data = data

    def getvalue(self) -> bytes:
        return self.data


@pytest.fixture
def test_env(tmp_path: Path):
    """Fixture providing initialized AppConfig, UserStore, and test users."""
    cfg = AppConfig()
    cfg.data_dir = str(tmp_path)
    cfg.server.edition = "server"
    cfg.server.require_auth = True
    cfg.server.session_idle_minutes = 15

    db_path = tmp_path / "app.db"
    store = UserStore(db_path)
    store.create_user("org1", "alice", "AlicePass12345!", "admin")
    store.create_user("org1", "bob", "BobPass12345!", "analyst")
    store.create_user("org1", "charlie", "CharliePass12345!", "viewer")
    store.create_user("org2", "david", "DavidPass12345!", "admin")

    admin_user = store.authenticate("org1", "alice", "AlicePass12345!")
    analyst_user = store.authenticate("org1", "bob", "BobPass12345!")
    viewer_user = store.authenticate("org1", "charlie", "CharliePass12345!")

    return {
        "cfg": cfg,
        "store": store,
        "db_path": db_path,
        "tmp_path": tmp_path,
        "admin": admin_user,
        "analyst": analyst_user,
        "viewer": viewer_user,
    }


def test_audit_log_direct_sanitization_and_isolation(tmp_path: Path) -> None:
    """Verify sanitization of credentials/API keys, multiline scrubbing, and strict org isolation."""
    db_path = tmp_path / "app.db"
    log = AuditLog(db_path)

    # 1. Sanitization of sensitive keys and API tokens
    assert _sanitize("User entered password=secret123") == "[redacted]"
    assert _sanitize("TOKEN expired") == "[redacted]"
    assert _sanitize("api_key=sk-proj-1234567890abcdef") == "[redacted]"
    assert _sanitize("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test") == "[redacted]"
    assert _sanitize("apikey: super-confidential-key") == "[redacted]"
    assert _sanitize("client credential = confidential") == "[redacted]"
    assert _sanitize("-----BEGIN RSA PRIVATE KEY----- abcde") == "[redacted]"
    assert _sanitize("Malicious prompt injection payload") == "[redacted]"
    assert _sanitize("Record contains ssn: 000-00-0000") == "[redacted]"

    # Multiline control character stripping (log injection defense)
    multiline = "First line\r\nSecond line\nThird line"
    assert _sanitize(multiline) == "First line Second line Third line"

    # Multiline containing secret
    multiline_secret = "Safe header\r\npassword = dangerous\nSafe footer"
    assert _sanitize(multiline_secret) == "[redacted]"

    # Safe text passes through
    assert _sanitize("Normal operation completed") == "Normal operation completed"
    assert _sanitize(None) is None

    # Length truncation at 300 characters
    long_text = "A" * 400
    assert len(_sanitize(long_text)) == 300

    # 2. Record events for two different organizations and unattributed
    log.record("org_a", "alice", "action_1", "Detail 1")
    log.record("org_a", "alice", "action_2", "password=foo")
    log.record("org_b", "bob", "action_3", "Detail 3")
    log.record("", "unauth_user", "action_4", "Missing org must be unattributed")

    # 3. Strict organization isolation
    events_a = log.list_for_org("org_a")
    assert len(events_a) == 2
    assert all(e["org_id"] == "org_a" for e in events_a)
    assert events_a[0]["action"] == "action_2"
    assert events_a[0]["detail"] == "[redacted]"
    assert events_a[1]["action"] == "action_1"

    events_b = log.list_for_org("org_b")
    assert len(events_b) == 1
    assert events_b[0]["org_id"] == "org_b"
    assert events_b[0]["username"] == "bob"

    # Missing org must NOT be recorded as "system"
    events_sys = log.list_for_org("system")
    assert len(events_sys) == 0

    events_unattr = log.list_for_org("unattributed")
    assert len(events_unattr) == 1
    assert events_unattr[0]["action"] == "action_4"

    # Empty org read must return empty list (fail-closed)
    assert log.list_for_org("") == []
    assert log.list_for_org("   ") == []


def test_audit_write_failure_policy(tmp_path: Path) -> None:
    """Verify that audit write failures raise AuditLogError without exposing secrets or paths."""
    # Direct to an unwritable directory path (SQLite cannot open a directory as a DB)
    invalid_db_path = tmp_path / "dir_as_db"
    invalid_db_path.mkdir(parents=True, exist_ok=True)
    log = AuditLog(invalid_db_path)

    with pytest.raises(AuditLogError) as exc_info:
        log.record("org1", "alice", "test_action", "secret_pass=foo")

    msg = str(exc_info.value)
    assert "Audit log write failure" in msg
    assert "secret_pass" not in msg
    assert str(invalid_db_path) not in msg

    # Empty action must raise AuditLogError
    with pytest.raises(AuditLogError, match="action must not be empty"):
        log.record("org1", "alice", "")


def test_login_auditing_success_and_failure(test_env) -> None:
    """Verify that successful and failed logins record audit events in app.db."""
    cfg = test_env["cfg"]
    db_path = test_env["db_path"]
    log = AuditLog(db_path)

    # A. Successful login
    state = MockSessionState({"user": None, "session_token": None, "last_activity": None})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "Organization id": "org1",
            "Username": "alice",
            "Password": "AlicePass12345!",
        }.get(label, "")
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Sign in"

        _login(cfg)

    events = log.list_for_org("org1")
    success_events = [e for e in events if e["action"] == "login_success"]
    assert len(success_events) == 1
    assert success_events[0]["username"] == "alice"
    assert "admin" in success_events[0]["detail"]
    assert "AlicePass12345!" not in success_events[0]["detail"]
    assert state.user is not None

    # B. Failed login with valid org
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "Organization id": "org1",
            "Username": "alice",
            "Password": "WrongPassword999!",
        }.get(label, "")
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Sign in"

        _login(cfg)

    events = log.list_for_org("org1")
    fail_events = [e for e in events if e["action"] == "login_failure"]
    assert len(fail_events) == 1
    assert fail_events[0]["username"] == "alice"
    assert "failed" in fail_events[0]["detail"].lower()
    assert "WrongPassword999!" not in fail_events[0]["detail"]

    # C. Failed login with missing org is categorized as 'unauthenticated', never 'system'
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "Organization id": "",
            "Username": "attacker",
            "Password": "WrongPassword999!",
        }.get(label, "")
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Sign in"

        _login(cfg)

    unauth_events = log.list_for_org("unauthenticated")
    assert any(e["action"] == "login_failure" and e["username"] == "attacker" for e in unauth_events)
    assert log.list_for_org("system") == []


def test_login_fail_closed_on_audit_write_error(test_env) -> None:
    """Verify that if the audit log is unavailable during login, sign-in fails closed."""
    cfg = test_env["cfg"]
    state = MockSessionState({"user": None, "session_token": None, "last_activity": None})

    mock_log = MagicMock()
    mock_log.record.side_effect = AuditLogError("Disk full")

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_audit_log", return_value=mock_log):
        mock_st.session_state = state
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "Organization id": "org1",
            "Username": "alice",
            "Password": "AlicePass12345!",
        }.get(label, "")
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Sign in"

        _login(cfg)

        # Must fail-closed: user state remains None
        assert state.user is None
        assert state.session_token is None
        mock_st.error.assert_called_with("Audit subsystem error: sign-in cannot proceed without audit logging.")


def test_logout_auditing(test_env) -> None:
    """Verify that user logout records an audit event and terminates the session."""
    cfg = test_env["cfg"]
    user = test_env["admin"]
    log = AuditLog(test_env["db_path"])

    state = MockSessionState({
        "user": user,
        "session_token": "valid-token-12345",
        "ingest": "some_dataset",
        "duckdb_console": "console",
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.sidebar.button.side_effect = lambda label, *args, **kwargs: label == "Sign out"

        _sidebar(cfg)

    events = log.list_for_org(user.org_id)
    logout_events = [e for e in events if e["action"] == "logout"]
    assert len(logout_events) == 1
    assert logout_events[0]["username"] == user.username
    assert state.user is None
    assert state.session_token is None


def test_session_timeout_auditing_and_identity_binding(test_env) -> None:
    """Verify idle expiration and SEC-ISO-01 session identity binding."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    log = AuditLog(test_env["db_path"])

    # 1. Idle expiration
    mgr = SessionManager(idle_timeout_minutes=15)
    token = mgr.create_session(user.id, user.org_id)
    # Fast forward session's last activity by 25 minutes
    mgr._sessions[token].last_activity = datetime.now(timezone.utc) - timedelta(minutes=25)

    state = MockSessionState({
        "user": user,
        "session_token": token,
        "ingest": "data",
        "duckdb_console": "console",
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_session_manager", return_value=mgr):
        mock_st.session_state = state
        timed_out = _check_session_timeout(cfg)

    assert timed_out is True
    assert state.user is None

    events = log.list_for_org(user.org_id)
    timeout_events = [e for e in events if e["action"] == "session_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0]["username"] == user.username
    assert "expired" in timeout_events[0]["detail"].lower()

    # 2. SEC-ISO-01 Identity Mismatch Binding:
    # A valid token exists, but belongs to user_id=99 (mismatched with in-state user.id=2)
    valid_other_token = mgr.create_session(user_id=99, org_id="org1")
    state_mismatch = MockSessionState({
        "user": user,
        "session_token": valid_other_token,
    })
    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_session_manager", return_value=mgr):
        mock_st.session_state = state_mismatch
        rejected = _check_session_timeout(cfg)

    assert rejected is True
    assert state_mismatch.user is None
    assert state_mismatch.session_token is None


def test_table_ingest_auditing_and_deduplication(test_env) -> None:
    """Verify that table ingestion is audited with structured metadata, deduplicated, and audited on failure."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    log = AuditLog(test_env["db_path"])

    csv_data = b"product,revenue,units\nWidget,100,5\nGadget,200,10\n"
    uploaded_file = MockUploadedFile("monthly_sales.csv", csv_data)

    state = MockSessionState({"ingest": None, "_last_ingested_file": None})

    # 1. Ingest valid table -> logs Table: 'monthly_sales.csv', Rows: 2, Cols: 3
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.file_uploader.return_value = uploaded_file

        _import_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    ingest_events = [e for e in events if e["action"] == "ingest_table"]
    assert len(ingest_events) == 1
    assert ingest_events[0]["username"] == user.username
    assert "Table: 'monthly_sales'" in ingest_events[0]["detail"]
    assert "Rows: 2" in ingest_events[0]["detail"]
    assert state.ingest is not None

    # 2. Ingest same table again (Streamlit script rerun simulation) -> must be deduplicated
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.file_uploader.return_value = uploaded_file

        _import_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    ingest_events = [e for e in events if e["action"] == "ingest_table"]
    assert len(ingest_events) == 1  # Deduplicated

    # 3. Ingestion failure on unsupported/malformed file
    corrupt_file = MockUploadedFile("corrupt.unknown_ext", b"\x00\x01\x02\x03")
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.file_uploader.return_value = corrupt_file

        _import_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    fail_events = [e for e in events if e["action"] == "ingest_table_failure"]
    assert len(fail_events) == 1
    assert fail_events[0]["username"] == user.username
    assert "File: 'corrupt'" in fail_events[0]["detail"]
    assert "Reason: UnsupportedFormatError" in fail_events[0]["detail"]


def test_sql_console_auditing_no_raw_query(test_env) -> None:
    """Verify that SQL console records structured row count and status, NEVER raw SQL query strings."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    log = AuditLog(test_env["db_path"])

    csv_data = b"product,revenue,units\nWidget,100,5\nGadget,200,10\n"
    ingest = load_tabular(csv_data, "sales.csv", cfg.limits)

    col1 = MagicMock()
    col1.button.side_effect = lambda label, *args, **kwargs: label == "Run Query"
    col2 = MagicMock()

    # 1. Successful read-only query
    raw_query = "SELECT product, revenue FROM data WHERE revenue > 50"
    state = MockSessionState({"ingest": ingest, "duckdb_console": None})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.columns.return_value = [col1, col2]
        mock_st.selectbox.return_value = "Run custom SQL query"
        mock_st.text_area.return_value = raw_query

        _sql_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    sql_events = [e for e in events if e["action"] == "sql_query"]
    assert len(sql_events) == 1
    assert sql_events[0]["username"] == user.username
    assert "Rows: 2" in sql_events[0]["detail"]
    assert "Truncated: False" in sql_events[0]["detail"]
    # RAW SQL QUERY MUST NEVER APPEAR IN AUDIT LOG
    assert raw_query not in sql_events[0]["detail"]
    assert "revenue > 50" not in sql_events[0]["detail"]

    # 2. Rejected query (attempting table-valued function SEC-SQL-01)
    rejected_query = "SELECT * FROM data, duckdb_functions()"
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.columns.return_value = [col1, col2]
        mock_st.selectbox.return_value = "Run custom SQL query"
        mock_st.text_area.return_value = rejected_query

        _sql_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    fail_events = [e for e in events if e["action"] == "sql_query_failure"]
    assert len(fail_events) == 1
    assert fail_events[0]["username"] == user.username
    assert "Status: rejected" in fail_events[0]["detail"]
    assert "Reason: ValidationError" in fail_events[0]["detail"]
    # REJECTED QUERY TEXT MUST NEVER APPEAR IN AUDIT LOG
    assert rejected_query not in fail_events[0]["detail"]


def test_sql_console_unauthorized_user_blocked(test_env) -> None:
    """Verify that a viewer without analyze permission cannot access SQL console."""
    cfg = test_env["cfg"]
    user = test_env["viewer"]
    # Temporarily remove analyze permission from viewer for testing unauthorized boundary
    with patch("private_analyst.ui.streamlit_app.can", return_value=False):
        with patch("private_analyst.ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MockSessionState({"ingest": None})
            _sql_tab(cfg, user)
            mock_st.error.assert_called_with("You do not have permission to execute analytical queries.")


def test_ml_workflows_auditing_structured_metadata(test_env) -> None:
    """Verify that all 5 ML workflows record allowlisted structured metadata without customer data."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    log = AuditLog(test_env["db_path"])

    df = pd.DataFrame({
        "salary_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "confidential_kpi": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target_cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })
    ingest = IngestResult(df, "confidential_ml.csv", [], "csv", 10, 3)

    # 1. Classification
    state = MockSessionState({"ingest": ingest})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.columns.side_effect = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
        mock_st.selectbox.side_effect = lambda label, *args, **kwargs: (
            "Classification (Decision Tree)" if "Analytical Task" in label
            else "target_cat"
        )
        mock_st.multiselect.return_value = ["salary_col", "confidential_kpi"]
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Train & Evaluate Classifier"

        _ml_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    assert any(e["action"] == "ml_classification" and "Features: 2" in e["detail"] for e in events)

    # 2. Regression
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.columns.side_effect = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
        mock_st.selectbox.side_effect = lambda label, *args, **kwargs: (
            "Linear Regression" if "Analytical Task" in label
            else "confidential_kpi"
        )
        mock_st.multiselect.return_value = ["salary_col"]
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Fit & Evaluate Regression"

        _ml_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    assert any(e["action"] == "ml_regression" and "Features: 1" in e["detail"] for e in events)

    # 3. Clustering
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.columns.side_effect = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
        mock_st.selectbox.return_value = "Clustering (K-Means)"
        mock_st.multiselect.return_value = ["salary_col", "confidential_kpi"]
        mock_st.slider.return_value = 2
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Run Clustering"

        _ml_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    assert any(e["action"] == "ml_clustering" and "k: 2" in e["detail"] for e in events)

    # 4. Anomaly Detection
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.columns.side_effect = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
        mock_st.selectbox.side_effect = lambda label, *args, **kwargs: (
            "Anomaly Detection (IQR Outliers)" if "Analytical Task" in label
            else "salary_col"
        )
        mock_st.slider.return_value = 1.5
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Detect Outliers"

        _ml_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    assert any(e["action"] == "anomaly_detection" and "Multiplier: 1.5" in e["detail"] for e in events)

    # 5. Linear Trend Forecasting
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.columns.side_effect = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
        mock_st.selectbox.side_effect = lambda label, *args, **kwargs: (
            "Linear Trend Forecasting" if "Analytical Task" in label
            else "salary_col"
        )
        mock_st.slider.return_value = 3
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Calculate Forecast"

        _ml_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    assert any(e["action"] == "trend_forecast" and "Horizon: 3" in e["detail"] for e in events)


def test_ask_tab_analysis_auditing_no_raw_prompt(test_env) -> None:
    """Verify that running analytical question operations records run_analysis without raw user prompts."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    log = AuditLog(test_env["db_path"])

    csv_data = b"product,revenue,units\nWidget,100,5\nGadget,200,10\n"
    ingest = load_tabular(csv_data, "sales.csv", cfg.limits)

    state = MockSessionState({
        "ingest": ingest,
        "last_result": None,
        "last_explanation": None,
    })

    confidential_prompt = "Find confidential revenue for secret client ACME?"
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.text_input.return_value = confidential_prompt
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Run analysis"

        _ask_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    analysis_events = [e for e in events if e["action"] == "run_analysis"]
    assert len(analysis_events) == 1
    assert analysis_events[0]["username"] == user.username
    assert "Operation: summarize" in analysis_events[0]["detail"]
    assert "Status: ok" in analysis_events[0]["detail"]
    # CONFIDENTIAL USER PROMPT MUST NOT BE LOGGED
    assert confidential_prompt not in analysis_events[0]["detail"]
    assert "secret client" not in analysis_events[0]["detail"]


def test_documents_tab_auditing_index_and_delete(test_env) -> None:
    """Verify document indexing and deletion record safe metadata and sanitized paths."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    log = AuditLog(test_env["db_path"])

    index = LocalTextIndex()
    state = MockSessionState({"index": index})

    # 1. Index document with potential path traversal in name
    txt_file = MockUploadedFile("../../etc/company_policy.txt", b"Local text indexing for confidential data.")
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.file_uploader.return_value = txt_file
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Index document"
        mock_st.text_input.return_value = ""

        _documents_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    idx_events = [e for e in events if e["action"] == "index_document"]
    assert len(idx_events) == 1
    # Only basename must be logged
    assert idx_events[0]["detail"] == "File: 'company_policy.txt', Passages: 1"

    # 2. Delete document
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.selectbox.return_value = "company_policy.txt"
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Delete"

        _documents_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    del_events = [e for e in events if e["action"] == "delete_document"]
    assert len(del_events) == 1
    assert del_events[0]["detail"] == "File: 'company_policy.txt'"


def test_reports_tab_export_requested_auditing(test_env) -> None:
    """Verify export buttons trigger accurately named export_requested events with format metadata."""
    cfg = test_env["cfg"]
    user = test_env["admin"]
    log = AuditLog(test_env["db_path"])

    csv_data = b"product,revenue,units\nWidget,100,5\nGadget,200,10\n"
    ingest = load_tabular(csv_data, "sales.csv", cfg.limits)

    mock_result = MagicMock(
        intent="summarize",
        status="success",
        observed={"source": "sales.csv"},
        calculated={"revenue_sum": 300},
    )
    state = MockSessionState({
        "ingest": ingest,
        "last_result": mock_result,
        "last_explanation": {"text": "Total revenue is 300."},
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        _reports_tab(cfg, user)

        # Trigger all registered download_button callbacks
        for call in mock_st.download_button.call_args_list:
            on_click = call.kwargs.get("on_click")
            if on_click:
                on_click()

    events = log.list_for_org(user.org_id)
    export_events = [e for e in events if e["action"] == "export_requested"]

    # Must accurately record 'export_requested' (not completed download)
    assert len(export_events) == 4
    details = [e["detail"] for e in export_events]
    assert any("Format: delimited" in d for d in details)
    assert any("Format: xlsx" in d for d in details)
    assert any("Format: md" in d for d in details)
    assert any("Format: pdf" in d for d in details)


def test_settings_tab_admin_actions_auditing(test_env) -> None:
    """Verify that administrative actions in Settings record structured audit events."""
    cfg = test_env["cfg"]
    user = test_env["admin"]
    log = AuditLog(test_env["db_path"])

    # 1. Create user
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = MockSessionState({"user": user})
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "New user organization id": "org1",
            "New username": "new_analyst_01",
            "New password": "SecurePassword123!",
        }.get(label, "")
        mock_st.selectbox.return_value = "analyst"
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Create user"

        _settings_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    create_events = [e for e in events if e["action"] == "create_user"]
    assert len(create_events) == 1
    assert "new_analyst_01" in create_events[0]["detail"]
    assert "SecurePassword123!" not in create_events[0]["detail"]

    # 2. Duplicate user creation failure
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = MockSessionState({"user": user})
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "New user organization id": "org1",
            "New username": "new_analyst_01",
            "New password": "SecurePassword123!",
        }.get(label, "")
        mock_st.selectbox.return_value = "analyst"
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Create user"

        _settings_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    create_fail_events = [e for e in events if e["action"] == "create_user_failure"]
    assert len(create_fail_events) == 1
    assert "new_analyst_01" in create_fail_events[0]["detail"]
    assert "Reason: IntegrityError" in create_fail_events[0]["detail"] or "IntegrityError" in create_fail_events[0]["detail"]

    # 3. Initialize master key store
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = MockSessionState({"user": user})
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "Master password to initialize key store (min 12 characters)": "SuperSecureMasterKey123!",
        }.get(label, "")
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Initialize Master Key Store"

        _settings_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    keystore_events = [e for e in events if e["action"] == "init_keystore"]
    assert len(keystore_events) == 1

    # 4. Create encrypted backup
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = MockSessionState({"user": user})
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "Backup encryption password (leave blank for standard zip)": "BackupEncryptionPassword123!",
        }.get(label, "")
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Generate Backup"

        _settings_tab(cfg, user)

    events = log.list_for_org(user.org_id)
    backup_events = [e for e in events if e["action"] == "create_backup"]
    assert len(backup_events) == 1
    assert "Archive:" in backup_events[0]["detail"]


def test_audit_attribution_spoofing_prevention(test_env) -> None:
    """Verify that audit records strictly take org_id and username from authenticated session context."""
    cfg = test_env["cfg"]
    authenticated_user = test_env["analyst"]  # org1, bob
    log = AuditLog(test_env["db_path"])

    csv_data = b"product,revenue,units\nWidget,100,5\n"
    uploaded_file = MockUploadedFile("monthly_sales.csv", csv_data)
    state = MockSessionState({"ingest": None, "_last_ingested_file": None})

    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        mock_st.file_uploader.return_value = uploaded_file

        _import_tab(cfg, authenticated_user)

    # Must be attributed to authenticated_user's org and username
    org1_events = log.list_for_org("org1")
    assert any(e["action"] == "ingest_table" and e["username"] == "bob" for e in org1_events)

    # Must NOT exist under other organizations
    org2_events = log.list_for_org("org2")
    assert len(org2_events) == 0


# ---------------------------------------------------------------------------
# Stage 2.1 Audit Resilience & Error Decoupling Tests
# ---------------------------------------------------------------------------


def test_logout_cleanup_on_audit_failure(test_env) -> None:
    """Fix 1: Session state must be purged during logout even if audit logging raises AuditLogError."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    mgr = SessionManager()
    token = mgr.create_session(user.id, user.org_id)

    mock_ingest = MagicMock()
    mock_console = MagicMock()
    state = MockSessionState({
        "user": user,
        "session_token": token,
        "ingest": mock_ingest,
        "duckdb_console": mock_console,
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_session_manager", return_value=mgr), \
         patch("private_analyst.ui.streamlit_app._get_audit_log") as mock_get_log:
        mock_st.session_state = state
        mock_st.sidebar.button.return_value = True

        mock_log = MagicMock()
        mock_log.record.side_effect = AuditLogError("Audit DB disk full")
        mock_get_log.return_value = mock_log

        _sidebar(cfg)

        # Verification: Session state was wiped in finally block despite audit write exception
        assert state["user"] is None
        assert state["session_token"] is None
        assert state["ingest"] is None
        assert state["duckdb_console"] is None
        assert mock_st.rerun.called
        assert mgr.validate_session(token) is None


def test_timeout_cleanup_on_audit_failure(test_env) -> None:
    """Fix 2: Expired session state must be wiped even if recording the timeout event fails."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]
    mgr = SessionManager(idle_timeout_minutes=15)
    token = mgr.create_session(user.id, user.org_id)

    # Force expiration by simulating invalid session
    state = MockSessionState({
        "user": user,
        "session_token": token,
        "ingest": MagicMock(),
        "duckdb_console": MagicMock(),
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_session_manager") as mock_get_mgr, \
         patch("private_analyst.ui.streamlit_app._get_audit_log") as mock_get_log:
        mock_st.session_state = state

        # Mock session manager indicating session is expired / invalid
        expired_mgr = MagicMock()
        expired_mgr.validate_session.return_value = None
        mock_get_mgr.return_value = expired_mgr

        # Mock audit log failing with AuditLogError
        mock_log = MagicMock()
        mock_log.record.side_effect = AuditLogError("Audit DB locked")
        mock_get_log.return_value = mock_log

        timed_out = _check_session_timeout(cfg)

        assert timed_out is True
        assert state["user"] is None
        assert state["session_token"] is None
        assert state["ingest"] is None
        assert state["duckdb_console"] is None
        expired_mgr.destroy_session.assert_called_once_with(token)
        mock_st.warning.assert_called_once()
        assert "Session expired" in mock_st.warning.call_args[0][0]


def test_sql_success_not_mislabeled_on_audit_failure(test_env) -> None:
    """Fix 3: Successful query completion must not be mislabeled as sql_query_failure if audit logging fails."""
    cfg = test_env["cfg"]
    user = test_env["analyst"]

    csv_data = b"col\n1\n2\n3\n"
    ingest = load_tabular(csv_data, "data.csv", cfg.limits)
    state = MockSessionState({
        "ingest": ingest,
        "duckdb_console": None,
    })

    mock_console = MagicMock()
    mock_console.list_tables.return_value = ["data"]
    mock_console.describe_table.return_value = [{"column_name": "col", "column_type": "BIGINT"}]
    mock_console.execute_query.return_value = QueryResult(
        columns=["col"],
        rows=[{"col": 1}, {"col": 2}, {"col": 3}],
        row_count=3,
        truncated=False,
        sql="SELECT * FROM data",
    )

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app.DuckDBConsole", return_value=mock_console), \
         patch("private_analyst.ui.streamlit_app._get_audit_log") as mock_get_log:
        mock_st.session_state = state
        col1 = MagicMock()
        col1.button.return_value = True
        mock_st.columns.return_value = [col1, MagicMock()]
        mock_st.text_area.return_value = "SELECT * FROM data"

        # Audit logger fails
        mock_log = MagicMock()
        mock_log.record.side_effect = AuditLogError("Audit database unavailable")
        mock_get_log.return_value = mock_log

        _sql_tab(cfg, user)

        # Verification:
        # 1. Query was executed successfully
        mock_console.execute_query.assert_called_once()
        # 2. Must NOT call st.error with failure
        mock_st.error.assert_not_called()
        # 3. Must warn that query succeeded but audit event could not be recorded
        mock_st.warning.assert_called_once()
        assert "Query returned 3 row(s), but the audit event could not be recorded" in mock_st.warning.call_args[0][0]
        # 4. Results dataframe was still presented to user
        mock_st.dataframe.assert_called()
        # 5. Must NOT record sql_query_failure
        for call in mock_log.record.call_args_list:
            assert call[0][2] != "sql_query_failure"


def test_user_creation_represented_on_audit_failure(test_env) -> None:
    """Fix 4: Successful user creation must not be reported as failure if subsequent audit logging fails."""
    cfg = test_env["cfg"]
    admin_user = test_env["admin"]
    store = test_env["store"]

    state = MockSessionState({"user": admin_user})

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_audit_log") as mock_get_log:
        mock_st.session_state = state
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "New user organization id": "org1",
            "New username": "resilient_user",
            "New password": "ValidPassword123!",
        }.get(label, "")
        mock_st.selectbox.return_value = "analyst"
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Create user"

        # Mock audit log failure
        mock_log = MagicMock()
        mock_log.record.side_effect = AuditLogError("Audit storage write failure")
        mock_get_log.return_value = mock_log

        _settings_tab(cfg, admin_user)

        # Verification:
        # 1. Did not report that user creation failed
        mock_st.error.assert_not_called()
        # 2. Displayed warning that user was created but audit event was not recorded
        mock_st.warning.assert_called_once()
        assert "User created, but the audit event could not be recorded" in mock_st.warning.call_args[0][0]
        # 3. User was indeed created in the database and can authenticate
        created_user = store.authenticate("org1", "resilient_user", "ValidPassword123!")
        assert created_user.username == "resilient_user"
        assert created_user.role == "analyst"


def test_export_callbacks_handle_audit_failure(test_env) -> None:
    """Fix 5: Export callbacks must not propagate unhandled exceptions if audit recording fails."""
    cfg = test_env["cfg"]
    user = test_env["admin"]

    csv_data = b"product,revenue,units\nWidget,100,5\n"
    ingest = load_tabular(csv_data, "sales.csv", cfg.limits)
    mock_result = MagicMock(
        intent="summarize",
        status="success",
        observed={"source": "sales.csv"},
        calculated={"revenue_sum": 100},
    )
    state = MockSessionState({
        "ingest": ingest,
        "last_result": mock_result,
        "last_explanation": {"text": "Explanation narrative."},
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_audit_log") as mock_get_log:
        mock_st.session_state = state

        # Mock audit logger failing
        mock_log = MagicMock()
        mock_log.record.side_effect = AuditLogError("Audit log table locked")
        mock_get_log.return_value = mock_log

        _reports_tab(cfg, user)

        # Trigger all download_button on_click callbacks; NONE must raise
        for call in mock_st.download_button.call_args_list:
            on_click = call.kwargs.get("on_click")
            if on_click:
                # Should cleanly catch and log warning without raising
                on_click()

    # Also test _safe_record_export_requested directly with unhandled Exception
    with patch("private_analyst.ui.streamlit_app._get_audit_log") as mock_get_log:
        mock_log = MagicMock()
        mock_log.record.side_effect = RuntimeError("Unexpected OS crash")
        mock_get_log.return_value = mock_log

        # Direct call must not raise
        _safe_record_export_requested(cfg, user, "test_target", "test_fmt", "extra info")


def test_genuine_sql_and_duplicate_user_failures_recorded(test_env) -> None:
    """Fix 6: Genuine SQL failures and duplicate-user failures still follow their correct error paths."""
    cfg = test_env["cfg"]
    user = test_env["admin"]
    log = AuditLog(test_env["db_path"])

    # 1. Genuine SQL query validation failure
    csv_data = b"col\n1\n2\n3\n"
    ingest = load_tabular(csv_data, "data.csv", cfg.limits)
    state = MockSessionState({
        "ingest": ingest,
        "duckdb_console": None,
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state
        col1 = MagicMock()
        col1.button.return_value = True
        mock_st.columns.return_value = [col1, MagicMock()]
        mock_st.text_area.return_value = "SELECT * FROM non_existent_table"

        _sql_tab(cfg, user)

        mock_st.error.assert_called_once()
        events = log.list_for_org(user.org_id)
        sql_fail = [e for e in events if e["action"] == "sql_query_failure"]
        assert len(sql_fail) == 1
        assert "Status: rejected" in sql_fail[0]["detail"]
        assert "non_existent_table" not in sql_fail[0]["detail"]

    # 2. Duplicate user creation failure
    state2 = MockSessionState({"user": user})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state2
        mock_st.text_input.side_effect = lambda label, *args, **kwargs: {
            "New user organization id": "org1",
            "New username": "alice",  # alice already exists in org1
            "New password": "AnotherPassword123!",
        }.get(label, "")
        mock_st.selectbox.return_value = "analyst"
        mock_st.button.side_effect = lambda label, *args, **kwargs: label == "Create user"

        _settings_tab(cfg, user)

        mock_st.error.assert_called_once()
        events = log.list_for_org(user.org_id)
        create_fail = [e for e in events if e["action"] == "create_user_failure"]
        assert len(create_fail) >= 1
        assert "alice" in create_fail[0]["detail"]
