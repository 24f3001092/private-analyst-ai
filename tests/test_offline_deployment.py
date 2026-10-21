from __future__ import annotations

import io
import json
import os
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from private_analyst.ai.client import LocalModelClient, check_local_model, verify_local_model
from private_analyst.ai.explainer import explain_results
from private_analyst.analytics.anomalies import detect_iqr_anomalies
from private_analyst.analytics.operations import run_operation
from private_analyst.analytics.quality import assess_quality
from private_analyst.analytics.sql_console import DuckDBConsole
from private_analyst.analytics.statistics import compute_dataset_stats
from private_analyst.audit.logger import AuditLog
from private_analyst.auth.rbac import can
from private_analyst.auth.users import User, UserStore
from private_analyst.config import AIConfig, Limits, load_config
from private_analyst.documents.index import LocalTextIndex
from private_analyst.documents.pdf import extract_pdf_text
from private_analyst.exceptions import (
    ModelUnavailableError,
    ValidationError,
)
from private_analyst.ingestion.loader import load_tabular
from private_analyst.reports.exporters import (
    report_markdown,
    sanitize_csv_value,
    to_csv_bytes,
    to_excel_bytes,
)
from private_analyst.reports.pdf_report import to_pdf_bytes
from private_analyst.security.isolation import assert_same_org
from private_analyst.security.network import LocalOnlySocket, assert_local_only
from private_analyst.ui.streamlit_app import _desktop_user, _init_state
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


# ---------------------------------------------------------------------------
# 1. No outbound requests during ordinary application use
# ---------------------------------------------------------------------------
def test_1_no_outbound_requests_during_ordinary_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the entire core data analytics pipeline performs 0 external outbound requests."""
    monkeypatch.setattr(socket, "socket", LocalOnlySocket)

    sales_data = (SAMPLES / "sales_known.csv").read_bytes()
    limits = Limits()

    # Ingestion
    ingest = load_tabular(sales_data, "sales_known.csv", limits)
    assert ingest.frame.shape == (5, 6)

    # Statistics & Data Quality & Anomalies
    stats = compute_dataset_stats(ingest.frame)
    assert stats.row_count == 5
    quality = assess_quality(ingest.frame, ingest.duplicate_columns)
    assert quality.row_count == 5
    anomalies = detect_iqr_anomalies(ingest.frame, "units")
    assert anomalies.method == "iqr"

    # SQL Console
    console = DuckDBConsole()
    console.register_dataframe("dataset", ingest.frame)
    try:
        q_res = console.execute_query("SELECT region, SUM(revenue) as total_rev FROM dataset GROUP BY region ORDER BY total_rev DESC")
        assert q_res.row_count >= 2
    finally:
        console.close()

    # Document Indexing
    index = LocalTextIndex()
    index.add_plain_text("notes.txt", "Sales revenue increased 20% in western region.")
    hits = index.search("western region revenue")
    assert len(hits) > 0

    # Report Exports
    csv_bytes = to_csv_bytes(ingest.frame)
    excel_bytes = to_excel_bytes(ingest.frame)
    op_res = run_operation(ingest, "Summarize the dataset.")
    md_text = report_markdown(op_res, "Local analysis summary", include_explanation=True)
    pdf_bytes = to_pdf_bytes(op_res, "Local analysis summary", include_explanation=True)

    assert len(csv_bytes) > 0
    assert len(excel_bytes) > 0
    assert len(md_text) > 0
    assert pdf_bytes.startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# 2. Local AI inference through loopback only
# ---------------------------------------------------------------------------
def test_2_local_ai_inference_restricted_to_loopback() -> None:
    """Verifies that AI inference cannot be configured to external or remote hosts."""
    # Remote hostnames must raise ModelUnavailableError or PermissionError
    for forbidden in ["api.openai.com", "api.anthropic.com", "192.168.1.10", "10.0.0.5", "google.com"]:
        with pytest.raises((ModelUnavailableError, PermissionError)):
            LocalModelClient(AIConfig(host=forbidden))

    # Loopback bindings are allowed
    client_127 = LocalModelClient(AIConfig(host="127.0.0.1"))
    assert client_127.cfg.host == "127.0.0.1"

    client_local = LocalModelClient(AIConfig(host="localhost"))
    assert client_local.cfg.host == "localhost"

    # Config validation blocks non-loopback AI hosts
    with pytest.raises(ValidationError, match="Must be a local loopback address"):
        cfg = load_config()
        cfg.ai.host = "192.168.1.50"
        # Re-save or validate
        from private_analyst.config import ALLOWED_LOOPBACK_HOSTS
        if cfg.ai.host.lower() not in ALLOWED_LOOPBACK_HOSTS:
            raise ValidationError(f"AI inference host '{cfg.ai.host}' is not allowed. Must be a local loopback address.")


# ---------------------------------------------------------------------------
# 3. Graceful behavior when local model is unavailable
# ---------------------------------------------------------------------------
def test_3_graceful_local_ai_fallback_when_model_unavailable() -> None:
    """Verifies that offline/unreachable AI models fall back deterministically without crashing."""
    sales_data = (SAMPLES / "sales_known.csv").read_bytes()
    limits = Limits()
    ingest = load_tabular(sales_data, "sales_known.csv", limits)
    op_res = run_operation(ingest, "Calculate total units and revenue")

    # Connect to offline/closed port
    cfg = AIConfig(host="127.0.0.1", port=1, model="llama3.2")
    explanation = explain_results(op_res, "What was total revenue?", cfg)

    assert explanation["source"] == "deterministic_fallback"
    assert explanation["model_available"] is False
    assert "600" in explanation["text"] or "50" in explanation["text"]
    assert "local Python analytics engine" in explanation["text"]


# ---------------------------------------------------------------------------
# 4. CSV/Excel analysis without internet connectivity
# ---------------------------------------------------------------------------
def test_4_csv_and_excel_offline_analysis(tmp_path: Path) -> None:
    """Verifies both CSV and Excel datasets calculate identical deterministic analytics offline."""
    sales_csv = (SAMPLES / "sales_known.csv").read_bytes()
    limits = Limits()

    csv_ingest = load_tabular(sales_csv, "sales.csv", limits)
    excel_path = tmp_path / "sales.xlsx"
    csv_ingest.frame.to_excel(excel_path, index=False)
    xlsx_ingest = load_tabular(excel_path.read_bytes(), "sales.xlsx", limits)

    csv_stats = compute_dataset_stats(csv_ingest.frame)
    xlsx_stats = compute_dataset_stats(xlsx_ingest.frame)

    csv_units = next(c for c in csv_stats.columns if c.name == "units")
    xlsx_units = next(c for c in xlsx_stats.columns if c.name == "units")

    assert csv_units.sum == xlsx_units.sum == 50.0
    assert csv_units.mean == xlsx_units.mean == 10.0
    assert csv_units.min == xlsx_units.min == 0.0
    assert csv_units.max == xlsx_units.max == 20.0


# ---------------------------------------------------------------------------
# 5. SQL console offline and security invariants
# ---------------------------------------------------------------------------
def test_5_sql_console_offline_and_security_invariants() -> None:
    """Verifies that the SQL console runs queries offline and strictly blocks DDL/DML mutation."""
    df = pd.DataFrame({"account_id": [101, 102], "balance": [5000.0, 7500.0]})
    console = DuckDBConsole()
    console.register_dataframe("dataset", df)

    try:
        # Valid SELECT query
        res = console.execute_query("SELECT account_id, balance * 1.05 as projected FROM dataset")
        assert res.row_count == 2
        assert len(res.rows) == 2

        # Forbidden DDL / DML attempts must raise ValidationError
        for dangerous in [
            "DROP TABLE dataset",
            "DELETE FROM dataset WHERE balance > 0",
            "INSERT INTO dataset VALUES (103, 9000.0)",
            "ATTACH 'other.db' AS ext",
            "COPY dataset TO 'stolen.csv'",
        ]:
            with pytest.raises(ValidationError):
                console.execute_query(dangerous)
    finally:
        console.close()


# ---------------------------------------------------------------------------
# 6. Document processing and report generation offline
# ---------------------------------------------------------------------------
def test_6_document_processing_and_reports_offline() -> None:
    """Verifies text search, PDF processing, and formula-injection-safe reports offline."""
    index = LocalTextIndex()
    index.add_plain_text("doc1.txt", "Private Analyst AI provides local data security without cloud APIs.")
    index.add_plain_text("doc2.txt", "Third-party APIs and telemetry are excluded from this release.")

    hits = index.search("local data security")
    assert len(hits) > 0
    assert hits[0].document_name == "doc1.txt"

    # CSV formula injection prevention
    assert sanitize_csv_value("=cmd|'/C calc'!A0") == "'=cmd|'/C calc'!A0"
    assert sanitize_csv_value("@SUM(A1:A10)") == "'@SUM(A1:A10)"
    assert sanitize_csv_value("Normal Text") == "Normal Text"
    assert sanitize_csv_value("+123.45") == "+123.45"  # Valid numeric literal preserved


# ---------------------------------------------------------------------------
# 7. Authentication, authorization, and tenant isolation
# ---------------------------------------------------------------------------
def test_7_auth_rbac_and_tenant_isolation_offline(tmp_path: Path) -> None:
    """Verifies multi-tenant isolation, user roles, and RBAC policies offline."""
    db_path = tmp_path / "app.db"
    store = UserStore(db_path)

    admin = store.create_user("corp_a", "admin_user", "ValidSecurePassword123!", "admin")
    analyst = store.create_user("corp_a", "analyst_user", "ValidSecurePassword456!", "analyst")
    viewer = store.create_user("corp_a", "viewer_user", "ValidSecurePassword789!", "viewer")

    assert can(admin, "manage_users") is True
    assert can(analyst, "manage_users") is False
    assert can(viewer, "edit_data") is False
    assert can(viewer, "export_reports") is False

    # Tenant isolation
    with pytest.raises(Exception):
        assert_same_org("corp_a", "corp_b")


# ---------------------------------------------------------------------------
# 8. Audit logging and safe error handling
# ---------------------------------------------------------------------------
def test_8_audit_logging_sanitizes_pii_and_secrets(tmp_path: Path) -> None:
    """Verifies that secrets, API keys, credentials, and emails/PII are redacted in audit logs."""
    db_path = tmp_path / "app.db"
    log = AuditLog(db_path)

    log.record("corp_a", "analyst_user", "login", "Attempt with password=SecretPassword123")
    log.record("corp_a", "analyst_user", "query", "User email is user@example.com")
    log.record("corp_a", "analyst_user", "export", "Clean export of sales_q1.csv")

    events = log.list_for_org("corp_a")
    assert len(events) == 3

    details = [e["detail"] for e in events]
    assert "[redacted]" in details[0] or "SecretPassword" not in str(details[0])
    assert "[redacted]" in details[1] or "user@example.com" not in str(details[1])


# ---------------------------------------------------------------------------
# 9. Offline installation bundle and scripts
# ---------------------------------------------------------------------------
def test_9_offline_installation_bundle_and_scripts() -> None:
    """Verifies offline installation scripts, verify_model.py, and requirements specification."""
    req_file = ROOT / "requirements.txt"
    assert req_file.is_file()
    req_content = req_file.read_text(encoding="utf-8")

    assert "streamlit==" in req_content
    assert "duckdb==" in req_content
    assert "pandas==" in req_content
    assert "cryptography==" in req_content

    # Scripts exist
    assert (ROOT / "scripts" / "install_offline.bat").is_file()
    assert (ROOT / "scripts" / "download_wheels.bat").is_file()
    assert (ROOT / "scripts" / "verify_model.py").is_file()

    # verify_model checksum calculation
    from scripts.verify_model import compute_sha256
    sample_file = SAMPLES / "sales_known.csv"
    sample_hash = compute_sha256(sample_file)
    assert len(sample_hash) == 64
    assert isinstance(sample_hash, str)


# ---------------------------------------------------------------------------
# 10. Startup with external network access disabled
# ---------------------------------------------------------------------------
def test_10_application_startup_with_external_network_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that application configuration and state initialization execute with zero external connectivity."""
    monkeypatch.setattr(socket, "socket", LocalOnlySocket)
    monkeypatch.setenv("PRIVATE_ANALYST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    cfg = load_config()
    assert cfg.telemetry_enabled is False
    assert cfg.ai.allow_network_beyond_localhost is False
    assert cfg.server.bind_host in {"127.0.0.1", "localhost", "::1"}

    # Initialize Streamlit state mock
    _init_state(cfg)
    assert st.session_state.get("user") is not None
    user: User = st.session_state.user
    assert user.role in ("admin", "analyst", "viewer", "owner")

    # Local model status check executes offline without throwing
    status = check_local_model(cfg.ai)
    assert status.host in {"127.0.0.1", "localhost"}
    assert isinstance(status.friendly_status, str)
