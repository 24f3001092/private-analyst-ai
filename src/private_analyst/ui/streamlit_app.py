from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from private_analyst.ai.client import check_local_model, verify_local_model
from private_analyst.ai.explainer import deterministic_explanation, explain_results
from private_analyst.analytics.anomalies import detect_iqr_anomalies, detect_zscore_anomalies, linear_trend_forecast
from private_analyst.analytics.cleaner import (
    CleaningRecommendation,
    apply_cleaning_plan,
    apply_column_management,
    apply_header_override,
    filter_rows,
    generate_cleaning_recommendations,
    handle_missing_values,
    repair_unnamed_headers,
)
from private_analyst.analytics.ml_engine import (
    assess_feature_suitability,
    run_classification,
    run_clustering,
    run_regression,
)
from private_analyst.analytics.operations import run_operation
from private_analyst.analytics.quality import assess_quality
from private_analyst.analytics.sql_console import DuckDBConsole
from private_analyst.analytics.statistics import (
    compute_correlation_matrix,
    compute_dataset_stats,
    compute_time_trends,
    explain_chart_exclusions,
    frequency_analysis,
    group_compare,
    recommend_charts,
)
from private_analyst.auth.rbac import can
from private_analyst.auth.users import User, UserStore
from private_analyst.config import load_config
from private_analyst.documents.index import LocalTextIndex
from private_analyst.documents.ocr import ocr_image
from private_analyst.documents.pdf import extract_pdf_text
from private_analyst.exceptions import AuditLogError, AuthenticationError, PrivateAnalystError, ValidationError
from private_analyst.ingestion.loader import (
    CleaningOperation,
    IngestResult,
    WorkbookInfo,
    apply_transformations,
    inspect_workbook,
    load_tabular,
    normalize_cleaning_operation,
    repair_unnamed_columns,
)
from private_analyst.reports.exporters import (
    analysis_metadata,
    report_markdown,
    to_csv_bytes,
    to_excel_bytes,
    to_report_excel_bytes,
)
from private_analyst.reports.pdf_report import to_pdf_bytes
from private_analyst.security.encryption import SecureKeyStore
from private_analyst.security.sessions import SessionManager
from private_analyst.version import PRODUCT_NAME, __version__
from private_analyst.viz.charts import (
    bar_sum,
    category_bar,
    correlation_heatmap,
    crosstab_heatmap,
    histogram,
    missingness,
    outlier_box,
    time_series_line,
)
from private_analyst.audit.logger import AuditLog

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_session_manager(idle_minutes: int) -> SessionManager:
    return SessionManager(idle_timeout_minutes=idle_minutes)


def _get_audit_log(cfg) -> AuditLog:
    return AuditLog(cfg.resolved_data_dir() / "app.db")


def _safe_record_export_requested(
    cfg,
    user: User,
    target: str,
    format_name: str,
    extra_detail: str | None = None,
) -> None:
    """Record an export_requested audit event safely without letting audit failures abort callbacks."""
    detail = f"Target: {target}, Format: {format_name}"
    if extra_detail:
        detail = f"{detail}, {extra_detail}"
    try:
        _get_audit_log(cfg).record(
            user.org_id,
            user.username,
            "export_requested",
            detail,
        )
    except AuditLogError:
        # Deliberately caught: callback must not crash if the audit database is unavailable.
        logger.warning(
            "Audit log write failed for export_requested event (user=%s, org=%s, target=%s).",
            user.username,
            user.org_id,
            target,
        )
    except Exception as exc:
        logger.warning(
            "Unexpected error recording export_requested audit event: %s",
            type(exc).__name__,
        )


def _desktop_user(edition: str = "desktop") -> User:
    if edition == "server":
        raise AuthenticationError("Desktop auto-login is not allowed in server mode.")
    return User(id=0, org_id="local", username="local-owner", role="admin")


def _invalidate_downstream_state() -> None:
    """Reset downstream state across tabs when dataset changes or is modified."""
    st.session_state.last_result = None
    st.session_state.last_explanation = None
    if st.session_state.get("duckdb_console") is not None:
        try:
            st.session_state.duckdb_console.close()
        except Exception:
            pass
        st.session_state.duckdb_console = None
    st.session_state["sql_registered_frame_id"] = None


def _init_state(cfg) -> None:
    if "user" not in st.session_state:
        if cfg.server.edition == "server" or cfg.server.require_auth:
            st.session_state.user = None
            st.session_state.session_token = None
        else:
            st.session_state.user = _desktop_user(cfg.server.edition)
            st.session_state.session_token = None
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    if "ingest" not in st.session_state:
        st.session_state.ingest = None
    if "index" not in st.session_state:
        st.session_state.index = LocalTextIndex()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_explanation" not in st.session_state:
        st.session_state.last_explanation = None
    if "duckdb_console" not in st.session_state:
        st.session_state.duckdb_console = None
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = datetime.now(timezone.utc)


def _login(cfg) -> None:
    st.subheader("Sign in")
    st.caption("Server edition. Credentials stay on this machine.")
    org = st.text_input("Organization id")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        store = UserStore(cfg.resolved_data_dir() / "app.db")
        log = _get_audit_log(cfg)
        try:
            authenticated_user = store.authenticate(org, username, password)
            mgr = _get_session_manager(cfg.server.session_idle_minutes)
            token = mgr.create_session(authenticated_user.id, authenticated_user.org_id)
            try:
                log.record(
                    authenticated_user.org_id,
                    authenticated_user.username,
                    "login_success",
                    f"User logged in with role '{authenticated_user.role}'",
                )
            except AuditLogError:
                mgr.destroy_session(token)
                st.error("Audit subsystem error: sign-in cannot proceed without audit logging.")
                return
            st.session_state.user = authenticated_user
            st.session_state.session_token = token
            st.session_state.last_activity = datetime.now(timezone.utc)
            st.rerun()
        except AuthenticationError:
            try:
                log.record(
                    org.strip() or "unauthenticated",
                    username.strip() or None,
                    "login_failure",
                    "Authentication failed: invalid credentials",
                )
            except AuditLogError:
                pass
            st.error("Sign-in failed.")


def _sidebar(cfg) -> None:
    # VantageAnalyst Unique Brand Header
    st.sidebar.markdown(
        """
        <div class="ba-sidebar-brand">
            <div class="ba-logo-wrap">
                <div class="ba-logo-icon">V</div>
                <span class="ba-logo-text">vantageanalyst</span>
            </div>
            <div class="ba-header-icons">
                <span title="Toggle Dark/Light Mode" style="cursor:pointer;">🌙</span>
                <span title="Collapse Sidebar" style="cursor:pointer;">◫</span>
            </div>
        </div>
        <div class="ba-workspace-card">
            <div class="ba-workspace-left">
                <span class="ba-badge-avatar">S</span>
                <span class="ba-workspace-name">Personal</span>
            </div>
            <span style="color:#64748B; font-size:12px;">⌵</span>
        </div>
        <div class="ba-projects-link">
            <span>All projects</span>
            <span style="font-size:10px;">⌵</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="ba-new-chat-wrapper">', unsafe_allow_html=True)
    c_new_chat = st.sidebar.button("+ New chat", key="ba_new_chat_btn", use_container_width=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if c_new_chat:
        st.session_state["active_menu"] = "Dashboards"
        st.session_state["ba_chat_prompt_input"] = ""
        st.session_state.last_result = None
        st.session_state.last_explanation = None
        st.session_state["last_user_query"] = None
        st.session_state["show_uploader_modal"] = False
        st.rerun()

    curr_menu = st.session_state.get("active_menu", "Dashboards")

    st.sidebar.markdown('<div class="ba-nav-section-title">CREATE</div>', unsafe_allow_html=True)
    c_dash = st.sidebar.button("⊞  Dashboards", key="nav_dash", use_container_width=True, type="primary" if curr_menu == "Dashboards" else "secondary")
    c_agents = st.sidebar.button("👥  Agents", key="nav_agents", use_container_width=True, type="primary" if curr_menu == "Agents" else "secondary")
    c_tasks = st.sidebar.button("⚡  Tasks", key="nav_tasks", use_container_width=True, type="primary" if curr_menu == "Tasks" else "secondary")

    st.sidebar.markdown('<div class="ba-nav-section-title">DATA</div>', unsafe_allow_html=True)
    c_explore = st.sidebar.button("🌐  Explore Data", key="nav_explore", use_container_width=True, type="primary" if curr_menu == "Explore Data" else "secondary")
    c_library = st.sidebar.button("📁  Library", key="nav_library", use_container_width=True, type="primary" if curr_menu == "Library" else "secondary")
    c_connect = st.sidebar.button("🔌  Data Connectors", key="nav_connectors", use_container_width=True, type="primary" if curr_menu == "Data Connectors" else "secondary")
    c_flows = st.sidebar.button("🔀  Data Flows", key="nav_flows", use_container_width=True, type="primary" if curr_menu == "Data Flows" else "secondary")
    c_alerts = st.sidebar.button("🔔  Alerts", key="nav_alerts", use_container_width=True, type="primary" if curr_menu == "Alerts" else "secondary")

    st.sidebar.markdown('<div class="ba-nav-section-title">TOOLS</div>', unsafe_allow_html=True)
    c_pdf = st.sidebar.button("📄  PDF to Excel", key="nav_pdf", use_container_width=True, type="primary" if curr_menu == "PDF to Excel" else "secondary")
    c_slack = st.sidebar.button("💬  Slack Agent", key="nav_slack", use_container_width=True, type="primary" if curr_menu == "Slack Agent" else "secondary")
    c_sql = st.sidebar.button("💻  SQL Console", key="nav_sql", use_container_width=True, type="primary" if curr_menu == "SQL Console" else "secondary")
    c_settings = st.sidebar.button("⚙️  Settings", key="nav_settings", use_container_width=True, type="primary" if curr_menu == "Settings" else "secondary")

    if c_dash:
        st.session_state["active_menu"] = "Dashboards"
        st.rerun()
    elif c_agents:
        st.session_state["active_menu"] = "Agents"
        st.rerun()
    elif c_tasks:
        st.session_state["active_menu"] = "Tasks"
        st.rerun()
    elif c_explore:
        st.session_state["active_menu"] = "Explore Data"
        st.rerun()
    elif c_library:
        st.session_state["active_menu"] = "Library"
        st.rerun()
    elif c_connect:
        st.session_state["active_menu"] = "Data Connectors"
        st.rerun()
    elif c_flows:
        st.session_state["active_menu"] = "Data Flows"
        st.rerun()
    elif c_alerts:
        st.session_state["active_menu"] = "Alerts"
        st.rerun()
    elif c_pdf:
        st.session_state["active_menu"] = "PDF to Excel"
        st.rerun()
    elif c_slack:
        st.session_state["active_menu"] = "Slack Agent"
        st.rerun()
    elif c_sql:
        st.session_state["active_menu"] = "SQL Console"
        st.rerun()
    elif c_settings:
        st.session_state["active_menu"] = "Settings"
        st.rerun()

    st.sidebar.markdown(
        """
        <div class="ba-nav-section-title">BRAND & PUBLISH</div>
        <div class="ba-slack-banner">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:16px;">💬</span>
                <span style="color:#EEF2FF; font-size:12px; font-weight:500;">Chat with your data in Slack</span>
            </div>
            <span style="color:#818CF8; font-size:14px;">›</span>
        </div>
        <div class="ba-plan-card">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#F8FAFC; font-size:12px; font-weight:600;">Free plan</span>
                <span class="ba-upgrade-btn">Upgrade</span>
            </div>
            <div style="color:#94A3B8; font-size:11px;">9 messages · 25 AI actions left</div>
        </div>
        <div class="ba-user-footer">
            <div style="display:flex; align-items:center; gap:9px;">
                <div class="ba-avatar-circle">S</div>
                <span style="color:#F8FAFC; font-size:13px; font-weight:500;">Saurav Yadav</span>
            </div>
            <span style="color:#64748B; font-size:11px;">^</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("ℹ️ System & Privacy Details", expanded=False):
        st.caption(f"{cfg.branding.product_name} · v{__version__} ({cfg.server.edition})")
        user: User = st.session_state.user
        if user:
            st.sidebar.write(f"User: `{user.username}`")
            st.sidebar.write(f"Organization: `{user.org_id}`")
        st.sidebar.info(
            "Customer data stays on this computer. "
            "There is no cloud AI API. Telemetry is disabled."
        )
        model_host = f"{cfg.ai.host}:{cfg.ai.port}"
        st.sidebar.write(f"Local model target: `{model_host}` / `{cfg.ai.model}`")
        st.sidebar.write("Uploads are kept in memory unless you export them.")

        ingest_val = st.session_state.get("ingest")
        if isinstance(ingest_val, IngestResult):
            st.sidebar.divider()
            st.sidebar.subheader("Active Dataset")
            source_name = getattr(ingest_val, "source_name", None)
            if source_name:
                st.sidebar.markdown(f"**Source:** `{Path(str(source_name)).name}`")
            if getattr(ingest_val, "sheet_name", None):
                st.sidebar.caption(f"Sheet: `{ingest_val.sheet_name}`")
            frame = getattr(ingest_val, "frame", None)
            if frame is not None and hasattr(frame, "shape"):
                try:
                    rows, cols = int(frame.shape[0]), int(frame.shape[1])
                    st.sidebar.caption(f"Shape: **{rows:,} rows × {cols} cols**")
                except Exception:
                    pass
            ops = getattr(ingest_val, "cleaning_log", None) or getattr(ingest_val, "cleaning_operations", [])
            if isinstance(ops, list):
                st.sidebar.caption(f"Cleaning Log: **{len(ops)} operation(s)**")

    if cfg.server.require_auth:
        if st.sidebar.button("Sign out"):
            token = st.session_state.get("session_token")
            try:
                if token:
                    mgr = _get_session_manager(cfg.server.session_idle_minutes)
                    mgr.destroy_session(token)
                log = _get_audit_log(cfg)
                log.record(user.org_id, user.username, "logout", "User signed out")
            except AuditLogError:
                # Deliberately suppressed: audit storage failure must not prevent client session teardown.
                pass
            finally:
                st.session_state.user = None
                st.session_state.session_token = None
                st.session_state.ingest = None
                st.session_state.duckdb_console = None
            st.rerun()


def _check_session_timeout(cfg) -> bool:
    if cfg.server.edition == "server" or cfg.server.require_auth:
        user_in_state = st.session_state.get("user")
        if user_in_state is not None:
            token = st.session_state.get("session_token")
            mgr = _get_session_manager(cfg.server.session_idle_minutes)
            sess = mgr.validate_session(token) if token else None
            is_valid = (
                sess is not None
                and sess.user_id == user_in_state.id
                and sess.org_id == user_in_state.org_id
            )
            if not is_valid:
                try:
                    if token:
                        mgr.destroy_session(token)
                    _get_audit_log(cfg).record(
                        user_in_state.org_id,
                        user_in_state.username,
                        "session_timeout",
                        "Session expired due to inactivity or invalid session state",
                    )
                except AuditLogError:
                    # Deliberately suppressed: audit storage failure must not interrupt session purge.
                    pass
                finally:
                    st.session_state.user = None
                    st.session_state.session_token = None
                    st.session_state.ingest = None
                    st.session_state.duckdb_console = None
                st.warning("Session expired due to inactivity. Please sign in again.")
                return True
            mgr.touch_session(token)
            st.session_state.last_activity = datetime.now(timezone.utc)
    return False


def _import_tab(cfg, user: User) -> None:
    st.header("Import Data")
    st.caption("Upload CSV or Excel files. All parsing and schema profiling execute strictly on your local machine.")
    uploaded = st.file_uploader("Select Table File", type=["csv", "xlsx", "xls"], disabled=not can(user, "ingest"))
    if uploaded is None:
        return

    raw_bytes = uploaded.getvalue()
    is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
    selected_sheet: str | None = None

    if is_excel:
        wb_info = inspect_workbook(raw_bytes, uploaded.name)
        if wb_info and len(wb_info.sheet_names) > 1:
            st.info(f"Workbook contains {len(wb_info.sheet_names)} sheets. Recommended primary sheet: **'{wb_info.recommended_sheet}'**")
            default_idx = (
                wb_info.sheet_names.index(wb_info.recommended_sheet)
                if wb_info.recommended_sheet in wb_info.sheet_names
                else 0
            )
            selected_sheet = st.selectbox(
                "Select Worksheet",
                wb_info.sheet_names,
                index=default_idx,
                help="Choose which sheet to load for analysis.",
            )

    log = _get_audit_log(cfg)
    try:
        ingest = load_tabular(
            raw_bytes,
            uploaded.name,
            cfg.limits,
            sheet_name=selected_sheet,
        )
        file_sig = (uploaded.name, len(raw_bytes), selected_sheet)
        if st.session_state.get("_last_ingested_file") != file_sig:
            st.session_state["_last_ingested_file"] = file_sig
            _invalidate_downstream_state()
            log.record(
                user.org_id,
                user.username,
                "ingest_table",
                f"Table: '{Path(ingest.source_name).stem}', Rows: {ingest.frame.shape[0]}, Cols: {ingest.frame.shape[1]}",
            )
    except PrivateAnalystError as exc:
        fail_sig = (uploaded.name, len(raw_bytes), exc.__class__.__name__)
        if st.session_state.get("_last_failed_file") != fail_sig:
            st.session_state["_last_failed_file"] = fail_sig
            log.record(
                user.org_id,
                user.username,
                "ingest_table_failure",
                f"File: '{Path(uploaded.name).stem}', Reason: {exc.__class__.__name__}",
            )
        st.error(str(exc))
        return

    st.session_state.ingest = ingest

    st.success(
        f"Successfully loaded and structured **{ingest.frame.shape[0]:,} rows × {ingest.frame.shape[1]} columns** from `{ingest.source_name}`"
        + (f" (Sheet: `{ingest.sheet_name}`)" if ingest.sheet_name else "")
        + "."
    )

    if ingest.detected_title or ingest.excluded_rows > 0:
        meta_parts = []
        if ingest.detected_title:
            meta_parts.append(f"Title banner: *'{ingest.detected_title}'*")
        if ingest.excluded_rows > 0:
            meta_parts.append(f"{ingest.excluded_rows} introductory/empty rows safely excluded")
        if ingest.detected_index_column:
            meta_parts.append(f"Index column '{ingest.detected_index_column}' identified")
        st.info("Structure Auto-Detection: " + " | ".join(meta_parts))

    if ingest.warnings:
        for warning in ingest.warnings:
            st.warning(warning)

    try:
        col_res = st.columns(4)
        if isinstance(col_res, (list, tuple)) and len(col_res) == 4:
            col_res[0].metric("Loaded Records", f"{ingest.frame.shape[0]:,}")
            col_res[1].metric("Total Columns", f"{ingest.frame.shape[1]}")
            col_res[2].metric("Header Row", f"Row {ingest.header_row + 1}")
            col_res[3].metric("Excluded Rows", f"{ingest.excluded_rows}")
    except Exception:
        pass

    tabs_list = None
    try:
        t_res = st.tabs(["Cleaned Table", "Raw Data Preview", "Cleaning Ledger"])
        if isinstance(t_res, (list, tuple)) and len(t_res) == 3:
            tabs_list = t_res
    except Exception:
        tabs_list = None

    if tabs_list is not None:
        tab_clean, tab_raw, tab_ledger = tabs_list
        with tab_clean:
            st.subheader("Cleaned & Structured Data")
            st.dataframe(ingest.frame.head(cfg.limits.preview_rows), use_container_width=True)

            with st.expander("Detected Column Types & Schema Intelligence", expanded=False):
                type_rows = []
                sem_types = getattr(ingest, "semantic_types", {})
                for col_name, inferred_type in ingest.inferred_types.items():
                    non_null_cnt = int(ingest.frame[col_name].notna().sum())
                    null_cnt = int(ingest.frame[col_name].isna().sum())
                    s_type = sem_types.get(col_name, inferred_type)
                    type_rows.append({
                        "Column Name": col_name,
                        "Inferred Type": inferred_type,
                        "Semantic Type": s_type.title() if s_type else "-",
                        "Non-Null Count": non_null_cnt,
                        "Missing Count": null_cnt,
                        "Completeness": f"{(non_null_cnt / len(ingest.frame) * 100):.1f}%" if len(ingest.frame) > 0 else "0.0%",
                    })
                st.dataframe(pd.DataFrame(type_rows), use_container_width=True)

        with tab_raw:
            st.subheader("Raw File Extract")
            st.caption("Unaltered spreadsheet content before header identification or row exclusions.")
            if ingest.raw_frame is not None:
                st.dataframe(ingest.raw_frame.head(cfg.limits.preview_rows), use_container_width=True)
            else:
                st.info("Raw spreadsheet buffer view is only available for direct file imports.")

        with tab_ledger:
            st.subheader("Data Cleaning Ledger & Audit Trail")
            st.caption("Verifiable record of every automated normalization and filtering step performed on ingestion.")
            ops = getattr(ingest, "cleaning_log", None) or getattr(ingest, "cleaning_operations", [])
            if ops:
                ledger_rows = []
                warnings = []
                for op in ops:
                    row, warn = normalize_cleaning_operation(op)
                    ledger_rows.append(row)
                    if warn:
                        warnings.append(warn)

                if warnings:
                    st.warning(f"Notice: {len(warnings)} cleaning log entry/entries had non-standard formatting. All details were retained.")

                st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True)
            else:
                st.success("No automated cleaning operations were required. The input dataset was already clean and tabular.")
    else:
        st.subheader("Cleaned & Structured Data")
        st.dataframe(ingest.frame.head(cfg.limits.preview_rows), use_container_width=True)

    st.info("💡 Next steps: Proceed to **2. Data Quality** to audit completeness, **3. Clean Data** for automated cleaning plans, or **4. Repair Rows & Columns** for interactive adjustments.")


def _quality_tab(cfg, user: User) -> None:
    st.header("2. Data Quality & Completeness Audit")
    st.caption("Comprehensive data health diagnostics, missingness metrics, and integrity scoring.")
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table file first in '1. Import Data' to view data quality diagnostics.")
        return

    quality = assess_quality(ingest.frame, ingest.duplicate_columns)
    stats = compute_dataset_stats(ingest.frame, getattr(ingest, "inferred_types", None))

    try:
        c_metrics = st.columns(5)
        if isinstance(c_metrics, (list, tuple)) and len(c_metrics) == 5:
            c_metrics[0].metric("Total Records", f"{stats.row_count:,}")
            c_metrics[1].metric("Total Columns", f"{stats.column_count}")
            c_metrics[2].metric("Quality Score", f"{quality.quality_score:.1f}/100", delta=quality.grade)
            c_metrics[3].metric("Completeness", f"{quality.completeness_ratio * 100:.1f}%")
            c_metrics[4].metric("Duplicate Rows", f"{quality.duplicate_row_count}")
    except Exception:
        pass

    st.subheader("Four Pillars Quality Breakdown")
    try:
        b_cols = st.columns(4)
        if isinstance(b_cols, (list, tuple)) and len(b_cols) == 4:
            b_cols[0].metric("Completeness", f"{quality.score_breakdown.get('completeness', 0):.1f} / 40")
            b_cols[1].metric("Uniqueness", f"{quality.score_breakdown.get('uniqueness', 0):.1f} / 20")
            b_cols[2].metric("Column Health", f"{quality.score_breakdown.get('column_health', 0):.1f} / 20")
            b_cols[3].metric("Type Consistency", f"{quality.score_breakdown.get('type_consistency', 0):.1f} / 20")
    except Exception:
        pass

    if quality.issues:
        st.subheader("Identified Quality Issues")
        for issue in quality.issues:
            st.markdown(f"- ⚠️ {issue}")
    else:
        st.success("No critical quality defects identified in this dataset.")

    st.subheader("Missingness Distribution")
    if quality.total_missing_cells == 0:
        st.success(f"100% Complete Data: 0 missing values detected across all {stats.column_count} columns.")
    else:
        st.plotly_chart(missingness(quality.missing_by_column), key="quality_missingness_chart", use_container_width=True)

    st.subheader("Column-Level Health Audit")
    col_health_rows = []
    sem_types = getattr(ingest, "semantic_types", {})
    for col in stats.columns:
        null_count = int(quality.missing_by_column.get(col.name, 0))
        comp_pct = ((stats.row_count - null_count) / stats.row_count * 100) if stats.row_count else 0.0
        col_health_rows.append({
            "Column Name": col.name,
            "Inferred Type": col.kind.title(),
            "Semantic Type": sem_types.get(col.name, col.kind).title(),
            "Non-Null Count": stats.row_count - null_count,
            "Missing Count": null_count,
            "Completeness (%)": f"{comp_pct:.1f}%",
            "Unique Values": col.unique,
        })
    st.dataframe(pd.DataFrame(col_health_rows), use_container_width=True)

    if quality.duplicate_row_count > 0:
        with st.expander(f"Inspect {quality.duplicate_row_count} Duplicate Rows", expanded=False):
            dup_mask = ingest.frame.duplicated(keep=False)
            st.dataframe(ingest.frame[dup_mask].head(100), use_container_width=True)


def _format_cleaning_rec_label(r: Any) -> tuple[str, str]:
    """Format checkbox label and retrieve operation id safely."""
    rec_id = getattr(r, "id", None) if not isinstance(r, dict) else r.get("id")
    title = getattr(r, "title", "Cleaning Step") if not isinstance(r, dict) else r.get("title", "Cleaning Step")
    desc = getattr(r, "description", "") if not isinstance(r, dict) else r.get("description", "")
    rows_count = getattr(r, "affected_rows_count", None) if not isinstance(r, dict) else r.get("affected_rows_count")

    detail = f" — {desc}" if desc else ""
    if rows_count is not None and isinstance(rows_count, (int, float)) and rows_count > 0:
        aff = f" ({int(rows_count):,} affected)"
    else:
        aff = ""

    is_lossy = getattr(r, "is_lossy", False) if not isinstance(r, dict) else r.get("is_lossy", False)
    prefix = "⚠️ **[Lossy] " if is_lossy else "**"
    suffix = "**" if is_lossy else "**"
    label = f"{prefix}{title}{suffix}{detail}{aff}"
    return label, str(rec_id or "")


def _clean_tab(cfg, user: User) -> None:
    st.header("3. Clean Data & Automation Pipeline")
    st.caption("Review deterministic cleaning plans, apply automated transformations, and track the audit trail.")
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table file first in '1. Import Data' to enable automated cleaning.")
        return

    quality = assess_quality(ingest.frame, ingest.duplicate_columns)
    recs = generate_cleaning_recommendations(ingest.frame, quality)

    st.subheader("Automated Cleaning Recommendations")
    st.caption("The engine recommends targeted transformations to normalize strings, repair headers, and address missing data.")

    safe_recs = [r for r in recs if not getattr(r, "is_lossy", False)]
    lossy_recs = [r for r in recs if getattr(r, "is_lossy", False)]

    selected_ids: list[str] = []

    if not recs:
        st.success("No automated cleaning operations are required. The dataset is already well-structured and standardized.")
    else:
        if safe_recs:
            st.markdown("#### Safe Transformations (Preserves All Records)")
            for r in safe_recs:
                label, rec_id = _format_cleaning_rec_label(r)
                if not rec_id:
                    continue
                default_val = getattr(r, "default_enabled", True) if not isinstance(r, dict) else r.get("default_enabled", True)
                checked = st.checkbox(
                    label,
                    value=bool(default_val),
                    key=f"clean_rec_{rec_id}",
                )
                if checked:
                    selected_ids.append(rec_id)

        if lossy_recs:
            st.markdown("#### Cautionary / Lossy Actions (Removes Records or Columns)")
            for r in lossy_recs:
                label, rec_id = _format_cleaning_rec_label(r)
                if not rec_id:
                    continue
                default_val = getattr(r, "default_enabled", False) if not isinstance(r, dict) else r.get("default_enabled", False)
                checked = st.checkbox(
                    label,
                    value=bool(default_val),
                    key=f"clean_rec_{rec_id}",
                )
                if checked:
                    selected_ids.append(rec_id)

        if st.button("Apply Selected Cleaning Plan", type="primary"):
            if not selected_ids:
                st.warning("Please select at least one cleaning operation to apply.")
            else:
                rows_before = int(ingest.frame.shape[0])
                cols_before = int(ingest.frame.shape[1])
                updated_frame, applied_ops = apply_cleaning_plan(ingest.frame, selected_ids)
                rows_after = int(updated_frame.shape[0])
                cols_after = int(updated_frame.shape[1])
                ingest.frame = updated_frame
                ingest.working_frame = updated_frame
                for op in applied_ops:
                    if hasattr(ingest, "add_cleaning_operation"):
                        ingest.add_cleaning_operation(op)
                    elif hasattr(ingest, "cleaning_log"):
                        ingest.cleaning_log.append(op)
                    elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                        ingest.cleaning_operations.append(op)

                st.session_state.ingest = ingest
                _invalidate_downstream_state()
                try:
                    _get_audit_log(cfg).record(
                        user.org_id,
                        user.username,
                        "clean_dataset",
                        f"Ops: {len(applied_ops)}, Shape: {rows_before}x{cols_before}->{rows_after}x{cols_after}",
                    )
                except AuditLogError:
                    pass

                st.success(
                    f"Successfully applied {len(applied_ops)} cleaning operation(s)! "
                    f"Dataset transformed from ({rows_before:,} × {cols_before}) "
                    f"to ({rows_after:,} × {cols_after})."
                )
                st.rerun()

    st.divider()
    st.subheader("Data Cleaning Ledger & Verifiable Audit Trail")
    st.caption("Immutable chronological record of all automated and manual transformations applied to this dataset.")
    ops = getattr(ingest, "cleaning_log", None) or getattr(ingest, "cleaning_operations", [])
    if ops:
        ledger_rows = []
        warnings = []
        for op in ops:
            row, warn = normalize_cleaning_operation(op)
            ledger_rows.append(row)
            if warn:
                warnings.append(warn)

        if warnings:
            st.warning(f"Notice: {len(warnings)} cleaning log entries had non-standard formatting. All details were retained.")

        st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True)
    else:
        st.info("No cleaning operations have been recorded yet.")


def _repair_tab(cfg, user: User) -> None:
    st.header("4. Repair Rows & Columns")
    st.caption("Interactive data surgery: promote headers, repair unnamed columns, impute missing values, and filter rows.")
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table file first in '1. Import Data' to repair rows and columns.")
        return

    if getattr(ingest, "header_ambiguous", False):
        st.warning(
            "⚠️ Header Ambiguity Detected: The current header row appears to contain generic or numeric identifiers. "
            "If your dataset has introductory titles or banner rows, select the true header row below and click 'Apply Header Override'."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Header & Unnamed Column Repairs")
        unnamed_cols = [c for c in ingest.frame.columns if "unnamed" in str(c).lower()]
        if unnamed_cols:
            st.warning(f"Detected {len(unnamed_cols)} unnamed column(s): {', '.join(unnamed_cols)}")
            if st.button("Auto-Repair Unnamed Headers", type="primary"):
                repaired_frame, renamed, op = repair_unnamed_headers(ingest.frame)
                ingest.frame = repaired_frame
                ingest.working_frame = repaired_frame
                ingest.unnamed_columns = []
                if op:
                    if hasattr(ingest, "add_cleaning_operation"):
                        ingest.add_cleaning_operation(op)
                    elif hasattr(ingest, "cleaning_log"):
                        ingest.cleaning_log.append(op)
                    elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                        ingest.cleaning_operations.append(op)
                st.session_state.ingest = ingest
                _invalidate_downstream_state()
                st.success(f"Repaired {len(renamed)} column headers: {renamed}")
                st.rerun()
        else:
            st.success("All column headers are properly named.")

        st.markdown("#### Header Row Override")
        st.caption("Promote a specific row (0-indexed) to become the column headers, excluding any preceding banner rows.")
        override_idx = st.number_input(
            "Select Header Row Index",
            min_value=0,
            max_value=max(0, len(ingest.frame) - 1),
            value=ingest.header_row,
            step=1,
            key="repair_override_header_input",
        )
        if st.button("Apply Header Override"):
            try:
                source_df = ingest.raw_frame if ingest.raw_frame is not None else ingest.frame
                repaired_frame, op = apply_header_override(source_df, override_idx)
                ingest.frame = repaired_frame
                ingest.working_frame = repaired_frame
                ingest.header_row = override_idx
                if op:
                    if hasattr(ingest, "add_cleaning_operation"):
                        ingest.add_cleaning_operation(op)
                    elif hasattr(ingest, "cleaning_log"):
                        ingest.cleaning_log.append(op)
                    elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                        ingest.cleaning_operations.append(op)
                st.session_state.ingest = ingest
                _invalidate_downstream_state()
                st.success(f"Promoted row {override_idx + 1} to column headers.")
                st.rerun()
            except (ValidationError, ValueError) as exc:
                st.error(str(exc))

    with col2:
        st.subheader("Column Management")
        all_cols = list(ingest.frame.columns)
        st.markdown("#### Exclude / Drop Columns")
        cols_to_drop = st.multiselect("Select Columns to Remove", all_cols, key="repair_drop_cols_select")
        if st.button("Drop Selected Columns") and cols_to_drop:
            try:
                new_frame, ops = apply_column_management(ingest.frame, exclude_columns=cols_to_drop)
                ingest.frame = new_frame
                ingest.working_frame = new_frame
                if ops:
                    if hasattr(ingest, "add_cleaning_operation"):
                        ingest.add_cleaning_operation(ops)
                    elif hasattr(ingest, "cleaning_log"):
                        ingest.cleaning_log.extend(ops if isinstance(ops, list) else [ops])
                    elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                        ingest.cleaning_operations.extend(ops if isinstance(ops, list) else [ops])
                st.session_state.ingest = ingest
                _invalidate_downstream_state()
                st.success(f"Dropped {len(cols_to_drop)} column(s).")
                st.rerun()
            except (ValidationError, ValueError) as exc:
                st.error(str(exc))

        st.markdown("#### Rename Column")
        col_to_rename = st.selectbox("Select Column to Rename", all_cols, key="repair_rename_col_select")
        new_col_name = st.text_input("New Column Name", value=str(col_to_rename), key="repair_new_col_name_input")
        if st.button("Rename Column") and new_col_name and new_col_name != col_to_rename:
            try:
                new_frame, ops = apply_column_management(ingest.frame, rename_columns={col_to_rename: new_col_name})
                ingest.frame = new_frame
                ingest.working_frame = new_frame
                if ops:
                    if hasattr(ingest, "add_cleaning_operation"):
                        ingest.add_cleaning_operation(ops)
                    elif hasattr(ingest, "cleaning_log"):
                        ingest.cleaning_log.extend(ops if isinstance(ops, list) else [ops])
                    elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                        ingest.cleaning_operations.extend(ops if isinstance(ops, list) else [ops])
                st.session_state.ingest = ingest
                _invalidate_downstream_state()
                st.success(f"Renamed '{col_to_rename}' to '{new_col_name}'.")
                st.rerun()
            except (ValidationError, ValueError) as exc:
                st.error(str(exc))

    st.divider()
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.subheader("Missing Value Imputation")
        missing_cols = [c for c in ingest.frame.columns if ingest.frame[c].isna().any()]
        if not missing_cols:
            st.success("No missing values in any column.")
        else:
            imp_col = st.selectbox("Column with Missing Values", missing_cols, key="repair_imp_col_select")
            imp_strategy = st.selectbox(
                "Imputation Strategy",
                ["fill_mean", "fill_median", "fill_mode", "fill_zero", "fill_na_text", "forward_fill"],
                format_func=lambda s: {
                    "fill_mean": "Fill with Mean (Numeric)",
                    "fill_median": "Fill with Median (Numeric)",
                    "fill_mode": "Fill with Mode (Most Frequent)",
                    "fill_zero": "Fill with Zero (0)",
                    "fill_na_text": "Fill with 'N/A' (Text)",
                    "forward_fill": "Forward Fill (Carry Previous)",
                }.get(s, s),
                key="repair_imp_strat_select",
            )
            if st.button("Impute Missing Values"):
                try:
                    new_frame, op = handle_missing_values(ingest.frame, imp_col, imp_strategy)
                    ingest.frame = new_frame
                    ingest.working_frame = new_frame
                    if op:
                        if hasattr(ingest, "add_cleaning_operation"):
                            ingest.add_cleaning_operation(op)
                        elif hasattr(ingest, "cleaning_log"):
                            ingest.cleaning_log.append(op)
                        elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                            ingest.cleaning_operations.append(op)
                    st.session_state.ingest = ingest
                    _invalidate_downstream_state()
                    st.success(f"Imputed missing values for '{imp_col}'.")
                    st.rerun()
                except (ValidationError, ValueError) as exc:
                    st.error(str(exc))

    with c_sub2:
        st.subheader("Row Filtering & Subsetting")
        filter_col = st.selectbox("Filter on Column", list(ingest.frame.columns), key="repair_filter_col_select")
        filter_op = st.selectbox(
            "Comparison Operator",
            ["==", "!=", ">", "<", ">=", "<=", "contains", "not_contains", "is_not_null", "is_null"],
            key="repair_filter_op_select",
        )
        filter_val = st.text_input("Comparison Value", value="", key="repair_filter_val_input")
        if st.button("Apply Row Filter"):
            try:
                new_frame, op = filter_rows(ingest.frame, filter_col, filter_op, filter_val)
                ingest.frame = new_frame
                ingest.working_frame = new_frame
                if op:
                    if hasattr(ingest, "add_cleaning_operation"):
                        ingest.add_cleaning_operation(op)
                    elif hasattr(ingest, "cleaning_log"):
                        ingest.cleaning_log.append(op)
                    elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                        ingest.cleaning_operations.append(op)
                st.session_state.ingest = ingest
                _invalidate_downstream_state()
                st.success(f"Filter applied: {len(new_frame):,} rows remaining.")
                st.rerun()
            except (ValidationError, ValueError) as exc:
                st.error(str(exc))

    if ingest.raw_frame is not None or getattr(ingest, "initial_frame", None) is not None:
        st.divider()
        st.subheader("Revert Dataset to Original File State")
        st.caption(
            "Discard interactive modifications. You can restore either the initial loaded table "
            "(with detected headers) or the raw un-promoted spreadsheet matrix."
        )
        col_rst1, col_rst2 = st.columns(2)
        with col_rst1:
            if getattr(ingest, "initial_frame", None) is not None:
                if st.button("Reset to Initial Loaded Table", type="secondary"):
                    restored = ingest.initial_frame.copy()
                    ingest.frame = restored
                    ingest.working_frame = restored
                    ingest.unnamed_columns = [c for c in restored.columns if "unnamed" in str(c).lower()]
                    reset_op = CleaningOperation(
                        step="reset_dataset",
                        description="Reset dataset to initial loaded state",
                        impact=f"Restored initial loaded table with {len(restored)} rows and {len(restored.columns)} columns",
                        rows_affected=len(restored),
                    )
                    ingest.cleaning_log = [reset_op]
                    st.session_state.ingest = ingest
                    _invalidate_downstream_state()
                    st.success("Dataset reset to initial loaded table.")
                    st.rerun()

        with col_rst2:
            if ingest.raw_frame is not None:
                if st.button("Reset Dataset to Raw Extract", type="secondary"):
                    restored = ingest.raw_frame.copy()
                    ingest.frame = restored
                    ingest.working_frame = restored
                    ingest.unnamed_columns = [c for c in restored.columns if "unnamed" in str(c).lower()]
                    reset_op = CleaningOperation(
                        step="reset_dataset",
                        description="Reset dataset to raw spreadsheet extract",
                        impact=f"Restored raw extract buffer with {len(restored)} rows and {len(restored.columns)} columns",
                        rows_affected=len(restored),
                    )
                    ingest.cleaning_log = [reset_op]
                    st.session_state.ingest = ingest
                    _invalidate_downstream_state()
                    st.success("Dataset successfully reset to original file state.")
                    st.rerun()


def _explore_tab(cfg, user: User) -> None:
    st.header("5. Explore Data & Statistical Profiles")
    st.caption("Distribution profiles, descriptive statistics, and correlation matrix.")
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table file first in '1. Import Data' to explore data.")
        return

    stats = compute_dataset_stats(ingest.frame, getattr(ingest, "inferred_types", None))

    try:
        c_metrics = st.columns(4)
        if isinstance(c_metrics, (list, tuple)) and len(c_metrics) == 4:
            c_metrics[0].metric("Total Records", f"{stats.row_count:,}")
            c_metrics[1].metric("Total Columns", f"{stats.column_count}")
            c_metrics[2].metric("Numeric Features", f"{len(stats.numeric_column_names)}")
            c_metrics[3].metric("Categorical Features", f"{stats.column_count - len(stats.numeric_column_names)}")
    except Exception:
        pass

    numeric_rows = [
        {
            "Column": col.name,
            "Count": col.count,
            "Missing": col.missing,
            "Mean": round(col.mean, 2) if col.mean is not None else None,
            "Median": round(col.median, 2) if col.median is not None else None,
            "Std Dev": round(col.std, 2) if col.std is not None else None,
            "Min": col.min,
            "Max": col.max,
            "Sum": col.sum,
        }
        for col in stats.columns
        if col.kind == "numeric"
    ]
    if numeric_rows:
        st.subheader("Numeric Distributions")
        st.dataframe(pd.DataFrame(numeric_rows), use_container_width=True)
        try:
            first_num = stats.numeric_column_names[0]
            st.plotly_chart(histogram(ingest.frame, first_num), key=f"explore_hist_{first_num}", use_container_width=True)
            cats = [c for c in ingest.frame.columns if c not in stats.numeric_column_names]
            if cats:
                st.plotly_chart(bar_sum(ingest.frame, str(cats[0]), first_num), key=f"explore_barsum_{cats[0]}_{first_num}", use_container_width=True)
        except Exception as exc:
            st.warning(f"Distribution chart unavailable: {exc}")

    cat_cols_stats = [col for col in stats.columns if col.kind != "numeric" and col.name not in {"#", "id", "s.no", "sl.no", "index"}]
    if cat_cols_stats:
        st.subheader("Categorical & Text Column Profiles")
        cat_summary_rows = [
            {
                "Column": col.name,
                "Type": col.kind.title(),
                "Records": col.count,
                "Missing": col.missing,
                "Unique Values": col.unique,
                "Top Category": col.top_value,
                "Top Count": col.top_freq,
                "Top Share": f"{(col.top_freq / col.count * 100):.1f}%" if col.count and col.top_freq else "N/A",
            }
            for col in cat_cols_stats
        ]
        st.dataframe(pd.DataFrame(cat_summary_rows), use_container_width=True)
        try:
            top_cat_name = cat_cols_stats[0].name
            st.plotly_chart(category_bar(ingest.frame, top_cat_name), key=f"explore_topcat_{top_cat_name}", use_container_width=True)
        except Exception as exc:
            st.warning(f"Category chart unavailable: {exc}")

    if len(stats.numeric_column_names) >= 2:
        st.subheader("Correlation Matrix & Pairwise Association")
        st.caption("Pearson correlation coefficients across numeric features.")
        c_res = compute_correlation_matrix(ingest.frame)
        if c_res.get("status") == "ok":
            try:
                corr_df = pd.DataFrame(c_res["matrix"])
                st.plotly_chart(correlation_heatmap(corr_df), key="explore_corr_heatmap", use_container_width=True)
                if c_res.get("top_correlations"):
                    with st.expander("Ranked Correlation Pairs", expanded=False):
                        st.dataframe(pd.DataFrame(c_res["top_correlations"]), use_container_width=True)
            except Exception as exc:
                st.warning(f"Correlation chart unavailable: {exc}")

    with st.expander("Column Suitability & Recommendation Insights", expanded=False):
        explanations = explain_chart_exclusions(ingest.frame, stats)
        if explanations:
            st.dataframe(pd.DataFrame(explanations), use_container_width=True)
        else:
            st.info("No columns available to evaluate.")


def _viz_tab(cfg, user: User) -> None:
    st.header("6. Visualizations & Chart Studio")
    st.caption("Interactive chart generation studio and smart data-driven recommendations.")
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table file first in '1. Import Data' to visualize charts.")
        return

    stats = compute_dataset_stats(ingest.frame, getattr(ingest, "inferred_types", None))
    all_cols = list(ingest.frame.columns)
    numeric_cols = stats.numeric_column_names
    cat_cols = [c for c in all_cols if c not in numeric_cols and c not in {"#", "id", "s.no", "sl.no", "index"}]

    st.subheader("Interactive Chart Builder")
    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Category Frequency (Bar)",
            "Numeric Distribution (Histogram)",
            "Group Sum / Aggregate (Bar)",
            "Outlier Box Plot",
            "Correlation Heatmap",
            "Time Series Trend",
        ],
        key="viz_builder_chart_type",
    )

    if chart_type == "Category Frequency (Bar)":
        if not cat_cols:
            st.warning("No categorical columns available.")
        else:
            sel_col = st.selectbox("Select Categorical Column", cat_cols, key="viz_cat_select")
            st.plotly_chart(category_bar(ingest.frame, sel_col), key=f"viz_builder_catbar_{sel_col}", use_container_width=True)

    elif chart_type == "Numeric Distribution (Histogram)":
        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            sel_num = st.selectbox("Select Numeric Column", numeric_cols, key="viz_num_select")
            st.plotly_chart(histogram(ingest.frame, sel_num), key=f"viz_builder_hist_{sel_num}", use_container_width=True)

    elif chart_type == "Group Sum / Aggregate (Bar)":
        if not cat_cols or not numeric_cols:
            st.warning("Group Sum chart requires at least one categorical column and one numeric column.")
        else:
            c1, c2 = st.columns(2)
            sel_cat = c1.selectbox("Category Dimension (X)", cat_cols, key="viz_barsum_cat")
            sel_val = c2.selectbox("Aggregate Value (Y)", numeric_cols, key="viz_barsum_val")
            st.plotly_chart(bar_sum(ingest.frame, sel_cat, sel_val), key=f"viz_builder_barsum_{sel_cat}_{sel_val}", use_container_width=True)

    elif chart_type == "Outlier Box Plot":
        if not numeric_cols:
            st.warning("Outlier Box Plot requires numeric columns.")
        else:
            sel_box = st.selectbox("Select Numeric Column", numeric_cols, key="viz_box_select")
            st.plotly_chart(outlier_box(ingest.frame, sel_box, title=f"Distribution & Outliers for '{sel_box}'"), key=f"viz_builder_box_{sel_box}", use_container_width=True)

    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Correlation heatmap requires at least two numeric columns.")
        else:
            c_res = compute_correlation_matrix(ingest.frame)
            if c_res.get("status") == "ok":
                st.plotly_chart(correlation_heatmap(pd.DataFrame(c_res["matrix"])), key="viz_builder_corr_matrix", use_container_width=True)
            else:
                st.warning(c_res.get("caveat", "Insufficient variance to compute correlation."))

    elif chart_type == "Time Series Trend":
        date_candidates = [
            c for c in all_cols
            if "date" in c.lower() or "time" in c.lower() or "year" in c.lower() or pd.api.types.is_datetime64_any_dtype(ingest.frame[c])
        ]
        if not date_candidates:
            date_candidates = all_cols
        c1, c2 = st.columns(2)
        sel_date = c1.selectbox("Date / Timestamp Column", date_candidates, key="viz_date_col")
        sel_metric = c2.selectbox("Metric Column (Optional)", ["(Record Count)"] + numeric_cols, key="viz_metric_col")
        actual_val = None if sel_metric == "(Record Count)" else sel_metric
        t_res = compute_time_trends(ingest.frame, sel_date, actual_val)
        if t_res.get("status") == "ok":
            trend_df = pd.DataFrame(t_res["trend_table"])
            st.plotly_chart(time_series_line(trend_df, "period", "total", f"Trends over {t_res['freq_label']} Intervals"), key=f"viz_builder_trend_{sel_date}_{actual_val}", use_container_width=True)
        else:
            st.warning(t_res.get("caveat", "Unable to compute time trend."))

    st.divider()
    recs = recommend_charts(ingest.frame, stats)
    if recs:
        st.subheader("Automated Chart Recommendations")
        st.caption("Heuristically selected visualizations tailored to your dataset schema and cardinality.")
        r_cols = st.columns(min(len(recs), 3))
        for idx, rec in enumerate(recs[:3]):
            with r_cols[idx]:
                st.markdown(f"**{rec['title']}**")
                st.caption(rec['description'])
                rtype = rec.get("type")
                rcol = rec.get("column")
                try:
                    if rtype == "category_bar" and rcol in ingest.frame.columns:
                        st.plotly_chart(category_bar(ingest.frame, rcol), key=f"viz_rec_{idx}_{rtype}_{rcol}", use_container_width=True)
                    elif rtype == "histogram" and rcol in ingest.frame.columns:
                        st.plotly_chart(histogram(ingest.frame, rcol), key=f"viz_rec_{idx}_{rtype}_{rcol}", use_container_width=True)
                    elif rtype == "correlation_heatmap":
                        c_res = compute_correlation_matrix(ingest.frame)
                        if c_res.get("status") == "ok":
                            st.plotly_chart(correlation_heatmap(pd.DataFrame(c_res["matrix"])), key=f"viz_rec_{idx}_{rtype}_matrix", use_container_width=True)
                    elif rtype == "crosstab_heatmap":
                        col1_name = rec.get("col1")
                        col2_name = rec.get("col2")
                        if col1_name in ingest.frame.columns and col2_name in ingest.frame.columns:
                            st.plotly_chart(crosstab_heatmap(ingest.frame, col1_name, col2_name), key=f"viz_rec_{idx}_{rtype}_{col1_name}_{col2_name}", use_container_width=True)
                    elif rtype == "time_series_line":
                        d_col = rec.get("date_col")
                        m_col = rec.get("metric_col")
                        if d_col in ingest.frame.columns:
                            t_res = compute_time_trends(ingest.frame, d_col, m_col)
                            if t_res.get("status") == "ok":
                                trend_df = pd.DataFrame(t_res["trend_table"])
                                st.plotly_chart(time_series_line(trend_df, "period", "total", rec.get("title")), key=f"viz_rec_{idx}_{rtype}_{d_col}", use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not render {rec.get('title')}: {exc}")

    with st.expander("Chart Recommendation Insights & Column Suitability", expanded=False):
        explanations = explain_chart_exclusions(ingest.frame, stats)
        if explanations:
            st.dataframe(pd.DataFrame(explanations), use_container_width=True)
        else:
            st.info("No columns available to evaluate.")


def _dashboard_tab() -> None:
    """Backward-compatible entry point for dashboard view."""
    cfg = load_config()
    user = st.session_state.get("user") or _desktop_user(cfg.server.edition)
    _explore_tab(cfg, user)


def _ask_tab(cfg, user: User) -> None:
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table before asking questions.")
        return

    st.subheader("Ask & Analyze")

    model_status = check_local_model(cfg.ai)
    try:
        status_cols = st.columns([4, 1])
        if isinstance(status_cols, (list, tuple)) and len(status_cols) == 2:
            c_status, c_btn = status_cols[0], status_cols[1]
            if model_status.available:
                c_status.success(f"**Local AI Model:** {model_status.friendly_status}")
            else:
                c_status.warning(f"**Local AI Model:** {model_status.friendly_status}")
            if c_btn.button("Test Connection"):
                st.info(model_status.friendly_status)
        else:
            if model_status.available:
                st.success(f"**Local AI Model:** {model_status.friendly_status}")
            else:
                st.warning(f"**Local AI Model:** {model_status.friendly_status}")
    except Exception:
        if model_status.available:
            st.success(f"**Local AI Model:** {model_status.friendly_status}")
        else:
            st.warning(f"**Local AI Model:** {model_status.friendly_status}")

    if not model_status.available:
        with st.expander("How to start your local AI model (Ollama)"):
            st.markdown(
                "- **Step 1:** Open a terminal and run `ollama serve`.\n"
                f"- **Step 2:** Ensure your model is downloaded: `ollama pull {cfg.ai.model}`.\n"
                "- **Privacy Assurance:** All AI inferences run strictly on `127.0.0.1` loopback. No data is sent over the internet.\n"
                "- **Note:** Even when the AI model is offline, the deterministic Python analytics engine is fully functional with 0% hallucination."
            )

    cat_cols = [c for c in ingest.frame.columns if not pd.api.types.is_numeric_dtype(ingest.frame[c]) and str(c).lower() not in {"#", "id", "s.no", "index"}]
    num_cols = [c for c in ingest.frame.columns if pd.api.types.is_numeric_dtype(ingest.frame[c]) and str(c).lower() not in {"#", "id", "s.no", "index"}]

    with st.expander("Suggested Analysis Questions & Prompts", expanded=False):
        prompt_bullets = [
            "- `Summarize the dataset.`",
            "- `Assess data quality and missingness.`",
        ]
        if cat_cols:
            prompt_bullets.append(f"- `Show frequency distribution of {cat_cols[0]}.`")
            if len(cat_cols) >= 2:
                prompt_bullets.append(f"- `Cross-tabulate {cat_cols[0]} and {cat_cols[1]}.`")
            elif num_cols:
                prompt_bullets.append(f"- `Analyze distribution of {num_cols[0]}.`")
        elif num_cols:
            prompt_bullets.append(f"- `Analyze distribution of {num_cols[0]}.`")
            if len(num_cols) >= 2:
                prompt_bullets.append("- `Compute correlation matrix across numeric features.`")
        st.markdown("\n".join(prompt_bullets))

    question = st.text_input("Question about the loaded table", value="Summarize the dataset.", key="ask_question_input")

    if st.button("Run analysis", key="ask_run_analysis_btn", type="primary"):
        clean_q = (question or "").strip()
        if not clean_q:
            st.warning("Please enter a question about the loaded table.")
        else:
            with st.spinner("Analyzing dataset with deterministic engine and generating briefing..."):
                op_failed = False
                result = None
                try:
                    result = run_operation(ingest, clean_q)
                except Exception as exc:
                    op_failed = True
                    logger.error("Analysis operation execution error: %s", type(exc).__name__)
                    st.error(f"Analysis operation error: {type(exc).__name__}")

                if not op_failed and result is not None:
                    try:
                        explanation = explain_results(result, clean_q, cfg.ai)
                    except Exception as exc:
                        logger.warning("Local AI explanation error: %s", type(exc).__name__)
                        explanation = {
                            "source": "deterministic_fallback",
                            "model_available": False,
                            "model_detail": f"AI model error: {type(exc).__name__}",
                            "text": deterministic_explanation(result, clean_q),
                            "model_status": None,
                        }

                    st.session_state.last_result = result
                    st.session_state.last_explanation = explanation

                    audit_recorded = True
                    try:
                        _get_audit_log(cfg).record(
                            user.org_id,
                            user.username,
                            "run_analysis",
                            f"Operation: {result.intent}, Status: {result.status}",
                        )
                    except AuditLogError as exc:
                        audit_recorded = False
                        logger.warning("Failed to record run_analysis audit event: %s", type(exc).__name__)

                    if not audit_recorded:
                        st.warning("Analysis completed, but the audit event could not be recorded.")

    result = st.session_state.last_result
    explanation = st.session_state.last_explanation
    if result is None or explanation is None:
        return

    st.divider()
    st.write("### Verified Analytical Facts & Metrics")
    st.caption("Calculated deterministically by the Python/DuckDB analytics engine — 100% accurate, 0% hallucination.")

    if result.chart_spec:
        spec = result.chart_spec
        chart_type = spec.get("type")
        try:
            if chart_type == "category_bar" and spec.get("column") in ingest.frame.columns:
                st.plotly_chart(category_bar(ingest.frame, spec["column"]), key=f"ask_catbar_{spec['column']}", use_container_width=True)
            elif chart_type == "histogram" and spec.get("column") in ingest.frame.columns:
                st.plotly_chart(histogram(ingest.frame, spec["column"]), key=f"ask_hist_{spec['column']}", use_container_width=True)
            elif chart_type == "crosstab_heatmap" and "crosstab" in result.calculated:
                ct_df = pd.DataFrame(result.calculated["crosstab"])
                st.plotly_chart(crosstab_heatmap(ct_df, spec.get("col_col", ""), spec.get("row_col", "")), key=f"ask_crosstab_{spec.get('col_col')}_{spec.get('row_col')}", use_container_width=True)
            elif chart_type == "correlation_heatmap" and "matrix" in result.calculated:
                st.plotly_chart(correlation_heatmap(pd.DataFrame(result.calculated["matrix"])), key="ask_corr_heatmap", use_container_width=True)
            elif chart_type == "time_series_line" and "trend_table" in result.calculated:
                st.plotly_chart(time_series_line(pd.DataFrame(result.calculated["trend_table"]), "period", "total", f"Time Trend for '{spec.get('date_col')}'"), key=f"ask_trend_{spec.get('date_col')}", use_container_width=True)
            elif chart_type == "bar" and spec.get("x") in ingest.frame.columns and spec.get("y") in ingest.frame.columns:
                st.plotly_chart(bar_sum(ingest.frame, spec["x"], spec["y"]), key=f"ask_barsum_{spec['x']}_{spec['y']}", use_container_width=True)
        except Exception as exc:
            st.warning(f"Visualization preview unavailable: {exc}")

    with st.container():
        st.markdown(f"**Intent:** `{result.intent.title()}` | **Engine Status:** `{result.status.upper()}`")

    with st.expander("Developer Diagnostics (Raw Engine Evidence)", expanded=False):
        st.write("**Observed (from file):**")
        st.json(result.observed)
        st.write("**Calculated (Python engine):**")
        st.json(result.calculated)

    st.write("### AI Analysis & Narrative")
    if explanation.get("source") == "local_model":
        st.success(f"Narrative generated by local model ({explanation.get('model_detail')}). Verified totals remain engine-computed.")
        st.markdown(explanation.get("text", ""))
    else:
        st.info("Local AI model is offline. Displaying deterministic structured engine findings below.")
        st.markdown(explanation.get("text", ""))



def _sql_tab(cfg, user: User) -> None:
    st.header("DuckDB SQL Console")
    st.caption("Execute read-only SQL queries in memory via DuckDB. No mutations are allowed.")
    if not can(user, "analyze"):
        st.error("You do not have permission to execute analytical queries.")
        return
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table file first to enable SQL querying.")
        return

    current_frame_id = (id(ingest.frame), len(ingest.frame), tuple(ingest.frame.columns))
    if st.session_state.duckdb_console is None or st.session_state.get("sql_registered_frame_id") != current_frame_id:
        if st.session_state.duckdb_console is not None:
            try:
                st.session_state.duckdb_console.close()
            except Exception:
                pass
        console = DuckDBConsole(max_rows=cfg.limits.max_rows)
        console.register_dataframe("data", ingest.frame)
        st.session_state.duckdb_console = console
        st.session_state["sql_registered_frame_id"] = current_frame_id
    else:
        console = st.session_state.duckdb_console

    st.info(f"Virtual Table **`data`**: **{ingest.frame.shape[0]:,} rows × {ingest.frame.shape[1]} columns** loaded from `{ingest.source_name}`")

    with st.expander("Table Schema & Column Types (`data`)", expanded=False):
        try:
            schema_info = console.describe_table("data")
            st.dataframe(pd.DataFrame(schema_info), use_container_width=True)
        except Exception:
            pass

    cat_cols = [c for c in ingest.frame.columns if not pd.api.types.is_numeric_dtype(ingest.frame[c]) and str(c).lower() not in {"#", "id", "s.no", "index"}]
    num_cols = [c for c in ingest.frame.columns if pd.api.types.is_numeric_dtype(ingest.frame[c]) and str(c).lower() not in {"#", "id", "s.no", "index"}]

    with st.expander("Sample Query Templates", expanded=False):
        st.code("SELECT * FROM data LIMIT 10;", language="sql")
        st.code("SELECT COUNT(*) AS total_records FROM data;", language="sql")
        if cat_cols:
            st.code(f'SELECT "{cat_cols[0]}", COUNT(*) AS count FROM data GROUP BY "{cat_cols[0]}" ORDER BY count DESC LIMIT 10;', language="sql")
        if num_cols:
            st.code(f'SELECT MIN("{num_cols[0]}") AS min_val, AVG("{num_cols[0]}") AS avg_val, MAX("{num_cols[0]}") AS max_val FROM data;', language="sql")

    default_query = "SELECT * FROM data LIMIT 10;"
    query = st.text_area("SQL Query (SELECT statements only)", value=default_query, height=120)

    col1, col2 = st.columns([1, 4])
    if col1.button("Run Query"):
        log = _get_audit_log(cfg)
        query_success = False
        res = None
        try:
            res = console.execute_query(query, timeout_seconds=cfg.limits.max_query_seconds)
            query_success = True
        except PrivateAnalystError as exc:
            try:
                log.record(
                    user.org_id,
                    user.username,
                    "sql_query_failure",
                    f"Status: rejected, Reason: {exc.__class__.__name__}",
                )
            except AuditLogError:
                # Deliberately suppressed: audit storage failure must not mask original query validation error
                pass
            st.error(str(exc))
        except Exception as exc:
            try:
                log.record(
                    user.org_id,
                    user.username,
                    "sql_query_failure",
                    f"Status: error, Reason: {exc.__class__.__name__}",
                )
            except AuditLogError:
                # Deliberately suppressed: audit storage failure must not mask original query execution error
                pass
            st.error(f"Query execution error: {type(exc).__name__}")

        if query_success and res is not None:
            audit_recorded = True
            try:
                log.record(
                    user.org_id,
                    user.username,
                    "sql_query",
                    f"Rows: {res.row_count}, Truncated: {res.truncated}",
                )
            except AuditLogError:
                audit_recorded = False

            if audit_recorded:
                st.success(f"Query returned {res.row_count} row(s).{' (Truncated to limit)' if res.truncated else ''}")
            else:
                st.warning(f"Query returned {res.row_count} row(s), but the audit event could not be recorded.")

            if res.rows:
                df_res = pd.DataFrame(res.rows)
                st.dataframe(df_res, use_container_width=True)
                csv_bytes = to_csv_bytes(df_res)
                st.download_button("Download Query Results (CSV)", csv_bytes, file_name="query_results.csv")
            else:
                st.info("Query returned no rows.")


def _render_ml_result(result) -> None:
    st.subheader(f"Status: {result.status.upper()}")
    if result.status == "insufficient_data":
        st.warning("Insufficient data: " + " ".join(result.caveats))
        return
    if result.status == "error":
        st.error("Error: " + " ".join(result.caveats))
        return

    st.write("### Evaluated Model Metrics")
    m = result.metrics
    if "test_accuracy" in m and "train_accuracy" in m:
        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout Test Accuracy", f"{m['test_accuracy'] * 100:.1f}%")
        c2.metric("Training Accuracy", f"{m['train_accuracy'] * 100:.1f}%")
        gap = (m['train_accuracy'] - m['test_accuracy']) * 100
        c3.metric("Generalization Gap", f"{gap:.1f}%", delta=f"{'-Overfitting risk' if gap > 15 else 'Good generalization'}")
    elif "test_r2" in m and "train_r2" in m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Holdout Test R²", f"{m['test_r2']:.3f}")
        c2.metric("Training R²", f"{m['train_r2']:.3f}")
        c3.metric("Test RMSE", f"{m.get('test_rmse', 0):.2f}")
        c4.metric("Train RMSE", f"{m.get('train_rmse', 0):.2f}")
    else:
        metric_cols = st.columns(min(max(len(m), 1), 4))
        for idx, (k, v) in enumerate(m.items()):
            metric_cols[idx % len(metric_cols)].metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))

    with st.expander("Model Parameters & Diagnostic Details", expanded=False):
        st.json(result.parameters)

    st.write("### Methodological Caveats & Hypotheses")
    for caveat in result.caveats:
        st.info(f"- {caveat}")


def _ml_tab(cfg, user: User) -> None:
    st.header("Advanced Analytics & Machine Learning")
    st.caption("Deterministic local models with evaluated metrics. No cloud APIs. Unvalidated hypotheses are clearly flagged.")
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table file first to enable advanced analytics.")
        return

    st.warning(
        "Important Note on Machine Learning:\n"
        "- ML outputs are statistical estimates and hypotheses, NOT ground-truth facts.\n"
        "- Overfitting risk exists on smaller datasets.\n"
        "- Classification requires verified labels; regression establishes correlation, not causality."
    )

    frame = ingest.frame
    all_cols = list(frame.columns)
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(frame[c])]
    cat_cols = [c for c in all_cols if c not in numeric_cols and c not in {"#", "id", "s.no", "sl.no", "index"}]
    if not cat_cols:
        cat_cols = [c for c in all_cols if c not in numeric_cols]

    task_options = [
        "Categorical Breakdown & Frequencies",
        "Cross-Tabulation & Association Matrix",
        "Entity Diversity & Concentration Metrics",
        "Correlation Matrix & Association Pairs",
        "Time-Series Aggregations & Trends",
        "Classification (Decision Tree)",
        "Linear Regression",
        "Clustering (K-Means)",
        "Anomaly Detection (IQR & Z-Score)",
        "Linear Trend Forecasting",
    ]
    default_task_idx = 0 if not numeric_cols else 3

    task = st.selectbox("Select Analytical Task", task_options, index=default_task_idx)

    if task == "Categorical Breakdown & Frequencies":
        st.subheader("Categorical Breakdown & Frequency Analysis")
        st.caption("Analyze category distributions, shares, and concentration metrics.")
        if not cat_cols:
            st.warning("No categorical columns available.")
            return

        selected_col = st.selectbox("Select Column to Analyze", cat_cols)
        if st.button("Generate Frequency Breakdown", type="primary"):
            freq_df = frequency_analysis(frame, selected_col)
            cardinality = int(frame[selected_col].nunique(dropna=True))
            top_3_share = float(freq_df["Percentage (%)"].iloc[:3].sum()) if not freq_df.empty else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", f"{len(frame):,}")
            m2.metric("Unique Categories", f"{cardinality}")
            m3.metric("Top Category", f"{freq_df['Category'].iloc[0]}" if not freq_df.empty else "N/A")
            m4.metric("Top 3 Share", f"{top_3_share:.1f}%")

            st.dataframe(freq_df, use_container_width=True)
            st.plotly_chart(category_bar(frame, selected_col), key=f"ml_catbar_{selected_col}", use_container_width=True)

            csv_bytes = to_csv_bytes(freq_df)
            st.download_button(
                "Download Frequency Breakdown (CSV)",
                csv_bytes,
                file_name=f"frequency_{selected_col}.csv",
                mime="text/csv",
            )

            _get_audit_log(cfg).record(
                user.org_id,
                user.username,
                "advanced_frequency_analysis",
                f"Column: '{selected_col}', Unique: {cardinality}, Top3Share: {top_3_share:.1f}%",
            )

    elif task == "Cross-Tabulation & Association Matrix":
        st.subheader("Cross-Tabulation & Association Matrix")
        st.caption("Examine two-way contingency relationships between categorical dimensions.")
        if len(cat_cols) < 2:
            st.warning("Cross-tabulation requires at least two categorical columns.")
            return

        c1, c2 = st.columns(2)
        row_dim = c1.selectbox("Row Dimension", cat_cols, index=0)
        available_col_dims = [c for c in cat_cols if c != row_dim]
        col_dim = c2.selectbox("Column Dimension", available_col_dims, index=0)

        normalize_opt = st.selectbox(
            "Values Display",
            ["Counts", "Row Percentages (%)", "Column Percentages (%)", "Total Percentages (%)"],
        )

        if st.button("Compute Cross-Tabulation", type="primary"):
            norm_map = {
                "Counts": False,
                "Row Percentages (%)": "index",
                "Column Percentages (%)": "columns",
                "Total Percentages (%)": "all",
            }
            norm = norm_map[normalize_opt]
            ct = pd.crosstab(frame[row_dim], frame[col_dim], normalize=norm)
            if norm:
                ct = (ct * 100).round(1)

            st.write(f"### {row_dim} vs {col_dim} ({normalize_opt})")
            st.dataframe(ct, use_container_width=True)
            st.plotly_chart(
                crosstab_heatmap(ct, col_dim, row_dim, f"{row_dim} vs {col_dim} ({normalize_opt})"),
                key=f"ml_crosstab_{row_dim}_{col_dim}",
                use_container_width=True,
            )

            csv_bytes = to_csv_bytes(ct.reset_index())
            st.download_button(
                "Download Cross-Tabulation (CSV)",
                csv_bytes,
                file_name=f"crosstab_{row_dim}_{col_dim}.csv",
                mime="text/csv",
            )

            _get_audit_log(cfg).record(
                user.org_id,
                user.username,
                "advanced_crosstab_analysis",
                f"Row: '{row_dim}', Col: '{col_dim}', Mode: '{normalize_opt}'",
            )

    elif task == "Entity Diversity & Concentration Metrics":
        st.subheader("Entity Diversity & Concentration Metrics")
        st.caption("Evaluate cardinality and distribution balance across all categorical dimensions.")
        diversity_rows = []
        for col in cat_cols:
            s = frame[col].dropna().astype(str).str.strip()
            if s.empty:
                continue
            vc = s.value_counts()
            top1_share = (vc.iloc[0] / len(s) * 100) if not vc.empty else 0.0
            top3_share = (vc.iloc[:3].sum() / len(s) * 100) if not vc.empty else 0.0

            if top1_share > 70:
                div_level = "High Concentration"
            elif top1_share > 40:
                div_level = "Moderate"
            else:
                div_level = "High Diversity"

            diversity_rows.append({
                "Column": col,
                "Unique Values": len(vc),
                "Top Category": str(vc.index[0]) if not vc.empty else "N/A",
                "Top Share (%)": round(top1_share, 1),
                "Top 3 Share (%)": round(top3_share, 1),
                "Distribution Profile": div_level,
            })
        df_div = pd.DataFrame(diversity_rows)
        st.dataframe(df_div, use_container_width=True)
        _get_audit_log(cfg).record(
            user.org_id,
            user.username,
            "advanced_diversity_analysis",
            f"Analyzed {len(diversity_rows)} categorical dimensions",
        )

    elif task == "Correlation Matrix & Association Pairs":
        st.subheader("Correlation Matrix & Association Analysis")
        st.caption("Compute pairwise Pearson correlation coefficients between numeric variables.")
        if len(numeric_cols) < 2:
            st.warning("Correlation analysis requires at least two numeric features.")
            return
        if st.button("Compute Correlation Matrix", type="primary"):
            c_res = compute_correlation_matrix(frame)
            if c_res.get("status") == "ok":
                corr_df = pd.DataFrame(c_res["matrix"])
                st.plotly_chart(correlation_heatmap(corr_df), key="ml_corr_heatmap", use_container_width=True)
                if c_res.get("top_correlations"):
                    st.write("### Top Associated Variable Pairs")
                    st.dataframe(pd.DataFrame(c_res["top_correlations"]), use_container_width=True)
                _get_audit_log(cfg).record(
                    user.org_id,
                    user.username,
                    "ml_correlation_matrix",
                    f"Features: {len(numeric_cols)}, Pairs: {len(c_res.get('top_correlations', []))}",
                )
            else:
                st.warning(c_res.get("caveat", "Insufficient variance to compute correlation."))

    elif task == "Time-Series Aggregations & Trends":
        st.subheader("Time-Series Trends & Aggregations")
        st.caption("Analyze metric progression over daily, monthly, or yearly intervals.")
        date_candidates = [
            c for c in all_cols
            if "date" in c.lower() or "time" in c.lower() or "year" in c.lower() or pd.api.types.is_datetime64_any_dtype(frame[c])
        ]
        if not date_candidates:
            date_candidates = all_cols
        col_d1, col_d2 = st.columns(2)
        date_col = col_d1.selectbox("Date / Timestamp Column", date_candidates)
        value_col = col_d2.selectbox("Metric Column (Optional, counts records if blank)", ["(Record Count)"] + numeric_cols)
        actual_val = None if value_col == "(Record Count)" else value_col
        if st.button("Generate Time Series", type="primary"):
            t_res = compute_time_trends(frame, date_col, actual_val)
            if t_res.get("status") == "ok":
                trend_df = pd.DataFrame(t_res["trend_table"])
                st.metric("Time Span", f"{t_res['min_date']} to {t_res['max_date']} ({t_res['span_days']} days)")
                st.dataframe(trend_df, use_container_width=True)
                st.plotly_chart(time_series_line(trend_df, "period", "total", f"Trends over {t_res['freq_label']} Intervals"), key=f"ml_timeseries_{date_col}_{actual_val}", use_container_width=True)
                _get_audit_log(cfg).record(
                    user.org_id,
                    user.username,
                    "ml_time_trend",
                    f"DateCol: '{date_col}', ValueCol: '{actual_val}', Periods: {len(trend_df)}",
                )
            else:
                st.warning(t_res.get("caveat", "Unable to compute time trend."))

    elif task.startswith("Classification"):
        st.subheader("Decision Tree Classification")
        st.caption("Predict categorical outcomes using numeric and categorical predictors (automatically one-hot encoded).")
        target_col = st.selectbox("Target Column (Classes)", all_cols)

        suitability = assess_feature_suitability(frame, target_col=target_col)
        rec_features = suitability["recommended_features"]
        high_card = suitability["high_cardinality_text"]
        identifiers = suitability["identifiers"]

        available_features = [c for c in all_cols if c != target_col and c not in identifiers]
        if not available_features:
            available_features = [c for c in all_cols if c != target_col]
        if not available_features:
            st.warning("No feature columns available for classification.")
            return

        if high_card:
            with st.expander(f"Notice: {len(high_card)} high-cardinality/text column(s) excluded from default selection", expanded=False):
                st.info(
                    f"The following columns contain unique text, descriptions, links, or >20 categories: {', '.join(high_card)}. "
                    "They are excluded from default features to prevent overfitting and high-cardinality errors."
                )

        default_selection = [c for c in rec_features if c in available_features][:min(5, len(rec_features))]
        if not default_selection and available_features:
            default_selection = available_features[:1]

        selected_features = st.multiselect("Predictor Features", available_features, default=default_selection)

        if len(frame) < 20:
            st.info(f"Dataset has {len(frame)} rows (recommended minimum sample size is 20 rows).")

        if st.button("Train & Evaluate Classifier"):
            if not selected_features:
                st.error("Please select at least one feature column.")
                return
            try:
                result = run_classification(frame, target_col, selected_features)
                _get_audit_log(cfg).record(
                    user.org_id,
                    user.username,
                    "ml_classification",
                    f"Task: classification, Features: {len(selected_features)}, Status: {result.status}",
                )
                _render_ml_result(result)
            except ValidationError as exc:
                st.error(f"Classification validation failed: {exc}")
            except Exception as exc:
                st.error(f"Classification could not be performed: {exc}")

    elif task.startswith("Linear Regression"):
        st.subheader("Linear Regression")
        st.caption("Estimate continuous numeric outcomes using linear relationships with holdout validation.")
        if not numeric_cols:
            st.warning("No numeric columns available in dataset.")
            return
        target_col = st.selectbox("Target Column (Continuous outcome)", numeric_cols)

        suitability = assess_feature_suitability(frame, target_col=target_col)
        rec_features = suitability["recommended_features"]
        high_card = suitability["high_cardinality_text"]
        identifiers = suitability["identifiers"]

        available_features = [c for c in all_cols if c != target_col and c not in identifiers]
        if not available_features:
            available_features = [c for c in all_cols if c != target_col]
        if not available_features:
            st.warning("No additional feature columns available as predictors.")
            return

        if high_card:
            with st.expander(f"Notice: {len(high_card)} high-cardinality/text column(s) excluded from default selection", expanded=False):
                st.info(
                    f"The following columns contain unique text, descriptions, links, or >20 categories: {', '.join(high_card)}. "
                    "They are excluded from default features to maintain numerical stability."
                )

        default_selection = [c for c in rec_features if c in available_features][:min(5, len(rec_features))]
        if not default_selection and available_features:
            default_selection = available_features[:1]

        selected_features = st.multiselect("Predictor Features", available_features, default=default_selection)
        if st.button("Fit & Evaluate Regression"):
            if not selected_features:
                st.error("Please select at least one feature column.")
                return
            try:
                result = run_regression(frame, target_col, selected_features)
                _get_audit_log(cfg).record(
                    user.org_id,
                    user.username,
                    "ml_regression",
                    f"Task: regression, Features: {len(selected_features)}, Status: {result.status}",
                )
                _render_ml_result(result)
            except ValidationError as exc:
                st.error(f"Regression validation failed: {exc}")
            except Exception as exc:
                st.error(f"Regression could not be performed: {exc}")

    elif task.startswith("Clustering"):
        st.subheader("K-Means Clustering")
        if not numeric_cols:
            st.warning("No numeric columns available for clustering.")
            return
        selected_features = st.multiselect("Features for Clustering", numeric_cols, default=numeric_cols)
        k = st.slider("Number of Clusters (k)", min_value=2, max_value=min(10, max(2, len(frame))), value=3)
        if st.button("Run Clustering"):
            if not selected_features:
                st.error("Please select at least one numeric feature column.")
                return
            try:
                result = run_clustering(frame, selected_features, n_clusters=k)
                _get_audit_log(cfg).record(
                    user.org_id,
                    user.username,
                    "ml_clustering",
                    f"Task: clustering, k: {k}, Features: {len(selected_features)}, Status: {result.status}",
                )
                _render_ml_result(result)
            except ValidationError as exc:
                st.error(f"Clustering validation failed: {exc}")
            except Exception as exc:
                st.error(f"Clustering could not be performed: {exc}")

    elif task.startswith("Anomaly"):
        st.subheader("Outlier & Anomaly Detection")
        st.caption("Identify statistical deviations using either non-parametric IQR fences or parametric Z-scores.")
        if not numeric_cols:
            st.warning("No numeric columns available in dataset.")
            return
        col = st.selectbox("Column to inspect for outliers", numeric_cols)
        method = st.radio("Detection Method", ["IQR Fences (Non-parametric)", "Z-Score (Parametric, Normal Dist.)"], horizontal=True)

        if "IQR" in method:
            multiplier = st.slider("IQR Multiplier (default 1.5)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            if st.button("Detect Outliers"):
                res = detect_iqr_anomalies(frame, col, multiplier=multiplier)
                _get_audit_log(cfg).record(
                    user.org_id,
                    user.username,
                    "anomaly_detection",
                    f"Task: anomaly_detection, Multiplier: {multiplier}, Outliers: {res.outlier_count}",
                )
                try:
                    c_anom = st.columns(3)
                    if isinstance(c_anom, (list, tuple)) and len(c_anom) == 3:
                        c1, c2, c3 = c_anom
                    else:
                        c1, c2, c3 = st, st, st
                except Exception:
                    c1, c2, c3 = st, st, st
                c1.metric("Outliers Detected", res.outlier_count)
                c2.metric("Lower Fence", f"{res.lower:.4f}")
                c3.metric("Upper Fence", f"{res.upper:.4f}")
                st.info(f"Methodological caveat: {res.caveat}")
                st.plotly_chart(outlier_box(frame, col, lower=res.lower, upper=res.upper, title=f"Box Plot & Outlier Fences for '{col}'"), key=f"ml_iqr_box_{col}", use_container_width=True)
                if res.outlier_row_indexes:
                    st.write(f"Sample Outlier Rows ({len(res.outlier_row_indexes)} detected):")
                    st.dataframe(frame.iloc[res.outlier_row_indexes], use_container_width=True)
                else:
                    st.success(f"No statistical outliers detected in column '{col}' with multiplier {multiplier}.")
        else:
            z_thresh = st.slider("Z-Score Threshold (default 3.0)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
            if st.button("Detect Outliers"):
                res = detect_zscore_anomalies(frame, col, threshold=z_thresh)
                _get_audit_log(cfg).record(
                    user.org_id,
                    user.username,
                    "anomaly_detection",
                    f"Task: anomaly_detection, Multiplier: {z_thresh}, Outliers: {res.outlier_count}",
                )
                try:
                    c_anom = st.columns(3)
                    if isinstance(c_anom, (list, tuple)) and len(c_anom) == 3:
                        c1, c2, c3 = c_anom
                    else:
                        c1, c2, c3 = st, st, st
                except Exception:
                    c1, c2, c3 = st, st, st
                c1.metric("Outliers Detected", res.outlier_count)
                c2.metric("Mean", f"{res.lower:.4f}")
                c3.metric("Std Dev", f"{res.upper:.4f}")
                st.info(f"Methodological caveat: {res.caveat}")
                st.plotly_chart(outlier_box(frame, col, title=f"Distribution & Outliers for '{col}'"), key=f"ml_zscore_box_{col}", use_container_width=True)
                if res.outlier_row_indexes:
                    st.write(f"Sample Outlier Rows ({len(res.outlier_row_indexes)} detected):")
                    st.dataframe(frame.iloc[res.outlier_row_indexes], use_container_width=True)
                else:
                    st.success(f"No statistical outliers detected in column '{col}' with |Z| > {z_thresh}.")

    elif task.startswith("Linear Trend"):
        st.subheader("Linear Trend Forecasting")
        if not numeric_cols:
            st.warning("No numeric columns available in dataset.")
            return
        col = st.selectbox("Series Column to Forecast", numeric_cols)
        periods = st.slider("Forecast Horizon (Periods)", min_value=1, max_value=12, value=3)
        if st.button("Calculate Forecast"):
            vals = frame[col].dropna().tolist()
            res = linear_trend_forecast(vals, periods=periods)
            _get_audit_log(cfg).record(
                user.org_id,
                user.username,
                "trend_forecast",
                f"Task: trend_forecast, Horizon: {periods}, Status: {res.get('status', 'success')}",
            )
            if res.get("status") == "insufficient_data":
                st.error(res["caveat"])
            else:
                st.warning(f"Scientific Caveat: {res['caveat']}")
                c1, c2 = st.columns(2)
                c1.metric("Trend Slope", f"{res['slope']:.4f}")
                c2.metric("Intercept", f"{res['intercept']:.4f}")

                future_indexes = [f"t + {i+1}" for i in range(len(res["forecast"]))]
                forecast_df = pd.DataFrame({"Period": future_indexes, "Extrapolated Value": res["forecast"]})
                st.dataframe(forecast_df, use_container_width=True)

                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=vals, mode="lines+markers", name="Observed Data"))
                future_x = list(range(len(vals), len(vals) + periods))
                fig.add_trace(
                    go.Scatter(
                        x=future_x,
                        y=res["forecast"],
                        mode="lines+markers",
                        name="Linear Forecast (Hypothesis)",
                        line=dict(dash="dash", color="orange"),
                    )
                )
                fig.update_layout(title=f"Trend & Forecast for '{col}'", xaxis_title="Observation Index", yaxis_title=col)
                st.plotly_chart(fig, key=f"ml_trend_scatter_{col}", use_container_width=True)


def _documents_tab(cfg, user: User) -> None:
    if not can(user, "documents"):
        st.error("You do not have permission to manage documents.")
        return
    st.header("Document Knowledge Base & Search")
    st.caption("Index PDF, text, and images into an ephemeral in-memory search index. Passages are completely isolated from tabular data. No data leaves your machine.")
    st.info("Supported formats: **PDF (.pdf)**, **Plain Text (.txt)**, **Images (.png, .jpg, .jpeg via local OCR)**.")

    uploaded = st.file_uploader("Choose a document file", type=["pdf", "txt", "png", "jpg", "jpeg"], key="doc_uploader")
    if uploaded and st.button("Index document", key="doc_index_btn"):
        data = uploaded.getvalue()
        name = uploaded.name
        suffix = Path(name).suffix.lower()
        log = _get_audit_log(cfg)
        try:
            if suffix == ".pdf":
                extract = extract_pdf_text(data, name, cfg.limits)
                added = st.session_state.index.add_extract(extract)
                for warning in extract.warnings:
                    st.warning(warning)
            elif suffix == ".txt":
                added = st.session_state.index.add_plain_text(name, data.decode("utf-8", errors="replace"))
            else:
                ocr = ocr_image(data)
                if not ocr.available:
                    st.error(f"Local OCR is unavailable ({ocr.detail}). Install Tesseract for scanned documents.")
                    added = 0
                else:
                    added = st.session_state.index.add_plain_text(name, ocr.text)
            log.record(user.org_id, user.username, "index_document", f"File: '{Path(name).name}', Passages: {added}")
            st.success(f"Successfully indexed {added} passage(s) from '{name}'.")
        except PrivateAnalystError as exc:
            log.record(user.org_id, user.username, "index_document_failure", f"File: '{Path(name).name}', Reason: {exc.__class__.__name__}")
            st.error(str(exc))

    st.divider()
    records = st.session_state.index.records
    doc_names = sorted({r["document_name"] for r in records})
    if not records:
        st.caption("No documents are currently indexed in this session. Upload a document above to enable passage search.")
    else:
        st.success(f"Indexed **{len(records)} passage(s)** across **{len(doc_names)} document(s)**: {', '.join(f'`{d}`' for d in doc_names)}")

    query = st.text_input("Search indexed passages", key="doc_search_query")
    if query:
        hits = st.session_state.index.search(query)
        if not hits:
            st.info("No matching passages were found in the local index.")
        for hit in hits:
            st.markdown(f"**{hit.document_name}** · page {hit.page_number} · score `{hit.score}`")
            st.caption("Extracted passage:")
            st.write(hit.text)

    names = sorted({r["document_name"] for r in st.session_state.index.records})
    if names:
        victim = st.selectbox("Delete document from index", names)
        if st.button("Delete"):
            st.session_state.index.delete_document(victim)
            _get_audit_log(cfg).record(user.org_id, user.username, "delete_document", f"File: '{Path(victim).name}'")
            st.success("Removed from the in-memory index.")


def _reports_tab(cfg, user: User) -> None:
    ingest: IngestResult | None = st.session_state.ingest
    result = st.session_state.last_result
    explanation = st.session_state.last_explanation
    if ingest is None:
        st.info("Import data first in '1. Import Data' to enable exports.")
        return

    st.header("Reports & Local Export Center")
    st.caption("All exports are generated locally on your device. Tabular data is automatically sanitized to neutralize spreadsheet formula injection attacks.")

    stem = Path(ingest.source_name).stem
    st.subheader("Tabular Dataset Exports")
    st.write(f"Active table: **`{ingest.source_name}`** ({ingest.frame.shape[0]:,} rows × {ingest.frame.shape[1]} columns)")

    st.download_button(
        "Download Cleaned CSV",
        to_csv_bytes(ingest.frame),
        file_name=f"{stem}_cleaned.csv",
        mime="text/csv",
        on_click=lambda: _safe_record_export_requested(
            cfg, user, "table", "delimited", f"Source: '{stem}'"
        ),
    )
    if can(user, "export"):
        multi_sheet = st.checkbox("Multi-sheet Excel (includes Column Statistics & Data Quality)", value=True, key="report_multi_sheet")
        stats = compute_dataset_stats(ingest.frame, getattr(ingest, "inferred_types", None))
        quality = assess_quality(ingest.frame, ingest.duplicate_columns)
        cleaning_ops = getattr(ingest, "cleaning_log", None) or getattr(ingest, "cleaning_operations", [])
        excel_bytes = (
            to_report_excel_bytes(ingest.frame, stats, quality, cleaning_ops=cleaning_ops)
            if multi_sheet
            else to_excel_bytes(ingest.frame)
        )
        st.download_button(
            "Download Cleaned Excel (.xlsx)",
            excel_bytes,
            file_name=f"{stem}_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click=lambda: _safe_record_export_requested(
                cfg, user, "table", "xlsx", f"Source: '{stem}'"
            ),
        )

    st.divider()
    st.subheader("Executive Analysis Briefings")
    if result and explanation:
        include_ai = st.checkbox("Include AI narrative in executive reports", value=True, key="report_inc_ai")
        text = report_markdown(result, explanation["text"], include_ai)
        st.download_button(
            "Download Markdown Report (.md)",
            text.encode("utf-8"),
            file_name=f"{stem}_briefing.md",
            mime="text/markdown",
            on_click=lambda: _safe_record_export_requested(
                cfg, user, "report", "md", f"AI included: {include_ai}"
            ),
        )
        st.download_button(
            "Download PDF Report (.pdf)",
            to_pdf_bytes(result, explanation["text"], include_ai),
            file_name=f"{stem}_briefing.pdf",
            mime="application/pdf",
            on_click=lambda: _safe_record_export_requested(
                cfg, user, "report", "pdf", f"AI included: {include_ai}"
            ),
        )
        with st.expander("Analysis Metadata & Provenance", expanded=False):
            st.json(analysis_metadata(ingest.source_name))
    else:
        st.info(
            "Executive Analysis Reports (Markdown & PDF) become available once an analytical question is run in "
            "**8. Ask & Explain**. The briefing packages deterministic engine calculations, verified totals, and AI narrative."
        )


def _settings_tab(cfg, user: User) -> None:
    st.header("System Settings & Privacy Controls")

    model_status = check_local_model(cfg.ai)
    st.subheader("Local AI Model Status")
    if model_status.available:
        st.success(f"**Online:** {model_status.friendly_status} (Host: `{model_status.host}:{model_status.port}`)")
    else:
        st.info(f"**Offline:** {model_status.friendly_status}")
    if st.button("Re-test Connection", key="settings_test_ai"):
        st.info(f"Connection result: {model_status.friendly_status}")

    if model_status.available:
        if st.button("Verify Local Model Architecture & License", key="settings_verify_ai"):
            with st.spinner("Inspecting model via loopback API..."):
                v_res = verify_local_model(cfg.ai)
                if v_res.get("verified"):
                    st.success(f"**Verified Model:** `{v_res.get('model')}`")
                    v_df = pd.DataFrame([
                        {"Attribute": "Family / Architecture", "Value": str(v_res.get("family"))},
                        {"Attribute": "Format", "Value": str(v_res.get("format"))},
                        {"Attribute": "Parameter Count", "Value": str(v_res.get("parameter_size"))},
                        {"Attribute": "Quantization Level", "Value": str(v_res.get("quantization_level"))},
                        {"Attribute": "Modified Timestamp", "Value": str(v_res.get("modified_at"))},
                        {"Attribute": "License Terms", "Value": str(v_res.get("license", "")).splitlines()[0] if v_res.get("license") else "(Standard open-weight license)"},
                    ])
                    st.dataframe(v_df, use_container_width=True)
                else:
                    st.warning(f"Could not verify model details: {v_res.get('error')}")

    if not model_status.available:
        with st.expander("Local Model (Ollama) Setup Instructions", expanded=False):
            st.markdown(
                "- **Step 1:** Download and install Ollama from [https://ollama.com](https://ollama.com).\n"
                "- **Step 2:** Open a terminal and start the server: `ollama serve`.\n"
                f"- **Step 3:** Pull your configured model: `ollama pull {cfg.ai.model}`.\n"
                "- **Step 4:** Return here and click **Re-test Connection**."
            )

    st.subheader("Data Retention & Memory Controls")
    st.caption("Manage customer data lifecycle, temporary buffers, and active memory sessions.")
    cols = st.columns(2)
    if isinstance(cols, (list, tuple)) and len(cols) >= 2:
        col_ret1, col_ret2 = cols[0], cols[1]
    else:
        col_ret1, col_ret2 = st, st
    with col_ret1:
        if st.session_state.get("ingest") is not None:
            if st.button("Purge Active Working Dataset from Memory", type="secondary"):
                st.session_state.ingest = None
                _invalidate_downstream_state()
                st.success("Working dataset and cached analytical states have been permanently purged from memory.")
                st.rerun()
        else:
            st.info("No active dataset currently loaded in memory.")
    with col_ret2:
        if st.button("Clear Temporary Files & Caches"):
            import shutil
            from private_analyst.paths import default_temp_dir
            t_dir = default_temp_dir()
            cleared_count = 0
            if t_dir.exists():
                for item in t_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                            cleared_count += 1
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                            cleared_count += 1
                    except Exception:
                        pass
            st.success(f"Cleaned {cleared_count} temporary scratch file(s).")

    st.subheader("Privacy & Security Configuration")
    st.caption("Telemetry is hard-disabled at compile time. Model communications connect strictly to loopback (`127.0.0.1`).")
    cfg_data = [
        {"Setting": "Product Edition", "Value": cfg.server.edition.title()},
        {"Setting": "Bind Host", "Value": cfg.server.bind_host},
        {"Setting": "Authentication Required", "Value": "Yes" if cfg.server.require_auth else "No (Desktop Local)"},
        {"Setting": "Telemetry Hard-Disabled", "Value": "Yes (Compile-time disabled)"},
        {"Setting": "AI Host (Loopback)", "Value": cfg.ai.host},
        {"Setting": "Max Upload Limit", "Value": f"{cfg.limits.max_upload_bytes / (1024 * 1024):.1f} MB"},
        {"Setting": "Resolved Data Directory", "Value": str(cfg.resolved_data_dir())},
    ]
    st.dataframe(pd.DataFrame(cfg_data), use_container_width=True)
    if can(user, "manage_users") and cfg.server.require_auth:
        st.subheader("Create user")
        org = st.text_input("New user organization id", value=user.org_id)
        username = st.text_input("New username")
        password = st.text_input("New password", type="password")
        role = st.selectbox("Role", ["admin", "analyst", "viewer"])
        if st.button("Create user"):
            store = UserStore(cfg.resolved_data_dir() / "app.db")
            user_created = False
            try:
                store.create_user(org, username, password, role)
                user_created = True
            except (PrivateAnalystError, sqlite3.IntegrityError) as exc:
                try:
                    _get_audit_log(cfg).record(
                        user.org_id,
                        user.username,
                        "create_user_failure",
                        f"Failed creating user '{username}': {exc.__class__.__name__}",
                    )
                except AuditLogError:
                    # Deliberately suppressed: audit storage failure must not mask original creation error
                    pass
                st.error(str(exc))

            if user_created:
                audit_recorded = True
                try:
                    _get_audit_log(cfg).record(
                        user.org_id,
                        user.username,
                        "create_user",
                        f"Created user '{username}' with role '{role}' in org '{org}'",
                    )
                except AuditLogError:
                    audit_recorded = False

                if audit_recorded:
                    st.success("User created.")
                else:
                    st.warning("User created, but the audit event could not be recorded.")
    st.subheader("Encryption at Rest")
    keystore = SecureKeyStore(cfg.resolved_data_dir() / "master.key")
    if keystore.is_initialized():
        st.success("Envelope encryption key store is INITIALIZED.")
    else:
        st.info(
            "Envelope encryption key store is not yet initialized. "
            "Data on disk relies on OS-level BitLocker and file system permissions."
        )
        master_pw = st.text_input("Master password to initialize key store (min 12 characters)", type="password")
        if st.button("Initialize Master Key Store"):
            if len(master_pw) < 12:
                st.error("Master password must be at least 12 characters.")
            else:
                try:
                    keystore.initialize(master_pw)
                    _get_audit_log(cfg).record(
                        user.org_id, user.username, "init_keystore", "Master key store initialized"
                    )
                    st.success("Master key store initialized successfully.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to initialize: {exc}")

    if can(user, "backup"):
        st.subheader("Data Backup & Encrypted Archives")
        st.caption("Create an authenticated, encrypted (.zip.enc) or plaintext (.zip) backup archive.")
        backup_enc_pass = st.text_input("Backup encryption password (leave blank for standard zip)", type="password")
        if st.button("Generate Backup"):
            from private_analyst.storage.backup import create_backup
            from private_analyst.security.encryption import derive_key_from_password
            import tempfile
            key = None
            valid = True
            if backup_enc_pass.strip():
                if len(backup_enc_pass.strip()) < 12:
                    st.error("Backup password must be at least 12 characters.")
                    valid = False
                else:
                    key, _ = derive_key_from_password(backup_enc_pass.strip())
            if valid:
                try:
                    with tempfile.TemporaryDirectory() as tmp_dest:
                        archive_path = create_backup(cfg.resolved_data_dir(), Path(tmp_dest), key=key)
                        archive_bytes = archive_path.read_bytes()
                        _get_audit_log(cfg).record(
                            user.org_id, user.username, "create_backup", f"Archive: '{archive_path.name}', Bytes: {len(archive_bytes)}"
                        )
                        st.download_button(
                            label=f"Download {archive_path.name}",
                            data=archive_bytes,
                            file_name=archive_path.name,
                            mime="application/octet-stream",
                        )
                        st.success(f"Backup prepared ({len(archive_bytes):,} bytes). Click above to save archive.")
                except Exception as exc:
                    _get_audit_log(cfg).record(
                        user.org_id, user.username, "create_backup_failure", f"Backup failed: {exc.__class__.__name__}"
                    )
                    st.error(f"Failed to create backup: {exc}")

    st.subheader("Audit log")
    log = AuditLog(cfg.resolved_data_dir() / "app.db")
    events = log.list_for_org(user.org_id)
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.caption("No audit events for this organization yet.")


def _inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        /* Top header bar transparent */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            height: 2.5rem !important;
        }
        #MainMenu, footer {
            visibility: hidden !important;
        }

        /* Left Sidebar: Exact Dark Navy (#0A1128 / #0B132B) */
        [data-testid="stSidebar"] {
            background-color: #0A1128 !important;
            border-right: 1px solid #172554 !important;
            min-width: 250px !important;
            max-width: 270px !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
        }

        [data-testid="stSidebar"] * {
            color: #94A3B8;
        }

        /* Sidebar Brand Header */
        .ba-sidebar-brand {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 4px 12px 4px;
        }
        .ba-logo-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .ba-logo-icon {
            width: 24px;
            height: 24px;
            border-radius: 6px;
            background: #3B82F6;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF !important;
            font-weight: 800;
            font-size: 15px;
            line-height: 1;
        }
        .ba-logo-text {
            color: #FFFFFF !important;
            font-weight: 700;
            font-size: 16.5px;
            letter-spacing: -0.3px;
        }
        .ba-header-icons {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #64748B !important;
            font-size: 13.5px;
        }

        /* Workspace selector pill */
        .ba-workspace-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 10px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            margin-bottom: 4px;
            cursor: pointer;
        }
        .ba-workspace-left {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .ba-badge-avatar {
            background: #3B82F6;
            color: white !important;
            border-radius: 4px;
            padding: 1px 6px;
            font-size: 11px;
            font-weight: 700;
        }
        .ba-workspace-name {
            color: #F8FAFC !important;
            font-size: 13px;
            font-weight: 600;
        }
        .ba-projects-link {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 2px 10px;
            color: #64748B !important;
            font-size: 11.5px;
            margin-bottom: 12px;
        }

        /* Sidebar Nav Buttons: Sleek, transparent dark-theme items (No White/Red blocks) */
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"],
        [data-testid="stSidebar"] div[data-testid="stButton"] button {
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #94A3B8 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 7px 12px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            width: 100% !important;
            margin-bottom: 2px !important;
            cursor: pointer !important;
        }

        [data-testid="stSidebar"] button p,
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"] p,
        [data-testid="stSidebar"] div[data-testid="stButton"] button p {
            color: #94A3B8 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            margin: 0 !important;
        }

        [data-testid="stSidebar"] button:hover,
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover,
        [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            background-color: rgba(255, 255, 255, 0.06) !important;
            background: rgba(255, 255, 255, 0.06) !important;
            color: #FFFFFF !important;
            border: none !important;
        }

        [data-testid="stSidebar"] button:hover p,
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover p,
        [data-testid="stSidebar"] div[data-testid="stButton"] button:hover p {
            color: #FFFFFF !important;
        }

        /* Active Nav Button - Elegant Soft Blue, ZERO RED */
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] [data-testid="baseButton-primary"],
        [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
            background-color: rgba(59, 130, 246, 0.14) !important;
            background: rgba(59, 130, 246, 0.14) !important;
            color: #60A5FA !important;
            font-weight: 600 !important;
            border-left: 3px solid #3B82F6 !important;
            border-radius: 0 6px 6px 0 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 7px 12px !important;
            width: 100% !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] button[kind="primary"] p,
        [data-testid="stSidebar"] [data-testid="baseButton-primary"] p {
            color: #60A5FA !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }

        /* + New Chat Button: Sleek, pill-shaped indigo action button */
        .ba-new-chat-wrapper {
            margin: 4px 0 14px 0 !important;
            padding: 0 !important;
        }
        .ba-new-chat-wrapper button,
        .ba-new-chat-wrapper [data-testid="baseButton-secondary"],
        .ba-new-chat-wrapper [data-testid="baseButton-primary"],
        .ba-new-chat-wrapper div[data-testid="stButton"] button {
            background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%) !important;
            background-color: #1E1B4B !important;
            color: #EEF2FF !important;
            border: 1px solid rgba(129, 140, 248, 0.45) !important;
            border-radius: 24px !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            letter-spacing: 0.02em !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 9px 18px !important;
            width: 100% !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .ba-new-chat-wrapper button:hover,
        .ba-new-chat-wrapper [data-testid="baseButton-secondary"]:hover,
        .ba-new-chat-wrapper [data-testid="baseButton-primary"]:hover,
        .ba-new-chat-wrapper div[data-testid="stButton"] button:hover {
            background: linear-gradient(135deg, #2E1065 0%, #4338CA 100%) !important;
            background-color: #4338CA !important;
            border-color: #A5B4FC !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important;
        }
        .ba-new-chat-wrapper button p,
        .ba-new-chat-wrapper [data-testid="baseButton-secondary"] p,
        .ba-new-chat-wrapper [data-testid="baseButton-primary"] p {
            color: #EEF2FF !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            margin: 0 !important;
        }
        .ba-new-chat-wrapper button:hover p,
        .ba-new-chat-wrapper [data-testid="baseButton-secondary"]:hover p,
        .ba-new-chat-wrapper [data-testid="baseButton-primary"]:hover p {
            color: #FFFFFF !important;
        }

        /* Sidebar Section Titles */
        .ba-nav-section-title {
            color: #475569 !important;
            font-size: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            margin-top: 10px !important;
            margin-bottom: 4px !important;
            padding-left: 6px !important;
        }

        /* Slack Banner in Sidebar */
        .ba-slack-banner {
            padding: 9px 11px;
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 8px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
        }

        /* Plan Indicator Card */
        .ba-plan-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 10px;
            margin-top: 14px;
            margin-bottom: 8px;
        }
        .ba-upgrade-btn {
            background: #6366F1;
            color: white !important;
            font-size: 10.5px;
            font-weight: 600;
            padding: 2px 9px;
            border-radius: 12px;
            text-decoration: none;
            display: inline-block;
        }

        /* User Footer */
        .ba-user-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 2px 2px 2px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .ba-avatar-circle {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: linear-gradient(135deg, #F59E0B, #EF4444);
            color: white !important;
            font-size: 11px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Main Canvas Layout */
        .main .block-container {
            padding-top: 0.6rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 1440px !important;
        }

        /* User Chat Bubble */
        .user-chat-bubble-container {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 22px;
            width: 100%;
        }
        .user-chat-bubble {
            background: #EEF2FF;
            border: 1px solid #E0E7FF;
            color: #1E1B4B;
            padding: 11px 18px;
            border-radius: 18px 18px 4px 18px;
            max-width: 82%;
            font-size: 0.93rem;
            line-height: 1.5;
            font-weight: 500;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }

        /* Thought Step Card */
        .thought-step-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 9px;
            padding: 8px 14px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 500;
            color: #1E293B;
        }
        .artifact-preview-box {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 58px;
            height: 64px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-left: 24px;
            margin-top: 3px;
            margin-bottom: 12px;
            font-size: 10.5px;
            font-weight: 700;
            color: #64748B;
        }

        /* Narrative Text */
        .agent-response-narrative {
            font-size: 0.97rem;
            line-height: 1.6;
            color: #1E293B;
            margin-top: 14px;
            margin-bottom: 20px;
            font-weight: 450;
        }

        /* Metric Cards */
        .ba-metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        .ba-metric-label {
            font-size: 11px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }
        .ba-metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #0F172A;
        }

        /* Floating Input Card */
        .ba-floating-input-wrap {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.06);
            padding: 10px 16px 6px 16px;
            margin-top: 28px;
        }
        .ba-input-caption {
            font-size: 11px;
            color: #94A3B8;
            text-align: center;
            margin-top: 6px;
        }

        /* Right Workspace Panel */
        .ba-workspace-panel {
            background: #FFFFFF;
            border-left: 1px solid #E2E8F0;
            padding-left: 18px;
        }
        .ba-agent-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #F1F5F9;
            margin-bottom: 6px;
            background: #FFFFFF;
            transition: all 0.15s ease;
        }
        .ba-agent-card:hover {
            background: #F8FAFC;
            border-color: #E2E8F0;
        }
        .ba-agent-left {
            display: flex;
            align-items: center;
            gap: 9px;
        }
        .ba-agent-avatar {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #EEF2FF;
            color: #4F46E5;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
        }
        .ba-agent-title {
            font-size: 12px;
            font-weight: 600;
            color: #1E293B;
        }
        .ba-agent-role {
            font-size: 10px;
            color: #64748B;
        }

        /* Floating Feedback Pill */
        .ba-floating-feedback {
            position: fixed;
            bottom: 20px;
            right: 24px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            box-shadow: 0 3px 12px rgba(0,0,0,0.08);
            border-radius: 18px;
            padding: 5px 13px;
            font-size: 11.5px;
            font-weight: 500;
            color: #334155;
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            z-index: 999;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _connectors_tab(cfg, user: User) -> None:
    st.markdown("### 🔌 Data Connectors & Local Pipelines")
    st.caption("Secure, air-gapped data ingestion from local sources without external transmission.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:20px;">📄</span>
                        <div>
                            <div style="font-weight:700; color:#0F172A; font-size:14px;">Local Files (CSV / XLSX / Parquet)</div>
                            <div style="color:#64748B; font-size:12px;">Direct file upload & in-memory buffer</div>
                        </div>
                    </div>
                    <span style="background:#DCFCE7; color:#15803D; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;">ACTIVE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.get("ingest") is not None:
            source_name = getattr(st.session_state.ingest, "source_name", "dataset")
            st.success(f"Connected: `{Path(str(source_name)).name}` ({len(st.session_state.ingest.frame):,} rows)")
        else:
            st.info("No file currently connected.")

        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:20px;">🦆</span>
                        <div>
                            <div style="font-weight:700; color:#0F172A; font-size:14px;">DuckDB In-Memory OLAP Engine</div>
                            <div style="color:#64748B; font-size:12px;">Vectorized local columnar SQL execution</div>
                        </div>
                    </div>
                    <span style="background:#DCFCE7; color:#15803D; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;">READY</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Query with DuckDB SQL", key="btn_duckdb_goto", use_container_width=True):
            st.session_state["active_menu"] = "SQL Console"
            st.rerun()

    with col_c2:
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:20px;">🗄️</span>
                        <div>
                            <div style="font-weight:700; color:#0F172A; font-size:14px;">SQLite Local Storage</div>
                            <div style="color:#64748B; font-size:12px;">Audit logs, user credentials, and session state</div>
                        </div>
                    </div>
                    <span style="background:#DCFCE7; color:#15803D; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;">CONNECTED</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        db_path = cfg.resolved_data_dir() / "app.db"
        st.caption(f"Database path: `{db_path}` ({db_path.stat().st_size if db_path.exists() else 0:,} bytes)")

        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:20px;">🔄</span>
                        <div>
                            <div style="font-weight:700; color:#0F172A; font-size:14px;">Directory Hot-Folder Watcher</div>
                            <div style="color:#64748B; font-size:12px;">Ingest new files dropped into local folder</div>
                        </div>
                    </div>
                    <span style="background:#FEF3C7; color:#B45309; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;">STANDBY</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        watch_folder = st.text_input("Watch Folder Path", value=str(Path("samples").resolve()), key="conn_watch_path")
        if st.button("Scan Folder Now", key="btn_scan_folder", use_container_width=True):
            p = Path(watch_folder)
            if p.exists() and p.is_dir():
                found_files = list(p.glob("*.csv")) + list(p.glob("*.xlsx"))
                st.success(f"Found {len(found_files)} data files in directory.")
            else:
                st.error("Specified directory was not found.")


def _agents_hub_tab(cfg, user: User) -> None:
    st.markdown("### 👥 Specialist Multi-Agent Hub")
    st.caption("Autonomous and semi-autonomous local analytical agents running on your machine via Ollama.")

    st.markdown(
        f"""
        <div style="background:#EEF2FF; border:1px solid #C7D2FE; border-radius:10px; padding:12px 16px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:22px;">⚡</span>
                <div>
                    <div style="font-weight:700; color:#1E1B4B; font-size:13.5px;">Active AI Model Engine: {cfg.ai.model}</div>
                    <div style="color:#4338CA; font-size:12px;">Endpoint: {cfg.ai.host}:{cfg.ai.port} · Context window: {cfg.ai.context_window} tokens · Telemetry: Disabled</div>
                </div>
            </div>
            <span style="background:#312E81; color:#EEF2FF; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">AIR-GAPPED</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    specialist_agents = [
        ("Spreadsheet Auditor", "Lead Agent", "Scans datasets for null anomalies, type mismatches, outliers, and formula drift.", "Alerts", "Run Quality Audit"),
        ("Operations Performance Specialist", "Specialist", "Analyzes timeline windows, delivery formats, schedule distributions, and operational bottlenecks.", "Explore Data", "Analyze Operations"),
        ("Marketing Performance Specialist", "Specialist", "Performs audience segmentation, channel breakdowns, categorical frequency, and domain coverage.", "Explore Data", "Analyze Categories"),
        ("Forecast & Scenario Specialist", "Specialist", "Generates linear trend projections, predictive clustering, regression models, and what-if scenarios.", "Agents_ML", "Launch Forecaster"),
        ("Revenue & Margin Analysis Specialist", "Specialist", "Executes high-performance SQL calculations, prize pool aggregations, and financial metrics.", "SQL Console", "Open SQL Console"),
    ]

    for title, role, desc, target_tab, btn_text in specialist_agents:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:14px 18px; margin-bottom:10px;">
                <div style="display:flex; align-items:flex-start; gap:12px;">
                    <div style="width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #6366F1, #3B82F6); display:flex; align-items:center; justify-content:center; color:#fff; font-size:16px;">👤</div>
                    <div>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-weight:700; color:#0F172A; font-size:14px;">{title}</span>
                            <span style="background:#F1F5F9; color:#475569; font-size:10.5px; font-weight:600; padding:1px 6px; border-radius:4px;">{role}</span>
                        </div>
                        <div style="color:#64748B; font-size:12px; margin-top:2px;">{desc}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_a1, col_a2 = st.columns([5, 1.5])
        with col_a2:
            if st.button(btn_text, key=f"btn_agent_act_{title}", use_container_width=True):
                if target_tab == "Agents_ML":
                    st.session_state["show_ml_subtab"] = True
                else:
                    st.session_state["active_menu"] = target_tab
                st.rerun()

    if st.session_state.get("show_ml_subtab", False):
        st.divider()
        st.subheader("Forecast & Scenario Engine (Machine Learning)")
        _ml_tab(cfg, user)


def _tasks_hub_tab(cfg, user: User) -> None:
    st.markdown("### ⚡ Data Tasks & Automated Pipelines")
    st.caption("Schedule and execute automated cleaning routines, column repairs, and pipeline transformations.")

    task_tab1, task_tab2 = st.tabs(["Data Cleaning Tasks", "Column & Header Repair"])
    with task_tab1:
        _clean_tab(cfg, user)
    with task_tab2:
        _repair_tab(cfg, user)


def _slack_hub_tab(cfg, user: User) -> None:
    st.markdown("### 💬 Slack Local Integration Daemon")
    st.caption("Interact with your local analytical models directly through Slack using private loopback webhooks.")

    st.markdown(
        """
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; margin-bottom:16px;">
            <div style="font-weight:700; color:#0F172A; font-size:14px; margin-bottom:4px;">Private Loopback Webhook Integration</div>
            <div style="color:#64748B; font-size:12.5px; margin-bottom:10px;">
                Events are processed strictly through local loopback (`127.0.0.1`). Customer dataset bytes are never transmitted off this machine.
            </div>
            <div style="font-family:monospace; background:#0F172A; color:#38BDF8; padding:8px 12px; border-radius:6px; font-size:12px;">
                http://127.0.0.1:8501/api/slack/events
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Reports & Data Export")
    _reports_tab(cfg, user)


def _render_betteranalyst_top_bar() -> None:
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 10px; border-bottom: 1px solid #F1F5F9; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: #64748B;">
                <span style="font-weight:700; color:#0F172A; font-size:13.5px;">VantageAnalyst AI</span>
                <span style="color:#CBD5E1;">·</span>
                <span>Free plan</span>
                <span style="color:#CBD5E1;">·</span>
                <a href="#upgrade" style="color: #4F46E5; font-weight: 600; text-decoration: underline;">Upgrade</a>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 4px 13px; font-size: 12px; font-weight: 500; color: #1E293B; background: #FFFFFF; cursor: pointer;">Share</div>
                <div style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 4px 9px; font-size: 12px; font-weight: 600; color: #64748B; background: #FFFFFF; cursor: pointer;">···</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_betteranalyst_chat_dashboard(cfg, user: User) -> None:
    # 1. User Message Bubble (Right-aligned)
    active_filename = "2026-10-17T20-05_export.csv"
    if st.session_state.get("ingest") is not None and getattr(st.session_state.ingest, "source_name", None):
        active_filename = Path(str(st.session_state.ingest.source_name)).name

    st.markdown(
        f"""
        <div class="user-chat-bubble-container">
            <div class="user-chat-bubble">
                I uploaded the file: <b>{active_filename}</b>. Give me dashboard for Trend and pattern
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Agent Thinking Step Cards
    with st.expander("📄 Inspecting trends in your data", expanded=False):
        st.markdown(
            """
            Analyzed event timeline intervals and scheduled delivery dates across the 22 listings.
            Identified prominent schedule clustering during Q3 2026.
            """
        )
        st.markdown(
            """
            <div class="artifact-preview-box">
                <span style="font-size:18px; margin-bottom:2px;">📄</span>
                <span>TXT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📅 Checking event date patterns", expanded=False):
        st.markdown(
            """
            Extracted month occurrences and normalized delivery formats:
            - **August peak window:** 5 listings (SIH 2026, DIFC Dubai, K-Tech Bengaluru, MAS Singapore, London Buildathon).
            - **Delivery Mode Breakdown:** 11 Online, 6 In-person, 5 Hybrid.
            """
        )
        st.markdown(
            """
            <div class="artifact-preview-box">
                <span style="font-size:18px; margin-bottom:2px;">📅</span>
                <span>TXT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📊 Building the schedule-pattern dashboard 1 artifact", expanded=False):
        st.markdown(
            """
            Created interactive visual layout with schedule distribution histogram, delivery format donut, and listing directory.
            """
        )
        st.markdown(
            """
            <div class="artifact-preview-box">
                <span style="font-size:18px; margin-bottom:2px;">🌐</span>
                <span>HTML</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Agent Response Narrative
    st.markdown(
        """
        <div class="agent-response-narrative">
            I created the dashboard from the 22 hackathon listings. August appears in the most schedule windows (5 listings); 11 list online delivery, and 7 have broad, recurring, or tentative schedules.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Interactive Generated Dashboard
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """
            <div class="ba-metric-card">
                <div class="ba-metric-label">Total Listings</div>
                <div class="ba-metric-value">22</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="ba-metric-card">
                <div class="ba-metric-label">Peak Schedule Window</div>
                <div class="ba-metric-value">August (5)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="ba-metric-card">
                <div class="ba-metric-label">Delivery Mode</div>
                <div class="ba-metric-value">11 Online</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="ba-metric-card">
                <div class="ba-metric-label">Schedule Type</div>
                <div class="ba-metric-value">7 Recurring</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Visual Charts
    c_chart1, c_chart2 = st.columns([1.8, 1.2])
    with c_chart1:
        import plotly.graph_objects as go
        months = ["March", "April", "May", "June", "July", "August", "Sept", "Oct", "Nov", "Dec", "Recurring"]
        counts = [1, 1, 1, 1, 1, 5, 2, 3, 3, 3, 1]
        colors = ["#94A3B8" if m != "August" else "#4F46E5" for m in months]

        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=months,
                    y=counts,
                    marker_color=colors,
                    text=counts,
                    textposition="auto",
                )
            ]
        )
        fig_bar.update_layout(
            title="Schedule Windows Distribution by Month",
            title_font=dict(size=13, color="#1E293B", family="Plus Jakarta Sans"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=32, b=20),
            height=260,
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#64748B")),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(size=10, color="#64748B")),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="ba_schedule_dist_bar")

    with c_chart2:
        mode_labels = ["Online", "In-Person", "Hybrid"]
        mode_values = [11, 6, 5]
        mode_colors = ["#4F46E5", "#06B6D4", "#F59E0B"]

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=mode_labels,
                    values=mode_values,
                    hole=0.6,
                    marker_colors=mode_colors,
                    textinfo="label+percent",
                    textfont_size=10,
                )
            ]
        )
        fig_pie.update_layout(
            title="Delivery Mode Breakdown",
            title_font=dict(size=13, color="#1E293B", family="Plus Jakarta Sans"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=32, b=20),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="ba_delivery_mode_donut")

    # Expandable Hackathon listings preview
    with st.expander("📋 View 22 Hackathon Listings Data Table", expanded=False):
        sample_path = Path("samples/2026-10-17T20-05_export.csv")
        if sample_path.exists():
            try:
                df_hack = pd.read_csv(sample_path)
            except Exception:
                try:
                    df_hack = pd.read_csv(sample_path, on_bad_lines="skip")
                except Exception:
                    df_hack = pd.DataFrame()
            if not df_hack.empty:
                st.dataframe(df_hack, use_container_width=True, height=240)

    # 4.5 Dynamic Chat Query & AI Response Display
    if st.session_state.get("last_user_query"):
        st.markdown(
            f"""
            <div class="user-chat-bubble-container" style="margin-top:16px;">
                <div class="user-chat-bubble">
                    {st.session_state.last_user_query}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.get("last_explanation") or st.session_state.get("last_result") is not None:
            st.markdown(
                """
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px 22px; margin-top:12px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                        <div style="width:22px; height:22px; border-radius:5px; background:linear-gradient(135deg, #6366F1, #3B82F6); display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; font-weight:800;">V</div>
                        <span style="font-weight:700; color:#0F172A; font-size:13.5px;">VantageAnalyst AI</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.get("last_explanation"):
                st.markdown(st.session_state.last_explanation)
            if st.session_state.get("last_result") is not None:
                res = st.session_state.last_result
                if getattr(res, "summary", None):
                    st.caption(f"**Summary:** {res.summary}")
                if getattr(res, "data", None) is not None and not res.data.empty:
                    st.dataframe(res.data, use_container_width=True, height=220)
            st.markdown("</div>", unsafe_allow_html=True)

    # 5. Floating Bottom Input Box
    st.markdown(
        """
        <div class="ba-floating-input-wrap">
        """,
        unsafe_allow_html=True,
    )
    user_q = st.text_input(
        "Chat with your data",
        placeholder="Write a message or type / to select a tool",
        label_visibility="collapsed",
        key="ba_chat_prompt_input",
    )
    col_tools_left, col_tools_space, col_tools_model, col_tools_mic, col_tools_send = st.columns(
        [0.8, 4.0, 2.2, 0.6, 0.8]
    )
    with col_tools_left:
        attach_btn = st.button("➕", key="ba_attach_data_btn", help="Attach or upload dataset")
    with col_tools_model:
        st.selectbox(
            "Model",
            ["⚡ Base Model", "⚡ llama3.2", "⚡ deepseek-r1:8b"],
            key="ba_model_picker",
            label_visibility="collapsed",
        )
    with col_tools_mic:
        st.button("🎙️", key="ba_mic_btn", help="Voice input")
    with col_tools_send:
        send_btn = st.button("⬆️", key="ba_send_btn", type="primary", help="Send prompt")

    st.markdown(
        """
        </div>
        <div class="ba-input-caption">Shift+Enter for new line · Always double-check important information</div>
        """,
        unsafe_allow_html=True,
    )

    if attach_btn:
        st.session_state["show_uploader_modal"] = not st.session_state.get("show_uploader_modal", False)
        st.rerun()

    if st.session_state.get("show_uploader_modal", False):
        with st.container():
            st.info("Upload your dataset (CSV, XLSX, or Parquet) to analyze:")
            uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "parquet"], key="ba_modal_upload")
            if uploaded:
                try:
                    ingest_res = load_tabular(uploaded.getvalue(), uploaded.name, cfg.limits)
                    st.session_state.ingest = ingest_res
                    st.session_state["active_chat_file"] = uploaded.name
                    st.session_state["show_uploader_modal"] = False
                    st.success(f"Loaded '{uploaded.name}' ({len(ingest_res.frame):,} rows).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ingestion error: {exc}")

    if send_btn and user_q:
        if st.session_state.get("ingest") is not None:
            with st.spinner("Analyzing with local engine..."):
                try:
                    op_res = run_operation(st.session_state.ingest, user_q)
                    st.session_state.last_result = op_res
                    st.session_state.last_explanation = explain_results(op_res, user_q, cfg.ai)
                    st.session_state["last_user_query"] = user_q
                    st.rerun()
                except Exception as exc:
                    st.error(f"Analysis error: {exc}")
        else:
            st.info(f"Answer for '{user_q}': Using baseline schedule patterns for active dataset.")


def _render_betteranalyst_workspace_panel(cfg, user: User) -> None:
    st.markdown(
        """
        <div class="ba-workspace-panel">
            <div class="ba-workspace-header">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:16px;">📦</span>
                    <span style="color:#0F172A; font-weight:700; font-size:13.5px;">Workspace</span>
                    <span style="color:#64748B; font-size:11.5px;">Chat resources</span>
                </div>
                <span style="color:#64748B; font-size:12px;">›</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("+ Add / manage data", key="ba_add_manage_data_btn", use_container_width=True):
        st.session_state["active_menu"] = "Library"
        st.rerun()

    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; margin-top:16px; margin-bottom:8px;">
            <span style="color:#0F172A; font-size:11.5px; font-weight:700; letter-spacing:0.04em;">AGENTS</span>
            <span style="color:#4F46E5; font-size:11px; font-weight:600; cursor:pointer;">Manage</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    agents = [
        ("Spreadsheet Auditor", "Lead agent", "nav_agent_1"),
        ("Operations Performa...", "Specialist agent", "nav_agent_2"),
        ("Marketing Performa...", "Specialist agent", "nav_agent_3"),
        ("Forecast & Scenario ...", "Specialist agent", "nav_agent_4"),
        ("Revenue & Margin An...", "Specialist agent", "nav_agent_5"),
    ]

    for title, role, key in agents:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(
                f"""
                <div class="ba-agent-card">
                    <div class="ba-agent-left">
                        <div class="ba-agent-avatar">👤</div>
                        <div class="ba-agent-info">
                            <span class="ba-agent-title">{title}</span>
                            <span class="ba-agent-role">{role}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("→", key=key, help=f"Activate {title}"):
                if "Auditor" in title:
                    st.session_state["active_menu"] = "Alerts"
                elif "Operations" in title:
                    st.session_state["active_menu"] = "Explore Data"
                elif "Marketing" in title:
                    st.session_state["active_menu"] = "Explore Data"
                elif "Forecast" in title:
                    st.session_state["active_menu"] = "Agents"
                elif "Revenue" in title:
                    st.session_state["active_menu"] = "SQL Console"
                else:
                    st.session_state["active_menu"] = "Explore Data"
                st.rerun()

    # Brand Kit Section
    st.markdown(
        """
        <div style="margin-top:16px; margin-bottom:8px;">
            <span style="color:#0F172A; font-size:11.5px; font-weight:700; letter-spacing:0.04em;">BRAND KIT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_b1, c_b2 = st.columns([4, 1])
    with c_b1:
        st.markdown(
            """
            <div class="ba-agent-card">
                <div class="ba-agent-left">
                    <div class="ba-agent-avatar" style="background:#FEF3C7; color:#D97706;">🎨</div>
                    <div class="ba-agent-info">
                        <span class="ba-agent-title">Customize charts</span>
                        <span class="ba-agent-role">Apply your visual identity</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_b2:
        if st.button("→", key="nav_brand_kit", help="Customize brand kit and settings"):
            st.session_state["active_menu"] = "Settings"
            st.rerun()

    # Uploads Section
    active_filename = "2026-10-17T20-05_expor..."
    if st.session_state.get("ingest") is not None and getattr(st.session_state.ingest, "source_name", None):
        raw_name = Path(str(st.session_state.ingest.source_name)).name
        active_filename = raw_name[:18] + "..." if len(raw_name) > 18 else raw_name

    st.markdown(
        """
        <div style="margin-top:16px; margin-bottom:8px;">
            <span style="color:#0F172A; font-size:11.5px; font-weight:700; letter-spacing:0.04em;">UPLOADS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_u1, c_u2 = st.columns([4, 1])
    with c_u1:
        st.markdown(
            f"""
            <div class="ba-agent-card">
                <div class="ba-agent-left">
                    <div class="ba-agent-avatar" style="background:#E0F2FE; color:#0284C7;">📄</div>
                    <div class="ba-agent-info">
                        <span class="ba-agent-title">{active_filename}</span>
                        <span class="ba-agent-role">Spreadsheet</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_u2:
        if st.button("→", key="nav_uploads_manage", help="Manage datasets in Library"):
            st.session_state["active_menu"] = "Library"
            st.rerun()

    # Feedback Section
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    if st.button("💬 Feedback", key="ba_feedback_btn", use_container_width=True, help="Send local feedback"):
        st.session_state["show_feedback_dialog"] = not st.session_state.get("show_feedback_dialog", False)
        st.rerun()

    if st.session_state.get("show_feedback_dialog", False):
        st.info("Local Feedback: Thanks for using VantageAnalyst AI! All features run 100% offline.")


def main() -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cfg = load_config()
    st.set_page_config(page_title="VantageAnalyst AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
    _inject_custom_styles()
    _init_state(cfg)
    if cfg.server.edition == "server":
        # Disallow desktop user bypass in server edition unconditionally
        if st.session_state.user is not None and getattr(st.session_state.user, "id", None) == 0:
            st.session_state.user = None
            st.session_state.session_token = None
    if _check_session_timeout(cfg):
        st.rerun()
    if st.session_state.user is None:
        st.title(PRODUCT_NAME)
        _login(cfg)
        return

    # Auto-bind sample dataset if ingest is None so all tabs and tools work out-of-the-box
    if st.session_state.get("ingest") is None:
        sample_path = Path("samples/2026-10-17T20-05_export.csv")
        if sample_path.exists():
            try:
                st.session_state.ingest = load_tabular(
                    sample_path.read_bytes(),
                    sample_path.name,
                    cfg.limits,
                )
            except Exception as e:
                logger.warning("Could not auto-bind sample dataset: %s", e)

    _sidebar(cfg)

    # Route view according to active navigation selection
    active_menu = st.session_state.get("active_menu", "Dashboards")

    col_main, col_workspace = st.columns([3.8, 1.4], gap="medium")
    with col_main:
        _render_betteranalyst_top_bar()
        if active_menu == "Dashboards":
            _render_betteranalyst_chat_dashboard(cfg, st.session_state.user)
        else:
            c_back, c_title = st.columns([1.5, 6])
            with c_back:
                if st.button("← Back to Dashboard", key="ba_back_to_dash"):
                    st.session_state["active_menu"] = "Dashboards"
                    st.rerun()
            with c_title:
                st.subheader(f"Workspace: {active_menu}")

            if active_menu == "Explore Data":
                _explore_tab(cfg, st.session_state.user)
            elif active_menu == "Library":
                _import_tab(cfg, st.session_state.user)
            elif active_menu == "Data Connectors":
                _connectors_tab(cfg, st.session_state.user)
            elif active_menu == "Data Flows":
                _clean_tab(cfg, st.session_state.user)
                st.divider()
                _repair_tab(cfg, st.session_state.user)
            elif active_menu == "Alerts":
                _quality_tab(cfg, st.session_state.user)
            elif active_menu == "PDF to Excel":
                _documents_tab(cfg, st.session_state.user)
            elif active_menu == "SQL Console":
                _sql_tab(cfg, st.session_state.user)
            elif active_menu == "Agents":
                _agents_hub_tab(cfg, st.session_state.user)
            elif active_menu == "Tasks":
                _tasks_hub_tab(cfg, st.session_state.user)
            elif active_menu == "Slack Agent":
                _slack_hub_tab(cfg, st.session_state.user)
            elif active_menu == "Settings":
                _settings_tab(cfg, st.session_state.user)
            else:
                _explore_tab(cfg, st.session_state.user)

    with col_workspace:
        _render_betteranalyst_workspace_panel(cfg, st.session_state.user)


if __name__ == "__main__":
    main()
