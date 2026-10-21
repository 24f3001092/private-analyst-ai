from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from private_analyst.ai.client import check_local_model
from private_analyst.ai.explainer import explain_results
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
    st.sidebar.title(cfg.branding.product_name)
    st.sidebar.caption(f"Version {__version__} · {cfg.server.edition} edition")
    user: User = st.session_state.user
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

    with col2:
        st.subheader("Column Management")
        all_cols = list(ingest.frame.columns)
        st.markdown("#### Exclude / Drop Columns")
        cols_to_drop = st.multiselect("Select Columns to Remove", all_cols, key="repair_drop_cols_select")
        if st.button("Drop Selected Columns") and cols_to_drop:
            new_frame, op = apply_column_management(ingest.frame, drop_columns=cols_to_drop)
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
            st.success(f"Dropped {len(cols_to_drop)} column(s).")
            st.rerun()

        st.markdown("#### Rename Column")
        col_to_rename = st.selectbox("Select Column to Rename", all_cols, key="repair_rename_col_select")
        new_col_name = st.text_input("New Column Name", value=str(col_to_rename), key="repair_new_col_name_input")
        if st.button("Rename Column") and new_col_name and new_col_name != col_to_rename:
            new_frame, op = apply_column_management(ingest.frame, rename_columns={col_to_rename: new_col_name})
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
            st.success(f"Renamed '{col_to_rename}' to '{new_col_name}'.")
            st.rerun()

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

    if ingest.raw_frame is not None:
        st.divider()
        st.subheader("Revert Dataset to Original File State")
        st.caption("Discard all interactive modifications and restore the original uncleaned spreadsheet buffer.")
        if st.button("Reset Dataset to Raw Extract", type="secondary"):
            ingest.frame = ingest.raw_frame.copy()
            ingest.working_frame = ingest.raw_frame.copy()
            ingest.unnamed_columns = [c for c in ingest.frame.columns if "unnamed" in str(c).lower()]
            reset_op = CleaningOperation(
                step="reset_dataset",
                description="Reset dataset to raw file extract",
                impact=f"Restored {len(ingest.frame)} rows and {len(ingest.frame.columns)} columns",
                rows_affected=len(ingest.frame),
            )
            if hasattr(ingest, "add_cleaning_operation"):
                ingest.add_cleaning_operation(reset_op)
            elif hasattr(ingest, "cleaning_log"):
                ingest.cleaning_log.append(reset_op)
            elif hasattr(ingest, "cleaning_operations") and isinstance(ingest.cleaning_operations, list):
                ingest.cleaning_operations.append(reset_op)
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
        first_num = stats.numeric_column_names[0]
        st.plotly_chart(histogram(ingest.frame, first_num), key=f"explore_hist_{first_num}", use_container_width=True)
        cats = [c for c in ingest.frame.columns if c not in stats.numeric_column_names]
        if cats:
            st.plotly_chart(bar_sum(ingest.frame, str(cats[0]), first_num), key=f"explore_barsum_{cats[0]}_{first_num}", use_container_width=True)

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
        top_cat_name = cat_cols_stats[0].name
        st.plotly_chart(category_bar(ingest.frame, top_cat_name), key=f"explore_topcat_{top_cat_name}", use_container_width=True)

    if len(stats.numeric_column_names) >= 2:
        st.subheader("Correlation Matrix & Pairwise Association")
        st.caption("Pearson correlation coefficients across numeric features.")
        c_res = compute_correlation_matrix(ingest.frame)
        if c_res.get("status") == "ok":
            corr_df = pd.DataFrame(c_res["matrix"])
            st.plotly_chart(correlation_heatmap(corr_df), key="explore_corr_heatmap", use_container_width=True)
            if c_res.get("top_correlations"):
                with st.expander("Ranked Correlation Pairs", expanded=False):
                    st.dataframe(pd.DataFrame(c_res["top_correlations"]), use_container_width=True)


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
                if rtype == "category_bar" and rcol in ingest.frame.columns:
                    st.plotly_chart(category_bar(ingest.frame, rcol), key=f"viz_rec_{idx}_{rtype}_{rcol}", use_container_width=True)
                elif rtype == "histogram" and rcol in ingest.frame.columns:
                    st.plotly_chart(histogram(ingest.frame, rcol), key=f"viz_rec_{idx}_{rtype}_{rcol}", use_container_width=True)
                elif rtype == "correlation_heatmap":
                    c_res = compute_correlation_matrix(ingest.frame)
                    if c_res.get("status") == "ok":
                        st.plotly_chart(correlation_heatmap(pd.DataFrame(c_res["matrix"])), key=f"viz_rec_{idx}_{rtype}_matrix", use_container_width=True)


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

    question = st.text_input("Question about the loaded table", value="Summarize the dataset.")

    if st.button("Run analysis"):
        result = run_operation(ingest, question)
        explanation = explain_results(result, question, cfg.ai)
        st.session_state.last_result = result
        st.session_state.last_explanation = explanation
        _get_audit_log(cfg).record(
            user.org_id,
            user.username,
            "run_analysis",
            f"Operation: {result.intent}, Status: {result.status}",
        )

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
        st.markdown(explanation["text"])
    else:
        st.info("Local AI model is offline. Displaying deterministic structured engine findings below.")
        st.markdown(explanation["text"])


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

    if not model_status.available:
        with st.expander("Local Model (Ollama) Setup Instructions", expanded=False):
            st.markdown(
                "- **Step 1:** Download and install Ollama from [https://ollama.com](https://ollama.com).\n"
                "- **Step 2:** Open a terminal and start the server: `ollama serve`.\n"
                f"- **Step 3:** Pull your configured model: `ollama pull {cfg.ai.model}`.\n"
                "- **Step 4:** Return here and click **Re-test Connection**."
            )

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
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1250px !important;
        }
        h1 {
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.02em !important;
        }
        h2 {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.4rem !important;
        }
        h3 {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            margin-top: 0.8rem !important;
            margin-bottom: 0.3rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 14px;
            font-size: 0.95rem;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cfg = load_config()
    st.set_page_config(page_title=cfg.branding.product_name, layout="wide")
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
    _sidebar(cfg)
    st.title(cfg.branding.product_name)
    tabs = st.tabs([
        "1. Import Data",
        "2. Data Quality",
        "3. Clean Data",
        "4. Repair Rows & Columns",
        "5. Explore Data",
        "6. Visualizations",
        "7. Advanced Analytics",
        "8. Ask & Explain",
        "9. SQL Console",
        "10. Documents",
        "11. Reports & Export",
        "12. Settings",
    ])
    with tabs[0]:
        _import_tab(cfg, st.session_state.user)
    with tabs[1]:
        _quality_tab(cfg, st.session_state.user)
    with tabs[2]:
        _clean_tab(cfg, st.session_state.user)
    with tabs[3]:
        _repair_tab(cfg, st.session_state.user)
    with tabs[4]:
        _explore_tab(cfg, st.session_state.user)
    with tabs[5]:
        _viz_tab(cfg, st.session_state.user)
    with tabs[6]:
        _ml_tab(cfg, st.session_state.user)
    with tabs[7]:
        _ask_tab(cfg, st.session_state.user)
    with tabs[8]:
        _sql_tab(cfg, st.session_state.user)
    with tabs[9]:
        _documents_tab(cfg, st.session_state.user)
    with tabs[10]:
        _reports_tab(cfg, st.session_state.user)
    with tabs[11]:
        _settings_tab(cfg, st.session_state.user)


if __name__ == "__main__":
    main()
