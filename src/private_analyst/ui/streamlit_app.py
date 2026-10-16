from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from private_analyst.ai.explainer import explain_results
from private_analyst.analytics.anomalies import detect_iqr_anomalies, linear_trend_forecast
from private_analyst.analytics.ml_engine import run_classification, run_clustering, run_regression
from private_analyst.analytics.operations import run_operation
from private_analyst.analytics.quality import assess_quality
from private_analyst.analytics.sql_console import DuckDBConsole
from private_analyst.analytics.statistics import compute_dataset_stats
from private_analyst.auth.rbac import can
from private_analyst.auth.users import User, UserStore
from private_analyst.config import load_config
from private_analyst.documents.index import LocalTextIndex
from private_analyst.documents.ocr import ocr_image
from private_analyst.documents.pdf import extract_pdf_text
from private_analyst.exceptions import AuditLogError, AuthenticationError, PrivateAnalystError
from private_analyst.ingestion.loader import IngestResult, load_tabular
from private_analyst.reports.exporters import analysis_metadata, report_markdown, to_csv_bytes, to_excel_bytes
from private_analyst.reports.pdf_report import to_pdf_bytes
from private_analyst.security.encryption import SecureKeyStore
from private_analyst.security.sessions import SessionManager
from private_analyst.version import PRODUCT_NAME, __version__
from private_analyst.viz.charts import bar_sum, histogram, missingness
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
    st.header("Import data")
    st.write("CSV and Excel files are parsed locally. They are not uploaded to a vendor.")
    uploaded = st.file_uploader("Table file", type=["csv", "xlsx"], disabled=not can(user, "ingest"))
    if uploaded is None:
        st.session_state.ingest = st.session_state.ingest
        return
    log = _get_audit_log(cfg)
    try:
        raw_bytes = uploaded.getvalue()
        ingest = load_tabular(raw_bytes, uploaded.name, cfg.limits)
        file_sig = (uploaded.name, len(raw_bytes))
        if st.session_state.get("_last_ingested_file") != file_sig:
            st.session_state["_last_ingested_file"] = file_sig
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
    st.success(f"Loaded {ingest.frame.shape[0]} rows × {ingest.frame.shape[1]} columns from {ingest.source_name}.")
    if ingest.warnings:
        for warning in ingest.warnings:
            st.warning(warning)
    st.subheader("Preview")
    st.dataframe(ingest.frame.head(cfg.limits.preview_rows), use_container_width=True)
    st.json(ingest.inferred_types)


def _dashboard_tab() -> None:
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table to see the dashboard.")
        return
    stats = compute_dataset_stats(ingest.frame)
    quality = assess_quality(ingest.frame, ingest.duplicate_columns)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", stats.row_count)
    c2.metric("Columns", stats.column_count)
    c3.metric("Duplicate rows", quality.duplicate_row_count)
    st.subheader("Data quality")
    st.write(quality.issues)
    st.plotly_chart(missingness(quality.missing_by_column), use_container_width=True)
    st.subheader("Numeric summaries")
    rows = [
        {
            "column": col.name,
            "count": col.count,
            "missing": col.missing,
            "mean": col.mean,
            "median": col.median,
            "std": col.std,
            "min": col.min,
            "max": col.max,
            "sum": col.sum,
        }
        for col in stats.columns
        if col.kind == "numeric"
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        first = stats.numeric_column_names[0]
        st.plotly_chart(histogram(ingest.frame, first), use_container_width=True)
        cats = [c for c in ingest.frame.columns if c not in stats.numeric_column_names]
        if cats:
            st.plotly_chart(bar_sum(ingest.frame, str(cats[0]), first), use_container_width=True)
    else:
        st.write("No numeric columns were detected.")


def _ask_tab(cfg, user: User) -> None:
    ingest: IngestResult | None = st.session_state.ingest
    if ingest is None:
        st.info("Import a table before asking questions.")
        return
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
    st.subheader("Evidence classes")
    st.write("Observed (from the file):", result.observed)
    st.write("Calculated (Python analytics engine):")
    st.json(result.calculated)
    if result.hypotheses:
        st.warning("Hypotheses / unvalidated methods: " + " ".join(result.hypotheses))
    if result.limitations:
        st.info("Limitations: " + " ".join(result.limitations))
    st.subheader("Explanation")
    if explanation["source"] == "local_model":
        st.success("Local model produced the narrative. Totals still come from Python.")
    else:
        st.warning(
            "No local model response was available. A deterministic explanation of the calculated JSON is shown instead."
        )
    st.write(explanation["text"])
    st.caption(f"Model detail: {explanation.get('model_detail')}")


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

    if st.session_state.duckdb_console is None:
        console = DuckDBConsole(max_rows=cfg.limits.max_rows)
        console.register_dataframe("data", ingest.frame)
        st.session_state.duckdb_console = console
    else:
        console = st.session_state.duckdb_console
        try:
            console.register_dataframe("data", ingest.frame)
        except Exception:
            pass

    st.subheader("Available Tables")
    tables = console.list_tables()
    st.write(f"Registered tables: {', '.join(f'`{t}`' for t in tables)}")

    with st.expander("Table Schema (`data`)"):
        try:
            schema_info = console.describe_table("data")
            st.dataframe(pd.DataFrame(schema_info), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not retrieve schema: {exc}")

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

    st.write("### Evaluated Metrics")
    st.json(result.metrics)

    st.write("### Model Parameters")
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

    task = st.selectbox(
        "Select Analytical Task",
        [
            "Classification (Decision Tree)",
            "Linear Regression",
            "Clustering (K-Means)",
            "Anomaly Detection (IQR Outliers)",
            "Linear Trend Forecasting",
        ],
    )

    frame = ingest.frame
    all_cols = list(frame.columns)
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(frame[c])]

    if task.startswith("Classification"):
        st.subheader("Decision Tree Classification")
        target_col = st.selectbox("Target Column (Classes)", all_cols)
        available_features = [c for c in numeric_cols if c != target_col]
        if not available_features:
            st.warning("No numeric feature columns available. Classification requires numeric predictors.")
            return
        selected_features = st.multiselect("Predictor Features", available_features, default=available_features)
        if st.button("Train & Evaluate Classifier"):
            if not selected_features:
                st.error("Please select at least one feature column.")
                return
            result = run_classification(frame, target_col, selected_features)
            _get_audit_log(cfg).record(
                user.org_id,
                user.username,
                "ml_classification",
                f"Task: classification, Features: {len(selected_features)}, Status: {result.status}",
            )
            _render_ml_result(result)

    elif task.startswith("Linear Regression"):
        st.subheader("Linear Regression")
        if not numeric_cols:
            st.warning("No numeric columns available in dataset.")
            return
        target_col = st.selectbox("Target Column (Continuous outcome)", numeric_cols)
        available_features = [c for c in numeric_cols if c != target_col]
        if not available_features:
            st.warning("No additional numeric feature columns available as predictors.")
            return
        selected_features = st.multiselect("Predictor Features", available_features, default=available_features)
        if st.button("Fit & Evaluate Regression"):
            if not selected_features:
                st.error("Please select at least one feature column.")
                return
            result = run_regression(frame, target_col, selected_features)
            _get_audit_log(cfg).record(
                user.org_id,
                user.username,
                "ml_regression",
                f"Task: regression, Features: {len(selected_features)}, Status: {result.status}",
            )
            _render_ml_result(result)

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
            result = run_clustering(frame, selected_features, n_clusters=k)
            _get_audit_log(cfg).record(
                user.org_id,
                user.username,
                "ml_clustering",
                f"Task: clustering, k: {k}, Features: {len(selected_features)}, Status: {result.status}",
            )
            _render_ml_result(result)

    elif task.startswith("Anomaly"):
        st.subheader("IQR Outlier & Anomaly Detection")
        if not numeric_cols:
            st.warning("No numeric columns available in dataset.")
            return
        col = st.selectbox("Column to inspect for outliers", numeric_cols)
        multiplier = st.slider("IQR Multiplier (default 1.5)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        if st.button("Detect Outliers"):
            res = detect_iqr_anomalies(frame, col, multiplier=multiplier)
            _get_audit_log(cfg).record(
                user.org_id,
                user.username,
                "anomaly_detection",
                f"Task: anomaly_detection, Multiplier: {multiplier}, Outliers: {res.outlier_count}",
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Outliers Detected", res.outlier_count)
            c2.metric("Lower Fence", f"{res.lower:.4f}")
            c3.metric("Upper Fence", f"{res.upper:.4f}")
            st.info(f"Methodological caveat: {res.caveat}")
            if res.outlier_row_indexes:
                st.write(f"Sample Outlier Rows ({len(res.outlier_row_indexes)} detected):")
                st.dataframe(frame.iloc[res.outlier_row_indexes], use_container_width=True)
            else:
                st.success(f"No statistical outliers detected in column '{col}' with multiplier {multiplier}.")

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
                st.plotly_chart(fig, use_container_width=True)


def _documents_tab(cfg, user: User) -> None:
    if not can(user, "documents"):
        st.error("You do not have permission to manage documents.")
        return
    st.write("Documents are indexed in memory for this session. Nothing is sent to an external search service.")
    uploaded = st.file_uploader("PDF, text, or image", type=["pdf", "txt", "png", "jpg", "jpeg"])
    if uploaded and st.button("Index document"):
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
            st.success(f"Indexed {added} passage(s) from {name}.")
        except PrivateAnalystError as exc:
            log.record(user.org_id, user.username, "index_document_failure", f"File: '{Path(name).name}', Reason: {exc.__class__.__name__}")
            st.error(str(exc))
    query = st.text_input("Search indexed passages")
    if query:
        hits = st.session_state.index.search(query)
        if not hits:
            st.write("No matching passages were found in the local index.")
        for hit in hits:
            st.markdown(f"**{hit.document_name}** · page {hit.page_number} · score {hit.score}")
            st.caption("Source evidence (extracted text):")
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
        st.info("Import data first.")
        return
    st.write("Exports are created locally. Choose what to include.")
    include_ai = st.checkbox("Include AI explanation in exports", value=False)
    st.download_button(
        "Download table CSV",
        to_csv_bytes(ingest.frame),
        file_name="table.csv",
        on_click=lambda: _safe_record_export_requested(
            cfg, user, "table", "delimited", f"Source: '{Path(ingest.source_name).stem}'"
        ),
    )
    if can(user, "export"):
        st.download_button(
            "Download table Excel",
            to_excel_bytes(ingest.frame),
            file_name="table.xlsx",
            on_click=lambda: _safe_record_export_requested(
                cfg, user, "table", "xlsx", f"Source: '{Path(ingest.source_name).stem}'"
            ),
        )
    if result and explanation:
        text = report_markdown(result, explanation["text"], include_ai)
        st.download_button(
            "Download markdown report",
            text.encode("utf-8"),
            file_name="report.md",
            on_click=lambda: _safe_record_export_requested(
                cfg, user, "report", "md", f"AI included: {include_ai}"
            ),
        )
        st.download_button(
            "Download PDF report",
            to_pdf_bytes(result, explanation["text"], include_ai),
            file_name="report.pdf",
            on_click=lambda: _safe_record_export_requested(
                cfg, user, "report", "pdf", f"AI included: {include_ai}"
            ),
        )
        st.json(analysis_metadata(ingest.source_name))
    else:
        st.caption("Run a question in Ask to enable analysis reports.")


def _settings_tab(cfg, user: User) -> None:
    st.write("Telemetry is hard-disabled. The model host must be loopback.")
    st.json(
        {
            "edition": cfg.server.edition,
            "bind_host": cfg.server.bind_host,
            "require_auth": cfg.server.require_auth,
            "telemetry_enabled": cfg.telemetry_enabled,
            "ai_host": cfg.ai.host,
            "data_dir": str(cfg.resolved_data_dir()),
            "max_upload_bytes": cfg.limits.max_upload_bytes,
        }
    )
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


def main() -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cfg = load_config()
    st.set_page_config(page_title=cfg.branding.product_name, layout="wide")
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
        "Import",
        "Dashboard",
        "Ask",
        "SQL Console",
        "Advanced Analytics",
        "Documents",
        "Reports",
        "Settings",
    ])
    with tabs[0]:
        _import_tab(cfg, st.session_state.user)
    with tabs[1]:
        _dashboard_tab()
    with tabs[2]:
        _ask_tab(cfg, st.session_state.user)
    with tabs[3]:
        _sql_tab(cfg, st.session_state.user)
    with tabs[4]:
        _ml_tab(cfg, st.session_state.user)
    with tabs[5]:
        _documents_tab(cfg, st.session_state.user)
    with tabs[6]:
        _reports_tab(cfg, st.session_state.user)
    with tabs[7]:
        _settings_tab(cfg, st.session_state.user)


if __name__ == "__main__":
    main()
