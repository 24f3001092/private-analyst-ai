"""Comprehensive offline acceptance test.

This test file verifies the full end-to-end workflow WITHOUT internet:
- File import (CSV and Excel)
- Statistics calculations with known expected answers
- Data quality assessment
- Anomaly detection
- Forecasting (labelled as hypothesis)
- Document processing (PDF, text)
- Document search with source references
- Report generation (CSV, Excel, Markdown, PDF)
- AI fallback when model is unavailable
- Network policy enforcement

Run with: python -m pytest tests/test_offline_acceptance.py -v
"""

from __future__ import annotations

import json
import socket
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from private_analyst.ai.client import LocalModelClient
from private_analyst.ai.explainer import explain_results
from private_analyst.analytics.anomalies import detect_iqr_anomalies, linear_trend_forecast
from private_analyst.analytics.operations import run_operation
from private_analyst.analytics.quality import assess_quality
from private_analyst.analytics.statistics import SAMPLE_STD_DDOF, compute_dataset_stats
from private_analyst.config import AIConfig, Limits, load_config
from private_analyst.documents.index import LocalTextIndex
from private_analyst.documents.pdf import extract_pdf_text
from private_analyst.exceptions import (
    FileTooLargeError,
    ModelUnavailableError,
    UnsupportedFormatError,
    ValidationError,
)
from private_analyst.ingestion.loader import load_tabular
from private_analyst.reports.exporters import to_csv_bytes, to_excel_bytes, report_markdown
from private_analyst.reports.pdf_report import to_pdf_bytes
from private_analyst.security.network import LocalOnlySocket, assert_local_only, dump_policy

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


# ---------- Known expected values for sales_known.csv ----------
# date,region,product,units,revenue,status
# 2024-01-01,East,Widget,10,100.0,ok
# 2024-01-02,West,Widget,20,200.0,ok
# 2024-01-03,East,Gadget,5,75.0,
# 2024-01-04,West,Gadget,15,225.0,ok
# 2024-01-05,East,Widget,0,0.0,cancelled
# ---
# units: [10, 20, 5, 15, 0] -> sum=50, mean=10, median=10, min=0, max=20
# revenue: [100, 200, 75, 225, 0] -> sum=600, mean=120
# status: 1 missing (Gadget/East row)

EXPECTED_UNITS_SUM = 50
EXPECTED_UNITS_MEAN = 10.0
EXPECTED_UNITS_MEDIAN = 10.0
EXPECTED_UNITS_MIN = 0
EXPECTED_UNITS_MAX = 20
EXPECTED_REVENUE_SUM = 600
EXPECTED_REVENUE_MEAN = 120.0
EXPECTED_STATUS_MISSING = 1
EXPECTED_ROW_COUNT = 5
EXPECTED_COL_COUNT = 6  # date, region, product, units, revenue, status


@pytest.fixture
def limits() -> Limits:
    return Limits(max_upload_bytes=10_000_000, max_rows=10_000, max_columns=50, preview_rows=10)


@pytest.fixture
def sales_csv() -> bytes:
    return (SAMPLES / "sales_known.csv").read_bytes()


@pytest.fixture
def sales_xlsx(tmp_path: Path) -> bytes:
    frame = pd.read_csv(SAMPLES / "sales_known.csv")
    path = tmp_path / "sales_known.xlsx"
    frame.to_excel(path, index=False)
    return path.read_bytes()


class TestOfflineCSVImport:
    """Step 1: CSV import and validation."""

    def test_csv_loads_successfully(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        assert ingest.frame.shape == (EXPECTED_ROW_COUNT, EXPECTED_COL_COUNT)
        assert ingest.source_name == "sales_known.csv"

    def test_csv_type_inference(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        assert ingest.inferred_types["units"] in ("integer", "float", "numeric_string")
        assert ingest.inferred_types["product"] == "text"


class TestOfflineExcelImport:
    """Step 2: Excel import matches CSV."""

    def test_excel_loads(self, sales_xlsx: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_xlsx, "sales_known.xlsx", limits)
        assert ingest.frame.shape[0] == EXPECTED_ROW_COUNT

    def test_excel_matches_csv_stats(self, sales_csv: bytes, sales_xlsx: bytes, limits: Limits) -> None:
        csv_stats = compute_dataset_stats(load_tabular(sales_csv, "a.csv", limits).frame)
        xlsx_stats = compute_dataset_stats(load_tabular(sales_xlsx, "a.xlsx", limits).frame)
        csv_units = next(c for c in csv_stats.columns if c.name == "units")
        xlsx_units = next(c for c in xlsx_stats.columns if c.name == "units")
        assert csv_units.sum == xlsx_units.sum == EXPECTED_UNITS_SUM


class TestOfflineStatistics:
    """Step 3: Statistics match independently calculated expected values."""

    def test_units_statistics(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        stats = compute_dataset_stats(ingest.frame)
        units = next(c for c in stats.columns if c.name == "units")

        assert units.sum == EXPECTED_UNITS_SUM
        assert units.mean == EXPECTED_UNITS_MEAN
        assert units.median == EXPECTED_UNITS_MEDIAN
        assert units.min == EXPECTED_UNITS_MIN
        assert units.max == EXPECTED_UNITS_MAX

        # Cross-verify with numpy
        arr = np.array([10, 20, 5, 15, 0], dtype=float)
        assert units.mean == float(np.mean(arr))
        assert units.std == float(np.std(arr, ddof=SAMPLE_STD_DDOF))

    def test_revenue_statistics(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        stats = compute_dataset_stats(ingest.frame)
        revenue = next(c for c in stats.columns if c.name == "revenue")
        assert revenue.sum == EXPECTED_REVENUE_SUM
        assert revenue.mean == EXPECTED_REVENUE_MEAN


class TestOfflineDataQuality:
    """Step 4: Data quality assessment."""

    def test_missing_values_detected(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        quality = assess_quality(ingest.frame, ingest.duplicate_columns)
        assert quality.missing_by_column["status"] == EXPECTED_STATUS_MISSING
        assert quality.row_count == EXPECTED_ROW_COUNT

    def test_quality_issues_reported(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        quality = assess_quality(ingest.frame, ingest.duplicate_columns)
        assert any("Missing" in issue or "missing" in issue.lower() for issue in quality.issues)


class TestOfflineAnomalies:
    """Step 5: Anomaly detection."""

    def test_iqr_anomalies_on_known_data(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        result = detect_iqr_anomalies(ingest.frame, "units")
        assert result.method == "iqr"
        assert "heuristic" in result.caveat.lower()

    def test_forecast_labelled_as_hypothesis(self) -> None:
        result = linear_trend_forecast([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["status"] == "hypothesis"
        assert "unvalidated" in result["caveat"].lower()
        assert len(result["forecast"]) == 3


class TestOfflineOperations:
    """Step 6: The operations router produces correct outputs."""

    def test_summarize(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        result = run_operation(ingest, "Calculate total and average of units.")
        units = next(c for c in result.calculated["columns"] if c["name"] == "units")
        assert units["sum"] == EXPECTED_UNITS_SUM
        assert units["mean"] == EXPECTED_UNITS_MEAN

    def test_quality_operation(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        result = run_operation(ingest, "Identify missing values")
        assert result.calculated["missing_by_column"]["status"] == EXPECTED_STATUS_MISSING


class TestOfflineDocuments:
    """Step 7: Document processing and search."""

    def test_text_indexing_and_search(self) -> None:
        index = LocalTextIndex()
        index.add_plain_text("memo.txt", "The quarterly revenue exceeded expectations in the eastern region.")
        hits = index.search("revenue eastern region")
        assert len(hits) > 0
        assert hits[0].document_name == "memo.txt"

    def test_pdf_creation_and_extraction(self) -> None:
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.drawString(72, 720, "Confidential sales analysis report.")
        c.save()
        pdf_data = buffer.getvalue()

        limits = Limits()
        extract = extract_pdf_text(pdf_data, "report.pdf", limits)
        assert any("sales" in p.text.lower() for p in extract.pages)

    def test_document_deletion_and_rebuild(self) -> None:
        index = LocalTextIndex()
        index.add_plain_text("keep.txt", "important data to keep")
        index.add_plain_text("remove.txt", "sensitive data to remove")
        removed = index.delete_document("remove.txt")
        assert removed >= 1
        index.rebuild()
        hits = index.search("sensitive data to remove")
        assert all(h.document_name != "remove.txt" for h in hits)


class TestOfflineReports:
    """Step 8: Report generation."""

    def test_csv_export(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        exported = to_csv_bytes(ingest.frame)
        restored = pd.read_csv(BytesIO(exported))
        assert restored.shape == ingest.frame.shape

    def test_excel_export(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        exported = to_excel_bytes(ingest.frame)
        restored = pd.read_excel(BytesIO(exported))
        assert restored.shape == ingest.frame.shape

    def test_pdf_report(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        result = run_operation(ingest, "Summarize the dataset.")
        pdf = to_pdf_bytes(result, "Test explanation", include_explanation=True)
        assert pdf[:5] == b"%PDF-"

    def test_markdown_report(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        result = run_operation(ingest, "Summarize the dataset.")
        md = report_markdown(result, "Test explanation", include_explanation=True)
        assert "summarize" in md.lower()


class TestOfflineAIFallback:
    """Step 9: AI explanation fallback when model is unavailable."""

    def test_deterministic_fallback(self, sales_csv: bytes, limits: Limits) -> None:
        ingest = load_tabular(sales_csv, "sales_known.csv", limits)
        result = run_operation(ingest, "Summarize the dataset.")
        # Use an unreachable port
        cfg = AIConfig(host="127.0.0.1", port=1, model="nonexistent")
        explanation = explain_results(result, "Summarize the dataset.", cfg)
        assert explanation["source"] == "deterministic_fallback"
        # The fallback must still contain the correct numbers
        assert "50" in explanation["text"] or str(EXPECTED_UNITS_SUM) in explanation["text"]


@pytest.mark.network
class TestOfflineNetworkPolicy:
    """Step 10: Network policy enforcement."""

    def test_non_local_host_blocked(self) -> None:
        with pytest.raises(PermissionError):
            assert_local_only("api.openai.com")

    def test_non_local_ai_host_rejected(self) -> None:
        with pytest.raises(ModelUnavailableError):
            LocalModelClient(AIConfig(host="example.com"))

    def test_policy_no_cloud_ai(self) -> None:
        policy = json.loads(dump_policy())
        assert policy["cloud_ai"] is False
        assert policy["developer_telemetry"] is False

    def test_config_forces_telemetry_off(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PRIVATE_ANALYST_DATA_DIR", str(tmp_path))
        cfg = load_config()
        assert cfg.telemetry_enabled is False
        assert cfg.ai.allow_network_beyond_localhost is False

    def test_socket_blocks_outbound(self, monkeypatch) -> None:
        monkeypatch.setattr(socket, "socket", LocalOnlySocket)
        with pytest.raises(OSError, match="blocked"):
            socket.create_connection(("93.184.216.34", 80), timeout=1)


class TestOfflineFileValidation:
    """Step 11: Invalid file handling."""

    def test_unsupported_format(self, limits: Limits) -> None:
        from private_analyst.ingestion.validators import validate_upload
        with pytest.raises(UnsupportedFormatError):
            validate_upload("virus.exe", 100, limits)

    def test_oversized_file(self) -> None:
        tiny_limits = Limits(max_upload_bytes=10)
        from private_analyst.ingestion.validators import validate_upload
        with pytest.raises(FileTooLargeError):
            validate_upload("data.csv", 100, tiny_limits)

    def test_empty_file(self, limits: Limits) -> None:
        with pytest.raises(ValidationError):
            load_tabular(b"", "empty.csv", limits)

    def test_path_traversal(self, limits: Limits) -> None:
        from private_analyst.ingestion.validators import validate_upload
        with pytest.raises(ValidationError):
            validate_upload("..\\..\\etc\\passwd.csv", 10, limits)
