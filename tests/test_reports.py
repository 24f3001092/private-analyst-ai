"""Tests for report generation — CSV, Excel, Markdown, and PDF exports."""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
import pytest

from private_analyst.analytics.operations import OperationResult, run_operation
from private_analyst.config import Limits
from private_analyst.ingestion.loader import load_tabular
from private_analyst.reports.exporters import (
    analysis_metadata,
    report_markdown,
    sanitize_csv_value,
    sanitize_dataframe_for_csv,
    to_csv_bytes,
    to_excel_bytes,
)
from private_analyst.reports.pdf_report import to_pdf_bytes


@pytest.fixture
def sample_result(sales_csv_bytes, limits: Limits) -> OperationResult:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    return run_operation(ingest, "Summarize the dataset.")


def test_csv_export_roundtrip(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    csv_bytes = to_csv_bytes(ingest.frame)
    assert isinstance(csv_bytes, bytes)
    restored = pd.read_csv(BytesIO(csv_bytes))
    assert restored.shape == ingest.frame.shape
    assert list(restored.columns) == list(ingest.frame.columns)


def test_excel_export_roundtrip(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    xlsx_bytes = to_excel_bytes(ingest.frame)
    assert isinstance(xlsx_bytes, bytes)
    restored = pd.read_excel(BytesIO(xlsx_bytes))
    assert restored.shape == ingest.frame.shape


def test_markdown_report_contains_intent(sample_result: OperationResult) -> None:
    md = report_markdown(sample_result, "Test explanation", include_explanation=True)
    assert "summarize" in md.lower()
    assert "Test explanation" in md


def test_markdown_report_excludes_explanation_when_not_included(sample_result: OperationResult) -> None:
    md = report_markdown(sample_result, "Secret explanation", include_explanation=False)
    assert "Secret explanation" not in md
    assert "chose not to export" in md.lower()


def test_pdf_report_is_valid_pdf(sample_result: OperationResult) -> None:
    pdf_bytes = to_pdf_bytes(sample_result, "Test explanation", include_explanation=True)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 100


def test_pdf_report_without_explanation(sample_result: OperationResult) -> None:
    pdf_bytes = to_pdf_bytes(sample_result, "Hidden text", include_explanation=False)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:5] == b"%PDF-"


def test_analysis_metadata_fields() -> None:
    meta = analysis_metadata("test_source.csv")
    assert meta["source_name"] == "test_source.csv"
    assert meta["engine"] == "local_python_analytics"
    assert meta["model_does_not_compute_totals"] is True
    assert "version" in meta
    assert "generated_at_utc" in meta


def test_csv_export_does_not_include_index(sales_csv_bytes, limits: Limits) -> None:
    """Verify that exports do not silently include row indices as a column."""
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    csv_bytes = to_csv_bytes(ingest.frame)
    restored = pd.read_csv(BytesIO(csv_bytes))
    # The original has specific columns; there should be no unnamed index col
    assert "Unnamed: 0" not in restored.columns


def test_csv_formula_injection_mitigation_dangerous_prefixes() -> None:
    """Verify dangerous formula prefixes (=, @, tab, CR, non-numeric +/-) are neutralized."""
    assert sanitize_csv_value("=1+1") == "'=1+1"
    assert sanitize_csv_value("=cmd|'/C calc'!A0") == "'=cmd|'/C calc'!A0"
    assert sanitize_csv_value("@SUM(A1:A10)") == "'@SUM(A1:A10)"
    assert sanitize_csv_value("\t=2+2") == "'\t=2+2"
    assert sanitize_csv_value("\r=3+3") == "'\r=3+3"
    assert sanitize_csv_value("+cmd") == "'+cmd"
    assert sanitize_csv_value("-cmd") == "'-cmd"
    assert sanitize_csv_value("  =indented_formula") == "'  =indented_formula"


def test_csv_formula_injection_preserves_legitimate_values() -> None:
    """Verify legitimate text and numbers are not corrupted by formula sanitization."""
    assert sanitize_csv_value("Ordinary Text") == "Ordinary Text"
    assert sanitize_csv_value("Widget-100") == "Widget-100"
    assert sanitize_csv_value("-42") == "-42"
    assert sanitize_csv_value("-42.50") == "-42.50"
    assert sanitize_csv_value("+100") == "+100"
    assert sanitize_csv_value("+3.14159") == "+3.14159"
    assert sanitize_csv_value("") == ""
    assert sanitize_csv_value("   ") == "   "
    assert sanitize_csv_value(123) == 123
    assert sanitize_csv_value(None) is None


def test_csv_export_mitigates_formula_injection_in_headers_and_cells() -> None:
    """Verify DataFrame CSV export sanitizes malicious cells and headers."""
    df = pd.DataFrame({
        "=danger_col": ["=1+1", "@calc", "safe text"],
        "normal_col": ["-cmd|'/c calc'!A0", -25.5, "+300"],
    })
    csv_bytes = to_csv_bytes(df)
    csv_text = csv_bytes.decode("utf-8")

    # Column header '=danger_col' must be escaped
    assert "'=danger_col" in csv_text
    # Cell '=1+1' must be escaped
    assert "'=1+1" in csv_text
    # Cell '@calc' must be escaped
    assert "'@calc" in csv_text
    # Malicious text formula '-cmd...' must be escaped
    assert "'-cmd" in csv_text
    # Valid number -25.5 must NOT be escaped
    assert "-25.5" in csv_text
    assert "'-25.5" not in csv_text
    # Valid number +300 must NOT be escaped
    assert "+300" in csv_text
    assert "'+300" not in csv_text


def test_csv_formula_injection_control_characters_and_whitespace() -> None:
    """Verify control characters, whitespace-prefixed formulas, and standalone symbols are handled."""
    # Control character prefixes (tab, carriage return)
    assert sanitize_csv_value("\tcalc") == "'\tcalc"
    assert sanitize_csv_value("  \tcalc") == "'  \tcalc"
    assert sanitize_csv_value("\rcalc") == "'\rcalc"
    assert sanitize_csv_value("\t") == "'\t"
    assert sanitize_csv_value("\r") == "'\r"

    # Multiline / newline-prefixed formulas
    assert sanitize_csv_value("\n=1+1") == "'\n=1+1"
    assert sanitize_csv_value("\r\n=1+1") == "'\r\n=1+1"
    assert sanitize_csv_value("\n-cmd") == "'\n-cmd"
    assert sanitize_csv_value(" \t=cmd") == "' \t=cmd"

    # Standalone formula trigger symbols (non-numeric)
    assert sanitize_csv_value("-") == "'-"
    assert sanitize_csv_value("+") == "'+"
    assert sanitize_csv_value("--") == "'--"

    # Legitimate numbers with leading whitespace or scientific notation
    assert sanitize_csv_value("  -42") == "  -42"
    assert sanitize_csv_value("  +42") == "  +42"
    assert sanitize_csv_value("-1e-4") == "-1e-4"
    assert sanitize_csv_value("+2.5e3") == "+2.5e3"


def test_csv_formula_injection_categorical_and_duplicate_columns() -> None:
    """Verify categorical string series and duplicate column DataFrames sanitize cleanly."""
    cat_series = pd.Series(["=formula_cat", "safe_cat", "-cmd"], dtype="category")
    df = pd.DataFrame({"cat_col": cat_series})
    clean = sanitize_dataframe_for_csv(df)
    csv_text = to_csv_bytes(clean).decode("utf-8")
    assert "'=formula_cat" in csv_text
    assert "'-cmd" in csv_text
    assert "safe_cat" in csv_text

    # Duplicate column names must not cause AttributeError or crash
    dup_df = pd.DataFrame([["=danger1", "=danger2"]], columns=["dup", "dup"])
    dup_clean = sanitize_dataframe_for_csv(dup_df)
    dup_csv = to_csv_bytes(dup_clean).decode("utf-8")
    assert "'=danger1" in dup_csv
    assert "'=danger2" in dup_csv
