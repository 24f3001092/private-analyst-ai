from __future__ import annotations

import pytest

from private_analyst.config import Limits
from private_analyst.exceptions import FileTooLargeError, UnsupportedFormatError, ValidationError
from private_analyst.ingestion.loader import load_tabular
from private_analyst.ingestion.validators import validate_upload
from private_analyst.analytics.quality import assess_quality
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_csv_and_missing_and_duplicates(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    quality = assess_quality(ingest.frame, ingest.duplicate_columns)
    assert quality.missing_by_column["status"] == 1
    assert quality.duplicate_row_count == 0


def test_duplicate_column_names(limits: Limits) -> None:
    data = (SAMPLES / "duplicate_columns.csv").read_bytes()
    ingest = load_tabular(data, "duplicate_columns.csv", limits)
    assert "region" in ingest.duplicate_columns
    assert "region.1" in ingest.frame.columns or "region__1" in ingest.frame.columns


def test_unsupported_format(limits: Limits) -> None:
    with pytest.raises(UnsupportedFormatError):
        validate_upload("notes.exe", 10, limits, kind="tabular")


def test_oversized_file(limits: Limits) -> None:
    tiny = Limits(max_upload_bytes=10)
    with pytest.raises(FileTooLargeError):
        validate_upload("a.csv", 11, tiny, kind="tabular")


def test_malformed_csv(limits: Limits) -> None:
    data = (SAMPLES / "malformed.csv").read_bytes()
    with pytest.raises(ValidationError):
        load_tabular(data, "malformed.csv", limits)


def test_malformed_excel(limits: Limits) -> None:
    with pytest.raises(ValidationError):
        load_tabular(b"this-is-not-xlsx", "bad.xlsx", limits)


def test_empty_table(limits: Limits) -> None:
    with pytest.raises(ValidationError):
        load_tabular(b"col_a,col_b\n", "empty.csv", limits)


def test_path_traversal_name(limits: Limits) -> None:
    with pytest.raises(ValidationError):
        validate_upload("..\\secret.csv", 10, limits)


def test_hackathon_workbook_inspection(limits: Limits) -> None:
    from private_analyst.ingestion.loader import inspect_workbook

    wb_path = SAMPLES / "Hackathon_List_2026-1.xlsx"
    if not wb_path.is_file():
        pytest.skip("Hackathon_List_2026-1.xlsx not found in samples")
    info = inspect_workbook(wb_path.read_bytes(), "Hackathon_List_2026-1.xlsx")
    assert info is not None
    assert "AI & Agentic Hackathons" in info.sheet_names
    assert "Coding Hackathons" in info.sheet_names
    assert "Quick Reference" in info.sheet_names
    assert info.recommended_sheet == "AI & Agentic Hackathons"


def test_hackathon_sheet1_header_detection_and_cleaning(limits: Limits) -> None:
    wb_path = SAMPLES / "Hackathon_List_2026-1.xlsx"
    if not wb_path.is_file():
        pytest.skip("Hackathon_List_2026-1.xlsx not found in samples")
    ingest = load_tabular(wb_path.read_bytes(), "Hackathon_List_2026-1.xlsx", limits)
    assert ingest.sheet_name == "AI & Agentic Hackathons"
    assert ingest.cleaned_shape == (22, 10)
    assert ingest.header_row == 2
    assert "AGENTIC AI" in (ingest.detected_title or "")
    assert ingest.detected_index_column == "#"
    assert "Hackathon Name" in ingest.frame.columns
    assert "Organizer" in ingest.frame.columns
    assert "Unnamed: 1" not in ingest.frame.columns


def test_hackathon_sheet2_coding_hackathons(limits: Limits) -> None:
    wb_path = SAMPLES / "Hackathon_List_2026-1.xlsx"
    if not wb_path.is_file():
        pytest.skip("Hackathon_List_2026-1.xlsx not found in samples")
    ingest = load_tabular(
        wb_path.read_bytes(),
        "Hackathon_List_2026-1.xlsx",
        limits,
        sheet_name="Coding Hackathons",
    )
    assert ingest.sheet_name == "Coding Hackathons"
    assert ingest.cleaned_shape == (20, 9)
    assert "Hackathon Name" in ingest.frame.columns
    assert "Timeline" in ingest.frame.columns


def test_hackathon_sheet3_quick_reference(limits: Limits) -> None:
    wb_path = SAMPLES / "Hackathon_List_2026-1.xlsx"
    if not wb_path.is_file():
        pytest.skip("Hackathon_List_2026-1.xlsx not found in samples")
    ingest = load_tabular(
        wb_path.read_bytes(),
        "Hackathon_List_2026-1.xlsx",
        limits,
        sheet_name="Quick Reference",
    )
    assert ingest.sheet_name == "Quick Reference"
    assert ingest.cleaned_shape[0] >= 18
    assert "Month" in ingest.frame.columns
    assert "Priority" in ingest.frame.columns
