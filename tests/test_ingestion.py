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
