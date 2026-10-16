from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from private_analyst.config import Limits

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture
def limits() -> Limits:
    return Limits(max_upload_bytes=1_000_000, max_rows=10_000, max_columns=50, preview_rows=10)


@pytest.fixture
def sales_csv_bytes() -> bytes:
    return (SAMPLES / "sales_known.csv").read_bytes()


@pytest.fixture
def sales_xlsx_bytes(tmp_path: Path) -> bytes:
    frame = pd.read_csv(SAMPLES / "sales_known.csv")
    path = tmp_path / "sales_known.xlsx"
    frame.to_excel(path, index=False)
    return path.read_bytes()
