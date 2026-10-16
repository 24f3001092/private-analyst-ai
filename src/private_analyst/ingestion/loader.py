from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from private_analyst.config import Limits
from private_analyst.exceptions import ResourceLimitError, ValidationError
from private_analyst.ingestion.validators import validate_upload


@dataclass
class IngestResult:
    frame: pd.DataFrame
    source_name: str
    source_kind: str
    warnings: list[str] = field(default_factory=list)
    duplicate_columns: list[str] = field(default_factory=list)
    inferred_types: dict[str, str] = field(default_factory=dict)
    persisted: bool = False


def _infer_types(frame: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_bool_dtype(series):
            out[str(col)] = "boolean"
        elif pd.api.types.is_integer_dtype(series):
            out[str(col)] = "integer"
        elif pd.api.types.is_float_dtype(series):
            out[str(col)] = "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            out[str(col)] = "datetime"
        else:
            coerced = pd.to_numeric(series, errors="coerce")
            if series.notna().any() and coerced.notna().sum() >= max(1, int(0.8 * series.notna().sum())):
                out[str(col)] = "numeric_string"
            else:
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                if series.notna().any() and parsed.notna().sum() >= max(1, int(0.8 * series.notna().sum())):
                    out[str(col)] = "datetime_string"
                else:
                    out[str(col)] = "text"
    return out


def _normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    import re

    seen: dict[str, int] = {}
    duplicates: list[str] = []
    new_cols: list[str] = []
    names = [str(col).strip() or "unnamed" for col in frame.columns]
    # pandas may already mangle duplicates to "region.1"
    bases = set(names)
    for name in names:
        mangled = re.fullmatch(r"(.+)\.(\d+)$", name)
        if mangled and mangled.group(1) in bases:
            duplicates.append(mangled.group(1))
        if name in seen:
            seen[name] += 1
            duplicates.append(name)
            new_cols.append(f"{name}__{seen[name]}")
        else:
            seen[name] = 0
            new_cols.append(name)
    frame = frame.copy()
    frame.columns = new_cols
    return frame, duplicates


def load_tabular(
    data: bytes,
    filename: str,
    limits: Limits,
) -> IngestResult:
    suffix = validate_upload(filename, len(data), limits, kind="tabular")
    buffer = io.BytesIO(data)
    warnings: list[str] = []
    try:
        if suffix == ".csv":
            frame = pd.read_csv(buffer, on_bad_lines="error", engine="python")
        else:
            frame = pd.read_excel(buffer, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 - convert parser errors
        raise ValidationError("The file could not be parsed as a table.") from exc

    if frame.shape[0] > limits.max_rows:
        raise ResourceLimitError(f"Row count exceeds the configured limit ({limits.max_rows}).")
    if frame.shape[1] > limits.max_columns:
        raise ResourceLimitError(f"Column count exceeds the configured limit ({limits.max_columns}).")
    if frame.empty:
        raise ValidationError("The table has no rows.")

    frame, duplicates = _normalize_columns(frame)
    if duplicates:
        warnings.append("Duplicate column names were renamed to keep them unique.")

    inferred = _infer_types(frame)
    return IngestResult(
        frame=frame,
        source_name=filename,
        source_kind="tabular",
        warnings=warnings,
        duplicate_columns=duplicates,
        inferred_types=inferred,
        persisted=False,
    )


def preview(frame: pd.DataFrame, rows: int) -> list[dict[str, Any]]:
    return frame.head(rows).where(frame.head(rows).notna(), None).to_dict(orient="records")
