from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from private_analyst.config import Limits
from private_analyst.exceptions import ResourceLimitError, ValidationError
from private_analyst.ingestion.validators import validate_upload

INDEX_COLUMN_CANDIDATES = {"#", "no", "id", "s.no", "sl.no", "sr.no", "index", "unnamed: 0"}


@dataclass
class WorkbookInfo:
    sheet_names: list[str]
    recommended_sheet: str
    sheet_previews: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class IngestResult:
    frame: pd.DataFrame
    source_name: str
    source_kind: str
    warnings: list[str] = field(default_factory=list)
    duplicate_columns: list[str] = field(default_factory=list)
    inferred_types: dict[str, str] = field(default_factory=dict)
    persisted: bool = False
    sheet_name: str | None = None
    available_sheets: list[str] = field(default_factory=list)
    header_row: int = 0
    detected_title: str | None = None
    detected_index_column: str | None = None
    original_shape: tuple[int, int] = (0, 0)
    cleaned_shape: tuple[int, int] = (0, 0)
    excluded_rows: int = 0
    original_column_names: dict[str, str] = field(default_factory=dict)


def inspect_workbook(data: bytes, filename: str) -> WorkbookInfo | None:
    """Inspect an Excel workbook to extract sheet names and determine the recommended primary sheet."""
    if not (filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls")):
        return None
    try:
        buffer = io.BytesIO(data)
        excel_file = pd.ExcelFile(buffer, engine="openpyxl")
        sheet_names = excel_file.sheet_names
        if not sheet_names:
            return None
        previews = {}
        best_sheet = sheet_names[0]
        max_cells = -1.0
        for s in sheet_names:
            try:
                df_peek = pd.read_excel(excel_file, sheet_name=s, header=None, nrows=25)
                cells = df_peek.dropna(how="all").size
                previews[s] = (len(df_peek), df_peek.shape[1])
                score = float(cells)
                if "quick" in s.lower() or "reference" in s.lower():
                    score *= 0.5
                if score > max_cells:
                    max_cells = score
                    best_sheet = s
            except Exception:
                previews[s] = (0, 0)
        return WorkbookInfo(
            sheet_names=sheet_names,
            recommended_sheet=best_sheet,
            sheet_previews=previews,
        )
    except Exception:
        return None


def detect_header_row_from_raw(df_raw: pd.DataFrame) -> tuple[int, str | None]:
    """Detect the real header row index in a raw DataFrame read with header=None.

    Identifies title banner rows, leading blank rows, and selects the row with
    the highest density of distinct descriptive column headers.
    Returns (header_row_index, title_text).
    """
    title_text = None
    num_rows = min(len(df_raw), 15)
    best_row = 0
    best_score = -1.0

    for r in range(num_rows):
        row = df_raw.iloc[r]
        non_null = row.dropna()
        non_null_count = len(non_null)
        if non_null_count == 0:
            continue

        # Check if this row is an introductory title banner (only 1 non-null value spanning multiple columns)
        if non_null_count == 1 and df_raw.shape[1] > 2:
            val = str(non_null.iloc[0]).strip()
            if len(val) > 3 and not title_text:
                title_text = val
            continue

        str_count = sum(1 for v in non_null if isinstance(v, str) and str(v).strip())
        unique_count = len(set(str(v).strip() for v in non_null))

        coverage = non_null_count / max(df_raw.shape[1], 1)
        str_ratio = str_count / max(non_null_count, 1)
        uniqueness = unique_count / max(non_null_count, 1)

        # Header score: prefers rows with high coverage, high string ratio, high uniqueness
        score = (coverage * 0.4) + (str_ratio * 0.4) + (uniqueness * 0.2)

        if score > best_score and non_null_count >= 2:
            best_score = score
            best_row = r

    return best_row, title_text


def _infer_types(frame: pd.DataFrame) -> dict[str, str]:
    """Infer semantic and physical data types for profiling and downstream analytics."""
    out: dict[str, str] = {}
    for col in frame.columns:
        series = frame[col]
        non_null = series.dropna()
        if non_null.empty:
            out[str(col)] = "empty"
            continue

        if pd.api.types.is_bool_dtype(series):
            out[str(col)] = "boolean"
        elif pd.api.types.is_integer_dtype(series):
            out[str(col)] = "integer"
        elif pd.api.types.is_float_dtype(series):
            out[str(col)] = "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            out[str(col)] = "datetime"
        else:
            str_series = non_null.astype(str).str.strip()

            # URL column detection
            url_matches = str_series.str.startswith(("http://", "https://")).sum()
            if url_matches >= max(1, int(0.7 * len(str_series))):
                out[str(col)] = "url"
                continue

            # Currency column detection
            has_currency = str_series.str.lower().str.contains(r"[₹\$€£¥]|inr|usd", regex=True).sum()
            if has_currency >= max(1, int(0.6 * len(str_series))):
                out[str(col)] = "currency"
                continue

            # Strict numeric string detection (require >=95% valid numbers)
            coerced = pd.to_numeric(str_series, errors="coerce")
            if coerced.notna().sum() >= max(1, int(0.95 * len(str_series))):
                out[str(col)] = "numeric_string"
                continue

            # Date parsing with strict threshold
            parsed = pd.to_datetime(str_series, errors="coerce", format="mixed")
            if parsed.notna().sum() >= max(1, int(0.85 * len(str_series))):
                out[str(col)] = "datetime_string"
                continue

            out[str(col)] = "text"
    return out


def _normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Clean column names, deduplicate uniquely, and retain an original mapping."""
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    new_cols: list[str] = []
    orig_map: dict[str, str] = {}

    names = [str(col).strip() or "unnamed_column" for col in frame.columns]
    bases = set(names)
    for name in names:
        mangled = re.fullmatch(r"(.+)\.(\d+)$", name)
        if mangled and mangled.group(1) in bases:
            duplicates.append(mangled.group(1))

    for raw_col in frame.columns:
        orig_name = str(raw_col).strip()
        name = orig_name if orig_name and not orig_name.startswith("Unnamed:") else "unnamed_column"
        if name in seen:
            seen[name] += 1
            if name not in duplicates:
                duplicates.append(name)
            clean_name = f"{name}__{seen[name]}"
        else:
            seen[name] = 0
            clean_name = name
        new_cols.append(clean_name)
        orig_map[clean_name] = orig_name

    frame = frame.copy()
    frame.columns = new_cols
    return frame, duplicates, orig_map


def load_tabular(
    data: bytes,
    filename: str,
    limits: Limits,
    sheet_name: str | None = None,
    header_row: int | None = None,
) -> IngestResult:
    """Load, validate, normalize, and profile tabular files (CSV and Excel)."""
    suffix = validate_upload(filename, len(data), limits, kind="tabular")
    buffer = io.BytesIO(data)
    warnings: list[str] = []
    available_sheets: list[str] = []
    selected_sheet = sheet_name
    detected_title = None

    try:
        if suffix == ".csv":
            try:
                df_raw = pd.read_csv(io.BytesIO(data), header=None, nrows=25, on_bad_lines="skip", engine="python")
                if header_row is None:
                    h_idx, detected_title = detect_header_row_from_raw(df_raw)
                else:
                    h_idx = header_row
            except Exception:
                h_idx = 0
            frame = pd.read_csv(io.BytesIO(data), header=h_idx, on_bad_lines="error", engine="python")
            raw_shape = (len(frame) + h_idx + 1, frame.shape[1])
        else:
            excel_file = pd.ExcelFile(buffer, engine="openpyxl")
            available_sheets = excel_file.sheet_names
            if not available_sheets:
                raise ValidationError("The Excel workbook contains no sheets.")

            if selected_sheet is None or selected_sheet not in available_sheets:
                info = inspect_workbook(data, filename)
                selected_sheet = info.recommended_sheet if info else available_sheets[0]

            df_raw = pd.read_excel(excel_file, sheet_name=selected_sheet, header=None)
            raw_shape = df_raw.shape
            if header_row is None:
                h_idx, detected_title = detect_header_row_from_raw(df_raw)
            else:
                h_idx = header_row

            frame = pd.read_excel(excel_file, sheet_name=selected_sheet, header=h_idx)

    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("The file could not be parsed as a table.") from exc

    if frame.shape[0] > limits.max_rows:
        raise ResourceLimitError(f"Row count exceeds the configured limit ({limits.max_rows}).")
    if frame.shape[1] > limits.max_columns:
        raise ResourceLimitError(f"Column count exceeds the configured limit ({limits.max_columns}).")

    # Clean whitespace and drop completely blank rows
    frame = frame.dropna(how="all").copy()

    # Strip string columns of leading/trailing spaces and convert empty strings to NA
    for col in frame.columns:
        if frame[col].dtype == object or isinstance(frame[col].dtype, pd.StringDtype):
            frame[col] = frame[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            frame[col] = frame[col].replace("", np.nan)

    frame = frame.dropna(how="all")

    if frame.empty:
        raise ValidationError("The table has no rows.")

    frame, duplicates, orig_map = _normalize_columns(frame)
    if duplicates:
        warnings.append("Duplicate column names were renamed to keep them unique.")

    if detected_title:
        warnings.append(f"Detected and separated title row: '{detected_title}'")

    detected_index = None
    first_col = frame.columns[0]
    if first_col.lower() in INDEX_COLUMN_CANDIDATES:
        detected_index = first_col
        warnings.append(f"Column '{first_col}' identified as row index / sequence number.")

    inferred = _infer_types(frame)
    excluded_rows = max(0, raw_shape[0] - frame.shape[0])

    return IngestResult(
        frame=frame,
        source_name=filename,
        source_kind="tabular",
        warnings=warnings,
        duplicate_columns=duplicates,
        inferred_types=inferred,
        persisted=False,
        sheet_name=selected_sheet,
        available_sheets=available_sheets,
        header_row=h_idx,
        detected_title=detected_title,
        detected_index_column=detected_index,
        original_shape=raw_shape,
        cleaned_shape=frame.shape,
        excluded_rows=excluded_rows,
        original_column_names=orig_map,
    )


def preview(frame: pd.DataFrame, rows: int) -> list[dict[str, Any]]:
    return frame.head(rows).where(frame.head(rows).notna(), None).to_dict(orient="records")
