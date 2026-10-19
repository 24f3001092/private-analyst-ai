from __future__ import annotations

from datetime import datetime, timezone
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
class CleaningOperation:
    step: str
    description: str
    impact: str
    affected_columns: list[str] = field(default_factory=list)
    rows_affected: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "applied"
    before_after: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "impact": self.impact,
            "affected_columns": list(self.affected_columns) if self.affected_columns else [],
            "rows_affected": self.rows_affected,
            "timestamp": self.timestamp,
            "status": self.status,
            "before_after": self.before_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | Any) -> CleaningOperation:
        if not isinstance(data, dict):
            return cls(
                step="unstructured_operation",
                description=str(data) if data is not None else "Empty operation",
                impact="Unstructured cleaning operation",
                affected_columns=[],
                rows_affected=0,
            )
        raw_step = data.get("step") or data.get("Step")
        raw_desc = data.get("description") or data.get("Description")
        raw_impact = data.get("impact") or data.get("Impact")
        raw_cols = data.get("affected_columns") or data.get("Affected Columns") or []
        raw_rows = data.get("rows_affected") if "rows_affected" in data else data.get("Rows Affected", 0)
        raw_ts = data.get("timestamp") or data.get("Timestamp")
        raw_status = data.get("status") or data.get("Status") or "applied"
        raw_ba = data.get("before_after") or data.get("Before / After") or ""

        if isinstance(raw_cols, str):
            cols = [c.strip() for c in raw_cols.split(",") if c.strip()]
        elif isinstance(raw_cols, (list, tuple, set)):
            cols = [str(c) for c in raw_cols]
        else:
            cols = []

        try:
            rows = int(raw_rows) if raw_rows is not None else 0
        except (ValueError, TypeError):
            rows = 0

        ts = str(raw_ts) if raw_ts else datetime.now(timezone.utc).isoformat()

        return cls(
            step=str(raw_step).strip() if raw_step else "unknown_step",
            description=str(raw_desc).strip() if raw_desc else "No description provided",
            impact=str(raw_impact).strip() if raw_impact else "No impact recorded",
            affected_columns=cols,
            rows_affected=rows,
            timestamp=ts,
            status=str(raw_status),
            before_after=str(raw_ba),
        )


def normalize_cleaning_operation(op: Any) -> tuple[dict[str, Any], str | None]:
    """Normalize a cleaning operation (dict, CleaningOperation, or arbitrary record) into a displayable dict.

    Returns:
        (entry_dict, warning_message_if_malformed)
    """
    warning = None
    if op is None:
        return {
            "Step": "Unknown Step",
            "Description": "Empty operation record",
            "Impact": "-",
            "Rows Affected": 0,
            "Affected Columns": "All",
            "Timestamp": "-",
        }, "Empty or null cleaning operation detected."

    if isinstance(op, CleaningOperation):
        step_str = op.step.replace("_", " ").strip().title() if op.step else "Unknown Step"
        ts_str = op.timestamp[:19].replace("T", " ") if getattr(op, "timestamp", None) else "-"
        cols_str = ", ".join(op.affected_columns) if op.affected_columns else "All"
        return {
            "Step": step_str,
            "Description": op.description or "No description provided",
            "Impact": op.impact or "-",
            "Rows Affected": op.rows_affected if isinstance(op.rows_affected, int) else 0,
            "Affected Columns": cols_str,
            "Timestamp": ts_str,
        }, None

    if isinstance(op, dict):
        raw_step = op.get("step") or op.get("Step")
        raw_desc = op.get("description") or op.get("Description")
        raw_impact = op.get("impact") or op.get("Impact")
        raw_cols = op.get("affected_columns") or op.get("Affected Columns")
        raw_rows = op.get("rows_affected") if "rows_affected" in op else op.get("Rows Affected")
        raw_ts = op.get("timestamp") or op.get("Timestamp")

        if not raw_step:
            warning = "Cleaning operation dictionary is missing a 'step' field."
            step_str = "Unknown Step"
        else:
            step_str = str(raw_step).replace("_", " ").strip().title()

        desc_str = str(raw_desc).strip() if raw_desc else "No description provided"
        impact_str = str(raw_impact).strip() if raw_impact else "-"

        if isinstance(raw_cols, (list, tuple, set)):
            cols_str = ", ".join(str(c) for c in raw_cols if str(c).strip()) if raw_cols else "All"
        elif isinstance(raw_cols, str) and raw_cols.strip():
            cols_str = raw_cols.strip()
        else:
            cols_str = "All"

        try:
            rows_num = int(raw_rows) if raw_rows is not None else 0
        except (ValueError, TypeError):
            rows_num = 0
            warning = f"Operation '{step_str}' had non-integer rows_affected: {raw_rows}"

        if raw_ts and isinstance(raw_ts, str) and len(raw_ts) >= 19:
            ts_str = raw_ts[:19].replace("T", " ")
        elif raw_ts:
            ts_str = str(raw_ts)
        else:
            ts_str = "-"

        return {
            "Step": step_str,
            "Description": desc_str,
            "Impact": impact_str,
            "Rows Affected": rows_num,
            "Affected Columns": cols_str,
            "Timestamp": ts_str,
        }, warning

    # Handle arbitrary object or unexpected type
    step_val = getattr(op, "step", None) or getattr(op, "name", "Malformed Operation")
    desc_val = getattr(op, "description", None) or str(op)
    impact_val = getattr(op, "impact", "-")
    cols_val = getattr(op, "affected_columns", "All")
    rows_val = getattr(op, "rows_affected", 0)
    ts_val = getattr(op, "timestamp", "-")

    cols_str = ", ".join(cols_val) if isinstance(cols_val, (list, tuple, set)) else str(cols_val)
    try:
        rows_num = int(rows_val)
    except (ValueError, TypeError):
        rows_num = 0

    return {
        "Step": str(step_val).replace("_", " ").strip().title(),
        "Description": str(desc_val),
        "Impact": str(impact_val),
        "Rows Affected": rows_num,
        "Affected Columns": cols_str,
        "Timestamp": str(ts_val)[:19].replace("T", " ") if isinstance(ts_val, str) else "-",
    }, f"Non-standard cleaning operation entry: {type(op).__name__}"


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
    cleaning_log: list[CleaningOperation] = field(default_factory=list)
    unnamed_columns: list[str] = field(default_factory=list)
    semantic_types: dict[str, str] = field(default_factory=dict)
    raw_preview: list[dict[str, Any]] = field(default_factory=list)
    raw_frame: pd.DataFrame | None = None
    quality_report: Any = None

    @property
    def working_frame(self) -> pd.DataFrame:
        return self.frame

    @working_frame.setter
    def working_frame(self, df: pd.DataFrame) -> None:
        self.frame = df

    @property
    def cleaning_operations(self) -> list[dict[str, Any]]:
        return [op.to_dict() if hasattr(op, "to_dict") else CleaningOperation.from_dict(op).to_dict() for op in self.cleaning_log]

    @cleaning_operations.setter
    def cleaning_operations(self, ops: list[Any]) -> None:
        self.cleaning_log = [
            op if isinstance(op, CleaningOperation) else CleaningOperation.from_dict(op)
            for op in (ops or [])
        ]

    def add_cleaning_operation(self, op: CleaningOperation | dict[str, Any] | Any) -> None:
        """Safely record a cleaning operation whether provided as a dataclass, dict, or raw record."""
        if isinstance(op, CleaningOperation):
            self.cleaning_log.append(op)
        elif isinstance(op, dict):
            self.cleaning_log.append(CleaningOperation.from_dict(op))
        else:
            self.cleaning_log.append(CleaningOperation.from_dict({"description": str(op)}))


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


def parse_currency(val: Any) -> tuple[float | None, str | None]:
    """Parse a currency string into a numeric float amount and symbol/code."""
    if val is None or pd.isna(val):
        return None, None
    s = str(val).strip()
    if not s:
        return None, None
    curr_match = re.search(r"([₹\$€£¥]|INR|USD|EUR|GBP|JPY)", s, re.IGNORECASE)
    symbol = curr_match.group(1).upper() if curr_match else None
    cleaned = re.sub(r"[₹\$€£¥,]|INR|USD|EUR|GBP|JPY", "", s, flags=re.IGNORECASE).strip()
    # Check for range: e.g. "500 - 1000"
    if "-" in cleaned or " to " in cleaned.lower():
        parts = re.split(r"-|\bto\b", cleaned, flags=re.IGNORECASE)
        try:
            return float(parts[0].strip()), symbol
        except (ValueError, IndexError):
            pass
    mult = 1.0
    s_lower = cleaned.lower()
    if s_lower.endswith("k"):
        mult = 1_000.0
        cleaned = cleaned[:-1].strip()
    elif "lakh" in s_lower or "lac" in s_lower:
        mult = 100_000.0
        cleaned = re.sub(r"lakh|lac", "", cleaned, flags=re.IGNORECASE).strip()
    elif "cr" in s_lower or "crore" in s_lower:
        mult = 10_000_000.0
        cleaned = re.sub(r"cr|crore", "", cleaned, flags=re.IGNORECASE).strip()
    elif s_lower.endswith("m"):
        mult = 1_000_000.0
        cleaned = cleaned[:-1].strip()
    try:
        return float(cleaned) * mult, symbol
    except ValueError:
        return None, symbol


def parse_percentage(val: Any) -> float | None:
    """Parse percentage string into a float value (e.g. '12.5%' -> 12.5, '(5%)' -> -5.0)."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    elif s.startswith("-"):
        negative = True
        s = s[1:].strip()
    is_pct_sign = False
    if s.endswith("%"):
        is_pct_sign = True
        s = s[:-1].strip()
    try:
        val_f = float(s.replace(",", ""))
        if not is_pct_sign and 0.0 < abs(val_f) <= 1.0:
            val_f = val_f * 100.0
        return -val_f if negative else val_f
    except ValueError:
        return None


def parse_numeric_range(val: Any) -> tuple[float | None, float | None, float | None]:
    """Parse a range string like '5,000 - 10,000', '10 to 20', or '50' into (min, max, mid)."""
    if val is None or pd.isna(val):
        return None, None, None
    s = str(val).strip()
    if not s:
        return None, None, None
    parts = re.split(r"\s*-\s*|\s+to\s+", s, flags=re.IGNORECASE)
    if len(parts) == 2:
        p1, _ = parse_currency(parts[0])
        p2, _ = parse_currency(parts[1])
        if p1 is not None and p2 is not None:
            low, high = min(p1, p2), max(p1, p2)
            return low, high, (low + high) / 2.0
    elif len(parts) == 1:
        p, _ = parse_currency(parts[0])
        if p is not None:
            return p, p, p
    return None, None, None


def _infer_semantic_types(frame: pd.DataFrame, base_types: dict[str, str]) -> dict[str, str]:
    """Infer rich semantic types (percentage, categorical, numeric_range, etc.)."""
    out: dict[str, str] = {}
    for col in frame.columns:
        b_type = base_types.get(str(col), "text")
        if b_type in ("integer", "float", "numeric_string"):
            out[str(col)] = "numeric"
        elif b_type in ("datetime", "datetime_string"):
            out[str(col)] = "datetime"
        elif b_type in ("currency", "url", "boolean", "empty"):
            out[str(col)] = b_type
        else:
            series = frame[col].dropna()
            if series.empty:
                out[str(col)] = "empty"
                continue
            str_series = series.astype(str).str.strip()
            # Percentage detection
            pct_matches = str_series.str.endswith("%").sum()
            if len(str_series) > 0 and pct_matches >= max(1, int(0.7 * len(str_series))):
                out[str(col)] = "percentage"
                continue
            # Numeric range detection
            range_matches = str_series.str.contains(r"^\s*[\$₹€£]?\s*\d+[\d,]*\s*(?:-|to)\s*[\$₹€£]?\s*\d+[\d,]*", regex=True, case=False).sum()
            if len(str_series) > 0 and range_matches >= max(1, int(0.6 * len(str_series))):
                out[str(col)] = "numeric_range"
                continue
            # Categorical detection: low unique values
            uniq = series.nunique()
            if 1 <= uniq <= 30 and len(str_series) >= uniq:
                out[str(col)] = "categorical"
                continue
            out[str(col)] = "text"
    return out


def repair_unnamed_columns(
    frame: pd.DataFrame,
    custom_names: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], CleaningOperation | None]:
    """Detect and rename columns named 'Unnamed: X' or 'unnamed_column', with user override support."""
    custom = custom_names or {}
    new_cols = []
    renamed: dict[str, str] = {}
    for idx, col in enumerate(frame.columns):
        col_str = str(col).strip()
        is_unnamed = col_str.startswith("Unnamed:") or col_str.startswith("unnamed_column")
        if is_unnamed:
            if col_str in custom and custom[col_str].strip():
                new_name = custom[col_str].strip()
            else:
                new_name = f"Column_{idx + 1}"
            renamed[col_str] = new_name
            new_cols.append(new_name)
        else:
            new_cols.append(col_str)

    if not renamed:
        return frame, {}, None

    updated = frame.copy()
    updated.columns = new_cols
    op = CleaningOperation(
        step="repair_unnamed_headers",
        description=f"Repaired {len(renamed)} unnamed header(s) with descriptive names",
        impact="Neutralized unnamed column artifacts from spreadsheet layout",
        affected_columns=list(renamed.values()),
        rows_affected=0,
    )
    return updated, renamed, op


def apply_transformations(
    frame: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    exclude_row_indices: list[int] | None = None,
    rename_cols: dict[str, str] | None = None,
    header_row: int | None = None,
    exclude_columns: list[str] | None = None,
    rename_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[CleaningOperation]]:
    """Apply interactive user adjustments: header override, column exclusion, row exclusion, and column renames."""
    updated = frame.copy()
    ops: list[CleaningOperation] = []

    if header_row is not None and isinstance(header_row, int) and 0 <= header_row < len(updated):
        new_headers = [str(x).strip() for x in updated.iloc[header_row]]
        updated = updated.iloc[header_row + 1:].copy().reset_index(drop=True)
        updated.columns = new_headers
        ops.append(
            CleaningOperation(
                step="override_header_row",
                description=f"Promoted row {header_row + 1} to column headers",
                impact=f"Reindexed table with row {header_row + 1} headers",
                rows_affected=header_row + 1,
            )
        )

    effective_exclude = exclude_cols or exclude_columns
    effective_rename = rename_cols or rename_columns

    if effective_exclude:
        valid_cols = [c for c in effective_exclude if c in updated.columns]
        if valid_cols:
            updated = updated.drop(columns=valid_cols)
            ops.append(
                CleaningOperation(
                    step="exclude_columns",
                    description=f"Excluded {len(valid_cols)} irrelevant or sensitive column(s)",
                    impact=f"Dropped columns: {', '.join(valid_cols)}",
                    affected_columns=valid_cols,
                )
            )

    if exclude_row_indices:
        valid_indices = [idx for idx in exclude_row_indices if idx in updated.index]
        if valid_indices:
            updated = updated.drop(index=valid_indices)
            ops.append(
                CleaningOperation(
                    step="exclude_rows",
                    description=f"Excluded {len(valid_indices)} irrelevant row(s)",
                    impact=f"Reduced row count from {len(frame)} to {len(updated)}",
                    rows_affected=len(valid_indices),
                )
            )

    if effective_rename:
        valid_renames = {k: v for k, v in effective_rename.items() if k in updated.columns and v.strip()}
        if valid_renames:
            updated = updated.rename(columns=valid_renames)
            ops.append(
                CleaningOperation(
                    step="rename_columns",
                    description=f"Renamed {len(valid_renames)} column(s)",
                    impact=f"Custom headers applied: {list(valid_renames.values())}",
                    affected_columns=list(valid_renames.values()),
                )
            )

    return updated, ops


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
    cleaning_log: list[CleaningOperation] = []

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
                df_raw = None
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

    if h_idx > 0:
        cleaning_log.append(
            CleaningOperation(
                step="header_selection",
                description=f"Selected row {h_idx + 1} (0-indexed {h_idx}) as table header row",
                impact=f"Skipped {h_idx} leading non-tabular or introductory rows",
                rows_affected=h_idx,
            )
        )

    if detected_title:
        cleaning_log.append(
            CleaningOperation(
                step="title_banner_separation",
                description=f"Detected and separated title banner: '{detected_title}'",
                impact="Preserved title context while excluding banner from table records",
                rows_affected=1,
            )
        )

    # Clean whitespace and drop completely blank rows
    init_rows = len(frame)
    frame = frame.dropna(how="all").copy()
    empty_dropped = init_rows - len(frame)
    if empty_dropped > 0:
        cleaning_log.append(
            CleaningOperation(
                step="drop_empty_rows",
                description=f"Dropped {empty_dropped} completely blank row(s)",
                impact="Eliminated empty spreadsheet rows from records",
                rows_affected=empty_dropped,
            )
        )

    # Strip string columns of leading/trailing spaces and convert empty strings to NA
    text_cols_stripped = []
    for col in frame.columns:
        if frame[col].dtype == object or isinstance(frame[col].dtype, pd.StringDtype):
            frame[col] = frame[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            frame[col] = frame[col].replace("", np.nan)
            text_cols_stripped.append(str(col))

    if text_cols_stripped:
        cleaning_log.append(
            CleaningOperation(
                step="strip_whitespace",
                description=f"Trimmed whitespace and converted empty strings to NA in {len(text_cols_stripped)} text column(s)",
                impact="Standardized string fields and ensured missing value consistency",
                affected_columns=text_cols_stripped,
            )
        )

    frame = frame.dropna(how="all")

    if frame.empty:
        raise ValidationError("The table has no rows.")

    frame, duplicates, orig_map = _normalize_columns(frame)
    if duplicates:
        warnings.append("Duplicate column names were renamed to keep them unique.")
        cleaning_log.append(
            CleaningOperation(
                step="deduplicate_columns",
                description=f"Renamed {len(duplicates)} duplicate column header(s)",
                impact=f"Appended unique suffixes to maintain distinct indexing: {duplicates}",
                affected_columns=duplicates,
            )
        )

    unnamed_cols = [
        str(c)
        for c in frame.columns
        if str(c).startswith("Unnamed:") or str(c) == "unnamed_column" or str(c).startswith("unnamed_column__")
    ]
    if unnamed_cols:
        cleaning_log.append(
            CleaningOperation(
                step="detect_unnamed_headers",
                description=f"Detected {len(unnamed_cols)} unnamed column header(s)",
                impact="Flagged for user review and repair",
                affected_columns=unnamed_cols,
            )
        )

    if detected_title:
        warnings.append(f"Detected and separated title row: '{detected_title}'")

    detected_index = None
    first_col = frame.columns[0]
    if first_col.lower() in INDEX_COLUMN_CANDIDATES:
        detected_index = first_col
        warnings.append(f"Column '{first_col}' identified as row index / sequence number.")

    inferred = _infer_types(frame)
    sem_types = _infer_semantic_types(frame, inferred)
    excluded_rows = max(0, raw_shape[0] - frame.shape[0])
    raw_prev = preview(df_raw, 10) if df_raw is not None else []

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
        cleaning_log=cleaning_log,
        unnamed_columns=unnamed_cols,
        semantic_types=sem_types,
        raw_preview=raw_prev,
        raw_frame=df_raw,
    )


def preview(frame: pd.DataFrame, rows: int) -> list[dict[str, Any]]:
    return frame.head(rows).where(frame.head(rows).notna(), None).to_dict(orient="records")
