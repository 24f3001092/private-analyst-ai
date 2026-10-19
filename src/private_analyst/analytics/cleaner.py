from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any

import numpy as np
import pandas as pd

from private_analyst.analytics.quality import DataQualityReport, assess_quality
from private_analyst.ingestion.loader import (
    CleaningOperation,
    parse_currency,
    parse_percentage,
)


@dataclass
class CleaningRecommendation:
    id: str
    title: str
    category: str
    description: str
    affected_columns: list[str] = field(default_factory=list)
    affected_rows_count: int = 0
    estimated_impact: str = ""
    is_lossy: bool = False
    default_enabled: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningPlanResult:
    recommendations: list[CleaningRecommendation]
    total_issues_found: int
    is_clean: bool


@dataclass
class CleaningExecutionSummary:
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    operations_applied: list[CleaningOperation] = field(default_factory=list)
    operations_skipped: list[str] = field(default_factory=list)


def generate_cleaning_recommendations(
    frame: pd.DataFrame,
    quality: DataQualityReport | None = None,
) -> list[CleaningRecommendation]:
    """Inspect dataset and generate deterministic, prioritized cleaning recommendations.

    Categorizes actions into safe (lossless) and lossy/confirmable operations.
    """
    recs: list[CleaningRecommendation] = []
    if frame.empty:
        return recs

    n_rows = len(frame)
    n_cols = len(frame.columns)

    # 1. Fully blank rows
    blank_row_mask = frame.isna().all(axis=1) | frame.apply(
        lambda row: all(str(v).strip() == "" or pd.isna(v) for v in row), axis=1
    )
    blank_row_count = int(blank_row_mask.sum())
    if blank_row_count > 0:
        recs.append(
            CleaningRecommendation(
                id="drop_empty_rows",
                title="Drop Fully Blank Rows",
                category="Structure",
                description=f"Identified {blank_row_count} completely blank row(s) containing no data values.",
                affected_columns=list(frame.columns),
                affected_rows_count=blank_row_count,
                estimated_impact=f"Reduces row count from {n_rows} to {n_rows - blank_row_count}.",
                is_lossy=False,
                default_enabled=True,
            )
        )

    # 2. Unnamed, blank, or artifact column headers
    unnamed_cols = [
        str(c)
        for c in frame.columns
        if str(c).strip().startswith("Unnamed:")
        or str(c).strip().lower().startswith("unnamed_column")
        or str(c).strip() == ""
    ]
    if unnamed_cols:
        recs.append(
            CleaningRecommendation(
                id="repair_unnamed_headers",
                title="Repair Unnamed Column Headers",
                category="Structure",
                description=f"Detected {len(unnamed_cols)} unnamed header(s) resulting from spreadsheet formatting: {', '.join(unnamed_cols[:5])}",
                affected_columns=unnamed_cols,
                affected_rows_count=0,
                estimated_impact="Assigns clean, deterministic names (e.g. column_1, column_2) to ensure predictable indexing.",
                is_lossy=False,
                default_enabled=True,
            )
        )

    # 3. Duplicate column headers (with mangled suffixes like .1 or __1)
    dup_cols = [str(c) for c in frame.columns if re.search(r"(__\d+|\.\d+)$", str(c))]
    if dup_cols:
        recs.append(
            CleaningRecommendation(
                id="deduplicate_headers",
                title="Standardize Duplicate Header Suffixes",
                category="Structure",
                description=f"Detected {len(dup_cols)} duplicate column name(s) with disambiguation suffixes.",
                affected_columns=dup_cols,
                affected_rows_count=0,
                estimated_impact="Standardizes duplicate column headers with clean canonical aliases.",
                is_lossy=False,
                default_enabled=True,
            )
        )

    # 4. Text whitespace trimming & empty string standardization
    text_cols_with_spaces = []
    text_spaces_rows = 0
    for col in frame.columns:
        s = frame[col]
        if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
            str_vals = s.dropna().astype(str)
            needs_trim = (str_vals != str_vals.str.strip()).sum()
            empty_strs = (str_vals == "").sum()
            if needs_trim > 0 or empty_strs > 0:
                text_cols_with_spaces.append(str(col))
                text_spaces_rows = max(text_spaces_rows, int(needs_trim + empty_strs))

    if text_cols_with_spaces:
        recs.append(
            CleaningRecommendation(
                id="strip_whitespace",
                title="Trim Text Whitespace & Normalize Empty Strings",
                category="Whitespace",
                description=f"Found unstripped whitespace or empty string cells in {len(text_cols_with_spaces)} text column(s).",
                affected_columns=text_cols_with_spaces,
                affected_rows_count=text_spaces_rows,
                estimated_impact="Eliminates leading/trailing spaces and converts blank text to NA for consistent missingness.",
                is_lossy=False,
                default_enabled=True,
            )
        )

    # 5. Exact duplicate rows
    dup_row_count = int(frame.duplicated().sum())
    if dup_row_count > 0:
        recs.append(
            CleaningRecommendation(
                id="deduplicate_rows",
                title="Remove Exact Duplicate Rows",
                category="Duplicates",
                description=f"Found {dup_row_count} exact duplicate row(s) matching other records across all fields.",
                affected_columns=list(frame.columns),
                affected_rows_count=dup_row_count,
                estimated_impact=f"Retains primary records and drops {dup_row_count} redundant rows.",
                is_lossy=True,
                default_enabled=False,  # Lossy: require user confirmation
            )
        )

    # 6. Numeric strings stored as text
    numeric_str_cols = []
    for col in frame.columns:
        s = frame[col]
        if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
            non_null = s.dropna()
            if len(non_null) >= 3:
                str_s = non_null.astype(str).str.strip().str.replace(",", "", regex=False)
                coerced = pd.to_numeric(str_s, errors="coerce")
                valid_ratio = coerced.notna().mean()
                if valid_ratio >= 0.90:
                    numeric_str_cols.append(str(col))

    if numeric_str_cols:
        recs.append(
            CleaningRecommendation(
                id="convert_numeric_strings",
                title="Convert Numeric Strings to Numbers",
                category="Types",
                description=f"Detected numeric numbers formatted as text strings in {len(numeric_str_cols)} column(s): {', '.join(numeric_str_cols)}.",
                affected_columns=numeric_str_cols,
                affected_rows_count=0,
                estimated_impact="Casts text columns to proper numeric types to enable aggregation and statistics.",
                is_lossy=False,
                default_enabled=True,
            )
        )

    # 7. Currency strings
    currency_cols = []
    for col in frame.columns:
        s = frame[col]
        if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
            non_null = s.dropna()
            if len(non_null) >= 2:
                sample = non_null.astype(str).str.strip()
                matches = sample.str.contains(r"[₹\$€£¥]|(?:inr|usd|eur|gbp)", regex=True, case=False).mean()
                if matches >= 0.60:
                    currency_cols.append(str(col))

    if currency_cols:
        recs.append(
            CleaningRecommendation(
                id="parse_currencies",
                title="Parse Currency Strings to Numeric Amounts",
                category="Types",
                description=f"Detected currency formatted values in: {', '.join(currency_cols)}.",
                affected_columns=currency_cols,
                affected_rows_count=0,
                estimated_impact="Extracts numeric float amounts while preserving currency indicators.",
                is_lossy=False,
                default_enabled=True,
            )
        )

    # 8. Percentage strings
    pct_cols = []
    for col in frame.columns:
        s = frame[col]
        if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
            non_null = s.dropna()
            if len(non_null) >= 2:
                sample = non_null.astype(str).str.strip()
                matches = sample.str.endswith("%").mean()
                if matches >= 0.70:
                    pct_cols.append(str(col))

    if pct_cols:
        recs.append(
            CleaningRecommendation(
                id="parse_percentages",
                title="Convert Percentages to Numeric Decimals",
                category="Types",
                description=f"Detected percentage strings in: {', '.join(pct_cols)}.",
                affected_columns=pct_cols,
                affected_rows_count=0,
                estimated_impact="Converts percentage values to numeric floats (e.g. '25%' -> 25.0).",
                is_lossy=False,
                default_enabled=True,
            )
        )

    # 9. Completely empty columns
    empty_cols = [str(c) for c in frame.columns if frame[c].isna().all()]
    if empty_cols:
        recs.append(
            CleaningRecommendation(
                id="drop_empty_cols",
                title="Drop Completely Empty Columns",
                category="Structure",
                description=f"Detected {len(empty_cols)} column(s) with 100% missing values: {', '.join(empty_cols)}.",
                affected_columns=empty_cols,
                affected_rows_count=0,
                estimated_impact=f"Removes {len(empty_cols)} non-informative empty column(s).",
                is_lossy=True,
                default_enabled=False,  # Require user confirmation
            )
        )

    # 10. Constant columns (single unique value)
    constant_cols = [
        str(c)
        for c in frame.columns
        if str(c) not in empty_cols and frame[c].nunique(dropna=True) == 1
    ]
    if constant_cols:
        recs.append(
            CleaningRecommendation(
                id="drop_constant_columns",
                title="Drop Constant Columns",
                category="Structure",
                description=f"Detected {len(constant_cols)} column(s) with identical value across all rows: {', '.join(constant_cols)}.",
                affected_columns=constant_cols,
                affected_rows_count=0,
                estimated_impact="Drops columns that provide zero variance or predictive value.",
                is_lossy=True,
                default_enabled=False,
            )
        )

    return recs


def apply_cleaning_plan(
    frame: pd.DataFrame,
    selected_op_ids: list[str],
    options: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[CleaningOperation]]:
    """Execute a user-approved cleaning plan deterministically and produce an audit trail."""
    opts = options or {}
    updated = frame.copy()
    ops: list[CleaningOperation] = []

    # 1. Drop empty rows
    if "drop_empty_rows" in selected_op_ids:
        init_rows = len(updated)
        blank_mask = updated.isna().all(axis=1) | updated.apply(
            lambda row: all(str(v).strip() == "" or pd.isna(v) for v in row), axis=1
        )
        if blank_mask.any():
            updated = updated[~blank_mask].copy().reset_index(drop=True)
            dropped = init_rows - len(updated)
            if dropped > 0:
                ops.append(
                    CleaningOperation(
                        step="drop_empty_rows",
                        description=f"Dropped {dropped} completely blank row(s)",
                        impact=f"Cleaned table row count from {init_rows} to {len(updated)}",
                        rows_affected=dropped,
                        status="applied",
                        before_after=f"{init_rows} rows -> {len(updated)} rows",
                    )
                )

    # 2. Repair unnamed headers
    if "repair_unnamed_headers" in selected_op_ids:
        repaired_df, renamed, op = repair_unnamed_headers(updated, opts.get("custom_column_names"))
        if op:
            updated = repaired_df
            ops.append(op)

    # 3. Strip text whitespace and convert empty strings to NA
    if "strip_whitespace" in selected_op_ids:
        stripped_cols = []
        for col in updated.columns:
            if updated[col].dtype == object or isinstance(updated[col].dtype, pd.StringDtype):
                s = updated[col].map(lambda v: v.strip() if isinstance(v, str) else v)
                s = s.replace("", np.nan)
                if not s.equals(updated[col]):
                    updated[col] = s
                    stripped_cols.append(str(col))

        if stripped_cols:
            ops.append(
                CleaningOperation(
                    step="strip_whitespace",
                    description=f"Trimmed text whitespace and normalized empty strings across {len(stripped_cols)} column(s)",
                    impact="Consistent string formats and standardized missing value representations",
                    affected_columns=stripped_cols,
                    rows_affected=len(updated),
                    status="applied",
                )
            )

    # 4. Convert numeric strings
    if "convert_numeric_strings" in selected_op_ids:
        converted_cols = []
        for col in updated.columns:
            s = updated[col]
            if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
                non_null = s.dropna()
                if len(non_null) >= 3:
                    str_s = non_null.astype(str).str.strip().str.replace(",", "", regex=False)
                    coerced = pd.to_numeric(str_s, errors="coerce")
                    if coerced.notna().mean() >= 0.90:
                        # Convert entire column
                        clean_col = pd.to_numeric(
                            s.astype(str).str.strip().str.replace(",", "", regex=False),
                            errors="coerce",
                        )
                        updated[col] = clean_col
                        converted_cols.append(str(col))

        if converted_cols:
            ops.append(
                CleaningOperation(
                    step="convert_numeric_strings",
                    description=f"Converted {len(converted_cols)} text column(s) to numeric: {', '.join(converted_cols)}",
                    impact="Enabled mathematical and statistical analysis for parsed numeric fields",
                    affected_columns=converted_cols,
                    status="applied",
                )
            )

    # 5. Parse currencies
    if "parse_currencies" in selected_op_ids:
        parsed_curr_cols = []
        for col in updated.columns:
            s = updated[col]
            if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
                non_null = s.dropna()
                if len(non_null) >= 2:
                    sample = non_null.astype(str).str.strip()
                    if sample.str.contains(r"[₹\$€£¥]|(?:inr|usd|eur|gbp)", regex=True, case=False).mean() >= 0.60:
                        parsed_series = s.map(lambda v: parse_currency(v)[0] if pd.notna(v) else np.nan)
                        if parsed_series.notna().sum() > 0:
                            updated[col] = parsed_series
                            parsed_curr_cols.append(str(col))

        if parsed_curr_cols:
            ops.append(
                CleaningOperation(
                    step="parse_currencies",
                    description=f"Parsed currency amounts to numeric floats in {len(parsed_curr_cols)} column(s)",
                    impact="Standardized monetary values for aggregation",
                    affected_columns=parsed_curr_cols,
                    status="applied",
                )
            )

    # 6. Parse percentages
    if "parse_percentages" in selected_op_ids:
        parsed_pct_cols = []
        for col in updated.columns:
            s = updated[col]
            if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
                non_null = s.dropna()
                if len(non_null) >= 2 and non_null.astype(str).str.strip().str.endswith("%").mean() >= 0.70:
                    parsed_series = s.map(lambda v: parse_percentage(v) if pd.notna(v) else np.nan)
                    if parsed_series.notna().sum() > 0:
                        updated[col] = parsed_series
                        parsed_pct_cols.append(str(col))

        if parsed_pct_cols:
            ops.append(
                CleaningOperation(
                    step="parse_percentages",
                    description=f"Parsed percentage text to numeric values in {len(parsed_pct_cols)} column(s)",
                    impact="Converted formatted percentage strings into numeric floats",
                    affected_columns=parsed_pct_cols,
                    status="applied",
                )
            )

    # 7. Deduplicate rows
    if "deduplicate_rows" in selected_op_ids:
        init_rows = len(updated)
        updated = updated.drop_duplicates().copy().reset_index(drop=True)
        dropped = init_rows - len(updated)
        if dropped > 0:
            ops.append(
                CleaningOperation(
                    step="deduplicate_rows",
                    description=f"Removed {dropped} duplicate row(s)",
                    impact=f"Reduced row count from {init_rows} to {len(updated)}",
                    rows_affected=dropped,
                    status="applied",
                    before_after=f"{init_rows} rows -> {len(updated)} rows",
                )
            )

    # 8. Drop empty columns
    if "drop_empty_cols" in selected_op_ids:
        empty_cols = [str(c) for c in updated.columns if updated[c].isna().all()]
        if empty_cols:
            updated = updated.drop(columns=empty_cols)
            ops.append(
                CleaningOperation(
                    step="drop_empty_cols",
                    description=f"Dropped {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}",
                    impact="Eliminated 100% missing columns from dataset",
                    affected_columns=empty_cols,
                    status="applied",
                )
            )

    # 9. Drop constant columns
    if "drop_constant_columns" in selected_op_ids:
        const_cols = [str(c) for c in updated.columns if updated[c].nunique(dropna=True) == 1]
        if const_cols:
            updated = updated.drop(columns=const_cols)
            ops.append(
                CleaningOperation(
                    step="drop_constant_columns",
                    description=f"Dropped {len(const_cols)} constant column(s): {', '.join(const_cols)}",
                    impact="Removed zero-variance columns",
                    affected_columns=const_cols,
                    status="applied",
                )
            )

    return updated, ops


def repair_unnamed_headers(
    frame: pd.DataFrame,
    custom_names: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], CleaningOperation | None]:
    """Auto-repair unnamed headers (e.g. 'Unnamed: 1' or 'unnamed_column') to clean deterministic names."""
    custom = custom_names or {}
    new_cols = []
    renamed: dict[str, str] = {}

    for idx, col in enumerate(frame.columns):
        col_str = str(col).strip()
        is_unnamed = (
            col_str.startswith("Unnamed:")
            or col_str.lower().startswith("unnamed_column")
            or col_str == ""
        )
        if is_unnamed:
            if col_str in custom and custom[col_str].strip():
                new_name = custom[col_str].strip()
            else:
                new_name = f"column_{idx + 1}"
            # Ensure new_name doesn't collide with existing columns
            cand_name = new_name
            k = 1
            while cand_name in new_cols or (cand_name in frame.columns and cand_name != col_str):
                cand_name = f"{new_name}_{k}"
                k += 1
            renamed[col_str] = cand_name
            new_cols.append(cand_name)
        else:
            new_cols.append(col_str)

    if not renamed:
        return frame, {}, None

    updated = frame.copy()
    updated.columns = new_cols
    op = CleaningOperation(
        step="repair_unnamed_headers",
        description=f"Repaired {len(renamed)} unnamed header(s) with deterministic names: {renamed}",
        impact="Eliminated spreadsheet layout unnamed column artifacts",
        affected_columns=list(renamed.values()),
        rows_affected=0,
        status="applied",
    )
    return updated, renamed, op


def apply_header_override(
    raw_frame: pd.DataFrame,
    header_row: int,
) -> tuple[pd.DataFrame, CleaningOperation]:
    """Reindex dataframe from raw table promoting the specified row to column headers."""
    if header_row < 0 or header_row >= len(raw_frame):
        raise ValueError(f"Header row index {header_row} is out of bounds (0 to {len(raw_frame)-1}).")

    raw_headers = [str(x).strip() or f"column_{i+1}" for i, x in enumerate(raw_frame.iloc[header_row])]
    # Deduplicate headers if identical
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in raw_headers:
        if h in seen:
            seen[h] += 1
            clean_headers.append(f"{h}__{seen[h]}")
        else:
            seen[h] = 0
            clean_headers.append(h)

    updated = raw_frame.iloc[header_row + 1:].copy().reset_index(drop=True)
    updated.columns = clean_headers

    # Drop blank rows resulting from promo
    init_r = len(updated)
    updated = updated.dropna(how="all").reset_index(drop=True)

    op = CleaningOperation(
        step="override_header_row",
        description=f"Promoted row {header_row + 1} (0-indexed {header_row}) to table column headers",
        impact=f"Reindexed {len(updated)} data rows using row {header_row + 1} as header schema",
        rows_affected=header_row + 1 + (init_r - len(updated)),
        status="applied",
    )
    return updated, op


def apply_column_management(
    frame: pd.DataFrame,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    rename_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[CleaningOperation]]:
    """Manage table schema: select columns to keep/exclude, or rename columns."""
    updated = frame.copy()
    ops: list[CleaningOperation] = []

    if include_columns is not None:
        valid_includes = [c for c in include_columns if c in updated.columns]
        if valid_includes and len(valid_includes) < len(updated.columns):
            dropped_cols = [c for c in updated.columns if c not in valid_includes]
            updated = updated[valid_includes].copy()
            ops.append(
                CleaningOperation(
                    step="filter_columns",
                    description=f"Retained {len(valid_includes)} column(s), dropped {len(dropped_cols)} column(s)",
                    impact=f"Filtered schema to: {valid_includes}",
                    affected_columns=dropped_cols,
                    status="applied",
                )
            )

    if exclude_columns:
        valid_excludes = [c for c in exclude_columns if c in updated.columns]
        if valid_excludes:
            updated = updated.drop(columns=valid_excludes)
            ops.append(
                CleaningOperation(
                    step="exclude_columns",
                    description=f"Excluded {len(valid_excludes)} column(s): {', '.join(valid_excludes)}",
                    impact="Removed selected columns from active working dataset",
                    affected_columns=valid_excludes,
                    status="applied",
                )
            )

    if rename_columns:
        valid_renames = {k: v.strip() for k, v in rename_columns.items() if k in updated.columns and v.strip() and v.strip() != k}
        if valid_renames:
            updated = updated.rename(columns=valid_renames)
            ops.append(
                CleaningOperation(
                    step="rename_columns",
                    description=f"Renamed {len(valid_renames)} column(s): {valid_renames}",
                    impact="Updated column labels in active working dataset",
                    affected_columns=list(valid_renames.values()),
                    status="applied",
                )
            )

    return updated, ops


def handle_missing_values(
    frame: pd.DataFrame,
    column: str,
    strategy: str,
    fill_value: Any = None,
) -> tuple[pd.DataFrame, CleaningOperation]:
    """Handle missing values for a specific column using a user-selected strategy."""
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    updated = frame.copy()
    initial_missing = int(updated[column].isna().sum())
    if initial_missing == 0:
        return updated, CleaningOperation(
            step="handle_missing_values",
            description=f"No missing values in '{column}'",
            impact="No changes made",
            affected_columns=[column],
            rows_affected=0,
            status="skipped",
        )

    if strategy == "drop_rows":
        updated = updated.dropna(subset=[column]).copy().reset_index(drop=True)
        op = CleaningOperation(
            step="drop_missing_rows",
            description=f"Dropped {initial_missing} row(s) containing missing values in '{column}'",
            impact=f"Reduced row count from {len(frame)} to {len(updated)}",
            affected_columns=[column],
            rows_affected=initial_missing,
            status="applied",
        )
    elif strategy == "impute_mean":
        mean_val = pd.to_numeric(updated[column], errors="coerce").mean()
        updated[column] = updated[column].fillna(mean_val)
        op = CleaningOperation(
            step="impute_mean",
            description=f"Imputed {initial_missing} missing values in '{column}' with mean ({mean_val:.2f})",
            impact=f"Replaced nulls with mean value {mean_val:.2f}",
            affected_columns=[column],
            rows_affected=initial_missing,
            status="applied",
        )
    elif strategy == "impute_median":
        med_val = pd.to_numeric(updated[column], errors="coerce").median()
        updated[column] = updated[column].fillna(med_val)
        op = CleaningOperation(
            step="impute_median",
            description=f"Imputed {initial_missing} missing values in '{column}' with median ({med_val:.2f})",
            impact=f"Replaced nulls with median value {med_val:.2f}",
            affected_columns=[column],
            rows_affected=initial_missing,
            status="applied",
        )
    elif strategy == "impute_mode":
        mode_val = updated[column].mode().dropna()
        fill = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        updated[column] = updated[column].fillna(fill)
        op = CleaningOperation(
            step="impute_mode",
            description=f"Imputed {initial_missing} missing values in '{column}' with mode ('{fill}')",
            impact=f"Replaced nulls with most frequent value '{fill}'",
            affected_columns=[column],
            rows_affected=initial_missing,
            status="applied",
        )
    elif strategy == "impute_constant":
        fill = fill_value if fill_value is not None else "Missing"
        updated[column] = updated[column].fillna(fill)
        op = CleaningOperation(
            step="impute_constant",
            description=f"Imputed {initial_missing} missing values in '{column}' with constant ('{fill}')",
            impact=f"Replaced nulls with constant '{fill}'",
            affected_columns=[column],
            rows_affected=initial_missing,
            status="applied",
        )
    else:
        raise ValueError(f"Unknown missing-value handling strategy: {strategy}")

    return updated, op


def filter_rows(
    frame: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> tuple[pd.DataFrame, CleaningOperation]:
    """Filter rows based on column condition."""
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    updated = frame.copy()
    init_rows = len(updated)

    if operator == "equals":
        mask = updated[column].astype(str) == str(value)
    elif operator == "not_equals":
        mask = updated[column].astype(str) != str(value)
    elif operator == "contains":
        mask = updated[column].astype(str).str.contains(str(value), case=False, na=False)
    elif operator == "greater_than":
        num_col = pd.to_numeric(updated[column], errors="coerce")
        mask = num_col > float(value)
    elif operator == "less_than":
        num_col = pd.to_numeric(updated[column], errors="coerce")
        mask = num_col < float(value)
    elif operator == "not_null":
        mask = updated[column].notna() & (updated[column].astype(str).str.strip() != "")
    elif operator == "is_null":
        mask = updated[column].isna() | (updated[column].astype(str).str.strip() == "")
    else:
        raise ValueError(f"Unsupported filter operator: {operator}")

    updated = updated[mask].copy().reset_index(drop=True)
    excluded = init_rows - len(updated)

    op = CleaningOperation(
        step="filter_rows",
        description=f"Filtered rows where '{column}' {operator} '{value}' (kept {len(updated)} rows, removed {excluded})",
        impact=f"Row count changed from {init_rows} to {len(updated)}",
        affected_columns=[column],
        rows_affected=excluded,
        status="applied",
        before_after=f"{init_rows} rows -> {len(updated)} rows",
    )
    return updated, op
