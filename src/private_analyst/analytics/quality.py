from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataQualityReport:
    row_count: int
    column_count: int
    missing_by_column: dict[str, int]
    missing_pct_by_column: dict[str, float]
    duplicate_row_count: int
    duplicate_column_names: list[str]
    constant_columns: list[str]
    mixed_type_columns: list[str]
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_quality(frame: pd.DataFrame, duplicate_columns: list[str] | None = None) -> DataQualityReport:
    missing = {str(c): int(frame[c].isna().sum()) for c in frame.columns}
    n = max(int(frame.shape[0]), 1)
    missing_pct = {k: round(100.0 * v / n, 4) for k, v in missing.items()}
    dup_rows = int(frame.duplicated().sum())
    constants: list[str] = []
    mixed: list[str] = []
    for col in frame.columns:
        series = frame[col]
        if series.nunique(dropna=True) <= 1:
            constants.append(str(col))
        if series.dtype == object:
            non_null = series.dropna()
            if non_null.empty:
                continue
            numericish = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if 0.1 < numericish < 0.9:
                mixed.append(str(col))
    issues: list[str] = []
    if any(v > 0 for v in missing.values()):
        issues.append("Missing values were detected in one or more columns.")
    if dup_rows:
        issues.append(f"{dup_rows} fully duplicated row(s) were detected.")
    if duplicate_columns:
        issues.append("Duplicate column names were present in the source file.")
    if constants:
        issues.append("One or more columns have a single distinct value.")
    if mixed:
        issues.append("Some text columns appear to mix numbers and non-numeric values.")
    if not issues:
        issues.append("No automated data-quality issues were flagged.")
    return DataQualityReport(
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        missing_by_column=missing,
        missing_pct_by_column=missing_pct,
        duplicate_row_count=dup_rows,
        duplicate_column_names=list(duplicate_columns or []),
        constant_columns=constants,
        mixed_type_columns=mixed,
        issues=issues,
    )
