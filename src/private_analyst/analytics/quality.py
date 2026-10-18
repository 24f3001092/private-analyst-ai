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
    empty_columns: list[str] = field(default_factory=list)
    quality_score: float = 100.0
    quality_grade: str = "Excellent"
    score_breakdown: dict[str, float] = field(default_factory=dict)
    column_types_summary: dict[str, str] = field(default_factory=dict)
    potential_duplicate_entities: dict[str, list[str]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        return self.quality_grade

    @property
    def total_missing_cells(self) -> int:
        return sum(self.missing_by_column.values())

    @property
    def completeness_ratio(self) -> float:
        total = self.row_count * self.column_count
        return (total - self.total_missing_cells) / total if total > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["grade"] = self.grade
        d["total_missing_cells"] = self.total_missing_cells
        d["completeness_ratio"] = self.completeness_ratio
        return d


def assess_quality(
    frame: pd.DataFrame,
    duplicate_columns: list[str] | None = None,
    inferred_types: dict[str, str] | None = None,
) -> DataQualityReport:
    """Assess data quality across completeness, uniqueness, column health, and type consistency."""
    n_rows = max(int(frame.shape[0]), 1)
    n_cols = max(int(frame.shape[1]), 1)
    total_cells = n_rows * n_cols

    missing = {str(c): int(frame[c].isna().sum()) for c in frame.columns}
    missing_pct = {k: round(100.0 * v / n_rows, 2) for k, v in missing.items()}
    total_missing = sum(missing.values())

    dup_rows = int(frame.duplicated().sum())
    constants: list[str] = []
    mixed: list[str] = []
    empty_cols: list[str] = []
    potential_dups: dict[str, list[str]] = {}

    for col in frame.columns:
        series = frame[col]
        non_null = series.dropna()
        if non_null.empty:
            empty_cols.append(str(col))
            continue
        if series.nunique(dropna=True) <= 1:
            constants.append(str(col))
        if series.dtype == object:
            numericish = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if 0.15 < numericish < 0.85:
                mixed.append(str(col))

            # Detect potential duplicate entities (differing only by whitespace or capitalization)
            str_vals = non_null.astype(str).str.strip()
            unique_raw = set(str_vals)
            lower_map: dict[str, set[str]] = {}
            for v in unique_raw:
                lower_map.setdefault(v.lower(), set()).add(v)
            dups = [f"{list(variants)[0]} / {list(variants)[1]}" for variants in lower_map.values() if len(variants) > 1]
            if dups:
                potential_dups[str(col)] = dups[:5]

    # Transparent mathematical score components (0 to 100 each)
    completeness_score = max(0.0, round(100.0 - (total_missing / total_cells * 100.0), 1))
    uniqueness_score = max(0.0, round(100.0 - (dup_rows / n_rows * 100.0), 1))

    col_health_penalty = (len(empty_cols) * 20.0 + len(constants) * 5.0 + len(duplicate_columns or []) * 10.0) / n_cols * 10.0
    column_health_score = max(0.0, round(100.0 - min(col_health_penalty, 100.0), 1))

    type_penalty = (len(mixed) * 25.0) / n_cols * 10.0
    type_consistency_score = max(0.0, round(100.0 - min(type_penalty, 100.0), 1))

    # Overall weighted score: Completeness 40%, Uniqueness 20%, Column Health 20%, Type Consistency 20%
    overall_score = round(
        0.40 * completeness_score
        + 0.20 * uniqueness_score
        + 0.20 * column_health_score
        + 0.20 * type_consistency_score,
        1,
    )

    if overall_score >= 90.0:
        grade = "Excellent"
    elif overall_score >= 75.0:
        grade = "Good"
    elif overall_score >= 60.0:
        grade = "Fair"
    else:
        grade = "Needs Attention"

    score_breakdown = {
        "completeness": completeness_score,
        "uniqueness": uniqueness_score,
        "column_health": column_health_score,
        "type_consistency": type_consistency_score,
    }

    issues: list[str] = []
    recommendations: list[str] = []

    if total_missing > 0:
        high_missing_cols = [k for k, v in missing_pct.items() if v >= 20.0]
        if high_missing_cols:
            issues.append(f"High missingness (>=20%) in {len(high_missing_cols)} column(s): {', '.join(high_missing_cols[:3])}")
            recommendations.append("Filter or impute columns with high missing rates before analytical aggregations.")
        else:
            issues.append(f"Minor missing values detected ({total_missing} cells across dataset).")
    if dup_rows:
        issues.append(f"{dup_rows} fully duplicated row(s) were detected.")
        recommendations.append("Deduplicate identical rows to prevent skewed counts.")
    if duplicate_columns:
        issues.append("Duplicate column names were renamed to prevent shadowing.")
    if empty_cols:
        issues.append(f"Completely empty column(s): {', '.join(empty_cols)}")
        recommendations.append("Drop empty columns before training models.")
    if mixed:
        issues.append(f"Mixed numeric and text values in: {', '.join(mixed)}")
        recommendations.append("Cast mixed columns explicitly to text or numeric.")
    if potential_dups:
        issues.append(f"Near-duplicate text entities detected in: {', '.join(potential_dups.keys())}")
        recommendations.append("Normalize capitalization and spacing in entity names.")
    if not issues:
        issues.append("No automated data-quality issues were flagged.")
        recommendations.append("Dataset is clean and ready for analysis.")

    return DataQualityReport(
        row_count=n_rows,
        column_count=n_cols,
        missing_by_column=missing,
        missing_pct_by_column=missing_pct,
        duplicate_row_count=dup_rows,
        duplicate_column_names=list(duplicate_columns or []),
        constant_columns=constants,
        mixed_type_columns=mixed,
        empty_columns=empty_cols,
        quality_score=overall_score,
        quality_grade=grade,
        score_breakdown=score_breakdown,
        column_types_summary=dict(inferred_types or {}),
        potential_duplicate_entities=potential_dups,
        issues=issues,
        recommendations=recommendations,
    )
