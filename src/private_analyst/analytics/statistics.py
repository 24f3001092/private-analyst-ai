"""Deterministic statistics. The language model must not invent these numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SAMPLE_STD_DDOF = 1


@dataclass
class ColumnStats:
    name: str
    kind: str  # numeric | categorical | text | url | currency | datetime | empty
    count: int
    missing: int
    unique: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | str | None = None
    max: float | str | None = None
    sum: float | None = None
    top_values: dict[str, int] = field(default_factory=dict)
    top_value: str | None = None
    top_freq: int | None = None
    min_length: int | None = None
    max_length: int | None = None
    mean_length: float | None = None
    evidence: str = "calculated_from_loaded_table"


@dataclass
class DatasetStats:
    row_count: int
    column_count: int
    columns: list[ColumnStats]
    numeric_column_names: list[str]
    categorical_column_names: list[str] = field(default_factory=list)
    text_column_names: list[str] = field(default_factory=list)
    url_column_names: list[str] = field(default_factory=list)
    currency_column_names: list[str] = field(default_factory=list)
    date_column_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def compute_column_stats(series: pd.Series, name: str, inferred_type: str | None = None) -> ColumnStats:
    missing = int(series.isna().sum())
    count = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))

    if count == 0:
        return ColumnStats(name=name, kind="empty", count=0, missing=missing, unique=0)

    if _is_numeric(series):
        finite = _finite_numeric(series)
        if finite.empty:
            return ColumnStats(
                name=name,
                kind="numeric",
                count=count,
                missing=missing,
                unique=unique,
            )
        arr = finite.to_numpy(dtype=float)
        return ColumnStats(
            name=name,
            kind="numeric",
            count=int(finite.shape[0]),
            missing=missing,
            unique=unique,
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr, ddof=SAMPLE_STD_DDOF)) if arr.size > 1 else 0.0,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            sum=float(np.sum(arr)),
        )

    non_null = series.dropna()
    as_text = non_null.astype(str).str.strip()
    lengths = as_text.str.len()
    val_counts = as_text.value_counts()
    top_5 = {str(k): int(v) for k, v in val_counts.head(5).items()}
    top_val = str(val_counts.index[0]) if not val_counts.empty else None
    top_freq = int(val_counts.iloc[0]) if not val_counts.empty else None

    if inferred_type in ("url", "currency", "datetime", "datetime_string"):
        kind = inferred_type
    else:
        kind = "categorical" if unique <= 25 else "text"

    return ColumnStats(
        name=name,
        kind=kind,
        count=count,
        missing=missing,
        unique=unique,
        min=str(as_text.min()) if not as_text.empty else None,
        max=str(as_text.max()) if not as_text.empty else None,
        top_values=top_5,
        top_value=top_val,
        top_freq=top_freq,
        min_length=int(lengths.min()) if not lengths.empty else None,
        max_length=int(lengths.max()) if not lengths.empty else None,
        mean_length=round(float(lengths.mean()), 1) if not lengths.empty else None,
    )


def compute_dataset_stats(frame: pd.DataFrame, inferred_types: dict[str, str] | None = None) -> DatasetStats:
    inferred = inferred_types or {}
    columns = [compute_column_stats(frame[col], str(col), inferred.get(str(col))) for col in frame.columns]
    numeric = [c.name for c in columns if c.kind == "numeric"]
    categorical = [c.name for c in columns if c.kind == "categorical"]
    text = [c.name for c in columns if c.kind == "text"]
    url = [c.name for c in columns if c.kind == "url"]
    currency = [c.name for c in columns if c.kind == "currency"]
    date_cols = [c.name for c in columns if c.kind in ("datetime", "datetime_string")]

    notes = [
        "Standard deviation uses the sample estimator (ddof=1).",
        "Statistics are computed only from the currently loaded table.",
        "Missing values are excluded from numeric summaries.",
    ]
    return DatasetStats(
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        columns=columns,
        numeric_column_names=numeric,
        categorical_column_names=categorical,
        text_column_names=text,
        url_column_names=url,
        currency_column_names=currency,
        date_column_names=date_cols,
        notes=notes,
    )


def frequency_analysis(frame: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    """Compute frequency distribution, percentages, and cumulative shares for any column."""
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' is not in the loaded table.")
    series = frame[column].dropna().astype(str).str.strip()
    if series.empty:
        return pd.DataFrame(columns=["Category", "Count", "Percentage (%)", "Cumulative (%)"])
    total = len(series)
    vc = series.value_counts().head(top_n).reset_index()
    vc.columns = ["Category", "Count"]
    vc["Percentage (%)"] = (vc["Count"] / total * 100.0).round(2)
    vc["Cumulative (%)"] = vc["Percentage (%)"].cumsum().round(2)
    return vc


def group_compare(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Group comparison: computes statistics for numeric values or cross-tabulation for categories."""
    if group_col not in frame.columns or value_col not in frame.columns:
        raise KeyError("Requested columns are not in the loaded table.")

    # If value column is numeric, compute aggregate stats
    if _is_numeric(frame[value_col]):
        numeric = _finite_numeric(frame[value_col])
        work = frame.loc[numeric.index, [group_col]].copy()
        work[value_col] = numeric
        grouped = work.groupby(group_col, dropna=False)[value_col].agg(
            count="count",
            total="sum",
            mean="mean",
            median="median",
            min="min",
            max="max",
            std=lambda s: float(np.std(s.to_numpy(dtype=float), ddof=SAMPLE_STD_DDOF)) if len(s) > 1 else 0.0,
        )
        return grouped.reset_index()

    # Otherwise, compute categorical cross-tabulation frequency
    crosstab = pd.crosstab(frame[group_col], frame[value_col], dropna=False, margins=True, margins_name="Total")
    return crosstab.reset_index()
