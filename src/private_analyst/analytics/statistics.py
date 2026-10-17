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
    kind: str
    count: int
    missing: int
    unique: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | str | None = None
    max: float | str | None = None
    sum: float | None = None
    evidence: str = "calculated_from_loaded_table"


@dataclass
class DatasetStats:
    row_count: int
    column_count: int
    columns: list[ColumnStats]
    numeric_column_names: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def compute_column_stats(series: pd.Series, name: str) -> ColumnStats:
    missing = int(series.isna().sum())
    count = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
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
    if non_null.empty:
        return ColumnStats(name=name, kind="non_numeric", count=0, missing=missing, unique=0)
    as_text = non_null.astype(str)
    return ColumnStats(
        name=name,
        kind="non_numeric",
        count=count,
        missing=missing,
        unique=unique,
        min=as_text.min(),
        max=as_text.max(),
    )


def compute_dataset_stats(frame: pd.DataFrame) -> DatasetStats:
    columns = [compute_column_stats(frame[col], str(col)) for col in frame.columns]
    numeric = [c.name for c in columns if c.kind == "numeric"]
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
        notes=notes,
    )


def group_compare(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    if group_col not in frame.columns or value_col not in frame.columns:
        raise KeyError("Requested columns are not in the loaded table.")
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
