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
    currency_symbol: str | None = None
    parsed_sum: float | None = None
    parsed_mean: float | None = None
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

    currency_sym = None
    parsed_sum = None
    parsed_mean = None
    if inferred_type in ("url", "currency", "datetime", "datetime_string"):
        kind = inferred_type
        if kind == "currency":
            from private_analyst.ingestion.loader import parse_currency
            parsed = [parse_currency(v) for v in non_null]
            valid_nums = [p[0] for p in parsed if p[0] is not None]
            symbols = [p[1] for p in parsed if p[1] is not None]
            if symbols:
                currency_sym = symbols[0]
            if valid_nums:
                parsed_sum = round(float(sum(valid_nums)), 2)
                parsed_mean = round(float(np.mean(valid_nums)), 2)
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
        currency_symbol=currency_sym,
        parsed_sum=parsed_sum,
        parsed_mean=parsed_mean,
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


def compute_correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> dict[str, Any]:
    """Compute pairwise correlation for numeric columns with non-zero variance."""
    numeric_cols = [c for c in frame.columns if _is_numeric(frame[c])]
    valid_cols = []
    for c in numeric_cols:
        series = _finite_numeric(frame[c])
        if series.nunique() > 1:
            valid_cols.append(c)

    if len(valid_cols) < 2:
        return {
            "status": "insufficient_features",
            "columns": valid_cols,
            "matrix": {},
            "top_correlations": [],
            "caveat": "Correlation analysis requires at least two numeric columns with variation.",
        }

    clean_sub = frame[valid_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean_sub) < 3:
        return {
            "status": "insufficient_data",
            "columns": valid_cols,
            "matrix": {},
            "top_correlations": [],
            "caveat": "At least 3 complete observations are required to compute correlations.",
        }

    corr_df = clean_sub.corr(method=method).round(3)
    pairs = []
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            c1, c2 = valid_cols[i], valid_cols[j]
            val = float(corr_df.loc[c1, c2])
            if np.isfinite(val):
                rel = (
                    "Strong Positive"
                    if val >= 0.7
                    else (
                        "Moderate Positive"
                        if val >= 0.4
                        else (
                            "Weak / None"
                            if val > -0.4
                            else ("Moderate Negative" if val > -0.7 else "Strong Negative")
                        )
                    )
                )
                pairs.append({
                    "col1": c1,
                    "col2": c2,
                    "correlation": val,
                    "abs_correlation": abs(val),
                    "relationship": rel,
                })
    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return {
        "status": "ok",
        "method": method,
        "columns": valid_cols,
        "matrix": corr_df.to_dict(),
        "top_correlations": pairs[:10],
        "caveat": "Correlation indicates linear association, not causation.",
    }


def compute_time_trends(
    frame: pd.DataFrame,
    date_col: str,
    value_col: str | None = None,
    freq: str = "auto",
) -> dict[str, Any]:
    """Aggregate records or numeric metrics over time periods."""
    if date_col not in frame.columns:
        raise KeyError(f"Date column '{date_col}' not found.")

    parsed_dates = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    valid_mask = parsed_dates.notna()
    if valid_mask.sum() < 2:
        return {
            "status": "insufficient_dates",
            "date_col": date_col,
            "records": [],
            "caveat": "Need at least 2 valid dates to compute time trends.",
        }

    sub = pd.DataFrame({"_date": parsed_dates[valid_mask]})
    min_date = sub["_date"].min()
    max_date = sub["_date"].max()
    span_days = int((max_date - min_date).days)

    if freq == "auto":
        if span_days <= 60:
            period_fmt = "%Y-%m-%d"
            freq_label = "Daily"
        elif span_days <= 730:
            period_fmt = "%Y-%m"
            freq_label = "Monthly"
        else:
            period_fmt = "%Y"
            freq_label = "Yearly"
    else:
        period_fmt = "%Y-%m"
        freq_label = "Monthly"

    sub["period"] = sub["_date"].dt.strftime(period_fmt)

    if value_col and value_col in frame.columns and _is_numeric(frame[value_col]):
        sub["value"] = pd.to_numeric(frame.loc[valid_mask, value_col], errors="coerce")
        agg = sub.groupby("period")["value"].agg(
            count="count",
            total="sum",
            mean="mean",
        ).reset_index()
    else:
        agg = sub.groupby("period").size().reset_index(name="count")
        agg["total"] = agg["count"]
        agg["mean"] = 1.0

    agg = agg.sort_values("period")
    return {
        "status": "ok",
        "date_col": date_col,
        "value_col": value_col,
        "freq_label": freq_label,
        "min_date": str(min_date)[:10],
        "max_date": str(max_date)[:10],
        "span_days": span_days,
        "trend_table": agg.to_dict(orient="records"),
        "caveat": "Time aggregations are based on parsed dates and exclude invalid date entries.",
    }


def _is_index_or_artifact(col_name: Any, series: pd.Series, c_stats: ColumnStats | None) -> tuple[bool, str]:
    """Determine if a column is an artifact, sequential index, constant, or uninformative identifier."""
    if c_stats is None:
        return False, ""

    if c_stats.kind == "empty" or c_stats.count == 0:
        return True, "Column is completely empty or missing all values."

    if c_stats.kind == "url":
        return True, "URL link column is not suited for distribution charting."

    if c_stats.unique <= 1:
        return True, f"Column is constant with only {c_stats.unique} unique value."

    col_str = str(col_name).strip()
    col_lower = col_str.lower()

    # Index candidates
    if col_lower in {"#", "id", "s.no", "sl.no", "index", "row_id", "row", "no", "_id"}:
        return True, f"Identified as row sequence or primary key identifier ('{col_name}')."

    # Unnamed spreadsheet artifacts
    if col_lower.startswith("unnamed:") or col_lower == "unnamed_column" or col_lower.startswith("unnamed_column__"):
        return True, f"Unnamed spreadsheet layout artifact ('{col_name}')."

    # Sequential row index numbers (e.g. 1, 2, 3...)
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) >= 3 and non_null.is_monotonic_increasing:
            diffs = non_null.diff().dropna()
            if (diffs == 1).all() and non_null.iloc[0] in (0, 1):
                return True, f"Sequential row index column with monotonic increments ('{col_name}')."

    # High-cardinality text (near-unique identifier)
    if c_stats.kind in ("categorical", "text"):
        if c_stats.count > 5:
            ratio = c_stats.unique / c_stats.count
            if ratio > 0.85 and c_stats.unique > 15:
                return True, f"High-cardinality free text / near-unique identifier ({c_stats.unique} distinct values across {c_stats.count} rows)."

    return False, ""


def recommend_charts(frame: pd.DataFrame, stats: DatasetStats | None = None) -> list[dict[str, Any]]:
    """Recommend the most informative visualizations based on column types and distributions."""
    if stats is None:
        stats = compute_dataset_stats(frame)
    recs: list[dict[str, Any]] = []

    # 1. Frequency distributions for top categorical features
    for col in (stats.categorical_column_names or []):
        c_stats = next((c for c in stats.columns if c.name == col), None)
        is_art, _ = _is_index_or_artifact(col, frame[col], c_stats)
        if is_art:
            continue
        if c_stats and 2 <= c_stats.unique <= 30:
            recs.append({
                "type": "category_bar",
                "column": col,
                "title": f"Distribution of {col}",
                "description": f"Displays counts across the {c_stats.unique} distinct values of '{col}'.",
                "priority": 1,
                "reason": f"Categorical feature with {c_stats.unique} distinct values.",
            })
            if len(recs) >= 2:
                break

    # 2. Histograms for numeric columns
    for col in (stats.numeric_column_names or []):
        c_stats = next((c for c in stats.columns if c.name == col), None)
        is_art, _ = _is_index_or_artifact(col, frame[col], c_stats)
        if is_art:
            continue
        if c_stats and c_stats.unique > 3:
            recs.append({
                "type": "histogram",
                "column": col,
                "title": f"Distribution of {col}",
                "description": f"Examines spread, skewness, and outliers in numeric feature '{col}'.",
                "priority": 2,
                "reason": f"Continuous numeric feature (range: {c_stats.min} to {c_stats.max}).",
            })
            if len(recs) >= 4:
                break

    # 3. Cross-tabulation / heatmap if >= 2 valid categorical
    cat_candidates = [
        c for c in stats.categorical_column_names
        if not _is_index_or_artifact(c, frame[c], next((s for s in stats.columns if s.name == c), None))[0]
    ]
    if len(cat_candidates) >= 2:
        recs.append({
            "type": "crosstab_heatmap",
            "row_col": cat_candidates[0],
            "col_col": cat_candidates[1],
            "title": f"{cat_candidates[0]} vs. {cat_candidates[1]} Contingency Matrix",
            "description": f"Highlights interactions and co-occurrences between '{cat_candidates[0]}' and '{cat_candidates[1]}'.",
            "priority": 3,
        })

    # 4. Correlation heatmap if >= 2 valid numeric
    num_candidates = [
        c for c in stats.numeric_column_names
        if not _is_index_or_artifact(c, frame[c], next((s for s in stats.columns if s.name == c), None))[0]
    ]
    if len(num_candidates) >= 2:
        recs.append({
            "type": "correlation_heatmap",
            "columns": num_candidates[:8],
            "title": "Numeric Feature Correlation Matrix",
            "description": f"Visualizes Pearson correlations across {min(len(num_candidates), 8)} numeric features.",
            "priority": 4,
        })

    # 5. Time series trend if date column present
    if stats.date_column_names:
        recs.append({
            "type": "time_series",
            "date_col": stats.date_column_names[0],
            "value_col": num_candidates[0] if num_candidates else None,
            "title": f"Timeline Trend ({stats.date_column_names[0]})",
            "description": f"Tracks event cadence or metric trajectory over '{stats.date_column_names[0]}'.",
            "priority": 5,
        })

    return recs


def explain_chart_exclusions(frame: pd.DataFrame, stats: DatasetStats | None = None) -> list[dict[str, Any]]:
    """Explain why each column is either recommended or excluded from automated charts."""
    if stats is None:
        stats = compute_dataset_stats(frame)
    explanations: list[dict[str, Any]] = []

    recommended_cols = set()
    for rec in recommend_charts(frame, stats):
        if "column" in rec:
            recommended_cols.add(rec["column"])
        if "row_col" in rec:
            recommended_cols.add(rec["row_col"])
        if "col_col" in rec:
            recommended_cols.add(rec["col_col"])
        if "columns" in rec:
            recommended_cols.update(rec["columns"])
        if "date_col" in rec:
            recommended_cols.add(rec["date_col"])

    for col in frame.columns:
        c_stats = next((s for s in stats.columns if s.name == col), None)
        kind = c_stats.kind if c_stats else "unknown"
        is_art, art_reason = _is_index_or_artifact(col, frame[col], c_stats)

        if col in recommended_cols:
            status = "Recommended"
            detail = f"Included in automated visualizations ({kind})."
        elif is_art:
            status = "Excluded"
            detail = art_reason
        elif kind in ("categorical", "text") and c_stats and c_stats.unique > 30:
            status = "Excluded"
            detail = f"High cardinality ({c_stats.unique} distinct categories; exceeds 30-category chart readability threshold)."
        elif kind == "numeric" and c_stats and c_stats.unique <= 3:
            status = "Excluded"
            detail = f"Low cardinality ({c_stats.unique} distinct values; better suited for filtering than continuous histogram)."
        else:
            status = "Available (Manual)"
            detail = f"Valid {kind} column available in Custom Visualization Builder."

        explanations.append({
            "Column": str(col),
            "Type": kind.title(),
            "Recommendation Status": status,
            "Explanation": detail,
        })

    return explanations
