"""Whitelist of analytical operations. No model-generated code is executed."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from private_analyst.analytics.anomalies import detect_iqr_anomalies, linear_trend_forecast
from private_analyst.analytics.ml_engine import run_classification, run_clustering, run_regression
from private_analyst.analytics.quality import assess_quality
from private_analyst.analytics.statistics import compute_dataset_stats, frequency_analysis, group_compare
from private_analyst.ingestion.loader import IngestResult

ALLOWED_INTENTS = {
    "summarize",
    "quality",
    "describe_column",
    "group_compare",
    "anomalies",
    "forecast",
    "limitations",
    "classify",
    "regression",
    "cluster",
    "frequency",
    "crosstab",
}


@dataclass
class OperationResult:
    intent: str
    status: str
    observed: dict[str, Any]
    calculated: dict[str, Any]
    hypotheses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    chart_spec: dict[str, Any] | None = None


def classify_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("missing", "duplicate", "quality", "inconsistent")):
        return "quality"
    if any(k in q for k in ("unusual", "anomal", "outlier")):
        return "anomalies"
    if "forecast" in q or "predict" in q:
        return "forecast"
    if "cluster" in q:
        return "cluster"
    if "classif" in q:
        return "classify"
    if "regress" in q:
        return "regression"
    if any(k in q for k in ("crosstab", "cross-tab", "contingency", " vs ", " versus ")):
        return "crosstab"
    if any(k in q for k in ("frequency", "frequencies", "distribution", "value count", "breakdown", "most common", "top ")):
        return "frequency"
    if any(k in q for k in ("compare", "group", "by region", "by product")):
        return "group_compare"
    if any(k in q for k in ("limitation", "caveat", "cannot")):
        return "limitations"
    if re.search(r"\b(mean|average|median|total|sum|min|max|std|standard deviation|summar)\b", q):
        return "summarize"
    return "summarize"


def _find_column_in_question(frame: pd.DataFrame, question: str) -> str | None:
    q = question.lower()
    for col in frame.columns:
        if str(col).lower() in q:
            return str(col)
    return None


def run_operation(ingest: IngestResult, question: str, intent: str | None = None) -> OperationResult:
    intent = intent or classify_intent(question)
    if intent not in ALLOWED_INTENTS:
        return OperationResult(
            intent="invalid",
            status="rejected",
            observed={},
            calculated={},
            limitations=["The request is not in the list of approved analytical operations."],
        )
    frame = ingest.frame
    stats = compute_dataset_stats(frame, getattr(ingest, "inferred_types", None))
    quality = assess_quality(frame, ingest.duplicate_columns)

    if intent == "quality":
        return OperationResult(
            intent=intent,
            status="ok",
            observed={"source_name": ingest.source_name, "rows": quality.row_count},
            calculated=quality.to_dict(),
            recommendations=["Review columns with missing values before using them in totals."],
        )
    if intent == "anomalies":
        if not stats.numeric_column_names:
            return OperationResult(
                intent=intent,
                status="unsupported_by_data",
                observed={"source_name": ingest.source_name},
                calculated={},
                limitations=["No numeric columns are available for IQR anomaly detection."],
            )
        column = stats.numeric_column_names[0]
        result = detect_iqr_anomalies(frame, column)
        return OperationResult(
            intent=intent,
            status="ok",
            observed={"column": column, "row_count": stats.row_count},
            calculated={
                "method": result.method,
                "lower": result.lower,
                "upper": result.upper,
                "outlier_count": result.outlier_count,
                "outlier_row_indexes": result.outlier_row_indexes,
            },
            hypotheses=[result.caveat],
        )
    if intent == "forecast":
        if not stats.numeric_column_names:
            return OperationResult(
                intent=intent,
                status="unsupported_by_data",
                observed={},
                calculated={},
                limitations=["Forecasting needs a numeric series."],
            )
        column = stats.numeric_column_names[0]
        values = pd.to_numeric(frame[column], errors="coerce").tolist()
        forecast = linear_trend_forecast([float(v) if v == v else float("nan") for v in values])
        return OperationResult(
            intent=intent,
            status=str(forecast.get("status")),
            observed={"column": column, "n": int(pd.to_numeric(frame[column], errors="coerce").notna().sum())},
            calculated=forecast,
            hypotheses=[str(forecast.get("caveat"))],
        )
    if intent == "frequency":
        cols = list(frame.columns)
        candidate = _find_column_in_question(frame, question)
        if candidate is None:
            informative = [
                c for c in (stats.categorical_column_names or cols)
                if c not in {"#", "id", "s.no", "sl.no", "index"}
            ]
            candidate = informative[0] if informative else cols[0]

        freq_df = frequency_analysis(frame, candidate)
        cardinality = int(frame[candidate].nunique(dropna=True))
        top_share = float(freq_df["Percentage (%)"].iloc[:3].sum()) if not freq_df.empty else 0.0

        return OperationResult(
            intent=intent,
            status="ok",
            observed={"column": candidate, "row_count": len(frame), "cardinality": cardinality},
            calculated={
                "column": candidate,
                "cardinality": cardinality,
                "top_3_share_pct": round(top_share, 1),
                "frequency_table": freq_df.to_dict(orient="records"),
            },
            chart_spec={"type": "category_bar", "column": candidate},
            recommendations=[f"Review the distribution of '{candidate}' across categories."],
        )
    if intent == "crosstab":
        cols = list(frame.columns)
        cat_cols = [
            c for c in (stats.categorical_column_names or cols)
            if c not in {"#", "id", "s.no", "sl.no", "index"} and frame[c].nunique(dropna=True) <= 25
        ]
        if len(cat_cols) < 2:
            cat_cols = [c for c in cols if frame[c].nunique(dropna=True) <= 50]

        if len(cat_cols) < 2:
            return OperationResult(
                intent=intent,
                status="unsupported_by_data",
                observed={},
                calculated={},
                limitations=["Cross-tabulation requires at least two categorical or grouping columns."],
            )

        col1, col2 = cat_cols[0], cat_cols[1]
        ct = group_compare(frame, col1, col2)
        return OperationResult(
            intent=intent,
            status="ok",
            observed={"row_col": col1, "col_col": col2, "row_count": len(frame)},
            calculated={
                "row_col": col1,
                "col_col": col2,
                "crosstab": ct.to_dict(),
            },
            chart_spec={"type": "crosstab_heatmap", "row_col": col1, "col_col": col2},
        )
    if intent == "group_compare":
        cols = list(frame.columns)
        non_numeric = [c for c in cols if c not in stats.numeric_column_names]
        if stats.numeric_column_names and non_numeric:
            grouped = group_compare(frame, str(non_numeric[0]), stats.numeric_column_names[0])
            return OperationResult(
                intent=intent,
                status="ok",
                observed={"group_col": str(non_numeric[0]), "value_col": stats.numeric_column_names[0]},
                calculated={"groups": grouped.to_dict(orient="records")},
                chart_spec={
                    "type": "bar",
                    "x": str(non_numeric[0]),
                    "y": stats.numeric_column_names[0],
                },
            )
        if len(non_numeric) >= 2:
            col1, col2 = str(non_numeric[0]), str(non_numeric[1])
            ct = group_compare(frame, col1, col2)
            return OperationResult(
                intent="crosstab",
                status="ok",
                observed={"row_col": col1, "col_col": col2, "fallback_from": "group_compare"},
                calculated={"crosstab": ct.to_dict()},
                chart_spec={"type": "crosstab_heatmap", "row_col": col1, "col_col": col2},
                recommendations=["Dataset has no numeric columns; performed categorical cross-tabulation instead."],
            )
        return OperationResult(
            intent=intent,
            status="unsupported_by_data",
            observed={},
            calculated={},
            limitations=["Group comparison needs at least one grouping column and one numeric column (or multiple categorical dimensions)."],
        )
    if intent == "classify":
        cols = list(frame.columns)
        non_numeric = [c for c in cols if c not in stats.numeric_column_names]
        target = non_numeric[0] if non_numeric else cols[-1]
        ml_res = run_classification(frame, target)
        return OperationResult(
            intent=intent,
            status=ml_res.status,
            observed={"target_column": target, "features": ml_res.parameters.get("feature_cols", [])},
            calculated=ml_res.metrics,
            hypotheses=ml_res.caveats,
            limitations=["Classification requires ground-truth labels and a representative sample."],
        )
    if intent == "regression":
        if not stats.numeric_column_names:
            return OperationResult(
                intent=intent,
                status="unsupported_by_data",
                observed={},
                calculated={},
                limitations=["Regression requires at least one numeric target column."],
            )
        target = stats.numeric_column_names[-1]
        ml_res = run_regression(frame, target)
        return OperationResult(
            intent=intent,
            status=ml_res.status,
            observed={"target_column": target, "features": ml_res.parameters.get("feature_cols", [])},
            calculated=ml_res.metrics,
            hypotheses=ml_res.caveats,
            limitations=["Regression estimates conditional expectations, not causal relationships."],
        )
    if intent == "cluster":
        if not stats.numeric_column_names:
            return OperationResult(
                intent=intent,
                status="unsupported_by_data",
                observed={},
                calculated={},
                limitations=["Clustering requires numeric feature columns."],
            )
        ml_res = run_clustering(frame, n_clusters=min(3, max(2, len(frame))))
        return OperationResult(
            intent=intent,
            status=ml_res.status,
            observed={"features": ml_res.parameters.get("feature_cols", []), "n_clusters": ml_res.parameters.get("n_clusters")},
            calculated=ml_res.metrics,
            hypotheses=ml_res.caveats,
            limitations=["Clusters are heuristic algorithmic partitions, not validated real-world segments."],
        )
    if intent == "limitations":
        return OperationResult(
            intent=intent,
            status="ok",
            observed={"source_name": ingest.source_name, "persisted": ingest.persisted},
            calculated={},
            limitations=[
                "Only the currently loaded file was analysed.",
                "The language model does not compute totals; Python does.",
                "Scanned documents require a local OCR engine.",
                "No claim is made that the installation is free of all security defects.",
            ],
        )

    chart_spec = None
    if stats.numeric_column_names:
        chart_spec = {"type": "histogram", "column": stats.numeric_column_names[0]}
    elif stats.categorical_column_names:
        candidate_cats = [
            c for c in stats.categorical_column_names
            if c not in {"#", "id", "s.no", "sl.no", "index"}
        ]
        top_cat = candidate_cats[0] if candidate_cats else stats.categorical_column_names[0]
        chart_spec = {"type": "category_bar", "column": top_cat}

    return OperationResult(
        intent="summarize",
        status="ok",
        observed={
            "source_name": ingest.source_name,
            "rows": stats.row_count,
            "columns": stats.column_count,
        },
        calculated=stats.to_dict(),
        recommendations=["Ask a follow-up about a specific column if you need a narrower summary."],
        chart_spec=chart_spec,
    )
