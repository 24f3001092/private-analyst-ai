from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from private_analyst.analytics.anomalies import detect_zscore_anomalies
from private_analyst.analytics.ml_engine import run_classification, run_regression
from private_analyst.analytics.operations import classify_intent, run_operation
from private_analyst.analytics.statistics import (
    compute_correlation_matrix,
    compute_dataset_stats,
    compute_time_trends,
    recommend_charts,
)
from private_analyst.config import Limits
from private_analyst.ingestion.loader import (
    apply_transformations,
    load_tabular,
    parse_currency,
    parse_numeric_range,
    parse_percentage,
    repair_unnamed_columns,
)
from private_analyst.reports.exporters import to_report_excel_bytes


def test_currency_parsing() -> None:
    assert parse_currency("$1,000") == (1000.0, "$")
    assert parse_currency("₹50,000") == (50000.0, "₹")
    assert parse_currency("€ 1.5M") == (1500000.0, "€")
    assert parse_currency("£2.5k") == (2500.0, "£")
    assert parse_currency("50 lakh")[0] == 5000000.0
    assert parse_currency("2.5 Cr")[0] == 25000000.0
    assert parse_currency("USD 1200") == (1200.0, "USD")
    assert parse_currency("Free") == (None, None)
    assert parse_currency(None) == (None, None)


def test_percentage_parsing() -> None:
    assert parse_percentage("25%") == 25.0
    assert parse_percentage("12.5 %") == 12.5
    assert parse_percentage("(5%)") == -5.0
    assert parse_percentage("-10.2%") == -10.2
    assert parse_percentage("0.45") == 45.0
    assert parse_percentage("invalid") is None
    assert parse_percentage(None) is None


def test_numeric_range_parsing() -> None:
    low, high, mid = parse_numeric_range("10-20")
    assert low == 10.0 and high == 20.0 and mid == 15.0

    low2, high2, mid2 = parse_numeric_range("100 to 200")
    assert low2 == 100.0 and high2 == 200.0 and mid2 == 150.0

    low3, high3, mid3 = parse_numeric_range("50")
    assert low3 == 50.0 and high3 == 50.0 and mid3 == 50.0

    assert parse_numeric_range("TBD")[0] is None


def test_repair_unnamed_columns() -> None:
    df = pd.DataFrame([
        [10, 20, 30],
        [40, 50, 60],
    ], columns=["Category", "Unnamed: 1", "Unnamed: 2"])

    repaired, renamed, op = repair_unnamed_columns(df)
    assert "Unnamed: 1" not in repaired.columns
    assert "Unnamed: 2" not in repaired.columns
    assert len(renamed) == 2
    assert op is not None
    assert "repair_unnamed_headers" == op.step


def test_apply_transformations() -> None:
    df = pd.DataFrame({
        "A": ["title", "hdr_a", 1, 2, 3],
        "B": ["meta", "hdr_b", 10, 20, 30],
        "C": ["drop", "drop_b", 100, 200, 300],
    })

    transformed, ops = apply_transformations(
        df,
        header_row=1,
        exclude_columns=["drop_b"],
        rename_columns={"hdr_a": "alpha"},
    )
    assert "alpha" in transformed.columns
    assert "drop_b" not in transformed.columns
    assert len(transformed) == 3
    assert len(ops) >= 2


def test_correlation_matrix_computation() -> None:
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + np.random.normal(0, 0.5, 50)
    z = -3 * x + np.random.normal(0, 0.5, 50)
    w = np.random.normal(0, 1, 50)

    df = pd.DataFrame({"x": x, "y": y, "z": z, "noise": w})
    res = compute_correlation_matrix(df)

    assert "matrix" in res
    assert "top_correlations" in res
    assert res["status"] == "ok"
    xy_pair = next(
        p for p in res["top_correlations"]
        if (p["col1"] == "x" and p["col2"] == "y") or (p["col1"] == "y" and p["col2"] == "x")
    )
    assert xy_pair["correlation"] > 0.9


def test_compute_time_trends() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    values = [100 + i * 15 for i in range(12)]
    df = pd.DataFrame({"record_date": dates, "revenue": values})

    trends = compute_time_trends(df, "record_date", "revenue", freq="M")
    assert trends["status"] == "ok"
    assert len(trends["trend_table"]) == 12
    assert trends["freq_label"] == "Monthly"


def test_recommend_charts() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "event_date": dates,
        "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
        "sales": [100, 150, 120, 180, 200, 220, 210, 240, 260, 280],
        "profit": [20, 30, 25, 40, 45, 50, 48, 55, 60, 65],
    })

    recommendations = recommend_charts(df)
    assert len(recommendations) >= 2
    types = [r["type"] for r in recommendations]
    assert any(t in types for t in ["category_bar", "histogram", "correlation_heatmap"])


def test_detect_zscore_anomalies() -> None:
    data = [10.0] * 50 + [1000.0]
    df = pd.DataFrame({"metric": data})

    res = detect_zscore_anomalies(df, "metric", threshold=3.0)
    assert res.outlier_count >= 1
    assert 50 in res.outlier_row_indexes
    assert "parametric" in res.caveat.lower() or "z-score" in res.caveat.lower()


def test_to_report_excel_bytes(limits: Limits) -> None:
    csv_bytes = b"region,sales,date\nEast,100,2024-01-01\nWest,200,2024-01-02\n"
    ingest = load_tabular(csv_bytes, "test.csv", limits)
    excel_bytes = to_report_excel_bytes(ingest)

    assert excel_bytes.startswith(b"PK\x03\x04")
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    assert "Cleaned Data" in xls.sheet_names


def test_ml_classification_with_categorical_features() -> None:
    np.random.seed(42)
    n = 60
    category = np.random.choice(["Tier1", "Tier2", "Tier3"], size=n)
    num_feat = np.random.normal(50, 10, size=n)
    target = ["High" if (c == "Tier1" or f > 52) else "Low" for c, f in zip(category, num_feat)]

    df = pd.DataFrame({
        "tier": category,
        "score": num_feat,
        "label": target,
    })

    res = run_classification(df, "label", ["tier", "score"])
    assert res.status == "evaluated"
    assert "train_accuracy" in res.metrics
    assert "test_accuracy" in res.metrics
    assert res.metrics["test_accuracy"] >= 0.0


def test_ml_regression_holdout_metrics() -> None:
    np.random.seed(42)
    n = 50
    x1 = np.random.uniform(10, 50, size=n)
    cat = np.random.choice(["A", "B"], size=n)
    y = 3.0 * x1 + (cat == "A") * 10.0 + np.random.normal(0, 1, size=n)

    df = pd.DataFrame({"x1": x1, "group": cat, "target": y})
    res = run_regression(df, "target", ["x1", "group"])

    assert res.status == "evaluated"
    assert "train_r2" in res.metrics
    assert "test_r2" in res.metrics
    assert "train_rmse" in res.metrics
    assert "test_rmse" in res.metrics


def test_operations_correlation_and_trend(limits: Limits) -> None:
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=20),
        "val_a": list(range(20)),
        "val_b": [2 * i + 1 for i in range(20)],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ingest = load_tabular(csv_bytes, "trends.csv", limits)

    res_corr = run_operation(ingest, "show correlation matrix between val_a and val_b")
    assert res_corr.intent == "correlation"
    assert res_corr.status == "ok"

    res_trend = run_operation(ingest, "what is the time trend over day for val_a")
    assert res_trend.intent == "trend"
    assert res_trend.status == "ok"
