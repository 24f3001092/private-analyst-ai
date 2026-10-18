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
    CleaningOperation,
    apply_transformations,
    load_tabular,
    normalize_cleaning_operation,
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


def test_cleaning_ledger_object_based_operations() -> None:
    op = CleaningOperation(
        step="drop_empty_rows",
        description="Dropped 4 completely blank rows",
        impact="Reduced table noise",
        affected_columns=["A", "B"],
        rows_affected=4,
    )
    row, warn = normalize_cleaning_operation(op)

    assert warn is None
    assert row["Step"] == "Drop Empty Rows"
    assert row["Description"] == "Dropped 4 completely blank rows"
    assert row["Impact"] == "Reduced table noise"
    assert row["Rows Affected"] == 4
    assert row["Affected Columns"] == "A, B"
    assert row["Timestamp"] != "-"


def test_cleaning_ledger_dict_based_operations() -> None:
    # Standard lowercase dictionary keys (as produced by to_dict() / JSON)
    dict_op = {
        "step": "header_selection",
        "description": "Selected row 3 as header row",
        "impact": "Skipped 2 introductory rows",
        "affected_columns": ["Title", "Notes"],
        "rows_affected": 2,
        "timestamp": "2026-10-18T12:30:00.000000Z",
    }
    row, warn = normalize_cleaning_operation(dict_op)

    assert warn is None
    assert row["Step"] == "Header Selection"
    assert row["Description"] == "Selected row 3 as header row"
    assert row["Impact"] == "Skipped 2 introductory rows"
    assert row["Rows Affected"] == 2
    assert row["Affected Columns"] == "Title, Notes"
    assert "2026-10-18 12:30:00" in row["Timestamp"]

    # Title-case dictionary keys
    title_dict = {
        "Step": "strip_whitespace",
        "Description": "Trimmed leading spaces",
        "Impact": "Clean strings",
        "Affected Columns": ["col_1"],
        "Rows Affected": 10,
    }
    row2, warn2 = normalize_cleaning_operation(title_dict)
    assert warn2 is None
    assert row2["Step"] == "Strip Whitespace"
    assert row2["Rows Affected"] == 10


def test_cleaning_ledger_missing_and_invalid_fields() -> None:
    # Missing 'step' key
    missing_step = {
        "description": "Operation without step identifier",
        "impact": "Unknown impact",
    }
    row1, warn1 = normalize_cleaning_operation(missing_step)
    assert row1["Step"] == "Unknown Step"
    assert row1["Description"] == "Operation without step identifier"
    assert warn1 is not None

    # Missing all optional fields
    bare_op = {"step": "bare_operation"}
    row2, warn2 = normalize_cleaning_operation(bare_op)
    assert warn2 is None
    assert row2["Step"] == "Bare Operation"
    assert row2["Description"] == "No description provided"
    assert row2["Impact"] == "-"
    assert row2["Rows Affected"] == 0
    assert row2["Affected Columns"] == "All"
    assert row2["Timestamp"] == "-"

    # Invalid non-integer rows_affected
    invalid_rows = {
        "step": "corrupt_count",
        "rows_affected": "not-a-number",
    }
    row3, warn3 = normalize_cleaning_operation(invalid_rows)
    assert row3["Rows Affected"] == 0
    assert warn3 is not None

    # None input
    row_none, warn_none = normalize_cleaning_operation(None)
    assert row_none["Step"] == "Unknown Step"
    assert warn_none is not None


def test_cleaning_ledger_malformed_entries() -> None:
    # Raw string entry
    row_str, warn_str = normalize_cleaning_operation("Some unstructured log message")
    assert row_str["Step"] == "Malformed Operation"
    assert "Some unstructured log message" in row_str["Description"]
    assert warn_str is not None

    # Arbitrary object
    class CustomOp:
        def __init__(self):
            self.step = "custom_step"
            self.description = "Custom description"
            self.impact = "Custom impact"
            self.rows_affected = 5

    row_obj, warn_obj = normalize_cleaning_operation(CustomOp())
    assert row_obj["Step"] == "Custom Step"
    assert row_obj["Rows Affected"] == 5


def test_cleaning_ledger_empty() -> None:
    ledger_rows = [normalize_cleaning_operation(op)[0] for op in []]
    assert len(ledger_rows) == 0


def test_ingest_result_cleaning_helpers(limits: Limits) -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ingest = load_tabular(csv_bytes, "test.csv", limits)

    initial_count = len(ingest.cleaning_log)

    # Add via CleaningOperation object
    ingest.add_cleaning_operation(CleaningOperation(
        step="manual_filter",
        description="Filtered test rows",
        impact="Reduced test rows",
    ))
    assert len(ingest.cleaning_log) == initial_count + 1

    # Add via dict
    ingest.add_cleaning_operation({
        "step": "manual_rename",
        "description": "Renamed A to Alpha",
        "impact": "Better naming",
    })
    assert len(ingest.cleaning_log) == initial_count + 2

    # Verify cleaning_operations property returns list of dicts
    ops = ingest.cleaning_operations
    assert isinstance(ops, list)
    assert len(ops) == initial_count + 2
    assert isinstance(ops[-1], dict)
    assert ops[-1]["step"] == "manual_rename"

    # Verify property setter
    ingest.cleaning_operations = [{"step": "reset_step", "description": "Reset op"}]
    assert len(ingest.cleaning_log) == 1
    assert ingest.cleaning_log[0].step == "reset_step"


def test_ui_import_tab_simulation_prevents_dict_attribute_error() -> None:
    """Simulate the exact UI loop in _import_tab() with mixed dict and object entries to verify no crash."""
    ops = [
        # Dataclass instance
        CleaningOperation(
            step="override_header_row",
            description="Promoted row 2 to headers",
            impact="Reindexed",
            rows_affected=2,
        ),
        # Standard dict (the exact case that previously triggered AttributeError: 'dict' object has no attribute 'step')
        {
            "step": "drop_empty_rows",
            "description": "Dropped 10 empty rows",
            "impact": "Clean data",
            "rows_affected": 10,
            "affected_columns": ["col1", "col2"],
        },
        # Dict with missing fields
        {
            "step": "repair_headers",
        },
        # Dict with no step
        {
            "description": "Orphan description",
        },
        # Raw string entry
        "Unexpected string entry",
    ]

    ledger_rows = []
    warnings = []
    for op in ops:
        row, warn = normalize_cleaning_operation(op)
        ledger_rows.append(row)
        if warn:
            warnings.append(warn)

    assert len(ledger_rows) == 5
    ledger_df = pd.DataFrame(ledger_rows)
    assert not ledger_df.empty
    assert list(ledger_df.columns) == [
        "Step",
        "Description",
        "Impact",
        "Rows Affected",
        "Affected Columns",
        "Timestamp",
    ]
    # Check that Step was title-cased and did not crash on dict
    assert ledger_df.iloc[0]["Step"] == "Override Header Row"
    assert ledger_df.iloc[1]["Step"] == "Drop Empty Rows"
    assert ledger_df.iloc[2]["Step"] == "Repair Headers"
    assert ledger_df.iloc[3]["Step"] == "Unknown Step"
    assert ledger_df.iloc[4]["Step"] == "Malformed Operation"
    assert len(warnings) >= 2


def test_to_report_excel_bytes_includes_cleaning_ledger(limits: Limits) -> None:
    df = pd.DataFrame({"X": [10, 20], "Y": [30, 40]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ingest = load_tabular(csv_bytes, "test.csv", limits)
    ingest.add_cleaning_operation(CleaningOperation(
        step="test_cleaning_step",
        description="Audit test operation",
        impact="Test impact",
    ))

    excel_bytes = to_report_excel_bytes(ingest)
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    assert "Cleaned Data" in xls.sheet_names
    assert "Cleaning Ledger" in xls.sheet_names
    ledger_sheet = pd.read_excel(xls, sheet_name="Cleaning Ledger")
    assert not ledger_sheet.empty
    assert "Step" in ledger_sheet.columns
    assert "Test Cleaning Step" in ledger_sheet["Step"].values
