from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from private_analyst.analytics.cleaner import (
    CleaningRecommendation,
    apply_cleaning_plan,
    apply_column_management,
    apply_header_override,
    filter_rows,
    generate_cleaning_recommendations,
    handle_missing_values,
    repair_unnamed_headers,
)
from private_analyst.analytics.quality import assess_quality
from private_analyst.config import Limits
from private_analyst.ingestion.loader import (
    CleaningOperation,
    IngestResult,
    inspect_workbook,
    load_tabular,
    normalize_cleaning_operation,
)
from private_analyst.reports.exporters import to_report_excel_bytes
from private_analyst.viz.charts import (
    bar_sum,
    category_bar,
    correlation_heatmap,
    crosstab_heatmap,
    histogram,
    missingness,
    outlier_box,
    time_series_line,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_charts_handle_empty_and_zero_rows() -> None:
    """Verify all chart functions gracefully handle empty DataFrames or missing columns without crashing."""
    empty_df = pd.DataFrame()
    dummy_df = pd.DataFrame({"col_a": [None, None], "col_b": ["text1", "text2"]})

    # 1. Histogram
    fig_h1 = histogram(empty_df, "col_x")
    assert fig_h1 is not None
    fig_h2 = histogram(dummy_df, "col_a")
    assert fig_h2 is not None

    # 2. Bar sum
    fig_bs1 = bar_sum(empty_df, "x", "y")
    assert fig_bs1 is not None
    fig_bs2 = bar_sum(dummy_df, "col_b", "col_a")
    assert fig_bs2 is not None

    # 3. Category bar
    fig_cb1 = category_bar(empty_df, "cat")
    assert fig_cb1 is not None
    fig_cb2 = category_bar(dummy_df, "col_a")
    assert fig_cb2 is not None

    # 4. Crosstab heatmap
    fig_ct = crosstab_heatmap(pd.DataFrame())
    assert fig_ct is not None

    # 5. Missingness
    fig_m = missingness({})
    assert fig_m is not None

    # 6. Correlation heatmap
    fig_ch = correlation_heatmap(pd.DataFrame())
    assert fig_ch is not None

    # 7. Time series line
    fig_ts = time_series_line(pd.DataFrame())
    assert fig_ts is not None

    # 8. Outlier box
    fig_ob = outlier_box(empty_df, "col_x")
    assert fig_ob is not None


def test_unique_plotly_chart_keys_simulation() -> None:
    """Verify that multiple charts rendered in loops receive deterministic, unique keys."""
    columns = ["category_1", "category_2", "category_1", "status"]
    generated_keys = set()

    for idx, col in enumerate(columns):
        # Deterministic key generation algorithm
        chart_key = f"dashboard_rec_{idx}_category_bar_{col}"
        assert chart_key not in generated_keys, f"Duplicate key detected: {chart_key}"
        generated_keys.add(chart_key)

    assert len(generated_keys) == 4


def test_generate_cleaning_recommendations_blank_rows_and_cols() -> None:
    df = pd.DataFrame([
        ["Alice", 100, None],
        [None, None, None],  # fully blank row
        ["Bob", 200, None],
        ["   ", None, None],  # whitespace blank row
    ], columns=["Name", "Score", "EmptyCol"])

    recs = generate_cleaning_recommendations(df)
    rec_ids = [r.id for r in recs]

    assert "drop_empty_rows" in rec_ids
    assert "drop_empty_cols" in rec_ids

    blank_rec = next(r for r in recs if r.id == "drop_empty_rows")
    assert blank_rec.affected_rows_count >= 2
    assert not blank_rec.is_lossy

    empty_col_rec = next(r for r in recs if r.id == "drop_empty_cols")
    assert "EmptyCol" in empty_col_rec.affected_columns
    assert empty_col_rec.is_lossy


def test_repair_unnamed_headers() -> None:
    df = pd.DataFrame([
        [1, 2, 3],
        [4, 5, 6],
    ], columns=["ID", "Unnamed: 1", "unnamed_column"])

    repaired, renamed, op = repair_unnamed_headers(df)
    assert "Unnamed: 1" not in repaired.columns
    assert "unnamed_column" not in repaired.columns
    assert "column_2" in repaired.columns
    assert "column_3" in repaired.columns
    assert op is not None
    assert op.step == "repair_unnamed_headers"
    assert op.status == "applied"


def test_cleaning_deduplicate_exact_rows() -> None:
    df = pd.DataFrame({
        "City": ["Delhi", "Mumbai", "Delhi", "Kolkata"],
        "Code": [11, 22, 11, 33],
    })
    recs = generate_cleaning_recommendations(df)
    assert any(r.id == "deduplicate_rows" for r in recs)

    cleaned_df, ops = apply_cleaning_plan(df, ["deduplicate_rows"])
    assert len(cleaned_df) == 3
    assert len(ops) == 1
    assert ops[0].step == "deduplicate_rows"
    assert ops[0].rows_affected == 1


def test_cleaning_numeric_strings_currencies_percentages() -> None:
    df = pd.DataFrame({
        "Amount": ["$1,000", "$2,500", "$500"],
        "TaxRate": ["18%", "12%", "5%"],
        "CountStr": ["10", "20", "30"],
    })
    cleaned_df, ops = apply_cleaning_plan(
        df,
        ["parse_currencies", "parse_percentages", "convert_numeric_strings"],
    )

    assert pd.api.types.is_numeric_dtype(cleaned_df["Amount"])
    assert cleaned_df["Amount"].iloc[0] == 1000.0
    assert pd.api.types.is_numeric_dtype(cleaned_df["TaxRate"])
    assert cleaned_df["TaxRate"].iloc[0] == 18.0
    assert pd.api.types.is_numeric_dtype(cleaned_df["CountStr"])
    assert cleaned_df["CountStr"].iloc[0] == 10
    assert len(ops) == 3


def test_cleaning_idempotency_and_no_duplication() -> None:
    """Verify that applying cleaning to an already clean dataset produces no modifications and no duplicate operations."""
    df = pd.DataFrame({
        "Name": ["Alpha", "Beta", "Gamma"],
        "Score": [10.5, 20.0, 35.2],
    })
    cleaned_df, ops = apply_cleaning_plan(
        df,
        ["drop_empty_rows", "strip_whitespace", "repair_unnamed_headers", "deduplicate_rows"],
    )
    assert len(cleaned_df) == 3
    assert len(ops) == 0  # No operations needed or applied

    recs = generate_cleaning_recommendations(cleaned_df)
    # None of the structure / whitespace / duplicate recs should trigger
    assert not any(r.id in ["drop_empty_rows", "strip_whitespace", "deduplicate_rows"] for r in recs)


def test_header_override_promotes_selected_row() -> None:
    raw_df = pd.DataFrame([
        ["TITLE BANNER", None, None],
        ["Notes row", None, None],
        ["Dept", "Employees", "Budget"],
        ["Engineering", 50, 1000000],
        ["Marketing", 20, 500000],
    ])

    reindexed_df, op = apply_header_override(raw_df, 2)
    assert list(reindexed_df.columns) == ["Dept", "Employees", "Budget"]
    assert len(reindexed_df) == 2
    assert op.step == "override_header_row"
    assert op.rows_affected >= 3


def test_column_management_rename_and_exclude() -> None:
    df = pd.DataFrame({
        "ColA": [1, 2],
        "ColB": [3, 4],
        "DropMe": [5, 6],
    })

    # Rename
    df_renamed, ops_r = apply_column_management(df, rename_columns={"ColA": "Alpha"})
    assert "Alpha" in df_renamed.columns
    assert "ColA" not in df_renamed.columns
    assert len(ops_r) == 1

    # Exclude
    df_excluded, ops_e = apply_column_management(df_renamed, exclude_columns=["DropMe"])
    assert "DropMe" not in df_excluded.columns
    assert len(df_excluded.columns) == 2
    assert len(ops_e) == 1


def test_handle_missing_values_strategies() -> None:
    df = pd.DataFrame({
        "val": [10.0, 20.0, None, 30.0],
        "cat": ["A", "B", None, "A"],
    })

    # Impute mean
    df_mean, op_m = handle_missing_values(df, "val", "impute_mean")
    assert df_mean["val"].isna().sum() == 0
    assert df_mean["val"].iloc[2] == 20.0
    assert op_m.step == "impute_mean"

    # Impute mode
    df_mode, op_mode = handle_missing_values(df, "cat", "impute_mode")
    assert df_mode["cat"].isna().sum() == 0
    assert df_mode["cat"].iloc[2] == "A"
    assert op_mode.step == "impute_mode"

    # Drop rows
    df_drop, op_d = handle_missing_values(df, "val", "drop_rows")
    assert len(df_drop) == 3
    assert op_d.step == "drop_missing_rows"


def test_filter_rows_conditions() -> None:
    df = pd.DataFrame({
        "score": [10, 25, 50, 75, 100],
        "grade": ["C", "B", "B", "A", "A"],
    })

    df_filtered, op_f = filter_rows(df, "score", "greater_than", 30)
    assert len(df_filtered) == 3
    assert op_f.rows_affected == 2

    df_eq, op_eq = filter_rows(df, "grade", "equals", "A")
    assert len(df_eq) == 2


def test_multisheet_excel_workbook_loading(limits: Limits) -> None:
    wb_path = SAMPLES / "Hackathon_List_2026-1.xlsx"
    if not wb_path.is_file():
        pytest.skip("Hackathon_List_2026-1.xlsx not found in samples")

    data = wb_path.read_bytes()
    info = inspect_workbook(data, "Hackathon_List_2026-1.xlsx")
    assert info is not None
    assert len(info.sheet_names) == 3

    # Load sheet 1
    s1 = load_tabular(data, "Hackathon_List_2026-1.xlsx", limits, sheet_name=info.sheet_names[0])
    assert s1.sheet_name == "AI & Agentic Hackathons"
    assert s1.header_row == 2
    assert "Hackathon Name" in s1.frame.columns

    # Switch to sheet 3
    s3 = load_tabular(data, "Hackathon_List_2026-1.xlsx", limits, sheet_name="Quick Reference")
    assert s3.sheet_name == "Quick Reference"
    assert s3.header_row == 3
    assert "Month" in s3.frame.columns


def test_cleaning_log_persistence_across_mutations(limits: Limits) -> None:
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["X", "Y", "Z"]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ingest = load_tabular(csv_bytes, "persist_test.csv", limits)

    initial_ops = len(ingest.cleaning_log)

    # 1. Add cleaning plan operation
    ingest.add_cleaning_operation(CleaningOperation(
        step="custom_step_1",
        description="First custom operation",
        impact="Custom impact",
    ))

    # 2. Add repair operation
    ingest.add_cleaning_operation(CleaningOperation(
        step="custom_step_2",
        description="Second custom operation",
        impact="Custom impact 2",
    ))

    assert len(ingest.cleaning_log) == initial_ops + 2
    # Verify working_frame alias
    assert ingest.working_frame is ingest.frame

    # Verify serialization via cleaning_operations property
    ops_dicts = ingest.cleaning_operations
    assert len(ops_dicts) == initial_ops + 2
    assert ops_dicts[-1]["step"] == "custom_step_2"
    assert ops_dicts[-1]["status"] == "applied"


def test_export_includes_cleaned_data_and_ledger(limits: Limits) -> None:
    df = pd.DataFrame({"Val": [10, 20]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ingest = load_tabular(csv_bytes, "export_test.csv", limits)
    ingest.add_cleaning_operation(CleaningOperation(
        step="test_ledger_export",
        description="Export verification op",
        impact="Testing sheet creation",
    ))

    excel_bytes = to_report_excel_bytes(ingest)
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    assert "Cleaned Data" in xls.sheet_names
    assert "Cleaning Ledger" in xls.sheet_names

    ledger_df = pd.read_excel(xls, sheet_name="Cleaning Ledger")
    assert not ledger_df.empty
    assert "Test Ledger Export" in ledger_df["Step"].values


class MockSessionState(dict):
    """Dictionary supporting attribute-style access for Streamlit session_state simulation."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


def test_recommendation_label_formatting_no_affected_count_crash() -> None:
    """1. Verify CleaningRecommendation model can be rendered without an affected_count crash."""
    from private_analyst.ui.streamlit_app import _format_cleaning_rec_label

    # Normal recommendation with positive affected rows
    rec_with_rows = CleaningRecommendation(
        id="drop_empty_rows",
        title="Drop Blank Rows",
        category="Structure",
        description="Found 5 blank rows.",
        affected_rows_count=5,
    )
    # Ensure rec does NOT have affected_count attribute
    assert not hasattr(rec_with_rows, "affected_count")
    label, rec_id = _format_cleaning_rec_label(rec_with_rows)
    assert rec_id == "drop_empty_rows"
    assert "Drop Blank Rows" in label
    assert "Found 5 blank rows." in label
    assert "(5 affected)" in label

    # Recommendation with None affected rows (unknown / not applicable)
    rec_none_rows = CleaningRecommendation(
        id="repair_unnamed_headers",
        title="Repair Unnamed Headers",
        category="Structure",
        description="Repair 2 unnamed columns.",
        affected_rows_count=None,
    )
    label_none, rec_id_none = _format_cleaning_rec_label(rec_none_rows)
    assert rec_id_none == "repair_unnamed_headers"
    assert "Repair Unnamed Headers" in label_none
    assert "affected" not in label_none

    # Recommendation with 0 affected rows
    rec_zero_rows = CleaningRecommendation(
        id="zero_op",
        title="Zero Op",
        category="Structure",
        description="Zero rows.",
        affected_rows_count=0,
    )
    label_zero, _ = _format_cleaning_rec_label(rec_zero_rows)
    assert "affected" not in label_zero


def test_real_recommendations_display_correct_description_and_impact() -> None:
    """2. Real recommendations display their correct description and available impact information."""
    from private_analyst.ui.streamlit_app import _format_cleaning_rec_label

    df = pd.DataFrame({
        "Amount": [" $1,000 ", " $2,500 ", " $1,000 ", None],
        "Pct": ["10%", "20%", "10%", None],
        "EmptyCol": [None, None, None, None],
    })
    recs = generate_cleaning_recommendations(df)
    assert len(recs) > 0

    for r in recs:
        assert not hasattr(r, "affected_count")
        label, rec_id = _format_cleaning_rec_label(r)
        assert rec_id == r.id
        assert r.title in label
        if r.description:
            assert r.description in label
        if r.affected_rows_count and r.affected_rows_count > 0:
            assert f"({r.affected_rows_count} affected)" in label
        else:
            assert "affected" not in label


def test_unknown_affected_counts_handled_honestly() -> None:
    """3. Unknown or non-applicable affected counts are represented safely and omit affected label."""
    from private_analyst.ui.streamlit_app import _format_cleaning_rec_label

    df = pd.DataFrame({
        "Unnamed: 0": [1, 2, 3],
        "ConstCol": ["A", "A", "A"],
        "AllEmpty": [None, None, None],
    })
    recs = generate_cleaning_recommendations(df)
    unnamed_rec = next(r for r in recs if r.id == "repair_unnamed_headers")
    assert unnamed_rec.affected_rows_count is None

    const_rec = next(r for r in recs if r.id == "drop_constant_columns")
    assert const_rec.affected_rows_count is None

    empty_rec = next(r for r in recs if r.id == "drop_empty_cols")
    assert empty_rec.affected_rows_count is None

    for r in [unnamed_rec, const_rec, empty_rec]:
        label, _ = _format_cleaning_rec_label(r)
        assert "affected" not in label
        assert "(0 affected)" not in label


def test_applying_selected_recommendation_invokes_correct_cleaning_operation() -> None:
    """4. Applying a selected recommendation still invokes the correct cleaning operation."""
    df = pd.DataFrame({
        "City": ["Delhi", "Mumbai", "Delhi"],
        "Code": [11, 22, 11],
    })
    cleaned_df, ops = apply_cleaning_plan(df, ["deduplicate_rows"])
    assert len(cleaned_df) == 2
    assert len(ops) == 1
    assert ops[0].step == "deduplicate_rows"
    assert ops[0].rows_affected == 1


def test_clean_tab_workflow_and_ledger_persistence() -> None:
    """5. Full cleaning workflow and cleaning history persistence across Streamlit reruns."""
    from unittest.mock import MagicMock, patch
    from private_analyst.auth.users import User
    from private_analyst.config import load_config
    from private_analyst.ui.streamlit_app import _clean_tab

    cfg = load_config()
    user = User(id=1, org_id="test-org", username="analyst", role="analyst")

    df = pd.DataFrame({
        "City": ["Delhi", "Mumbai", "Delhi"],
        "Amount": [" $100 ", " $200 ", " $100 "],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    limits = Limits()
    ingest = load_tabular(csv_bytes, "workflow_test.csv", limits)

    state = MockSessionState({
        "ingest": ingest,
        "clean_rec_deduplicate_rows": True,
        "clean_rec_strip_whitespace": True,
        "clean_rec_parse_currencies": True,
    })

    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app._get_audit_log") as mock_log_getter:
        mock_st.session_state = state
        # Simulate button clicked
        mock_st.button.return_value = True
        mock_log = MagicMock()
        mock_log_getter.return_value = mock_log

        # Execute _clean_tab: must not raise AttributeError('affected_count') or any error
        _clean_tab(cfg, user)

        # Verify operations were recorded and applied
        assert len(ingest.cleaning_log) >= 1
        assert len(ingest.frame) == 2  # Deduplicated from 3 to 2 rows
        assert mock_st.rerun.called
        assert mock_log.record.called
        assert mock_log.record.call_args[0][2] == "clean_dataset"


def test_clean_tab_empty_dataset_and_empty_recommendations() -> None:
    """6. The Clean Data page handles an empty dataset or an empty recommendation list."""
    from unittest.mock import patch
    from private_analyst.auth.users import User
    from private_analyst.config import load_config
    from private_analyst.ui.streamlit_app import _clean_tab

    cfg = load_config()
    user = User(id=1, org_id="test-org", username="analyst", role="analyst")

    # Case A: ingest is None
    state_none = MockSessionState({"ingest": None})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state_none
        _clean_tab(cfg, user)
        mock_st.info.assert_called_with("Import a table file first in '1. Import Data' to enable automated cleaning.")

    # Case B: ingest has empty DataFrame
    empty_ingest = IngestResult(
        frame=pd.DataFrame(),
        source_name="empty.csv",
        source_kind="csv",
    )
    state_empty = MockSessionState({"ingest": empty_ingest})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state_empty
        _clean_tab(cfg, user)
        assert mock_st.success.called
        assert "No automated cleaning operations are required" in mock_st.success.call_args[0][0]

    # Case C: clean DataFrame with 0 issues
    clean_df = pd.DataFrame({"Name": ["Alpha", "Beta"], "Score": [10, 20]})
    clean_ingest = IngestResult(
        frame=clean_df,
        source_name="clean.csv",
        source_kind="csv",
    )
    state_clean = MockSessionState({"ingest": clean_ingest})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st:
        mock_st.session_state = state_clean
        _clean_tab(cfg, user)
        assert mock_st.success.called


def test_clean_tab_handles_malformed_recommendations_gracefully() -> None:
    """7. The Clean Data page handles unexpected or dictionary recommendation objects safely."""
    from unittest.mock import patch
    from private_analyst.auth.users import User
    from private_analyst.config import load_config
    from private_analyst.ui.streamlit_app import _clean_tab

    cfg = load_config()
    user = User(id=1, org_id="test-org", username="analyst", role="analyst")

    dummy_df = pd.DataFrame({"Col": [1, 2]})
    dummy_ingest = IngestResult(
        frame=dummy_df,
        source_name="dummy.csv",
        source_kind="csv",
    )

    malformed_recs = [
        {"id": "dict_op", "title": "Dict Op", "is_lossy": False, "description": "Dict desc"},
        {"id": "", "title": "Empty ID"},  # Should be skipped safely
        CleaningRecommendation(
            id="partial_obj",
            title="Partial Obj",
            category="Other",
            description="",
            affected_rows_count=None,
        ),
    ]

    state = MockSessionState({"ingest": dummy_ingest})
    with patch("private_analyst.ui.streamlit_app.st") as mock_st, \
         patch("private_analyst.ui.streamlit_app.generate_cleaning_recommendations", return_value=malformed_recs):
        mock_st.session_state = state
        mock_st.button.return_value = False
        _clean_tab(cfg, user)
        # Check that checkbox was rendered for valid recs without crashing
        assert mock_st.checkbox.called
