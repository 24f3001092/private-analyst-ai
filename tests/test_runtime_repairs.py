from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from private_analyst.analytics.cleaner import (
    apply_column_management,
    apply_header_override,
    repair_unnamed_headers,
)
from private_analyst.analytics.statistics import (
    compute_dataset_stats,
    explain_chart_exclusions,
    recommend_charts,
)
from private_analyst.exceptions import ValidationError
from private_analyst.ingestion.loader import (
    CleaningOperation,
    IngestResult,
    check_header_ambiguity,
    normalize_cleaning_operation,
)
from private_analyst.ui.streamlit_app import _invalidate_downstream_state
import streamlit as st


# ---------------------------------------------------------------------------
# 1. Column management with valid columns
# ---------------------------------------------------------------------------
def test_column_management_valid_drop_and_rename() -> None:
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alpha", "Beta", "Gamma"],
        "score": [10.5, 20.0, 30.2],
        "category": ["A", "B", "A"],
    })

    # Drop column using exclude_columns and alias drop_columns
    df_dropped, ops1 = apply_column_management(df, exclude_columns=["category"])
    assert list(df_dropped.columns) == ["id", "name", "score"]
    assert len(ops1) == 1
    assert ops1[0].step == "exclude_columns"
    assert "category" in ops1[0].description
    assert df_dropped["score"].dtype == df["score"].dtype

    df_dropped2, ops2 = apply_column_management(df, drop_columns=["id"])
    assert list(df_dropped2.columns) == ["name", "score", "category"]
    assert len(ops2) == 1

    # Rename column
    df_renamed, ops3 = apply_column_management(df, rename_columns={"score": "final_score"})
    assert list(df_renamed.columns) == ["id", "name", "final_score", "category"]
    assert len(ops3) == 1
    assert ops3[0].step == "rename_columns"
    assert df_renamed["final_score"].iloc[0] == 10.5


# ---------------------------------------------------------------------------
# 2. Column management with empty selection
# ---------------------------------------------------------------------------
def test_column_management_empty_selection() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    df_no_op, ops = apply_column_management(df)
    assert df_no_op.equals(df)
    assert ops == []

    df_empty_drop, ops = apply_column_management(df, exclude_columns=[])
    assert df_empty_drop.equals(df)
    assert ops == []

    df_empty_rename, ops = apply_column_management(df, rename_columns={})
    assert df_empty_rename.equals(df)
    assert ops == []


# ---------------------------------------------------------------------------
# 3. Invalid and duplicate column names
# ---------------------------------------------------------------------------
def test_column_management_invalid_and_duplicate_names() -> None:
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})

    # Nonexistent column to drop
    with pytest.raises(ValidationError, match=r"exist in the table|not found in dataset"):
        apply_column_management(df, exclude_columns=["nonexistent"])

    # Nonexistent column to include
    with pytest.raises(ValidationError, match=r"not found in dataset"):
        apply_column_management(df, include_columns=["col_a", "ghost"])

    # Empty new column name in rename
    with pytest.raises(ValidationError, match=r"cannot be empty or whitespace"):
        apply_column_management(df, rename_columns={"col_a": "   "})

    # Duplicate target names
    with pytest.raises(ValidationError, match=r"Duplicate target column names"):
        apply_column_management(df, rename_columns={"col_a": "dup", "col_b": "dup"})

    # Target collision with unrenamed column
    with pytest.raises(ValidationError, match=r"already exists"):
        apply_column_management(df, rename_columns={"col_a": "col_b"})


# ---------------------------------------------------------------------------
# 4. Dropping all columns raises ValidationError
# ---------------------------------------------------------------------------
def test_column_management_drop_all_columns_raises() -> None:
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with pytest.raises(ValidationError, match=r"Cannot drop all columns"):
        apply_column_management(df, exclude_columns=["col1", "col2"])

    with pytest.raises(ValidationError, match=r"Include column list cannot be empty"):
        apply_column_management(df, include_columns=[])


# ---------------------------------------------------------------------------
# 5. Resetting modified dataset to original state
# ---------------------------------------------------------------------------
def test_reset_dataset_to_original_state() -> None:
    raw_df = pd.DataFrame({
        0: ["AGENTIC AI TITLE BANNER", "Header1", "Val1"],
        1: ["", "Header2", "Val2"],
    })
    initial_working_df = pd.DataFrame({
        "Header1": ["Val1"],
        "Header2": ["Val2"],
    })

    ingest = IngestResult(
        frame=initial_working_df.copy(),
        source_kind="csv",
        raw_frame=raw_df.copy(),
        initial_frame=initial_working_df.copy(),
        source_name="test.csv",
        detected_title="AGENTIC AI TITLE BANNER",
    )

    # Modify working frame
    modified_df, _ = apply_column_management(ingest.frame, exclude_columns=["Header2"])
    ingest.frame = modified_df
    ingest.working_frame = modified_df
    assert list(ingest.frame.columns) == ["Header1"]

    # Reset to initial loaded state
    restored_initial = ingest.initial_frame.copy()
    ingest.frame = restored_initial
    ingest.working_frame = restored_initial
    reset_op = CleaningOperation(
        step="reset_dataset",
        description="Reset dataset to initial loaded state",
        impact=f"Restored initial loaded table with {len(restored_initial)} rows and {len(restored_initial.columns)} columns",
        rows_affected=len(restored_initial),
    )
    ingest.cleaning_log = [reset_op]

    assert list(ingest.frame.columns) == ["Header1", "Header2"]
    assert len(ingest.cleaning_log) == 1
    assert ingest.cleaning_log[0].step == "reset_dataset"

    # Reset to raw extract
    restored_raw = ingest.raw_frame.copy()
    ingest.frame = restored_raw
    ingest.working_frame = restored_raw
    reset_op_raw = CleaningOperation(
        step="reset_dataset",
        description="Reset dataset to raw spreadsheet extract",
        impact=f"Restored raw extract buffer with {len(restored_raw)} rows and {len(restored_raw.columns)} columns",
        rows_affected=len(restored_raw),
    )
    ingest.cleaning_log = [reset_op_raw]

    assert ingest.frame.shape == (3, 2)
    assert raw_df.shape == (3, 2)  # raw_df unmutated


# ---------------------------------------------------------------------------
# 6. Repeated resets
# ---------------------------------------------------------------------------
def test_repeated_resets_idempotent() -> None:
    raw_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ingest = IngestResult(
        frame=raw_df.copy(),
        source_kind="csv",
        raw_frame=raw_df.copy(),
        initial_frame=raw_df.copy(),
        source_name="data.csv",
    )

    for i in range(3):
        # Mutate
        ingest.frame = ingest.frame[["a"]]
        assert list(ingest.frame.columns) == ["a"]

        # Reset
        ingest.frame = ingest.initial_frame.copy()
        ingest.working_frame = ingest.initial_frame.copy()
        reset_op = CleaningOperation(
            step="reset_dataset",
            description=f"Reset attempt {i+1}",
            impact="Restored",
            rows_affected=len(ingest.frame),
        )
        ingest.cleaning_log = [reset_op]

        assert list(ingest.frame.columns) == ["a", "b"]
        assert len(ingest.cleaning_log) == 1


# ---------------------------------------------------------------------------
# 7. Missing or malformed cleaning-operation metadata
# ---------------------------------------------------------------------------
def test_normalize_cleaning_operation_robustness() -> None:
    # 1. Proper CleaningOperation object
    op_obj = CleaningOperation(
        step="drop_duplicates",
        description="Removed duplicate records",
        impact="2 rows removed",
        rows_affected=2,
    )
    row, warn = normalize_cleaning_operation(op_obj)
    assert row["Step"] == "Drop Duplicates"
    assert row["Rows Affected"] == 2
    assert warn is None

    # 2. Dictionary format
    op_dict = {
        "step": "repair_headers",
        "description": "Repaired unnamed headers",
        "impact": "1 header renamed",
        "rows_affected": 0,
    }
    row, warn = normalize_cleaning_operation(op_dict)
    assert row["Step"] == "Repair Headers"
    assert warn is None

    # 3. Malformed dictionary with missing fields
    sparse_dict = {"step": "custom_step"}
    row, warn = normalize_cleaning_operation(sparse_dict)
    assert row["Step"] == "Custom Step"
    assert row["Description"] == "No description provided"
    assert row["Rows Affected"] == 0

    # 4. Non-dictionary / unexpected type
    row, warn = normalize_cleaning_operation("just a string")
    assert row["Step"] == "Malformed Operation"
    assert "just a string" in row["Description"]
    assert warn is not None


# ---------------------------------------------------------------------------
# 8. Numeric-looking and duplicate headers
# ---------------------------------------------------------------------------
def test_numeric_and_duplicate_headers_handling() -> None:
    # Row with numeric headers: [1, 2, 3]
    df_raw = pd.DataFrame([
        [1, 2, 3],
        ["val1", "val2", "val3"],
        ["val4", "val5", "val6"],
    ])

    df_promoted, op = apply_header_override(df_raw, 0)
    assert list(df_promoted.columns) == ["1", "2", "3"]
    assert len(df_promoted) == 2

    # Row with duplicate and missing/nan headers
    df_raw_dups = pd.DataFrame([
        ["Score", "Score", np.nan, None, "Score"],
        [10, 20, 30, 40, 50],
    ])
    df_promoted2, op2 = apply_header_override(df_raw_dups, 0)
    expected_cols = ["Score", "Score__1", "unnamed_column__3", "unnamed_column__4", "Score__2"]
    assert list(df_promoted2.columns) == expected_cols


# ---------------------------------------------------------------------------
# 9. Ragged CSV rows and introductory banner rows
# ---------------------------------------------------------------------------
def test_header_ambiguity_detection() -> None:
    # Ambiguous: row 0 is a single title cell with empty values
    ambiguous_df = pd.DataFrame([
        ["AGENTIC AI HACKATHONS 2026", np.nan, np.nan, np.nan],
        ["", "", "", ""],
        ["#", "Hackathon Name", "Organizer", "Prize"],
        ["1", "Global AI Hack", "Google", "10000"],
    ])
    is_amb, reason = check_header_ambiguity(ambiguous_df, header_row=0)
    assert is_amb is True
    assert "mostly blank" in str(reason)

    # Mostly numeric row 0
    numeric_header_df = pd.DataFrame([
        [101, 102, 103, 104],
        ["A", "B", "C", "D"],
    ])
    is_amb2, reason2 = check_header_ambiguity(numeric_header_df, header_row=0)
    assert is_amb2 is True
    assert "numeric" in str(reason2)

    # Informative header row
    clean_df = pd.DataFrame([
        ["Customer_ID", "First_Name", "Last_Name", "Revenue"],
        ["C01", "Jane", "Doe", 5000],
    ])
    is_amb3, reason3 = check_header_ambiguity(clean_df, header_row=0)
    assert is_amb3 is False
    assert reason3 is None


# ---------------------------------------------------------------------------
# 10. Chart recommendations with high-cardinality, numeric, and constant cols
# ---------------------------------------------------------------------------
def test_chart_recommendations_and_suitability() -> None:
    n_rows = 25
    df = pd.DataFrame({
        "1": list(range(1, n_rows + 1)),                   # Numeric sequence / index artifact
        "#": list(range(1, n_rows + 1)),                   # Sequential counter
        "Unnamed: 2": [f"Val_{i}" for i in range(n_rows)],  # Unnamed artifact
        "Event_Name": [f"Event {i}" for i in range(n_rows)],# High-cardinality near-unique text
        "Constant_Status": ["Active"] * n_rows,            # Constant column
        "Category": ["Web", "AI", "Mobile", "AI", "Web"] * 5,  # Low cardinality category
        "Prize_Amount": [float(i * 1000) for i in range(n_rows)],  # Numeric metric
    })

    stats = compute_dataset_stats(df)
    recs = recommend_charts(df, stats)

    rec_columns = [r.get("column") for r in recs if "column" in r]

    # Artifacts must NEVER be recommended
    assert "1" not in rec_columns
    assert "#" not in rec_columns
    assert "Unnamed: 2" not in rec_columns
    assert "Event_Name" not in rec_columns
    assert "Constant_Status" not in rec_columns

    # Category and Numeric must be recommended
    assert "Category" in rec_columns
    assert "Prize_Amount" in rec_columns

    # Verify explain_chart_exclusions explains exclusions
    explanations = explain_chart_exclusions(df, stats)
    reasons = {e["Column"]: e["Explanation"] for e in explanations}

    assert "row sequence or primary key" in reasons["#"] or "Index or sequential row counter" in reasons["#"]
    assert "Sequential row index" in reasons["1"] or "Index or sequential row counter" in reasons["1"]
    assert "Unnamed spreadsheet layout artifact" in reasons["Unnamed: 2"] or "Unnamed column artifact" in reasons["Unnamed: 2"]
    assert "high-cardinality" in reasons["Event_Name"].lower()
    assert "constant" in reasons["Constant_Status"].lower()
    assert "Included in automated visualizations" in reasons["Category"]
    assert "Included in automated visualizations" in reasons["Prize_Amount"]


# ---------------------------------------------------------------------------
# 11. State invalidation after successful dataset changes
# ---------------------------------------------------------------------------
def test_state_invalidation_lifecycle() -> None:
    st.session_state.last_result = {"some": "data"}
    st.session_state.last_explanation = "Previous explanation"
    st.session_state["sql_registered_frame_id"] = "old_frame_id"

    class FakeDuckDBConsole:
        def __init__(self):
            self.closed = False
        def close(self):
            self.closed = True

    fake_console = FakeDuckDBConsole()
    st.session_state.duckdb_console = fake_console

    _invalidate_downstream_state()

    assert st.session_state.last_result is None
    assert st.session_state.last_explanation is None
    assert st.session_state.duckdb_console is None
    assert st.session_state["sql_registered_frame_id"] is None
    assert fake_console.closed is True


# ---------------------------------------------------------------------------
# 12. Failure handling without corrupting active dataset
# ---------------------------------------------------------------------------
def test_failure_handling_preserves_active_dataset() -> None:
    original_df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": [4, 5, 6],
    })
    working_df = original_df.copy()

    # Attempt an illegal drop
    with pytest.raises(ValidationError):
        apply_column_management(working_df, exclude_columns=["col_a", "col_b"])

    # Ensure working_df is untouched
    assert working_df.equals(original_df)
    assert list(working_df.columns) == ["col_a", "col_b"]
    assert working_df.shape == (3, 2)
