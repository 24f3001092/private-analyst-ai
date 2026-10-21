from __future__ import annotations

import io
import pandas as pd
import pytest

from private_analyst.ai.explainer import explain_results
from private_analyst.analytics.ml_engine import (
    assess_feature_suitability,
    run_classification,
    run_regression,
)
from private_analyst.analytics.operations import run_operation
from private_analyst.analytics.sql_console import DuckDBConsole
from private_analyst.config import AIConfig, Limits
from private_analyst.exceptions import ValidationError
from private_analyst.ingestion.loader import detect_delimiter, load_tabular
from private_analyst.reports.exporters import to_csv_bytes, to_report_excel_bytes


def make_synthetic_hackathon_csv() -> bytes:
    """Create synthetic CSV modeling Hackathon_List_2026-1 layout with title banner and blank line."""
    lines = [
        "AGENTIC AI / AI HACKATHONS — INDIA + GLOBAL (2026)",
        "",
        "#,Hackathon Name,Organizer,Theme / Focus,Dates / Window,Location / Mode,Prizes / Grants,Eligibility,Application Link,Status",
    ]
    organizers = ["OpenAI", "Google", "Anthropic", "Microsoft"]
    themes = ["Autonomous Agents", "Multimodal RAG", "Code Generation", "Agent Tooling"]
    for i in range(1, 23):
        lines.append(
            f"{i},Hackathon {i},{organizers[i % 4]},{themes[i % 4]},March 2026,Virtual,{i * 5000},Open to all,https://example.com/{i},Open"
        )
    return "\n".join(lines).encode("utf-8")


def test_ragged_csv_with_title_banner_and_blank_rows() -> None:
    data = make_synthetic_hackathon_csv()
    limits = Limits()
    result = load_tabular(data, "Hackathon_List_2026-1.csv", limits)

    assert result.frame.shape == (22, 10)
    assert result.header_row == 2
    assert "AGENTIC AI / AI HACKATHONS" in result.detected_title
    assert list(result.frame.columns) == [
        "#",
        "Hackathon Name",
        "Organizer",
        "Theme / Focus",
        "Dates / Window",
        "Location / Mode",
        "Prizes / Grants",
        "Eligibility",
        "Application Link",
        "Status",
    ]
    assert result.raw_frame is not None
    assert result.raw_frame.shape == (25, 10)
    assert result.excluded_rows >= 2


def test_detect_delimiter_heuristics() -> None:
    comma_data = "a,b,c\n1,2,3"
    semi_data = "a;b;c\n1;2;3"
    tab_data = "a\tb\tc\n1\t2\t3"
    pipe_data = "a|b|c\n1|2|3"

    assert detect_delimiter(comma_data) == ","
    assert detect_delimiter(semi_data) == ";"
    assert detect_delimiter(tab_data) == "\t"
    assert detect_delimiter(pipe_data) == "|"


def test_feature_suitability_assessment() -> None:
    df = pd.DataFrame({
        "#": list(range(1, 26)),
        "Hackathon Name": [f"Event {i}" for i in range(1, 26)],
        "Organizer": ["OrgA", "OrgB", "OrgC", "OrgD", "OrgE"] * 5,
        "Score": [float(i * 3.5) for i in range(1, 26)],
        "Constant": ["Same"] * 25,
    })

    suitability = assess_feature_suitability(df, target_col="Score")

    assert suitability["suitable_numeric"] == []  # Score was target
    assert "Organizer" in suitability["suitable_categorical"]
    assert "Hackathon Name" in suitability["high_cardinality_text"]
    assert "#" in suitability["identifiers"]
    assert "Constant" in suitability["constant"]
    assert suitability["recommended_features"] == ["Organizer"]


def test_classification_high_cardinality_rejection() -> None:
    df = pd.DataFrame({
        "Hackathon Name": [f"Event {i}" for i in range(1, 26)],
        "Organizer": ["OrgA", "OrgB"] * 12 + ["OrgA"],
        "Target": ["Yes", "No"] * 12 + ["Yes"],
    })

    with pytest.raises(ValidationError) as exc:
        run_classification(df, target_col="Target", feature_cols=["Hackathon Name"])
    assert "too many distinct values" in str(exc.value)

    # Valid run with suitable features
    res = run_classification(df, target_col="Target", feature_cols=["Organizer"])
    assert res.task == "classification"
    assert res.status in ("evaluated", "hypothesis")


def test_sql_console_virtual_table_and_schema() -> None:
    data = make_synthetic_hackathon_csv()
    result = load_tabular(data, "Hackathon_List_2026-1.csv", Limits())

    console = DuckDBConsole(max_rows=1000)
    console.register_dataframe("data", result.frame)

    tables = console.list_tables()
    assert "data" in tables

    schema = console.describe_table("data")
    col_names = [col["name"] for col in schema]
    assert "Hackathon Name" in col_names
    assert "Theme / Focus" in col_names

    res = console.execute_query('SELECT count(*) AS total_rows FROM data;')
    assert res.row_count == 1
    assert res.rows[0]["total_rows"] == 22

    res_limit = console.execute_query('SELECT "#", "Organizer" FROM data LIMIT 5;')
    assert res_limit.row_count == 5
    assert len(res_limit.rows) == 5

    console.close()


def test_offline_explainer_fallback_narrative() -> None:
    data = make_synthetic_hackathon_csv()
    result = load_tabular(data, "Hackathon_List_2026-1.csv", Limits())
    op_res = run_operation(result, "Summarize the dataset.")

    offline_cfg = AIConfig(host="127.0.0.1", port=59999)
    exp = explain_results(op_res, "Summarize the dataset.", offline_cfg)

    assert exp["source"] == "deterministic_fallback"
    assert exp["model_available"] is False
    assert "Analysis Summary" in exp["text"]
    assert "Observed Facts" in exp["text"]
    assert "URLError" not in exp["text"]  # Offline status handled cleanly without raw dump


def test_csv_and_report_export_protection() -> None:
    df_dangerous = pd.DataFrame({
        "Name": ["=CMD|' /C calc'!A0", "@SUM(1+1)", "-15.5", "+42", "+CMD()"],
        "Value": [10, 20, 30, 40, 50],
    })

    csv_bytes = to_csv_bytes(df_dangerous)
    csv_text = csv_bytes.decode("utf-8")

    assert "'=CMD" in csv_text
    assert "'@SUM" in csv_text
    assert "'-15.5" not in csv_text  # Legitimate negative number intact
    assert "'+42" not in csv_text  # Legitimate positive number intact
    assert "'+CMD" in csv_text  # Non-numeric + prefix sanitized
