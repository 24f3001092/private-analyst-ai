from __future__ import annotations

import io

import pandas as pd
import pytest

from private_analyst.analytics.sql_console import DuckDBConsole, QueryResult
from private_analyst.exceptions import ValidationError


def test_select_query_returns_correct_results(sales_csv_bytes: bytes) -> None:
    frame = pd.read_csv(io.BytesIO(sales_csv_bytes))
    console = DuckDBConsole()
    console.register_dataframe("sales", frame)
    result = console.execute_query("SELECT SUM(units) FROM sales")
    assert isinstance(result, QueryResult)
    assert result.row_count == 1
    assert not result.truncated
    assert len(result.columns) == 1
    first_val = list(result.rows[0].values())[0]
    assert first_val == 50


def test_insert_rejected(sales_csv_bytes: bytes) -> None:
    frame = pd.read_csv(io.BytesIO(sales_csv_bytes))
    console = DuckDBConsole()
    console.register_dataframe("sales", frame)
    with pytest.raises(ValidationError):
        console.execute_query(
            "INSERT INTO sales VALUES ('2024-01-06', 'East', 'Widget', 10, 100.0, 'ok')"
        )


def test_drop_rejected(sales_csv_bytes: bytes) -> None:
    frame = pd.read_csv(io.BytesIO(sales_csv_bytes))
    console = DuckDBConsole()
    console.register_dataframe("sales", frame)
    with pytest.raises(ValidationError):
        console.execute_query("DROP TABLE sales")


def test_describe_table(sales_csv_bytes: bytes) -> None:
    frame = pd.read_csv(io.BytesIO(sales_csv_bytes))
    console = DuckDBConsole()
    console.register_dataframe("sales", frame)
    desc = console.describe_table("sales")
    assert isinstance(desc, list)
    col_names = [col["name"] for col in desc]
    assert "date" in col_names
    assert "region" in col_names
    assert "product" in col_names
    assert "units" in col_names
    assert "revenue" in col_names
    assert "status" in col_names

    for col in desc:
        assert "name" in col
        assert "type" in col
        assert isinstance(col["name"], str)
        assert isinstance(col["type"], str)


def test_list_tables() -> None:
    console = DuckDBConsole()
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [3, 4]})
    console.register_dataframe("table_alpha", df1)
    console.register_dataframe("table_beta", df2)
    tables = console.list_tables()
    assert "table_alpha" in tables
    assert "table_beta" in tables


def test_empty_query() -> None:
    console = DuckDBConsole()
    with pytest.raises(ValidationError):
        console.execute_query("")
    with pytest.raises(ValidationError):
        console.execute_query("   ")
    with pytest.raises(ValidationError):
        console.execute_query("   \n\t  ")
    with pytest.raises(ValidationError):
        console.execute_query("-- just a comment\n")


def test_row_limit() -> None:
    console = DuckDBConsole()
    frame = pd.DataFrame({"val": list(range(20_000))})
    console.register_dataframe("large_table", frame)
    result = console.execute_query("SELECT * FROM large_table")
    assert result.truncated is True
    assert result.row_count == 10_000
    assert len(result.rows) == 10_000


def test_update_rejected() -> None:
    console = DuckDBConsole()
    df = pd.DataFrame({"x": [1, 2]})
    console.register_dataframe("t", df)
    with pytest.raises(ValidationError):
        console.execute_query("UPDATE t SET x = 99")


def test_delete_rejected() -> None:
    console = DuckDBConsole()
    df = pd.DataFrame({"x": [1, 2]})
    console.register_dataframe("t", df)
    with pytest.raises(ValidationError):
        console.execute_query("DELETE FROM t")


def test_multiple_statements_rejected() -> None:
    console = DuckDBConsole()
    df = pd.DataFrame({"x": [1, 2]})
    console.register_dataframe("t", df)
    with pytest.raises(ValidationError):
        console.execute_query("SELECT * FROM t; DROP TABLE t;")


def test_with_cte_query_allowed(sales_csv_bytes: bytes) -> None:
    frame = pd.read_csv(io.BytesIO(sales_csv_bytes))
    console = DuckDBConsole()
    console.register_dataframe("sales", frame)
    query = """
    WITH aggregated AS (
        SELECT region, SUM(units) as total_units
        FROM sales
        GROUP BY region
    )
    SELECT region, total_units FROM aggregated ORDER BY total_units DESC
    """
    result = console.execute_query(query)
    assert result.row_count == 2
    assert not result.truncated
    assert "region" in result.columns
    assert "total_units" in result.columns


def test_describe_nonexistent_table() -> None:
    console = DuckDBConsole()
    with pytest.raises(ValidationError):
        console.describe_table("nonexistent")


def test_invalid_table_name() -> None:
    console = DuckDBConsole()
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValidationError):
        console.register_dataframe("table with spaces", df)
    with pytest.raises(ValidationError):
        console.register_dataframe("123bad_identifier", df)


def test_syntax_error() -> None:
    console = DuckDBConsole()
    df = pd.DataFrame({"x": [1]})
    console.register_dataframe("t", df)
    with pytest.raises(ValidationError):
        console.execute_query("SELECT FROM WHERE")


def test_query_result_to_dict() -> None:
    result = QueryResult(
        columns=["a"],
        rows=[{"a": 1}],
        row_count=1,
        truncated=False,
        sql="SELECT 1 as a",
    )
    d = result.to_dict()
    assert d["columns"] == ["a"]
    assert d["rows"] == [{"a": 1}]
    assert d["row_count"] == 1
    assert d["truncated"] is False
    assert d["sql"] == "SELECT 1 as a"
