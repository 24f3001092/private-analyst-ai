"""Security regression tests for DuckDB SQL Console sandbox and file-access prevention.

Verifies that:
- Filesystem reading functions (read_csv, read_parquet, read_text, read_json, scan_csv) are blocked.
- Direct string literal table sources (FROM 'path/to/file') are blocked.
- Synthetic canary files on disk cannot be read or leaked.
- Remote URL access and httpfs extensions are blocked.
- Mutation/DDL/DML/administrative operations (ATTACH, LOAD, INSTALL, PRAGMA, SET) are blocked.
- Multi-statement injection with comments and whitespace tricks are blocked.
- Unregistered tables are blocked.
- Query timeouts and memory limits are enforced.
- Approved SELECT queries, aggregations, CTEs, and joins continue to function normally.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from private_analyst.analytics.sql_console import DuckDBConsole
from private_analyst.exceptions import ResourceLimitError, ValidationError


@pytest.fixture
def canary_file(tmp_path: Path) -> Path:
    """Create a temporary synthetic canary file with a unique secret token."""
    canary = tmp_path / "synthetic_canary.txt"
    canary.write_text("CANARY_TOKEN_98765_DO_NOT_LEAK\nSECRET_DATA,12345\n", encoding="utf-8")
    return canary


@pytest.fixture
def sales_console(sales_csv_bytes: bytes) -> DuckDBConsole:
    frame = pd.read_csv(io.BytesIO(sales_csv_bytes))
    console = DuckDBConsole()
    console.register_dataframe("sales", frame)
    return console


class TestDuckDBFileSystemAccessBlocked:
    """Verify that file access is completely denied through all DuckDB table functions."""

    def test_read_csv_blocked_with_canary(self, sales_console: DuckDBConsole, canary_file: Path) -> None:
        canary_path = str(canary_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM read_csv('{canary_path}')")

    def test_read_csv_auto_blocked(self, sales_console: DuckDBConsole, canary_file: Path) -> None:
        canary_path = str(canary_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM read_csv_auto('{canary_path}')")

    def test_read_parquet_blocked(self, sales_console: DuckDBConsole, tmp_path: Path) -> None:
        parquet_file = tmp_path / "canary.parquet"
        parquet_file.write_bytes(b"PAR1fake_parquet_content")
        path_str = str(parquet_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM read_parquet('{path_str}')")

    def test_read_json_blocked(self, sales_console: DuckDBConsole, tmp_path: Path) -> None:
        json_file = tmp_path / "canary.json"
        json_file.write_text('{"token": "CANARY_JSON_123"}', encoding="utf-8")
        path_str = str(json_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM read_json('{path_str}')")

    def test_read_blob_blocked(self, sales_console: DuckDBConsole, canary_file: Path) -> None:
        canary_path = str(canary_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM read_blob('{canary_path}')")

    def test_read_text_blocked(self, sales_console: DuckDBConsole, canary_file: Path) -> None:
        canary_path = str(canary_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM read_text('{canary_path}')")

    def test_direct_string_path_table_source_blocked(self, sales_console: DuckDBConsole, canary_file: Path) -> None:
        canary_path = str(canary_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM '{canary_path}'")

    def test_direct_quoted_path_table_source_blocked(self, sales_console: DuckDBConsole, canary_file: Path) -> None:
        canary_path = str(canary_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f'SELECT * FROM "{canary_path}"')

    def test_join_with_file_blocked(self, sales_console: DuckDBConsole, canary_file: Path) -> None:
        canary_path = str(canary_file).replace("\\", "/")
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"SELECT * FROM sales JOIN read_csv('{canary_path}') ON 1=1")

    def test_glob_function_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM glob('*.*')")


class TestDuckDBRemoteAndExtensionAccessBlocked:
    """Verify that remote network sources and extension loading are prevented."""

    def test_http_url_read_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM read_csv('http://127.0.0.1:8000/canary.csv')")

    def test_https_url_read_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM read_csv('https://example.com/data.parquet')")

    def test_install_extension_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("INSTALL httpfs")

    def test_load_extension_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("LOAD httpfs")

    def test_attach_database_blocked(self, sales_console: DuckDBConsole, tmp_path: Path) -> None:
        db_file = tmp_path / "other.db"
        with pytest.raises(ValidationError):
            sales_console.execute_query(f"ATTACH '{db_file}' AS other_db")

    def test_detach_database_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("DETACH other_db")


class TestDuckDBMetadataAndCatalogProbingBlocked:
    """Verify that internal system/catalog exploration is blocked."""

    def test_duckdb_settings_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM duckdb_settings()")

    def test_duckdb_secrets_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM duckdb_secrets()")

    def test_pragma_database_list_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM pragma_database_list()")

    def test_unregistered_table_rejected(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to table 'secret_table' is not allowed"):
            sales_console.execute_query("SELECT * FROM secret_table")

    def test_schema_qualified_internal_table_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM information_schema.tables")


class TestDuckDBEvasionAndFormattingTricks:
    """Verify that comment tricks, case tricks, whitespace, and null bytes are rejected."""

    def test_null_byte_rejected(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM sales\x00; DROP TABLE sales;")

    def test_semicolon_hidden_in_comment_rejected(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SELECT * FROM sales; -- comment \n DROP TABLE sales;")

    def test_case_variation_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("dRoP tAbLe sales")

    def test_set_configuration_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("SET enable_external_access = true")

    def test_pragma_mutation_blocked(self, sales_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            sales_console.execute_query("PRAGMA threads = 10")


class TestDuckDBResourceLimits:
    """Verify query timeouts and result truncation."""

    def test_query_timeout_interrupts_execution(self, sales_console: DuckDBConsole) -> None:
        # A slow recursive/cross-join Cartesian product query with short timeout
        slow_query = """
        WITH RECURSIVE nums(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM nums WHERE x < 5000000
        )
        SELECT COUNT(*) FROM nums, nums as nums2
        """
        with pytest.raises(ResourceLimitError):
            sales_console.execute_query(slow_query, timeout_seconds=1)


class TestApprovedQueriesFunctionNormally:
    """Verify that legitimate analytical queries work without false positives."""

    def test_aggregation_and_grouping(self, sales_console: DuckDBConsole) -> None:
        result = sales_console.execute_query(
            "SELECT region, SUM(revenue) AS total_rev, COUNT(*) AS cnt FROM sales GROUP BY region ORDER BY region"
        )
        assert result.row_count == 2
        assert "region" in result.columns
        assert "total_rev" in result.columns
        rows = {r["region"]: r["total_rev"] for r in result.rows}
        assert rows["East"] == 175.0  # 100 + 75 + 0
        assert rows["West"] == 425.0  # 200 + 225

    def test_filtering_and_order_by(self, sales_console: DuckDBConsole) -> None:
        result = sales_console.execute_query(
            "SELECT product, units FROM sales WHERE units > 5 ORDER BY units DESC"
        )
        assert result.row_count == 3
        assert result.rows[0]["units"] == 20

    def test_safe_self_join(self, sales_console: DuckDBConsole) -> None:
        result = sales_console.execute_query(
            "SELECT a.region, a.product, b.units FROM sales a JOIN sales b ON a.product = b.product AND a.region != b.region"
        )
        assert result.row_count > 0

    def test_scalar_select_without_from(self, sales_console: DuckDBConsole) -> None:
        result = sales_console.execute_query("SELECT 42 AS answer, 'safe' AS label")
        assert result.rows[0]["answer"] == 42
        assert result.rows[0]["label"] == "safe"


@pytest.fixture
def data_console() -> DuckDBConsole:
    console = DuckDBConsole()
    console.register_dataframe("data", pd.DataFrame({"id": [1, 2], "val": [10, 20]}))
    return console


class TestSecSql01And02Regression:
    """Regression tests for SEC-SQL-01 (comma joins) and SEC-SQL-02 (schema qualification)."""

    def test_comma_join_duckdb_functions_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="cannot be used as a table source"):
            data_console.execute_query("SELECT * FROM data, duckdb_functions()")

    def test_comma_join_duckdb_keywords_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="cannot be used as a table source"):
            data_console.execute_query("SELECT * FROM data, duckdb_keywords()")

    def test_comma_join_subquery_and_invalid_table_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to table 'invalid_table' is not allowed"):
            data_console.execute_query("SELECT * FROM data, (SELECT 1), invalid_table")

    def test_unknown_schema_qualified_table_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to schema-qualified source 'unknown_schema.data' is not permitted"):
            data_console.execute_query("SELECT * FROM unknown_schema.data")

    def test_main_schema_registered_table_allowed(self, data_console: DuckDBConsole) -> None:
        result = data_console.execute_query("SELECT * FROM main.data")
        assert result.row_count == 2
        assert result.columns == ["id", "val"]

    def test_main_schema_unregistered_table_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to schema-qualified source 'main.invalid_table' is not permitted"):
            data_console.execute_query("SELECT * FROM main.invalid_table")

    def test_quoted_unknown_schema_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to schema-qualified source"):
            data_console.execute_query('SELECT * FROM "unknown_schema"."data"')

    def test_quoted_main_schema_allowed(self, data_console: DuckDBConsole) -> None:
        result = data_console.execute_query('SELECT * FROM "main"."data"')
        assert result.row_count == 2

    def test_unquoted_main_quoted_table_allowed(self, data_console: DuckDBConsole) -> None:
        result = data_console.execute_query('SELECT * FROM main."data"')
        assert result.row_count == 2

    def test_quoted_main_unquoted_table_allowed(self, data_console: DuckDBConsole) -> None:
        result = data_console.execute_query('SELECT * FROM "main".data')
        assert result.row_count == 2

    def test_comment_bypass_attempts_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError):
            data_console.execute_query("SELECT * FROM data, /* bypass */ duckdb_functions()")
        with pytest.raises(ValidationError):
            data_console.execute_query("SELECT * FROM data, -- comment\n duckdb_functions()")
        with pytest.raises(ValidationError):
            data_console.execute_query("SELECT * FROM data, /* sub */ (SELECT 1), /* inv */ invalid_table")

    def test_nested_subquery_in_where_unregistered_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to table 'secret_table' is not allowed"):
            data_console.execute_query("SELECT * FROM data WHERE id IN (SELECT id FROM secret_table)")

    def test_nested_subquery_in_where_registered_allowed(self, data_console: DuckDBConsole) -> None:
        result = data_console.execute_query("SELECT * FROM data WHERE id IN (SELECT id FROM data WHERE val > 10)")
        assert result.row_count == 1
        assert result.rows[0]["id"] == 2

    def test_nested_subquery_in_from_unregistered_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to table 'secret_table' is not allowed"):
            data_console.execute_query("SELECT * FROM (SELECT * FROM secret_table)")

    def test_deeply_nested_subquery_unregistered_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to table 'secret_table' is not allowed"):
            data_console.execute_query("SELECT * FROM (SELECT * FROM (SELECT * FROM secret_table))")

    def test_nested_subquery_in_join_on_condition_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to table 'secret_table' is not allowed"):
            data_console.execute_query("SELECT * FROM data d1 JOIN data d2 ON d1.id = (SELECT max(id) FROM secret_table)")

    def test_scalar_subquery_in_select_list_unregistered_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="Access to table 'secret_table' is not allowed"):
            data_console.execute_query("SELECT (SELECT max(id) FROM secret_table) AS m FROM data")

    def test_scalar_subquery_in_select_list_registered_allowed(self, data_console: DuckDBConsole) -> None:
        result = data_console.execute_query("SELECT (SELECT max(id) FROM data) AS m FROM data")
        assert result.row_count == 2
        assert result.rows[0]["m"] == 2

    def test_cte_with_unregistered_table_blocked(self, data_console: DuckDBConsole) -> None:
        query = "WITH bad_cte AS (SELECT * FROM secret_table) SELECT * FROM bad_cte"
        with pytest.raises(ValidationError, match="Access to table 'secret_table' is not allowed"):
            data_console.execute_query(query)

    def test_cte_with_comma_join_function_blocked(self, data_console: DuckDBConsole) -> None:
        query = "WITH bad_cte AS (SELECT * FROM data, duckdb_functions()) SELECT * FROM bad_cte"
        with pytest.raises(ValidationError, match="cannot be used as a table source"):
            data_console.execute_query(query)

    def test_quoted_cte_with_comma_join_invalid_table_blocked(self, data_console: DuckDBConsole) -> None:
        query = 'WITH "bad_cte" AS (SELECT * FROM data, (SELECT 1), invalid_table) SELECT * FROM "bad_cte"'
        with pytest.raises(ValidationError, match="Access to table 'invalid_table' is not allowed"):
            data_console.execute_query(query)

    def test_quoted_cte_valid_query_allowed(self, data_console: DuckDBConsole) -> None:
        query = 'WITH "valid_cte" AS (SELECT id, val FROM data) SELECT * FROM "valid_cte"'
        result = data_console.execute_query(query)
        assert result.row_count == 2

    def test_table_valued_functions_blocked(self, data_console: DuckDBConsole) -> None:
        with pytest.raises(ValidationError, match="cannot be used as a table source"):
            data_console.execute_query("SELECT * FROM range(10)")
        with pytest.raises(ValidationError, match="cannot be used as a table source"):
            data_console.execute_query("SELECT * FROM generate_series(1, 10)")

    def test_extract_function_with_from_keyword_allowed(self, data_console: DuckDBConsole) -> None:
        result = data_console.execute_query("SELECT EXTRACT(YEAR FROM CAST('2024-05-15' AS DATE)) AS yr FROM data")
        assert result.row_count == 2
        assert result.rows[0]["yr"] == 2024
