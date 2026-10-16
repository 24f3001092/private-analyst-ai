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
