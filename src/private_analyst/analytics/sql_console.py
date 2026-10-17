"""DuckDB in-memory SQL console for analytical queries with strict sandbox and validation."""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass
from typing import Any

import duckdb
import pandas as pd

from private_analyst.exceptions import ResourceLimitError, ValidationError

DEFAULT_MAX_ROWS: int = 10_000
DEFAULT_TIMEOUT_SECONDS: int = 30

DISALLOWED_MUTATION_KEYWORDS: frozenset[str] = frozenset({
    "ALTER",
    "ATTACH",
    "CALL",
    "CHECKPOINT",
    "COPY",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "EXECUTE",
    "EXPORT",
    "IMPORT",
    "INSERT",
    "INSTALL",
    "INTO",
    "LOAD",
    "MERGE",
    "PRAGMA",
    "PREPARE",
    "RESET",
    "REINDEX",
    "SET",
    "SHOW",
    "SUMMARIZE",
    "TRUNCATE",
    "UPDATE",
    "UPSERT",
    "USE",
    "VACUUM",
})

# Functions that perform file I/O, external network calls, or internal catalog probing
DISALLOWED_FUNCTIONS: frozenset[str] = frozenset({
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_blob",
    "read_text",
    "scan_csv",
    "scan_parquet",
    "scan_json",
    "glob",
    "httpfs",
    "parquet_scan",
    "sqlite_scan",
    "postgres_scan",
    "duckdb_settings",
    "duckdb_variables",
    "duckdb_secrets",
    "duckdb_memory",
    "duckdb_tables",
    "duckdb_views",
    "duckdb_indexes",
    "pragma_database_list",
    "pragma_table_info",
    "query_table",
    "current_setting",
})


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool
    sql: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_comments(sql: str) -> str:
    """Remove SQL comments (-- and /* */) while preserving string literals."""
    out: list[str] = []
    i = 0
    n = len(sql)
    in_single_quote = False
    in_double_quote = False

    while i < n:
        char = sql[i]
        next_char = sql[i + 1] if i + 1 < n else ""

        if in_single_quote:
            out.append(char)
            if char == "'":
                if next_char == "'":
                    out.append(next_char)
                    i += 2
                    continue
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            out.append(char)
            if char == '"':
                if next_char == '"':
                    out.append(next_char)
                    i += 2
                    continue
                in_double_quote = False
            i += 1
            continue

        if char == "-" and next_char == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            out.append(" ")
            continue

        if char == "/" and next_char == "*":
            i += 2
            while i < n:
                if sql[i] == "*" and (i + 1 < n and sql[i + 1] == "/"):
                    i += 2
                    break
                i += 1
            out.append(" ")
            continue

        if char == "'":
            in_single_quote = True
            out.append(char)
            i += 1
            continue

        if char == '"':
            in_double_quote = True
            out.append(char)
            i += 1
            continue

        out.append(char)
        i += 1

    return "".join(out).strip()


def _split_statements(sql: str) -> list[str]:
    """Split SQL into statements by semicolon outside quotes and comments."""
    statements: list[str] = []
    current: list[str] = []
    i = 0
    n = len(sql)
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        char = sql[i]
        next_char = sql[i + 1] if i + 1 < n else ""

        if in_line_comment:
            current.append(char)
            if char == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_single_quote:
            current.append(char)
            if char == "'":
                if next_char == "'":
                    current.append(next_char)
                    i += 2
                    continue
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            current.append(char)
            if char == '"':
                if next_char == '"':
                    current.append(next_char)
                    i += 2
                    continue
                in_double_quote = False
            i += 1
            continue

        if char == "-" and next_char == "-":
            in_line_comment = True
            current.append(char)
            current.append(next_char)
            i += 2
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            current.append(char)
            current.append(next_char)
            i += 2
            continue

        if char == "'":
            in_single_quote = True
            current.append(char)
            i += 1
            continue

        if char == '"':
            in_double_quote = True
            current.append(char)
            i += 1
            continue

        if char == ";":
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
            i += 1
            continue

        current.append(char)
        i += 1

    remaining = "".join(current).strip()
    if remaining:
        statements.append(remaining)

    return statements


def _extract_unquoted_tokens(stmt: str) -> list[str]:
    """Extract identifier and keyword tokens outside string literals and double quotes."""
    cleaned = _strip_comments(stmt)
    out: list[str] = []
    i = 0
    n = len(cleaned)
    in_single = False
    in_double = False

    while i < n:
        c = cleaned[i]
        next_c = cleaned[i + 1] if i + 1 < n else ""
        if in_single:
            if c == "'":
                if next_c == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if in_double:
            if c == '"':
                if next_c == '"':
                    i += 2
                    continue
                in_double = False
            i += 1
            continue
        if c == "'":
            in_single = True
            i += 1
            continue
        if c == '"':
            in_double = True
            i += 1
            continue
        out.append(c)
        i += 1

    text = "".join(out)
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)


def _extract_cte_names(cleaned_sql: str) -> set[str]:
    """Extract CTE names defined in WITH [RECURSIVE] cte_name[(cols)] AS (...) clauses."""
    ctes: set[str] = set()
    pattern = r"\b(?:WITH\s+(?:RECURSIVE\s+)?|,)\s*([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\")(?:\s*\([^)]*\))?\s+AS\s*\("
    matches = re.finditer(pattern, cleaned_sql, re.IGNORECASE)
    for m in matches:
        ctes.add(m.group(1).strip('"').lower())
    return ctes


@dataclass
class _SQLToken:
    text: str
    kind: str  # 'word', 'string', 'double_quote', 'symbol'


CLAUSE_TERMINATORS: frozenset[str] = frozenset({
    "WHERE",
    "GROUP",
    "HAVING",
    "WINDOW",
    "QUALIFY",
    "ORDER",
    "LIMIT",
    "OFFSET",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "FETCH",
})

JOIN_KEYWORDS: frozenset[str] = frozenset({
    "JOIN",
    "INNER",
    "LEFT",
    "RIGHT",
    "FULL",
    "CROSS",
    "NATURAL",
    "OUTER",
})


def _tokenize_sql(sql: str) -> list[_SQLToken]:
    """Tokenize SQL into word, quoted string, double-quoted identifier, and symbol tokens."""
    tokens: list[_SQLToken] = []
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == "'":
            start = i
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            tokens.append(_SQLToken(sql[start:i], "string"))
            continue
        if c == '"':
            start = i
            i += 1
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            tokens.append(_SQLToken(sql[start:i], "double_quote"))
            continue
        if c in "(),;.":
            tokens.append(_SQLToken(c, "symbol"))
            i += 1
            continue
        if c.isalnum() or c in "_":
            start = i
            while i < n and (sql[i].isalnum() or sql[i] in "_"):
                i += 1
            tokens.append(_SQLToken(sql[start:i], "word"))
            continue
        tokens.append(_SQLToken(c, "symbol"))
        i += 1
    return tokens


def _find_matching_paren(tokens: list[_SQLToken], start: int) -> int:
    """Given tokens[start] is '(', return index of matching ')'."""
    depth = 1
    i = start + 1
    n = len(tokens)
    while i < n and depth > 0:
        if tokens[i].kind == "symbol" and tokens[i].text == "(":
            depth += 1
        elif tokens[i].kind == "symbol" and tokens[i].text == ")":
            depth -= 1
        i += 1
    return i - 1


def _extract_table_item(
    tokens: list[_SQLToken], i: int, n: int
) -> tuple[str | None, int, list[_SQLToken] | None]:
    """Parse a single table item starting at index i.

    Returns (source_name, next_index, subquery_tokens_or_None).
    """
    if i >= n:
        return None, i, None

    tok = tokens[i]

    # Subquery in parentheses: (SELECT ...)
    if tok.kind == "symbol" and tok.text == "(":
        depth = 1
        sub_start = i + 1
        i += 1
        while i < n and depth > 0:
            if tokens[i].kind == "symbol" and tokens[i].text == "(":
                depth += 1
            elif tokens[i].kind == "symbol" and tokens[i].text == ")":
                depth -= 1
            i += 1
        sub_tokens = tokens[sub_start : i - 1]
        # Skip alias if present
        if i < n and tokens[i].kind == "word" and tokens[i].text.upper() == "AS":
            i += 2
        elif (
            i < n
            and tokens[i].kind in ("word", "double_quote")
            and tokens[i].text.upper() not in CLAUSE_TERMINATORS
            and tokens[i].text.upper() not in JOIN_KEYWORDS
            and tokens[i].text not in (",", ")", ";", "ON", "USING")
        ):
            i += 1
        return None, i, sub_tokens

    # String literal table source e.g. FROM 'secret.csv'
    if tok.kind == "string":
        return tok.text, i + 1, None

    # Identifier, multipart identifier, or function call
    if tok.kind in ("word", "double_quote"):
        parts = [tok.text]
        i += 1
        while i < n and tokens[i].kind == "symbol" and tokens[i].text == ".":
            i += 1
            if i < n and tokens[i].kind in ("word", "double_quote"):
                parts.append(tokens[i].text)
                i += 1
            else:
                parts.append("")
                break

        name = ".".join(parts)

        # Check if function call e.g. duckdb_functions() or schema.func()
        if i < n and tokens[i].kind == "symbol" and tokens[i].text == "(":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if tokens[i].kind == "symbol" and tokens[i].text == "(":
                    depth += 1
                elif tokens[i].kind == "symbol" and tokens[i].text == ")":
                    depth -= 1
                i += 1
            # Skip alias if present
            if i < n and tokens[i].kind == "word" and tokens[i].text.upper() == "AS":
                i += 2
            elif (
                i < n
                and tokens[i].kind in ("word", "double_quote")
                and tokens[i].text.upper() not in CLAUSE_TERMINATORS
                and tokens[i].text.upper() not in JOIN_KEYWORDS
                and tokens[i].text not in (",", ")", ";", "ON", "USING")
            ):
                i += 1
            return f"{name}()", i, None

        # Plain or schema-qualified identifier
        # Skip alias if present
        if i < n and tokens[i].kind == "word" and tokens[i].text.upper() == "AS":
            i += 2
        elif (
            i < n
            and tokens[i].kind in ("word", "double_quote")
            and tokens[i].text.upper() not in CLAUSE_TERMINATORS
            and tokens[i].text.upper() not in JOIN_KEYWORDS
            and tokens[i].text not in (",", ")", ";", "ON", "USING")
        ):
            i += 1
        return name, i, None

    return None, i + 1, None


def _extract_from_tokens(tokens: list[_SQLToken]) -> list[str]:
    """Recursively extract all table sources from a token stream."""
    sources: list[str] = []
    n = len(tokens)
    i = 0

    while i < n:
        tok = tokens[i]

        # Parenthesized expression or subquery
        if tok.kind == "symbol" and tok.text == "(":
            close_idx = _find_matching_paren(tokens, i)
            sub_tokens = tokens[i + 1 : close_idx]
            has_query = any(t.kind == "word" and t.text.upper() in ("SELECT", "WITH") for t in sub_tokens)
            if has_query:
                sources.extend(_extract_from_tokens(sub_tokens))
            i = close_idx + 1
            continue

        upper = tok.text.upper() if tok.kind == "word" else ""

        if upper == "FROM":
            i += 1
            # Read first table item after FROM
            item, i, sub = _extract_table_item(tokens, i, n)
            if item:
                sources.append(item)
            if sub:
                sources.extend(_extract_from_tokens(sub))

            # Continue reading comma-separated or join-separated tables in this FROM clause
            while i < n:
                t = tokens[i]
                if t.kind == "symbol" and t.text in (";", ")"):
                    break
                if t.kind == "word" and t.text.upper() in CLAUSE_TERMINATORS:
                    break

                # Comma separates tables
                if t.kind == "symbol" and t.text == ",":
                    i += 1
                    item, i, sub = _extract_table_item(tokens, i, n)
                    if item:
                        sources.append(item)
                    if sub:
                        sources.extend(_extract_from_tokens(sub))
                    continue

                # JOIN keyword
                if t.kind == "word" and t.text.upper() in JOIN_KEYWORDS:
                    while i < n and tokens[i].kind == "word" and tokens[i].text.upper() in JOIN_KEYWORDS:
                        i += 1
                    item, i, sub = _extract_table_item(tokens, i, n)
                    if item:
                        sources.append(item)
                    if sub:
                        sources.extend(_extract_from_tokens(sub))
                    continue

                # ON or USING condition in JOIN
                if t.kind == "word" and t.text.upper() in ("ON", "USING"):
                    i += 1
                    while i < n:
                        if tokens[i].kind == "symbol" and tokens[i].text == "(":
                            close_idx = _find_matching_paren(tokens, i)
                            sub_tokens = tokens[i + 1 : close_idx]
                            if any(x.kind == "word" and x.text.upper() in ("SELECT", "WITH") for x in sub_tokens):
                                sources.extend(_extract_from_tokens(sub_tokens))
                            i = close_idx + 1
                            continue
                        if tokens[i].kind == "symbol" and tokens[i].text in (",", ";", ")"):
                            break
                        if tokens[i].kind == "word" and (
                            tokens[i].text.upper() in JOIN_KEYWORDS or tokens[i].text.upper() in CLAUSE_TERMINATORS
                        ):
                            break
                        i += 1
                    continue

                i += 1
            continue

        i += 1

    return sources


def _extract_table_sources(cleaned_sql: str) -> list[str]:
    """Tokenize SQL and extract all referenced table sources."""
    tokens = _tokenize_sql(cleaned_sql)
    return _extract_from_tokens(tokens)


def _validate_table_references(cleaned_sql: str, allowed_tables: set[str]) -> None:
    """Ensure that all table sources refer exclusively to allowed tables/CTEs."""
    sources = _extract_table_sources(cleaned_sql)
    allowed_lower = {t.lower() for t in allowed_tables}

    if not sources:
        return

    for target in sources:
        # Check for direct file paths or string literals
        if target.startswith("'") or ("/" in target or "\\" in target):
            raise ValidationError("Direct file paths or string literals cannot be used as table sources.")

        # Check for function call in table position e.g. duckdb_functions()
        if target.endswith("()"):
            func_name = target[:-2].strip("\"'")
            raise ValidationError(f"Function call '{func_name}' cannot be used as a table source.")

        # Schema-qualified name
        if "." in target:
            parts = target.split(".")
            if len(parts) == 2:
                schema = parts[0].strip("\"'").lower()
                tbl = parts[1].strip("\"'").lower()
                if schema == "main" and tbl in allowed_lower:
                    continue
            raise ValidationError(f"Access to schema-qualified source '{target}' is not permitted.")

        # Plain identifier
        clean_target = target.strip("\"'").lower()
        if clean_target not in allowed_lower:
            display_allowed = ", ".join(sorted(allowed_tables)) if allowed_tables else "none"
            raise ValidationError(
                f"Access to table '{clean_target}' is not allowed. Only registered tables ({display_allowed}) can be queried."
            )


def _validate_select_statement(sql: str, allowed_tables: set[str] | None = None) -> str:
    """Validate that the query contains exactly one read-only SELECT statement against approved tables."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValidationError("SQL query cannot be empty.")

    # Reject null bytes
    if "\x00" in sql:
        raise ValidationError("Null bytes are not allowed in SQL query.")

    statements = _split_statements(sql)
    if not statements:
        raise ValidationError("SQL query cannot be empty.")

    if len(statements) > 1:
        raise ValidationError("Multiple SQL statements are not permitted.")

    stmt = statements[0]
    cleaned = _strip_comments(stmt)
    if not cleaned:
        raise ValidationError("SQL query cannot be empty.")

    tokens = _extract_unquoted_tokens(cleaned)
    if not tokens:
        raise ValidationError("SQL query contains no valid statements.")

    first_keyword = tokens[0].upper()
    if first_keyword not in ("SELECT", "WITH"):
        raise ValidationError(
            f"Only SELECT statements are allowed (found '{first_keyword}')."
        )

    if first_keyword == "WITH":
        has_select = any(t.upper() == "SELECT" for t in tokens)
        if not has_select:
            raise ValidationError("Common Table Expressions (WITH) must conclude with a SELECT query.")

    for token in tokens:
        upper = token.upper()
        if upper in DISALLOWED_MUTATION_KEYWORDS:
            raise ValidationError(
                f"Disallowed operation or keyword in query: '{upper}'."
            )
        lower = token.lower()
        if lower in DISALLOWED_FUNCTIONS:
            raise ValidationError(
                f"Disallowed function in query: '{token}'."
            )

    # Check CTE names and allowed tables
    cte_names = _extract_cte_names(cleaned)
    effective_allowed = set(allowed_tables or set()) | cte_names
    _validate_table_references(cleaned, effective_allowed)

    return stmt


class DuckDBConsole:
    """In-memory DuckDB analytical console for safe, read-only SQL queries.
    
    Hardened with:
    - enable_external_access=False (disables file reading, writing, and networking)
    - lock_configuration=True (prevents altering configuration)
    - max_memory='512MB' (caps memory usage)
    - Strict table-source allowlist checking against registered datasets
    - Read-only AST token validator
    """

    def __init__(self, max_rows: int = DEFAULT_MAX_ROWS) -> None:
        self._max_rows = max_rows
        self._conn = duckdb.connect(
            database=":memory:",
            config={
                "enable_external_access": False,
                "lock_configuration": True,
                "max_memory": "512MB",
            },
        )
        self._registered_tables: set[str] = set()

    @property
    def max_rows(self) -> int:
        return self._max_rows

    def register_dataframe(self, name: str, frame: pd.DataFrame) -> None:
        """Register a pandas DataFrame as a virtual table in DuckDB."""
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Table name must be a non-empty string.")
        clean_name = name.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", clean_name):
            raise ValidationError(
                f"Invalid table name: '{name}'. Table names must be valid identifiers."
            )
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError("Registered object must be a pandas DataFrame.")

        try:
            self._conn.register(clean_name, frame)
            self._registered_tables.add(clean_name)
        except Exception as exc:
            raise ValidationError("Failed to register DataFrame as a virtual table.") from exc

    def execute_query(self, sql: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> QueryResult:
        """Execute a validated SELECT query and return structured results."""
        if timeout_seconds <= 0:
            raise ValidationError("timeout_seconds must be a positive integer.")

        validated_sql = _validate_select_statement(sql, allowed_tables=self._registered_tables)

        timer = threading.Timer(float(timeout_seconds), self._conn.interrupt)
        timer.daemon = True
        timer.start()

        try:
            res = self._conn.execute(validated_sql)

            description = res.description
            columns = [desc[0] for desc in description] if description else []

            raw_rows = res.fetchmany(self._max_rows + 1)
            truncated = False
            if len(raw_rows) > self._max_rows:
                truncated = True
                raw_rows = raw_rows[: self._max_rows]

            rows: list[dict] = [dict(zip(columns, row)) for row in raw_rows]
            row_count = len(rows)

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=row_count,
                truncated=truncated,
                sql=validated_sql,
            )
        except duckdb.PermissionException as exc:
            raise ValidationError("File system and external network access are disabled.") from exc
        except duckdb.InvalidInputException as exc:
            raise ValidationError("Query operation is not permitted.") from exc
        except duckdb.InterruptException as exc:
            raise ResourceLimitError(f"Query timed out after {timeout_seconds} seconds.") from exc
        except duckdb.OutOfMemoryException as exc:
            raise ResourceLimitError("Query exceeded memory limits.") from exc
        except duckdb.ParserException as exc:
            raise ValidationError("SQL syntax error.") from exc
        except duckdb.CatalogException as exc:
            raise ValidationError("Referenced table or column not found.") from exc
        except duckdb.BinderException as exc:
            raise ValidationError("Query binding error: check table and column names.") from exc
        except duckdb.Error as exc:
            msg = str(exc).lower()
            if "interrupt" in msg or "timeout" in msg or "cancel" in msg:
                raise ResourceLimitError(f"Query timed out after {timeout_seconds} seconds.") from exc
            if "permission" in msg or "disabled" in msg:
                raise ValidationError("File system and external network access are disabled.") from exc
            raise ValidationError("SQL query execution failed.") from exc
        except ValidationError:
            raise
        except ResourceLimitError:
            raise
        except Exception as exc:
            raise ValidationError("An unexpected error occurred during query execution.") from exc
        finally:
            timer.cancel()

    def list_tables(self) -> list[str]:
        """Return a sorted list of registered table names."""
        try:
            return sorted(list(self._registered_tables))
        except Exception as exc:
            raise ValidationError("Failed to list database tables.") from exc

    def describe_table(self, name: str) -> list[dict]:
        """Describe table schema, returning column names and types."""
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Table name must be a non-empty string.")
        clean_name = name.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", clean_name):
            raise ValidationError(f"Invalid table name: '{name}'.")

        if clean_name not in self._registered_tables:
            raise ValidationError(f"Table '{clean_name}' does not exist.")

        try:
            res = self._conn.execute(f"DESCRIBE {clean_name}")
            description = res.description or []
            col_names = [d[0].lower() for d in description]
            name_idx = col_names.index("column_name") if "column_name" in col_names else 0
            type_idx = col_names.index("column_type") if "column_type" in col_names else 1

            rows = res.fetchall()
            col_info: list[dict] = []
            for row in rows:
                col_name = str(row[name_idx])
                col_type = str(row[type_idx])
                col_info.append({
                    "name": col_name,
                    "type": col_type,
                    "column_name": col_name,
                    "column_type": col_type,
                })
            return col_info
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"Failed to describe table '{clean_name}'.") from exc

    def close(self) -> None:
        """Close the DuckDB connection."""
        if hasattr(self, "_conn") and self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> DuckDBConsole:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
