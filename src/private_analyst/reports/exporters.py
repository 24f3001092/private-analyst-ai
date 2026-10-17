from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd

from private_analyst.analytics.operations import OperationResult
from private_analyst.version import PRODUCT_NAME, __version__


def analysis_metadata(source_name: str) -> dict:
    return {
        "product": PRODUCT_NAME,
        "version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "engine": "local_python_analytics",
        "model_does_not_compute_totals": True,
    }


DANGEROUS_FORMULA_PREFIXES = ("=", "@", "\t", "\r")


def sanitize_csv_value(val: object) -> object:
    """Sanitize string values to neutralize CSV/DDE formula injection attacks.

    Prepends a single quote (') to cells starting with '=', '@', tab, carriage return,
    or '+' / '-' when not part of a valid numeric literal.
    """
    if not isinstance(val, str):
        return val
    stripped = val.lstrip()
    if not stripped:
        return val
    lead = stripped[0]
    if lead in DANGEROUS_FORMULA_PREFIXES:
        return f"'{val}"
    if lead in ("+", "-"):
        try:
            float(stripped)
            return val
        except ValueError:
            return f"'{val}"
    return val


def sanitize_dataframe_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sanitized copy of the DataFrame safe for CSV export."""
    clean = frame.copy()
    clean.columns = [sanitize_csv_value(col) if isinstance(col, str) else col for col in clean.columns]
    for col in clean.columns:
        series = clean[col]
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            clean[col] = series.map(lambda v: sanitize_csv_value(v) if isinstance(v, str) else v)
    return clean


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    clean_frame = sanitize_dataframe_for_csv(frame)
    return clean_frame.to_csv(index=False).encode("utf-8")


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


def report_markdown(result: OperationResult, explanation: str, include_explanation: bool) -> str:
    lines = [
        f"# {PRODUCT_NAME} report",
        "",
        f"Intent: {result.intent}",
        f"Status: {result.status}",
        "",
        "## Observed",
        str(result.observed),
        "",
        "## Calculated",
        str(result.calculated)[:8000],
        "",
        "## Hypotheses",
        *result.hypotheses,
        "",
        "## Recommendations",
        *result.recommendations,
        "",
        "## Limitations",
        *result.limitations,
    ]
    if include_explanation:
        lines.extend(["", "## Explanation", explanation])
    else:
        lines.extend(["", "## Explanation", "(User chose not to export the AI explanation.)"])
    return "\n".join(lines)
