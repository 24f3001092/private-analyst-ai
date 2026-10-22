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


DANGEROUS_FORMULA_PREFIXES = ("=", "@")
DANGEROUS_CONTROL_PREFIXES = ("\t", "\r")


def sanitize_csv_value(val: object) -> object:
    """Sanitize string values to neutralize CSV/DDE formula injection attacks.

    Prepends a single quote (') to cells starting with '=', '@', tab, carriage return,
    or '+' / '-' when not part of a valid numeric literal.
    """
    if not isinstance(val, str):
        return val
    if not val:
        return val
    space_trimmed = val.lstrip(" ")
    if space_trimmed and space_trimmed[0] in DANGEROUS_CONTROL_PREFIXES:
        return f"'{val}"
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
    for i in range(clean.shape[1]):
        series = clean.iloc[:, i]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            clean.isetitem(i, series.map(lambda v: sanitize_csv_value(v) if isinstance(v, str) else v))
    return clean


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    clean_frame = sanitize_dataframe_for_csv(frame)
    return clean_frame.to_csv(index=False).encode("utf-8")


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


def to_report_excel_bytes(
    frame: pd.DataFrame | Any,
    stats_dict: dict | Any | None = None,
    quality_dict: dict | Any | None = None,
    cleaning_ops: list[Any] | None = None,
) -> bytes:
    """Generate a multi-tab Excel workbook with sanitized data, column stats, and data quality metrics."""
    actual_frame = frame.frame if hasattr(frame, "frame") else frame
    buffer = io.BytesIO()
    clean_frame = sanitize_dataframe_for_csv(actual_frame)
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        clean_frame.to_excel(writer, sheet_name="Cleaned Data", index=False)

        # Handle stats_dict whether it is a dict or a DatasetStats dataclass
        columns_data = None
        if stats_dict is not None:
            if isinstance(stats_dict, dict) and "columns" in stats_dict:
                columns_data = stats_dict["columns"]
            elif hasattr(stats_dict, "columns"):
                columns_data = [
                    {
                        "name": c.name,
                        "kind": c.kind,
                        "count": c.count,
                        "missing": c.missing,
                        "unique": c.unique,
                        "mean": c.mean,
                        "median": c.median,
                        "std": c.std,
                        "min": c.min,
                        "max": c.max,
                        "top_value": c.top_value,
                        "top_freq": c.top_freq,
                    }
                    for c in stats_dict.columns
                ]

        if columns_data:
            stats_rows = []
            for c in columns_data:
                stats_rows.append({
                    "Column": c.get("name"),
                    "Type": c.get("kind"),
                    "Count": c.get("count"),
                    "Missing": c.get("missing"),
                    "Unique": c.get("unique"),
                    "Mean": c.get("mean"),
                    "Median": c.get("median"),
                    "Std": c.get("std"),
                    "Min": c.get("min"),
                    "Max": c.get("max"),
                    "Top Value": c.get("top_value"),
                    "Top Freq": c.get("top_freq"),
                })
            pd.DataFrame(stats_rows).to_excel(writer, sheet_name="Column Statistics", index=False)

        # Handle quality_dict whether it is a dict or a QualityAssessment dataclass
        if quality_dict is not None:
            if hasattr(quality_dict, "quality_score"):
                q_score = quality_dict.quality_score
                q_grade = getattr(quality_dict, "grade", "")
                r_count = getattr(quality_dict, "row_count", 0)
                c_count = getattr(quality_dict, "column_count", 0)
                m_cells = getattr(quality_dict, "total_missing_cells", 0)
                d_rows = getattr(quality_dict, "duplicate_row_count", 0)
            else:
                q_score = quality_dict.get("quality_score")
                q_grade = quality_dict.get("grade", quality_dict.get("quality_grade"))
                r_count = quality_dict.get("row_count")
                c_count = quality_dict.get("column_count")
                m_cells = quality_dict.get("total_missing_cells")
                d_rows = quality_dict.get("duplicate_row_count", quality_dict.get("duplicate_rows"))

            q_rows = [
                {"Metric": "Quality Score", "Value": q_score},
                {"Metric": "Quality Grade", "Value": q_grade},
                {"Metric": "Total Rows", "Value": r_count},
                {"Metric": "Total Columns", "Value": c_count},
                {"Metric": "Total Missing Cells", "Value": m_cells},
                {"Metric": "Duplicate Rows", "Value": d_rows},
            ]
            pd.DataFrame(q_rows).to_excel(writer, sheet_name="Data Quality", index=False)

        # Handle cleaning ledger if available on the ingest object or passed explicitly
        ops_to_use = cleaning_ops or getattr(frame, "cleaning_log", None) or getattr(frame, "cleaning_operations", None)
        if ops_to_use:
            from private_analyst.ingestion.loader import normalize_cleaning_operation
            c_rows = [normalize_cleaning_operation(op)[0] for op in ops_to_use]
            if c_rows:
                pd.DataFrame(c_rows).to_excel(writer, sheet_name="Cleaning Ledger", index=False)
    return buffer.getvalue()


def report_markdown(
    result: OperationResult,
    explanation: str,
    include_explanation: bool,
    source_name: str | None = None,
    quality_score: float | None = None,
) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# {PRODUCT_NAME} report",
        "",
        f"**Generated:** `{now_str}` | **Product Version:** `v{__version__}`",
        f"**Intent:** `{result.intent}` | **Status:** `{result.status}`",
    ]
    if source_name:
        lines.append(f"**Dataset Source:** `{source_name}`")
    if quality_score is not None:
        lines.append(f"**Data Quality Score:** `{quality_score:.1f} / 100`")

    lines.extend([
        "",
        "---",
        "",
        "## Executive Overview",
        f"This report presents deterministic local analytical findings for intent **{result.intent}**.",
        "Calculations are performed strictly in Python on your local device without external telemetry.",
        "",
        "## Observed",
    ])

    if result.observed:
        for k, v in result.observed.items():
            lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    else:
        lines.append("- *(No specific attributes recorded)*")

    lines.extend([
        "",
        "## Calculated",
    ])

    calc = result.calculated
    if calc:
        if "quality_score" in calc:
            lines.append(f"- **Overall Quality Score:** {calc.get('quality_score')} / 100 ({calc.get('grade')})")
            lines.append(f"- **Total Missing Cells:** {calc.get('total_missing_cells', 0):,}")
            lines.append(f"- **Duplicate Rows:** {calc.get('duplicate_rows', 0)}")
        elif "columns" in calc and isinstance(calc["columns"], list):
            lines.append(f"- **Total Rows:** {calc.get('row_count')}")
            lines.append(f"- **Total Columns:** {calc.get('column_count')}")
            lines.append("")
            lines.append("| Column | Type | Missing | Unique | Summary Stats |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for c in calc["columns"]:
                name = c.get("name")
                kind = c.get("kind")
                miss = c.get("missing", 0)
                uniq = c.get("unique", 0)
                if kind == "numeric":
                    summary = f"mean={c.get('mean')}, median={c.get('median')}, sum={c.get('sum')}"
                elif kind == "currency" and c.get("parsed_sum") is not None:
                    summary = f"total={c.get('currency_symbol') or ''}{c.get('parsed_sum'):,}"
                else:
                    top = c.get("top_value")
                    summary = f"top='{top}' ({c.get('top_freq')}x)" if top else "N/A"
                lines.append(f"| `{name}` | {kind} | {miss} | {uniq} | {summary} |")
        elif "frequency_table" in calc:
            lines.append("| Category | Count | Percentage (%) | Cumulative (%) |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for r in calc.get("frequency_table", [])[:15]:
                cat = r.get("Category", r.get("category"))
                cnt = r.get("Count", r.get("count"))
                pct = r.get("Percentage (%)", r.get("percentage"))
                cum = r.get("Cumulative (%)", r.get("cumulative"))
                lines.append(f"| {cat} | {cnt} | {pct}% | {cum}% |")
        elif "top_correlations" in calc:
            lines.append("| Feature 1 | Feature 2 | Pearson Correlation | Relationship |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for p in calc.get("top_correlations", []):
                lines.append(f"| `{p.get('col1')}` | `{p.get('col2')}` | `{p.get('correlation'):+.3f}` | {p.get('relationship')} |")
        elif "trend_table" in calc:
            lines.append(f"| Period ({calc.get('freq_label', 'Time')}) | Event Count | Total Value |")
            lines.append("| :--- | :--- | :--- |")
            for r in calc.get("trend_table", [])[:20]:
                lines.append(f"| {r.get('period')} | {r.get('count')} | {r.get('total')} |")
        elif "outlier_count" in calc:
            lines.append(f"- **Outliers Detected:** {calc.get('outlier_count')}")
            lines.append(f"- **Detection Method:** {calc.get('method')}")
            lines.append(f"- **Acceptable Threshold Range:** [{calc.get('lower')}, {calc.get('upper')}]")
        else:
            for k, v in calc.items():
                if isinstance(v, (int, float, str, bool)):
                    lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    else:
        lines.append("- *(No calculation metrics available)*")

    lines.extend([
        "",
        "## Hypotheses",
        *([f"- {h}" for h in result.hypotheses] if result.hypotheses else ["- *(None)*"]),
        "",
        "## Recommendations",
        *([f"- {r}" for r in result.recommendations] if result.recommendations else ["- *(None)*"]),
        "",
        "## Limitations",
        *([f"- {l}" for l in result.limitations] if result.limitations else ["- *(None)*"]),
    ])

    if include_explanation:
        lines.extend(["", "## Explanation", explanation])
    else:
        lines.extend(["", "## Explanation", "(User chose not to export the AI explanation.)"])

    return "\n".join(lines)
