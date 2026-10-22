from __future__ import annotations

from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_fig(message: str, title: str = "Visualization Unavailable") -> go.Figure:
    """Return a clean fallback figure displaying an informational message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def histogram(frame: pd.DataFrame, column: str) -> go.Figure:
    if frame is None or frame.empty or column not in frame.columns:
        return _empty_fig(f"No records available for column '{column}'.", f"Distribution of {column}")

    s = frame[column].dropna()
    if s.empty:
        return _empty_fig(f"Column '{column}' contains only missing values.", f"Distribution of {column}")

    # Handle numeric and categorical safely
    fig = px.histogram(frame, x=column, title=f"Distribution of {column}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title=column, yaxis_title="Count")
    return fig


def bar_sum(frame: pd.DataFrame, x: str, y: str) -> go.Figure:
    if frame is None or frame.empty or x not in frame.columns or y not in frame.columns:
        return _empty_fig(f"Cannot aggregate '{y}' by '{x}': missing columns or empty data.", f"Sum of {y} by {x}")

    clean_df = frame[[x, y]].dropna().copy()
    if clean_df.empty:
        return _empty_fig(f"No non-null data pairs for '{x}' and '{y}'.", f"Sum of {y} by {x}")

    clean_df[y] = pd.to_numeric(clean_df[y], errors="coerce")
    clean_df = clean_df.dropna(subset=[y])
    if clean_df.empty:
        return _empty_fig(f"Column '{y}' contains no numeric values to sum.", f"Sum of {y} by {x}")

    grouped = clean_df.groupby(x, dropna=False)[y].sum().reset_index()
    fig = px.bar(grouped, x=x, y=y, title=f"Sum of {y} by {x}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def category_bar(frame: pd.DataFrame, column: str, top_n: int = 15) -> go.Figure:
    if frame is None or frame.empty or column not in frame.columns:
        return _empty_fig(f"No records available for column '{column}'.", f"Category Breakdown: {column}")

    s = frame[column].dropna()
    if s.empty:
        return _empty_fig(f"Column '{column}' contains only missing values.", f"Category Breakdown: {column}")

    counts = s.astype(str).str.strip().value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]
    total = len(s)
    counts["Percentage"] = (counts["Count"] / total * 100).round(1) if total > 0 else 0.0

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        text="Count",
        hover_data={"Percentage": ":.1f%"},
        title=f"Category Breakdown: {column} (Top {len(counts)})",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis_title=column,
        yaxis_title="Count",
        xaxis_tickangle=-30 if len(counts) > 4 else 0,
    )
    return fig


def crosstab_heatmap(
    ct_df: pd.DataFrame,
    x_title: str = "",
    y_title: str = "",
    title: str = "Cross-Tabulation Matrix",
) -> go.Figure:
    if ct_df is None or ct_df.empty:
        return _empty_fig("Cross-tabulation table is empty.", title)

    fig = px.imshow(
        ct_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x=x_title, y=y_title, color="Count"),
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
    return fig


def missingness(missing_by_column: dict[str, int]) -> go.Figure:
    if not missing_by_column:
        return _empty_fig("No column missingness data available.", "Missing Values by Column")

    data = pd.DataFrame(
        {"column": list(missing_by_column.keys()), "missing": list(missing_by_column.values())}
    )
    fig = px.bar(
        data,
        x="column",
        y="missing",
        text="missing",
        title="Missing Values by Column",
        labels={"column": "Column", "missing": "Missing Count"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis_tickangle=-35 if len(data) > 4 else 0,
        yaxis=dict(rangemode="tozero"),
    )
    return fig


def correlation_heatmap(
    corr_df: pd.DataFrame | dict[str, dict[str, float]],
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    df = pd.DataFrame(corr_df) if isinstance(corr_df, dict) else corr_df
    if df is None or df.empty:
        return _empty_fig("No numeric features available to compute correlations.", title)

    fig = px.imshow(
        df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title=title,
        labels=dict(color="Correlation"),
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
    return fig


def time_series_line(
    trend_df: pd.DataFrame,
    date_col: str = "period",
    val_col: str = "count",
    title: str = "Temporal Trend",
) -> go.Figure:
    if trend_df is None or trend_df.empty or date_col not in trend_df.columns or val_col not in trend_df.columns:
        return _empty_fig("No time-series records available to plot.", title)

    fig = px.line(
        trend_df,
        x=date_col,
        y=val_col,
        markers=True,
        title=title,
        labels={date_col: "Period", val_col: val_col.title()},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=30), yaxis=dict(rangemode="tozero"))
    return fig


def outlier_box(
    frame: pd.DataFrame,
    column: str,
    lower: float | None = None,
    upper: float | None = None,
    title: str | None = None,
) -> go.Figure:
    t = title or f"Outlier Analysis: {column}"
    if frame is None or frame.empty or column not in frame.columns:
        return _empty_fig(f"No records available for column '{column}'.", t)

    s = pd.to_numeric(frame[column], errors="coerce").dropna()
    if s.empty:
        return _empty_fig(f"Column '{column}' contains no numeric values for outlier analysis.", t)

    fig = px.box(frame, y=column, points="all", title=t)
    if lower is not None and isinstance(lower, (int, float)):
        fig.add_hline(y=float(lower), line_dash="dash", line_color="orange", annotation_text=f"Lower ({float(lower):.1f})")
    if upper is not None and isinstance(upper, (int, float)):
        fig.add_hline(y=float(upper), line_dash="dash", line_color="red", annotation_text=f"Upper ({float(upper):.1f})")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
    return fig
