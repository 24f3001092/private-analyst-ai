from __future__ import annotations

import pandas as pd
import plotly.express as px


def histogram(frame: pd.DataFrame, column: str):
    fig = px.histogram(frame, x=column, title=f"Distribution of {column}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title=column, yaxis_title="Count")
    return fig


def bar_sum(frame: pd.DataFrame, x: str, y: str):
    grouped = frame.groupby(x, dropna=False)[y].sum().reset_index()
    fig = px.bar(grouped, x=x, y=y, title=f"Sum of {y} by {x}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def category_bar(frame: pd.DataFrame, column: str, top_n: int = 15):
    counts = frame[column].dropna().astype(str).str.strip().value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]
    total = len(frame[column].dropna())
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
):
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


def missingness(missing_by_column: dict[str, int]):
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


def correlation_heatmap(corr_df: pd.DataFrame | dict[str, dict[str, float]], title: str = "Feature Correlation Matrix"):
    df = pd.DataFrame(corr_df) if isinstance(corr_df, dict) else corr_df
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


def time_series_line(trend_df: pd.DataFrame, date_col: str = "period", val_col: str = "count", title: str = "Temporal Trend"):
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
):
    t = title or f"Outlier Analysis: {column}"
    fig = px.box(frame, y=column, points="all", title=t)
    if lower is not None and isinstance(lower, (int, float)):
        fig.add_hline(y=float(lower), line_dash="dash", line_color="orange", annotation_text=f"Lower ({float(lower):.1f})")
    if upper is not None and isinstance(upper, (int, float)):
        fig.add_hline(y=float(upper), line_dash="dash", line_color="red", annotation_text=f"Upper ({float(upper):.1f})")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
    return fig
