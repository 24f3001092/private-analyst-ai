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
