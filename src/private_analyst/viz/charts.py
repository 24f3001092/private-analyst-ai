from __future__ import annotations

import pandas as pd
import plotly.express as px


def histogram(frame: pd.DataFrame, column: str):
    return px.histogram(frame, x=column, title=f"Distribution of {column}")


def bar_sum(frame: pd.DataFrame, x: str, y: str):
    grouped = frame.groupby(x, dropna=False)[y].sum().reset_index()
    return px.bar(grouped, x=x, y=y, title=f"Sum of {y} by {x}")


def missingness(missing_by_column: dict[str, int]):
    data = pd.DataFrame(
        {"column": list(missing_by_column.keys()), "missing": list(missing_by_column.values())}
    )
    return px.bar(data, x="column", y="missing", title="Missing values by column")
