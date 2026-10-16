from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from private_analyst.analytics.statistics import _finite_numeric, _is_numeric


@dataclass
class AnomalyResult:
    column: str
    method: str
    lower: float
    upper: float
    outlier_count: int
    outlier_row_indexes: list[int]
    caveat: str


def detect_iqr_anomalies(frame: pd.DataFrame, column: str, multiplier: float = 1.5) -> AnomalyResult:
    if column not in frame.columns:
        raise KeyError("Column is not in the loaded table.")
    series = frame[column]
    if not _is_numeric(series):
        raise ValueError("IQR anomaly detection requires a numeric column.")
    finite = _finite_numeric(series)
    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (finite < lower) | (finite > upper)
    indexes = [int(i) for i in finite.index[mask].tolist()]
    return AnomalyResult(
        column=column,
        method="iqr",
        lower=lower,
        upper=upper,
        outlier_count=len(indexes),
        outlier_row_indexes=indexes[:200],
        caveat=(
            "IQR outliers are a heuristic, not confirmed errors. "
            "They are not a factual finding of fraud or failure."
        ),
    )


def linear_trend_forecast(values: list[float], periods: int = 3) -> dict:
    """Unvalidated linear extrapolation. Labelled as a hypothesis, not a fact."""
    import numpy as np

    if periods < 1 or periods > 24:
        raise ValueError("Forecast horizon is outside the allowed range.")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return {
            "status": "insufficient_data",
            "forecast": [],
            "caveat": "A linear forecast needs at least three finite observations.",
        }
    x = np.arange(arr.size, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    future_x = np.arange(arr.size, arr.size + periods, dtype=float)
    forecast = [float(intercept + slope * xi) for xi in future_x]
    return {
        "status": "hypothesis",
        "method": "ordinary_least_squares_linear",
        "slope": float(slope),
        "intercept": float(intercept),
        "forecast": forecast,
        "caveat": "This is an unvalidated linear extrapolation, not a measured future outcome.",
    }
