from __future__ import annotations

import numpy as np
import pandas as pd

from private_analyst.analytics.statistics import SAMPLE_STD_DDOF, compute_dataset_stats
from private_analyst.config import Limits
from private_analyst.ingestion.loader import load_tabular


def test_known_csv_statistics_match_numpy(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    stats = compute_dataset_stats(ingest.frame)
    units = ingest.frame["units"].to_numpy(dtype=float)
    revenue = ingest.frame["revenue"].to_numpy(dtype=float)

    units_stats = next(c for c in stats.columns if c.name == "units")
    revenue_stats = next(c for c in stats.columns if c.name == "revenue")

    assert stats.row_count == 5
    assert units_stats.sum == 50
    assert units_stats.mean == float(np.mean(units))
    assert units_stats.median == float(np.median(units))
    assert units_stats.min == 0
    assert units_stats.max == 20
    assert units_stats.std == float(np.std(units, ddof=SAMPLE_STD_DDOF))

    assert revenue_stats.sum == 600
    assert revenue_stats.mean == float(np.mean(revenue))
    assert abs(revenue_stats.std - float(np.std(revenue, ddof=SAMPLE_STD_DDOF))) < 1e-9


def test_excel_matches_csv(sales_csv_bytes, sales_xlsx_bytes, limits: Limits) -> None:
    csv_stats = compute_dataset_stats(load_tabular(sales_csv_bytes, "a.csv", limits).frame)
    xlsx_stats = compute_dataset_stats(load_tabular(sales_xlsx_bytes, "a.xlsx", limits).frame)
    csv_units = next(c for c in csv_stats.columns if c.name == "units")
    xlsx_units = next(c for c in xlsx_stats.columns if c.name == "units")
    assert csv_units.sum == xlsx_units.sum
    assert csv_units.mean == xlsx_units.mean
    assert csv_units.std == xlsx_units.std


def test_missing_values_excluded_from_mean() -> None:
    frame = pd.DataFrame({"x": [1.0, None, 3.0]})
    stats = compute_dataset_stats(frame)
    col = stats.columns[0]
    assert col.missing == 1
    assert col.mean == 2.0
    assert col.count == 2


def test_categorical_column_statistics_and_frequency() -> None:
    from private_analyst.analytics.statistics import frequency_analysis

    df = pd.DataFrame({
        "mode": ["Online", "Online", "Online", "In-Person", "In-Person"],
        "domain": ["AI", "AI", "Open Source", "Robotics", "AI"],
    })
    stats = compute_dataset_stats(df)
    mode_col = next(c for c in stats.columns if c.name == "mode")
    assert mode_col.kind == "categorical"
    assert mode_col.unique == 2
    assert mode_col.top_value == "Online"
    assert mode_col.top_freq == 3

    freq = frequency_analysis(df, "mode")
    assert list(freq["Category"]) == ["Online", "In-Person"]
    assert list(freq["Count"]) == [3, 2]
    assert list(freq["Percentage (%)"]) == [60.0, 40.0]
    assert list(freq["Cumulative (%)"]) == [60.0, 100.0]


def test_quality_scoring_comprehensive() -> None:
    from private_analyst.analytics.quality import assess_quality

    df_perfect = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    q_perf = assess_quality(df_perfect)
    assert q_perf.quality_score == 100.0
    assert q_perf.grade == "Excellent"
    assert q_perf.score_breakdown["completeness"] == 100.0
    assert q_perf.total_missing_cells == 0
    assert q_perf.completeness_ratio == 1.0

    df_flawed = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", None]})
    q_flawed = assess_quality(df_flawed)
    assert q_flawed.quality_score < 100.0
    assert q_flawed.duplicate_row_count == 1
    assert q_flawed.grade in {"Good", "Fair", "Needs Attention"}
