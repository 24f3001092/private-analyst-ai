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
