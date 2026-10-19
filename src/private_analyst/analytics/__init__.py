from private_analyst.analytics.statistics import ColumnStats, DatasetStats, compute_dataset_stats
from private_analyst.analytics.quality import DataQualityReport, assess_quality
from private_analyst.analytics.anomalies import AnomalyResult, detect_iqr_anomalies
from private_analyst.analytics.ml_engine import (
    MLResult,
    evaluate_model_performance,
    run_classification,
    run_clustering,
    run_regression,
)
from private_analyst.analytics.cleaner import (
    CleaningExecutionSummary,
    CleaningPlanResult,
    CleaningRecommendation,
    apply_cleaning_plan,
    apply_column_management,
    apply_header_override,
    filter_rows,
    generate_cleaning_recommendations,
    handle_missing_values,
    repair_unnamed_headers,
)
from private_analyst.analytics.sql_console import DuckDBConsole, QueryResult


__all__ = [
    "ColumnStats",
    "DatasetStats",
    "compute_dataset_stats",
    "DataQualityReport",
    "assess_quality",
    "AnomalyResult",
    "detect_iqr_anomalies",
    "MLResult",
    "evaluate_model_performance",
    "run_classification",
    "run_clustering",
    "run_regression",
    "DuckDBConsole",
    "QueryResult",
    "CleaningRecommendation",
    "CleaningPlanResult",
    "CleaningExecutionSummary",
    "generate_cleaning_recommendations",
    "apply_cleaning_plan",
    "repair_unnamed_headers",
    "apply_header_override",
    "apply_column_management",
    "handle_missing_values",
    "filter_rows",
]

