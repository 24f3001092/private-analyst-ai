from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from private_analyst.analytics.ml_engine import (
    MLResult,
    evaluate_model_performance,
    run_classification,
    run_clustering,
    run_regression,
)
from private_analyst.exceptions import ValidationError


def test_classification_basic() -> None:
    # Synthetic DataFrame with clear binary labels
    rng = np.random.default_rng(42)
    n = 60
    # Class 0: low feature values; Class 1: high feature values
    f1_class0 = rng.normal(loc=0.0, scale=0.5, size=n // 2)
    f2_class0 = rng.normal(loc=0.0, scale=0.5, size=n // 2)
    f1_class1 = rng.normal(loc=10.0, scale=0.5, size=n // 2)
    f2_class1 = rng.normal(loc=10.0, scale=0.5, size=n // 2)

    features1 = np.concatenate([f1_class0, f1_class1])
    features2 = np.concatenate([f2_class0, f2_class1])
    targets = np.array([0] * (n // 2) + [1] * (n // 2))

    df = pd.DataFrame({"feat1": features1, "feat2": features2, "target": targets})

    result = run_classification(df, target_col="target")

    assert isinstance(result, MLResult)
    assert result.task == "classification"
    assert result.status == "evaluated"
    assert result.evidence_class == "model_output_not_ground_truth"

    required_keys = {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    assert required_keys.issubset(result.metrics.keys())
    assert result.metrics["accuracy"] > 0.5
    assert result.metrics["precision"] > 0.5
    assert result.metrics["recall"] > 0.5
    assert result.metrics["f1"] > 0.5
    assert len(result.caveats) >= 1


def test_classification_insufficient_data() -> None:
    # Fewer than 20 rows
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    result = run_classification(df, target_col="target")
    assert result.task == "classification"
    assert result.status == "insufficient_data"
    assert len(result.caveats) >= 1


def test_classification_single_class() -> None:
    # 30 rows but all belonging to class 1
    df = pd.DataFrame({
        "feat1": np.linspace(0, 10, 30),
        "feat2": np.linspace(10, 20, 30),
        "target": [1] * 30,
    })
    result = run_classification(df, target_col="target")
    assert result.task == "classification"
    assert result.status == "error"
    assert len(result.caveats) >= 1


def test_regression_basic() -> None:
    # Synthetic linear data: y = 3 * x + 7
    rng = np.random.default_rng(42)
    x = np.linspace(0, 20, 80)
    noise = rng.normal(0, 0.2, 80)
    y = 3.0 * x + 7.0 + noise

    df = pd.DataFrame({"x": x, "y": y})
    result = run_regression(df, target_col="y")

    assert isinstance(result, MLResult)
    assert result.task == "regression"
    assert result.status == "evaluated"
    assert result.metrics["r2"] > 0.5
    assert "mae" in result.metrics
    assert "rmse" in result.metrics
    assert "coefficients" in result.metrics
    assert len(result.caveats) >= 1


def test_regression_non_numeric_target() -> None:
    df = pd.DataFrame({
        "x": np.linspace(0, 10, 30),
        "target": ["category_a", "category_b"] * 15,
    })
    with pytest.raises(ValidationError):
        run_regression(df, target_col="target")


def test_clustering_basic() -> None:
    # 3 clearly separated clusters of 20 points each
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(20, 2))
    c2 = rng.normal(loc=[10.0, 10.0], scale=0.5, size=(20, 2))
    c3 = rng.normal(loc=[20.0, 20.0], scale=0.5, size=(20, 2))
    data = np.vstack([c1, c2, c3])

    df = pd.DataFrame(data, columns=["x1", "x2"])
    result = run_clustering(df, n_clusters=3)

    assert isinstance(result, MLResult)
    assert result.task == "clustering"
    assert result.status == "hypothesis"
    assert len(result.metrics["cluster_centers"]) == 3
    assert len(result.metrics["cluster_sizes"]) == 3
    assert len(set(result.metrics["cluster_labels"])) == 3
    assert "silhouette_score" in result.metrics
    assert result.metrics["silhouette_score"] > 0.5
    assert len(result.caveats) >= 1


def test_clustering_invalid_k() -> None:
    df = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30, dtype=float)})
    with pytest.raises(ValueError):
        run_clustering(df, n_clusters=0)

    with pytest.raises(ValueError):
        run_clustering(df, n_clusters=1)

    with pytest.raises(ValueError):
        run_clustering(df, n_clusters=21)


def test_all_results_have_caveats() -> None:
    rng = np.random.default_rng(42)

    # 1. Classification (evaluated)
    df_clf = pd.DataFrame({
        "x": rng.normal(0, 1, 40),
        "y": [0] * 20 + [1] * 20,
    })
    res_clf = run_classification(df_clf, target_col="y")
    assert len(res_clf.caveats) >= 1
    assert all(isinstance(c, str) and len(c) > 0 for c in res_clf.caveats)

    # 2. Classification (insufficient_data)
    res_clf_insuf = run_classification(df_clf.iloc[:10], target_col="y")
    assert len(res_clf_insuf.caveats) >= 1

    # 3. Classification (error - single class)
    df_single = pd.DataFrame({"x": np.arange(30), "y": [0] * 30})
    res_clf_err = run_classification(df_single, target_col="y")
    assert len(res_clf_err.caveats) >= 1

    # 4. Regression (evaluated)
    df_reg = pd.DataFrame({
        "x": np.linspace(0, 10, 40),
        "y": 2.0 * np.linspace(0, 10, 40) + rng.normal(0, 0.1, 40),
    })
    res_reg = run_regression(df_reg, target_col="y")
    assert len(res_reg.caveats) >= 1
    assert all(isinstance(c, str) and len(c) > 0 for c in res_reg.caveats)

    # 5. Clustering (hypothesis)
    df_clust = pd.DataFrame({"x": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)})
    res_clust = run_clustering(df_clust, n_clusters=3)
    assert len(res_clust.caveats) >= 1
    assert all(isinstance(c, str) and len(c) > 0 for c in res_clust.caveats)

    # Verify evidence class on all
    for r in [res_clf, res_clf_insuf, res_clf_err, res_reg, res_clust]:
        assert r.evidence_class == "model_output_not_ground_truth"


def test_evaluate_model_performance() -> None:
    y_true_clf = [0, 1, 0, 1, 1]
    y_pred_clf = [0, 1, 0, 0, 1]
    clf_metrics = evaluate_model_performance(y_true_clf, y_pred_clf, task="classification")
    assert clf_metrics["accuracy"] == 0.8
    assert "precision" in clf_metrics
    assert "recall" in clf_metrics
    assert "f1" in clf_metrics
    assert "confusion_matrix" in clf_metrics

    y_true_reg = [1.0, 2.0, 3.0, 4.0]
    y_pred_reg = [1.0, 2.0, 3.0, 4.0]
    reg_metrics = evaluate_model_performance(y_true_reg, y_pred_reg, task="regression")
    assert reg_metrics["r2"] == 1.0
    assert reg_metrics["mae"] == 0.0
    assert reg_metrics["rmse"] == 0.0

    with pytest.raises(ValueError):
        evaluate_model_performance(y_true_clf, y_pred_clf, task="unsupported")


def test_missing_and_nan_handling() -> None:
    # 25 rows, some with NaN
    x = [float(i) for i in range(25)]
    x[5] = np.nan
    y = [float(2 * i) for i in range(25)]
    y[10] = np.nan

    df = pd.DataFrame({"x": x, "y": y})
    # 23 complete rows remain (> 20)
    result = run_regression(df, target_col="y")
    assert result.status == "evaluated"
    assert result.metrics["r2"] > 0.99
