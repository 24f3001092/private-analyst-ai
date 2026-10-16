from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from private_analyst.analytics.statistics import _is_numeric
from private_analyst.exceptions import ValidationError


@dataclass
class MLResult:
    task: str  # classification, regression, clustering
    status: str  # evaluated, hypothesis, insufficient_data, error
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    caveats: list[str]
    evidence_class: str = "model_output_not_ground_truth"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_model_performance(y_true: Any, y_pred: Any, task: str = "classification") -> dict[str, Any]:
    """Computes appropriate metrics for classification or regression.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Estimated target values from model.
    task : str
        Either 'classification' or 'regression'.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed performance metrics.
    """
    if task == "classification":
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        return {
            "accuracy": float(accuracy_score(y_t, y_p)),
            "precision": float(precision_score(y_t, y_p, average="macro", zero_division=0)),
            "recall": float(recall_score(y_t, y_p, average="macro", zero_division=0)),
            "f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_t, y_p).tolist(),
        }

    if task == "regression":
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        mse = float(mean_squared_error(y_t, y_p))
        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_t, y_p))
        return {
            "r2": r2,
            "R²": r2,
            "mae": mae,
            "MAE": mae,
            "rmse": rmse,
            "RMSE": rmse,
            "mse": mse,
        }

    raise ValueError("Unsupported task. Must be 'classification' or 'regression'.")


def run_classification(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> MLResult:
    """Run deterministic decision tree classification.

    Parameters
    ----------
    frame : pd.DataFrame
        Loaded dataset.
    target_col : str
        Name of target column.
    feature_cols : list[str] | None
        List of feature column names; if None, uses all numeric columns except target.

    Returns
    -------
    MLResult
        Classification result containing metrics, parameters, and caveats.
    """
    if target_col not in frame.columns:
        raise ValidationError("Target column is not in the loaded table.")

    if feature_cols is None:
        selected_features = [col for col in frame.columns if col != target_col and _is_numeric(frame[col])]
    else:
        for col in feature_cols:
            if col not in frame.columns:
                raise ValidationError("One or more feature columns are not in the loaded table.")
            if not _is_numeric(frame[col]):
                raise ValidationError("Classification feature columns must be numeric.")
        selected_features = [col for col in feature_cols if col != target_col]

    if not selected_features:
        return MLResult(
            task="classification",
            status="error",
            metrics={},
            parameters={"target_col": target_col, "feature_cols": []},
            caveats=["No valid numeric feature columns available for classification."],
        )

    # Handle NaN values by dropping rows with NaN in relevant columns
    cols_to_use = [target_col] + selected_features
    clean_df = frame[cols_to_use].dropna()

    # Validate that there are enough rows (at least 20)
    if int(clean_df.shape[0]) < 20:
        return MLResult(
            task="classification",
            status="insufficient_data",
            metrics={},
            parameters={
                "target_col": target_col,
                "feature_cols": selected_features,
                "row_count": int(clean_df.shape[0]),
            },
            caveats=["Classification requires at least 20 complete rows of data."],
        )

    # Validate target has at least 2 distinct values
    unique_target_count = int(clean_df[target_col].nunique())
    if unique_target_count < 2:
        return MLResult(
            task="classification",
            status="error",
            metrics={},
            parameters={
                "target_col": target_col,
                "feature_cols": selected_features,
                "row_count": int(clean_df.shape[0]),
                "unique_classes": unique_target_count,
            },
            caveats=["Classification target column must contain at least 2 distinct classes."],
        )

    X = clean_df[selected_features]
    y = clean_df[target_col]

    # Train/test split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = evaluate_model_performance(y_test, y_pred, task="classification")

    caveats = [
        "Decision tree models are prone to overfitting, particularly on small sample sizes.",
        "Performance metrics are evaluated on an 80/20 train/test holdout split and may vary with different splits.",
        "This model output is a statistical estimate and must not be treated as ground truth or unvalidated fact.",
    ]

    parameters: dict[str, Any] = {
        "model": "DecisionTreeClassifier",
        "target_col": target_col,
        "feature_cols": selected_features,
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classes": [str(c) for c in clf.classes_],
    }

    return MLResult(
        task="classification",
        status="evaluated",
        metrics=metrics,
        parameters=parameters,
        caveats=caveats,
    )


def run_regression(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> MLResult:
    """Run deterministic linear regression.

    Parameters
    ----------
    frame : pd.DataFrame
        Loaded dataset.
    target_col : str
        Name of numeric target column.
    feature_cols : list[str] | None
        List of feature column names; if None, uses all numeric columns except target.

    Returns
    -------
    MLResult
        Regression result containing metrics, coefficients, parameters, and caveats.
    """
    if target_col not in frame.columns:
        raise ValidationError("Target column is not in the loaded table.")

    if not _is_numeric(frame[target_col]):
        raise ValidationError("Target column must be numeric for regression.")

    if feature_cols is None:
        selected_features = [col for col in frame.columns if col != target_col and _is_numeric(frame[col])]
    else:
        for col in feature_cols:
            if col not in frame.columns:
                raise ValidationError("One or more feature columns are not in the loaded table.")
            if not _is_numeric(frame[col]):
                raise ValidationError("Regression feature columns must be numeric.")
        selected_features = [col for col in feature_cols if col != target_col]

    if not selected_features:
        return MLResult(
            task="regression",
            status="error",
            metrics={},
            parameters={"target_col": target_col, "feature_cols": []},
            caveats=["No valid numeric feature columns available for regression."],
        )

    # Handle NaN values by dropping rows with NaN in relevant columns
    cols_to_use = [target_col] + selected_features
    clean_df = frame[cols_to_use].dropna()

    if int(clean_df.shape[0]) < 20:
        return MLResult(
            task="regression",
            status="insufficient_data",
            metrics={},
            parameters={
                "target_col": target_col,
                "feature_cols": selected_features,
                "row_count": int(clean_df.shape[0]),
            },
            caveats=["Regression requires at least 20 complete rows of data."],
        )

    X = clean_df[selected_features]
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    metrics = evaluate_model_performance(y_test, y_pred, task="regression")
    coefficients = {str(col): float(coef) for col, coef in zip(selected_features, reg.coef_)}
    metrics["coefficients"] = coefficients
    metrics["intercept"] = float(reg.intercept_)

    caveats = [
        "Linear regression assumes linear relationships and absence of strong multicollinearity.",
        "Performance metrics are evaluated on an 80/20 train/test holdout split and do not guarantee future accuracy.",
        "Regression coefficients describe statistical association in the sample, not causal mechanisms.",
        "This model output is a statistical estimate and must not be treated as ground truth or unvalidated fact.",
    ]

    parameters: dict[str, Any] = {
        "model": "LinearRegression",
        "target_col": target_col,
        "feature_cols": selected_features,
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "coefficients": coefficients,
        "intercept": float(reg.intercept_),
    }

    return MLResult(
        task="regression",
        status="evaluated",
        metrics=metrics,
        parameters=parameters,
        caveats=caveats,
    )


def run_clustering(
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_clusters: int = 3,
) -> MLResult:
    """Run KMeans clustering with feature standardization.

    Parameters
    ----------
    frame : pd.DataFrame
        Loaded dataset.
    feature_cols : list[str] | None
        List of numeric feature column names; if None, uses all numeric columns.
    n_clusters : int
        Number of clusters to form (must be between 2 and 20).

    Returns
    -------
    MLResult
        Clustering result containing cluster labels, silhouette score, centers, sizes, and caveats.
    """
    if not isinstance(n_clusters, int) or n_clusters < 2 or n_clusters > 20:
        raise ValueError("Number of clusters (n_clusters) must be an integer between 2 and 20.")

    if feature_cols is None:
        selected_features = [col for col in frame.columns if _is_numeric(frame[col])]
    else:
        for col in feature_cols:
            if col not in frame.columns:
                raise ValidationError("One or more feature columns are not in the loaded table.")
            if not _is_numeric(frame[col]):
                raise ValidationError("Clustering feature columns must be numeric.")
        selected_features = list(feature_cols)

    if not selected_features:
        return MLResult(
            task="clustering",
            status="error",
            metrics={},
            parameters={"n_clusters": n_clusters, "feature_cols": []},
            caveats=["No valid numeric feature columns available for clustering."],
        )

    # Handle NaN values by dropping rows with NaN in relevant columns
    clean_df = frame[selected_features].dropna()

    if int(clean_df.shape[0]) < n_clusters:
        return MLResult(
            task="clustering",
            status="insufficient_data",
            metrics={},
            parameters={
                "n_clusters": n_clusters,
                "feature_cols": selected_features,
                "row_count": int(clean_df.shape[0]),
            },
            caveats=["Clustering requires at least as many complete rows as clusters."],
        )

    # Use StandardScaler for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df[selected_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    if int(clean_df.shape[0]) > n_clusters and len(set(labels)) > 1:
        sil_score = float(silhouette_score(X_scaled, labels))
    else:
        sil_score = 0.0

    unscaled_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers = [[float(v) for v in row] for row in unscaled_centers]
    scaled_centers = [[float(v) for v in row] for row in kmeans.cluster_centers_]
    cluster_sizes = {int(i): int(np.sum(labels == i)) for i in range(n_clusters)}
    cluster_labels = [int(lbl) for lbl in labels]

    metrics: dict[str, Any] = {
        "cluster_labels": cluster_labels,
        "silhouette_score": sil_score,
        "cluster_centers": cluster_centers,
        "cluster_sizes": cluster_sizes,
        "scaled_cluster_centers": scaled_centers,
        "inertia": float(kmeans.inertia_),
    }

    caveats = [
        "Clusters are algorithmically generated groupings (hypotheses), not confirmed real-world categories or ground truth.",
        "K-Means is sensitive to the choice of k and assumes roughly spherical clusters.",
        "Features were standardized using StandardScaler; scale differences between features affect cluster formation.",
        "This model output is exploratory and must not be treated as ground truth or unvalidated fact.",
    ]

    parameters: dict[str, Any] = {
        "model": "KMeans",
        "n_clusters": n_clusters,
        "feature_cols": selected_features,
        "row_count": int(clean_df.shape[0]),
        "random_state": 42,
    }

    return MLResult(
        task="clustering",
        status="hypothesis",
        metrics=metrics,
        parameters=parameters,
        caveats=caveats,
    )
