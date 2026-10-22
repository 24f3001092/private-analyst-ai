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


def _prepare_ml_matrix(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None,
    task: str = "classification",
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], dict[str, Any]]:
    """Validate, clean, and encode feature columns for machine learning models."""
    caveats: list[str] = []
    metadata: dict[str, Any] = {}

    if feature_cols is None:
        selected_features = [col for col in frame.columns if col != target_col and _is_numeric(frame[col])]
    else:
        for col in feature_cols:
            if col not in frame.columns:
                raise ValidationError("One or more feature columns are not in the loaded table.")
        selected_features = [col for col in feature_cols if col != target_col]

    if not selected_features:
        return pd.DataFrame(), pd.Series(dtype=object), [], caveats, metadata

    cols_to_use = [target_col] + selected_features
    clean_df = frame[cols_to_use].dropna().copy()
    if clean_df.empty:
        return pd.DataFrame(), pd.Series(dtype=object), [], caveats, metadata

    encoded_parts: list[pd.DataFrame] = []
    final_selected: list[str] = []

    for col in selected_features:
        series = clean_df[col]
        if _is_numeric(series):
            # Check target leakage: if feature column is identical to target
            if series.equals(clean_df[target_col]):
                caveats.append(
                    f"Excluded feature '{col}' because its values are identical to target column '{target_col}' (target leakage)."
                )
                continue
            encoded_parts.append(clean_df[[col]].astype(float))
            final_selected.append(col)
        else:
            uniq = int(series.nunique())
            if uniq > 20:
                raise ValidationError(
                    f"Feature column '{col}' is not numeric and has too many distinct values ({uniq}) for one-hot encoding. Select numeric or low-cardinality categorical features."
                )
            dummies = pd.get_dummies(series, prefix=col, drop_first=True, dtype=float)
            encoded_parts.append(dummies)
            final_selected.append(col)
            caveats.append(
                f"Categorical feature '{col}' was automatically one-hot encoded into {len(dummies.columns)} binary indicators."
            )

    if not encoded_parts:
        return pd.DataFrame(), pd.Series(dtype=object), [], caveats, metadata

    X = pd.concat(encoded_parts, axis=1)
    y = clean_df[target_col]
    metadata["row_count"] = int(clean_df.shape[0])
    metadata["encoded_columns"] = list(X.columns)

    return X, y, final_selected, caveats, metadata


def assess_feature_suitability(
    frame: pd.DataFrame,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Assess and categorize dataset columns for suitability in machine learning tasks."""
    id_candidates = {"#", "id", "s.no", "sl.no", "sr.no", "index", "no", "unnamed: 0"}
    suitable_numeric: list[str] = []
    suitable_categorical: list[str] = []
    high_cardinality_text: list[str] = []
    identifiers: list[str] = []
    constant_cols: list[str] = []

    total_rows = len(frame)

    for col in frame.columns:
        if target_col is not None and col == target_col:
            continue

        c_str = str(col).lower().strip()
        series = frame[col]
        non_null = series.dropna()
        n_unique = int(non_null.nunique())

        if n_unique <= 1:
            constant_cols.append(col)
            continue

        if c_str in id_candidates:
            identifiers.append(col)
            continue

        if _is_numeric(series):
            if total_rows > 10 and n_unique == total_rows and c_str in {"#", "no", "id", "num"}:
                identifiers.append(col)
            else:
                suitable_numeric.append(col)
        else:
            if 2 <= n_unique <= 20:
                suitable_categorical.append(col)
            else:
                high_cardinality_text.append(col)

    recommended = suitable_numeric + suitable_categorical

    return {
        "suitable_numeric": suitable_numeric,
        "suitable_categorical": suitable_categorical,
        "high_cardinality_text": high_cardinality_text,
        "identifiers": identifiers,
        "constant": constant_cols,
        "recommended_features": recommended,
    }


def run_classification(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> MLResult:
    """Run deterministic decision tree classification with train/test metrics and categorical encoding."""
    if target_col not in frame.columns:
        raise ValidationError("Target column is not in the loaded table.")

    X, y, selected_features, prep_caveats, metadata = _prepare_ml_matrix(
        frame, target_col, feature_cols, task="classification"
    )

    if not selected_features or X.empty:
        return MLResult(
            task="classification",
            status="error",
            metrics={},
            parameters={"target_col": target_col, "feature_cols": []},
            caveats=["No valid feature columns available for classification."],
        )

    # Validate row count
    if int(X.shape[0]) < 20:
        return MLResult(
            task="classification",
            status="insufficient_data",
            metrics={},
            parameters={
                "target_col": target_col,
                "feature_cols": selected_features,
                "row_count": int(X.shape[0]),
            },
            caveats=["Classification requires at least 20 complete rows of data."],
        )

    # Validate target has at least 2 distinct classes
    unique_target_count = int(y.nunique())
    if unique_target_count < 2:
        return MLResult(
            task="classification",
            status="error",
            metrics={},
            parameters={
                "target_col": target_col,
                "feature_cols": selected_features,
                "row_count": int(X.shape[0]),
                "unique_classes": unique_target_count,
            },
            caveats=["Classification target column must contain at least 2 distinct classes."],
        )

    # Train/test split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    metrics = evaluate_model_performance(y_test, y_pred_test, task="classification")
    train_metrics = evaluate_model_performance(y_train, y_pred_train, task="classification")

    metrics["train_accuracy"] = train_metrics["accuracy"]
    metrics["test_accuracy"] = metrics["accuracy"]
    metrics["train_f1"] = train_metrics["f1"]
    metrics["test_f1"] = metrics["f1"]
    metrics["feature_importances"] = {
        str(col): round(float(imp), 4) for col, imp in zip(X.columns, clf.feature_importances_)
    }

    caveats = list(prep_caveats)
    # Check class imbalance
    class_dist = y.value_counts(normalize=True)
    if float(class_dist.min()) < 0.15:
        minority_class = class_dist.idxmin()
        caveats.append(
            f"Class imbalance detected: minority class '{minority_class}' represents only {class_dist.min() * 100:.1f}% of instances."
        )

    # Check overfitting
    acc_diff = metrics["train_accuracy"] - metrics["test_accuracy"]
    if acc_diff > 0.15:
        caveats.append(
            f"Model shows noticeable overfitting: Train Accuracy is {metrics['train_accuracy'] * 100:.1f}%, while Test Accuracy is {metrics['test_accuracy'] * 100:.1f}%."
        )

    caveats.extend([
        "Decision tree models are prone to overfitting, particularly on small sample sizes.",
        "Performance metrics are evaluated on an 80/20 train/test holdout split and may vary with different splits.",
        "This model output is a statistical estimate and must not be treated as ground truth or unvalidated fact.",
    ])

    parameters: dict[str, Any] = {
        "model": "DecisionTreeClassifier",
        "target_col": target_col,
        "feature_cols": selected_features,
        "encoded_columns": list(X.columns),
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
    """Run deterministic linear regression with train/test metrics and categorical encoding."""
    if target_col not in frame.columns:
        raise ValidationError("Target column is not in the loaded table.")

    if not _is_numeric(frame[target_col]):
        raise ValidationError("Target column must be numeric for regression.")

    X, y, selected_features, prep_caveats, metadata = _prepare_ml_matrix(
        frame, target_col, feature_cols, task="regression"
    )

    if not selected_features or X.empty:
        return MLResult(
            task="regression",
            status="error",
            metrics={},
            parameters={"target_col": target_col, "feature_cols": []},
            caveats=["No valid numeric feature columns available for regression."],
        )

    if int(X.shape[0]) < 20:
        return MLResult(
            task="regression",
            status="insufficient_data",
            metrics={},
            parameters={
                "target_col": target_col,
                "feature_cols": selected_features,
                "row_count": int(X.shape[0]),
            },
            caveats=["Regression requires at least 20 complete rows of data."],
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred_test = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)

    metrics = evaluate_model_performance(y_test, y_pred_test, task="regression")
    train_metrics = evaluate_model_performance(y_train, y_pred_train, task="regression")

    metrics["train_r2"] = train_metrics["r2"]
    metrics["test_r2"] = metrics["r2"]
    metrics["train_rmse"] = train_metrics["rmse"]
    metrics["test_rmse"] = metrics["rmse"]
    coefficients = {str(col): float(coef) for col, coef in zip(X.columns, reg.coef_)}
    metrics["coefficients"] = coefficients
    metrics["intercept"] = float(reg.intercept_)

    caveats = list(prep_caveats)
    r2_diff = metrics["train_r2"] - metrics["test_r2"]
    if r2_diff > 0.15:
        caveats.append(
            f"Model indicates generalization gap: Train R² is {metrics['train_r2']:.3f}, but Test R² dropped to {metrics['test_r2']:.3f}."
        )

    caveats.extend([
        "Linear regression assumes linear relationships and absence of strong multicollinearity.",
        "Performance metrics are evaluated on an 80/20 train/test holdout split and do not guarantee future accuracy.",
        "Regression coefficients describe statistical association in the sample, not causal mechanisms.",
        "This model output is a statistical estimate and must not be treated as ground truth or unvalidated fact.",
    ])

    parameters: dict[str, Any] = {
        "model": "LinearRegression",
        "target_col": target_col,
        "feature_cols": selected_features,
        "encoded_columns": list(X.columns),
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
