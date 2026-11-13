"""Leakage-safe Telco customer churn modelling with SMOTE, Random Forest and XGBoost."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "Churn"
ID_COLUMNS = ["customerID"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV and normalize TotalCharges whitespace to numeric missing values."""
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r} in dataset")
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"].replace(r"^\s*$", np.nan, regex=True),
            errors="coerce",
        )
    return frame


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Separate target, drop ID, and identify numeric/categorical predictors."""
    if TARGET not in frame:
        raise ValueError(f"Missing required target column {TARGET!r}")
    clean = frame.copy()
    y = clean[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Target must contain only 'No' and 'Yes' values")
    drop_cols = [c for c in ID_COLUMNS + [TARGET] if c in clean.columns]
    X = clean.drop(columns=drop_cols)
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y.astype("int64"), numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Impute numerics and ordinal-encode categoricals before SMOTENC.

    SMOTENC is used rather than vanilla SMOTE because the source data contains
    nominal fields. It synthesizes categorical values as categories instead of
    interpolating fractional one-hot indicators.
    """
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_post_smote_encoder(n_numeric: int, n_categorical: int) -> ColumnTransformer:
    """One-hot encode nominal features after SMOTENC has resampled them."""
    categorical_indices = list(range(n_numeric, n_numeric + n_categorical))
    numeric_indices = list(range(n_numeric))
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_indices),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ],
        remainder="drop",
    )


def get_model_feature_names(
    fitted_pipeline: Any, numeric_cols: list[str], categorical_cols: list[str]
) -> list[str]:
    """Recover readable post-encoding feature names for feature-importance plots."""
    ordinal = (
        fitted_pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"].named_steps["ordinal"]
    )
    onehot = fitted_pipeline.named_steps["encoder"].named_transformers_["categorical"]
    names = [f"numeric__{name}" for name in numeric_cols]
    for source_name, ordinal_categories, encoded_categories in zip(
        categorical_cols, ordinal.categories_, onehot.categories_
    ):
        for encoded_value in encoded_categories:
            index = int(round(float(encoded_value)))
            if 0 <= index < len(ordinal_categories):
                category_label = str(ordinal_categories[index])
            else:
                category_label = "unknown"
            names.append(f"categorical__{source_name}_{category_label}")
    return names


def build_pipeline(
    model_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
    model_params: dict[str, Any] | None = None,
) -> ImbPipeline:
    """Create a preprocessing -> optional SMOTE -> estimator pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    params = dict(model_params or {})
    key = model_name.lower().replace("_", "").replace("-", "")
    if key in {"randomforest", "rf"}:
        estimator_params = {
            "n_estimators": 350,
            "min_samples_leaf": 2,
            "class_weight": None,
            "random_state": random_state,
            "n_jobs": 1,
        }
        estimator_params.update(params)
        estimator = RandomForestClassifier(**estimator_params)
    elif key in {"xgboost", "xgb"}:
        estimator_params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": 1,
        }
        estimator_params.update(params)
        estimator = XGBClassifier(**estimator_params)
    else:
        raise ValueError("model_name must be 'random_forest' or 'xgboost'")

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if use_smote:
        categorical_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))
        steps.append(("smote", SMOTENC(categorical_features=categorical_indices, random_state=random_state, k_neighbors=5)))
    steps.append(("encoder", build_post_smote_encoder(len(numeric_cols), len(categorical_cols))))
    steps.append(("classifier", estimator))
    return ImbPipeline(steps=steps)


def search_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    n_iter: int = 3,
    cv_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Tune a SMOTE pipeline with stratified CV and ROC-AUC as the objective."""
    key = model_name.lower().replace("_", "").replace("-", "")
    pipe = build_pipeline(key, numeric_cols, categorical_cols, random_state=random_state)
    if key in {"randomforest", "rf"}:
        param_distributions = {
            "classifier__n_estimators": [250, 350, 450],
            "classifier__max_depth": [None, 12, 20],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.7],
        }
    elif key in {"xgboost", "xgb"}:
        param_distributions = {
            "classifier__n_estimators": [200, 300, 400],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.08],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            "classifier__reg_lambda": [1.0, 2.0, 5.0],
        }
    else:
        raise ValueError("model_name must be 'random_forest' or 'xgboost'")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate threshold-based metrics and rank-based ROC-AUC / average precision."""
    probabilities = estimator.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1": float(f1_score(y_test, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, probabilities)),
        "Average Precision": float(average_precision_score(y_test, probabilities)),
    }


def split_dataset(
    X: pd.DataFrame, y: pd.Series, *, test_size: float = 0.20, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
