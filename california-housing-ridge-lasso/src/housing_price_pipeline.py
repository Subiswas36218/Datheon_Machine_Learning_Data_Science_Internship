"""Leakage-safe California housing price regression pipeline.

The project intentionally uses a small, transparent scikit-learn stack:
LinearRegression, Ridge, Lasso, and GridSearchCV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "median_house_value"
CATEGORICAL_FEATURES = ["ocean_proximity"]
REQUIRED_COLUMNS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    TARGET, "ocean_proximity",
]
ALPHA_GRIDS = {"Ridge": [0.1, 1.0, 10.0, 100.0], "Lasso": [100.0, 300.0, 1000.0, 3000.0]}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the supplied California Housing CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values.")
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable household ratios and interaction features.

    This is row-wise deterministic feature engineering: it does not learn
    statistics from the full dataset, so applying it before splitting does
    not leak target or validation information.
    """
    data = frame.copy()
    safe_households = data["households"].replace(0, np.nan)
    safe_rooms = data["total_rooms"].replace(0, np.nan)
    safe_population = data["population"].replace(0, np.nan)

    data["rooms_per_household"] = data["total_rooms"] / safe_households
    data["bedrooms_per_room"] = data["total_bedrooms"] / safe_rooms
    data["population_per_household"] = data["population"] / safe_households
    data["rooms_per_person"] = data["total_rooms"] / safe_population

    data["income_x_rooms_per_household"] = (
        data["median_income"] * data["rooms_per_household"]
    )
    data["income_x_housing_age"] = (
        data["median_income"] * data["housing_median_age"]
    )
    data["latitude_x_longitude"] = data["latitude"] * data["longitude"]
    return data


def get_feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical predictor columns, excluding target."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in data.columns]
    numeric = [
        column for column in data.columns
        if column != TARGET and column not in categorical
        and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not numeric and not categorical:
        raise ValueError("No usable predictor columns were found.")
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build train-fitted imputing, scaling, and encoding transformations."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model_pipelines(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    """Construct OLS, Ridge, and Lasso pipelines sharing identical preprocessing."""
    models: dict[str, Pipeline] = {}
    estimators: dict[str, Any] = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10_000, tol=1e-2, precompute=True, selection="random", random_state=42),
    }
    for name, estimator in estimators.items():
        models[name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("regressor", estimator),
            ]
        )
    return models


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, and R² in original target units."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_experiment(
    dataset_path: str | Path,
    output_dir: str | Path = "outputs",
    random_state: int = 42,
    cv: int = 3,
    alpha_grid: list[float] | None = None,
) -> dict[str, Any]:
    """Fit all models, tune Ridge/Lasso with CV, and persist tabular artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(dataset_path)
    data = engineer_features(raw)

    numeric_features, categorical_features = get_feature_columns(data)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    pipelines = build_model_pipelines(numeric_features, categorical_features)
    fitted_models: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for model_name, pipeline in pipelines.items():
        if model_name == "OLS":
            pipeline.fit(X_train, y_train)
            fitted = pipeline
            best_alpha = np.nan
            cv_rmse = np.nan
        else:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid={"regressor__alpha": (alpha_grid if alpha_grid is not None else ALPHA_GRIDS[model_name])},
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=1,
                refit=True,
                return_train_score=True,
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_alpha = float(search.best_params_["regressor__alpha"])
            cv_rmse = float(-search.best_score_)
            for params, mean_score, std_score, mean_train in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
                search.cv_results_["mean_train_score"],
            ):
                tuning_rows.append({
                    "model": model_name,
                    "alpha": float(params["regressor__alpha"]),
                    "mean_cv_rmse": float(-mean_score),
                    "std_cv_rmse": float(std_score),
                    "mean_train_rmse": float(-mean_train),
                })

        predictions = fitted.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        rows.append({
            "model": model_name,
            **metrics,
            "best_alpha": best_alpha,
            "best_cv_rmse": cv_rmse,
        })
        fitted_models[model_name] = fitted

    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output_dir / "hyperparameter_search.csv", index=False)

    prediction_frame = pd.DataFrame({
        "actual": y_test.to_numpy(),
        **{name: model.predict(X_test) for name, model in fitted_models.items()},
    })
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    best_name = str(comparison.loc[0, "model"])
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    # Export coefficient table for linear estimators; OHE feature names are retained.
    preprocessor = best_model.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()
    coefficients = np.asarray(best_model.named_steps["regressor"].coef_).ravel()
    coefficient_frame = pd.DataFrame({
        "feature": transformed_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
    }).sort_values("absolute_coefficient", ascending=False)
    coefficient_frame.to_csv(output_dir / "best_model_coefficients.csv", index=False)

    return {
        "raw_data": raw,
        "data": data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": fitted_models,
        "comparison": comparison,
        "predictions": prediction_frame,
        "tuning": pd.DataFrame(tuning_rows),
        "best_model_name": best_name,
        "best_model": best_model,
        "best_predictions": best_predictions,
        "best_residuals": y_test.to_numpy() - best_predictions,
    }
