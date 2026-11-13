from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_pipeline import (  # noqa: E402
    build_pipeline,
    evaluate_model,
    load_dataset,
    prepare_features,
)

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"


def test_dataset_cleaning_and_target():
    frame = load_dataset(DATA_PATH)
    assert frame.shape == (7043, 21)
    assert pd.api.types.is_numeric_dtype(frame["TotalCharges"])
    assert frame["TotalCharges"].isna().sum() == 11
    X, y, numeric, categorical = prepare_features(frame)
    assert "customerID" not in X.columns
    assert y.value_counts().to_dict() == {0: 5174, 1: 1869}
    assert "TotalCharges" in numeric
    assert "Contract" in categorical


def test_pipeline_places_smote_after_preprocessing():
    frame = load_dataset(DATA_PATH)
    X, y, numeric, categorical = prepare_features(frame)
    pipeline = build_pipeline("random_forest", numeric, categorical)
    assert list(pipeline.named_steps) == ["preprocessor", "smote", "encoder", "classifier"]


def test_small_end_to_end_fit_and_metrics():
    frame = load_dataset(DATA_PATH)
    X, y, numeric, categorical = prepare_features(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    # Keep the test fast while validating the complete data/preprocess/SMOTE/model path.
    model = build_pipeline(
        "random_forest", numeric, categorical,
        model_params={"n_estimators": 20, "max_depth": 8},
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert set(metrics) == {
        "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Average Precision"
    }
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_invalid_target_is_rejected():
    frame = load_dataset(DATA_PATH)
    frame.loc[0, "Churn"] = "unknown"
    try:
        prepare_features(frame)
    except ValueError as exc:
        assert "only 'No' and 'Yes'" in str(exc)
    else:
        raise AssertionError("Invalid target label should be rejected")
