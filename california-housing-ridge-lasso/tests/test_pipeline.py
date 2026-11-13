from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from housing_price_pipeline import (  # noqa: E402
    TARGET,
    REQUIRED_COLUMNS,
    build_model_pipelines,
    engineer_features,
    get_feature_columns,
    load_dataset,
    run_experiment,
)


class TestHousingPricePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = REPO_ROOT / "data" / "housing.csv"
        cls.raw = load_dataset(cls.data_path)

    def test_supplied_dataset_schema_and_missingness(self) -> None:
        self.assertEqual(len(self.raw), 20_640)
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(self.raw.columns))
        self.assertEqual(int(self.raw["total_bedrooms"].isna().sum()), 207)
        self.assertEqual(int(self.raw[TARGET].isna().sum()), 0)
        self.assertEqual(self.raw["ocean_proximity"].nunique(), 5)

    def test_engineered_features_are_present_and_row_count_is_preserved(self) -> None:
        featured = engineer_features(self.raw)
        expected = {
            "rooms_per_household",
            "bedrooms_per_room",
            "population_per_household",
            "rooms_per_person",
            "income_x_rooms_per_household",
            "income_x_housing_age",
            "latitude_x_longitude",
        }
        self.assertTrue(expected.issubset(featured.columns))
        self.assertEqual(len(featured), len(self.raw))
        self.assertTrue(np.isfinite(featured["rooms_per_household"]).all())

    def test_model_factory_contains_recommended_estimators(self) -> None:
        featured = engineer_features(self.raw)
        numeric, categorical = get_feature_columns(featured)
        pipelines = build_model_pipelines(numeric, categorical)
        self.assertEqual(set(pipelines), {"OLS", "Ridge", "Lasso"})
        self.assertIsInstance(pipelines["OLS"].named_steps["regressor"], LinearRegression)
        self.assertIsInstance(pipelines["Ridge"].named_steps["regressor"], Ridge)
        self.assertIsInstance(pipelines["Lasso"].named_steps["regressor"], Lasso)
        self.assertIn("ocean_proximity", categorical)

    def test_end_to_end_experiment_writes_metrics_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_experiment(
                dataset_path=self.data_path,
                output_dir=temporary_directory,
                random_state=42,
                cv=2,
                alpha_grid=[100.0, 300.0],
            )
            comparison = result["comparison"]
            self.assertEqual(set(comparison["model"]), {"OLS", "Ridge", "Lasso"})
            self.assertTrue((comparison["RMSE"] > 0).all())
            self.assertTrue(comparison["R2"].between(-1, 1).all())
            self.assertEqual(len(result["predictions"]), 4_128)
            self.assertTrue((Path(temporary_directory) / "model_comparison.csv").exists())
            self.assertTrue((Path(temporary_directory) / "hyperparameter_search.csv").exists())
            self.assertTrue((Path(temporary_directory) / "test_predictions.csv").exists())
            self.assertTrue((Path(temporary_directory) / "best_model_coefficients.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
