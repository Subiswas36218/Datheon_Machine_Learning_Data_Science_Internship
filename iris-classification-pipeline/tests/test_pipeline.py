"""Smoke tests for the Iris classification project (standard-library unittest)."""
import tempfile
import unittest
from pathlib import Path

from src.iris_classification import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_models,
    load_dataset,
    run_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "iris.csv"


class IrisPipelineTests(unittest.TestCase):
    def test_dataset_schema_and_shape(self):
        data = load_dataset(DATA_PATH)
        self.assertEqual(len(data), 150)
        self.assertEqual(FEATURE_COLUMNS + [TARGET_COLUMN], list(data.columns))
        self.assertEqual(data[TARGET_COLUMN].nunique(), 3)
        self.assertFalse(data.isna().any().any())

    def test_model_factory(self):
        models = build_models()
        self.assertEqual(set(models), {"Logistic Regression", "Decision Tree"})
        for model in models.values():
            self.assertIn("scaler", model.named_steps)
            self.assertIn("classifier", model.named_steps)

    def test_end_to_end_pipeline_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir)
            metrics = run_pipeline(DATA_PATH, output)
            self.assertEqual(len(metrics), 2)
            self.assertTrue((output / "classification_metrics.csv").exists())
            self.assertTrue((output / "classification_report.txt").exists())
            self.assertTrue((output / "pairplot.png").exists())


if __name__ == "__main__":
    unittest.main()
