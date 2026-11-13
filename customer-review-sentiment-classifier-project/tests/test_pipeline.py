from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_pipeline import (  # noqa: E402
    binary_training_data, build_pipeline, load_dataset, predict_sentiment,
    preprocess_text, evaluate_with_oof,
)


class SentimentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(ROOT / "data" / "sentiment_reviews.csv")

    def test_dataset_schema_and_labels(self):
        self.assertEqual(len(self.data), 10)
        self.assertEqual(set(self.data.columns), {"review_id", "review_text", "sentiment"})
        self.assertEqual(self.data["sentiment"].value_counts().to_dict(),
                         {"positive": 5, "negative": 4, "neutral": 1})

    def test_preprocessing_removes_html_and_keeps_negation(self):
        clean = preprocess_text("<b>NOT</b> good, but durable!")
        self.assertNotIn("<b>", clean)
        self.assertIn("not", clean.split())
        self.assertIn("good", clean.split())

    def test_binary_subset_excludes_single_neutral_row(self):
        binary = binary_training_data(self.data)
        self.assertEqual(len(binary), 9)
        self.assertEqual(set(binary["sentiment"]), {"positive", "negative"})

    def test_models_cv_and_live_prediction(self):
        metrics, oof, models = evaluate_with_oof(self.data, n_splits=3)
        self.assertEqual(set(metrics["model"]), {"multinomial_nb", "logistic_regression"})
        self.assertEqual(len(oof), 9)
        self.assertTrue(metrics["accuracy"].between(0, 1).all())
        result = predict_sentiment("This product is excellent and works perfectly.",
                                   models["multinomial_nb"])
        self.assertEqual(len(result), 1)
        self.assertIn(result.loc[0, "predicted_sentiment"], {"positive", "negative"})
        self.assertTrue(0 <= result.loc[0, "confidence"] <= 1)


if __name__ == "__main__":
    unittest.main()
