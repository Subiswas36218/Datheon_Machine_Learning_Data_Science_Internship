"""Train and benchmark the review sentiment models."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_pipeline import evaluate_with_oof, load_dataset, save_models


def main() -> None:
    data = load_dataset(ROOT / "data" / "sentiment_reviews.csv")
    metrics, predictions, models = evaluate_with_oof(data)
    report_dir = ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_dir / "cross_validation_metrics.csv", index=False)
    predictions.to_csv(report_dir / "out_of_fold_predictions.csv", index=False)
    save_models(models, ROOT / "models")
    print("Cross-validation benchmark (small-sample demonstration):")
    print(metrics.to_string(index=False))
    print(f"\nModels saved to {ROOT / 'models'}")


if __name__ == "__main__":
    main()
