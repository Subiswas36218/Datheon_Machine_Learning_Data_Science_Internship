"""End-to-end Iris classification pipeline using the task's recommended stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
BOUNDARY_FEATURES = ["petal_length", "petal_width"]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and validate the schema needed by the analysis."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    data = pd.read_csv(csv_path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "CSV does not match the expected Iris schema. "
            f"Missing columns: {missing_columns}. Found: {list(data.columns)}"
        )

    data = data[required].copy()
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"Missing values detected:\n{missing[missing > 0]}")

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("At least two target classes are required.")
    return data


def build_models() -> dict[str, Pipeline]:
    """Create the two required scaled classification pipelines."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]
        ),
    }


def make_pairplot(data: pd.DataFrame, output_dir: Path) -> None:
    """Save a pairplot of all feature pairs, colored by species."""
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.pairplot(
        data,
        vars=FEATURE_COLUMNS,
        hue=TARGET_COLUMN,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.75, "s": 38},
    )
    grid.fig.suptitle("Iris Feature Pairplot by Species", y=1.02)
    grid.savefig(output_dir / "pairplot.png", dpi=160, bbox_inches="tight")
    plt.close(grid.fig)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred,
    class_labels: list[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Save a labeled confusion-matrix heatmap."""
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    filename = model_name.lower().replace(" ", "_") 
    fig.savefig(output_dir / f"confusion_matrix_{filename}.png", dpi=160)
    plt.close(fig)


def plot_decision_boundary(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: list[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Plot a 2D slice of the trained four-feature model.

    The two petal features vary over the mesh. The two sepal features are fixed at
    their training medians. Predictions still pass through the full trained pipeline.
    """
    x_feature, y_feature = BOUNDARY_FEATURES
    x_min, x_max = X_train[x_feature].min() - 0.5, X_train[x_feature].max() + 0.5
    y_min, y_max = X_train[y_feature].min() - 0.5, X_train[y_feature].max() + 0.5
    xx, yy = __import__("numpy").meshgrid(
        __import__("numpy").linspace(x_min, x_max, 240),
        __import__("numpy").linspace(y_min, y_max, 240),
    )

    grid = pd.DataFrame(
        {
            feature: [X_train[feature].median()] * xx.size
            for feature in FEATURE_COLUMNS
        }
    )
    grid[x_feature] = xx.ravel()
    grid[y_feature] = yy.ravel()
    predictions = model.predict(grid).reshape(xx.shape)

    class_to_number = {label: index for index, label in enumerate(class_labels)}
    numeric_regions = pd.DataFrame(predictions.ravel()).iloc[:, 0].map(class_to_number)
    numeric_regions = numeric_regions.to_numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, numeric_regions, alpha=0.22, levels=len(class_labels) - 1)
    sns.scatterplot(
        data=X_test.assign(species=y_test.to_numpy()),
        x=x_feature,
        y=y_feature,
        hue="species",
        hue_order=class_labels,
        style="species",
        s=75,
        edgecolor="black",
        ax=ax,
    )
    fixed_values = ", ".join(
        f"{feature}={X_train[feature].median():.2f}"
        for feature in FEATURE_COLUMNS
        if feature not in BOUNDARY_FEATURES
    )
    ax.set_title(f"Decision Boundary Slice — {model_name}")
    ax.set_xlabel(x_feature.replace("_", " ").title())
    ax.set_ylabel(y_feature.replace("_", " ").title())
    ax.text(
        0.02, 0.98, f"Fixed at training medians: {fixed_values}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.legend(title="True species", loc="best")
    fig.tight_layout()
    filename = model_name.lower().replace(" ", "_")
    fig.savefig(output_dir / f"decision_boundary_{filename}.png", dpi=160)
    plt.close(fig)


def run_pipeline(data_path: Path, output_dir: Path) -> pd.DataFrame:
    """Run exploration, model training, evaluation, and visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(data_path)

    print("\nDATASET OVERVIEW")
    print(f"Rows: {len(data)} | Features: {len(FEATURE_COLUMNS)}")
    print("\nClass distribution:")
    print(data[TARGET_COLUMN].value_counts().sort_index())
    print("\nDescriptive statistics:")
    print(data[FEATURE_COLUMNS].describe().round(3))

    make_pairplot(data, output_dir)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    class_labels = sorted(y.unique())

    print(f"\nTraining rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    summary_rows = []
    report_sections = []

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        weighted = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        summary_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "macro_precision": macro[0],
                "macro_recall": macro[1],
                "macro_f1": macro[2],
                "weighted_precision": weighted[0],
                "weighted_recall": weighted[1],
                "weighted_f1": weighted[2],
            }
        )

        report = classification_report(
            y_test, y_pred, labels=class_labels, zero_division=0
        )
        report_sections.append(f"{model_name}\n{'=' * len(model_name)}\n{report}")
        print(f"\n{model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)

        filename = model_name.lower().replace(" ", "_")
        plot_confusion_matrix(y_test, y_pred, class_labels, model_name, output_dir)
        plot_decision_boundary(
            model, X_train, X_test, y_test, class_labels, model_name, output_dir
        )

    metrics = pd.DataFrame(summary_rows).sort_values("weighted_f1", ascending=False)
    metrics.to_csv(output_dir / "classification_metrics.csv", index=False)
    (output_dir / "classification_report.txt").write_text(
        "\n\n".join(report_sections), encoding="utf-8"
    )

    print("\nMODEL COMPARISON")
    print(metrics.round(4).to_string(index=False))
    print(f"\nArtifacts saved to: {output_dir.resolve()}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Iris classification models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/iris.csv"),
        help="Path to the supplied Iris CSV (default: data/iris.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated plots and reports (default: outputs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data, args.output)
