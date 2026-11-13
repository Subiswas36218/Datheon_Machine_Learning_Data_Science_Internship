"""NLP sentiment classification using NLTK, TF-IDF and scikit-learn."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import joblib
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
TARGET_COLUMN = "sentiment"
TEXT_COLUMN = "review_text"
ID_COLUMN = "review_id"
BINARY_LABELS = ("negative", "positive")

# Compact fallback used only when the NLTK stopwords corpus is unavailable.
# Negation terms are intentionally not removed because they reverse polarity.
_FALLBACK_STOPWORDS = set("""
a an the and or but if while of at by for with about against between into through
during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t can will just don
should now is am are was were be been being have has had do does did this that
these those it its i me my we our you your he she they them their what which who
whom as until because also really quite would could might must shall may
""".split())
_NEGATIONS = {"no", "not", "nor", "never", "n't"}
_TOKENIZER = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")
_LEMMATIZER = WordNetLemmatizer()


def ensure_nltk_resources(download: bool = False) -> dict[str, bool]:
    """Check (and optionally download) NLTK stopwords/WordNet resources.

    The pipeline can run offline with a compact stopword fallback and without
    lemmatization. For full preprocessing, install NLTK's stopwords and wordnet
    corpora using download=True or the documented nltk.download commands.
    """
    if download:
        for package in ("stopwords", "wordnet", "omw-1.4"):
            nltk.download(package, quiet=True)
    try:
        from nltk.corpus import stopwords
        stopword_set = set(stopwords.words("english"))
        source = "nltk"
    except LookupError:
        stopword_set = set(_FALLBACK_STOPWORDS)
        source = "fallback"
    stopword_set -= _NEGATIONS
    try:
        _LEMMATIZER.lemmatize("reviews")
        wordnet_available = True
    except LookupError:
        wordnet_available = False
    return {
        "nltk_stopwords_available": source == "nltk",
        "wordnet_available": wordnet_available,
    }


_RESOURCE_STATUS = ensure_nltk_resources(download=False)
try:
    from nltk.corpus import stopwords as _nltk_stopwords
    _STOPWORDS = set(_nltk_stopwords.words("english")) - _NEGATIONS
except LookupError:
    _STOPWORDS = set(_FALLBACK_STOPWORDS) - _NEGATIONS


def preprocess_text(text: object) -> str:
    """Lowercase, remove HTML/punctuation, tokenize, stopword-filter and lemmatize.

    If WordNet is not installed, the function safely skips lemmatization and
    reports this through ``ensure_nltk_resources``. This permits offline use.
    """
    if pd.isna(text):
        return ""
    value = html.unescape(str(text))
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("’", "'").lower()
    # Expand common negation contractions before tokenization.
    value = re.sub(r"\bcan't\b", "can not", value)
    value = re.sub(r"\bcannot\b", "can not", value)
    value = re.sub(r"n't\b", " not", value)
    tokens = _TOKENIZER.tokenize(value)
    cleaned: list[str] = []
    for token in tokens:
        if token in _STOPWORDS and token not in _NEGATIONS:
            continue
        if _RESOURCE_STATUS["wordnet_available"]:
            token = _LEMMATIZER.lemmatize(token)
        cleaned.append(token)
    return " ".join(cleaned)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the supplied review CSV."""
    frame = pd.read_csv(path)
    required = {ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").astype(str)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip().str.lower()
    if not set(frame[TARGET_COLUMN].unique()).issubset({"positive", "negative", "neutral"}):
        raise ValueError("sentiment must contain positive, negative, and/or neutral labels")
    return frame


def binary_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the two target classes requested by the task; retain neutral in EDA."""
    binary = frame[frame[TARGET_COLUMN].isin(BINARY_LABELS)].copy()
    counts = binary[TARGET_COLUMN].value_counts()
    if len(counts) != 2 or counts.min() < 3:
        raise ValueError("At least three examples per binary class are needed for 3-fold CV")
    return binary.reset_index(drop=True)


def build_vectorizer() -> TfidfVectorizer:
    """Build a fold-safe TF-IDF vectorizer with a maximum of 5,000 terms."""
    return TfidfVectorizer(
        preprocessor=preprocess_text,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )


def build_pipeline(model_name: str) -> Pipeline:
    """Return a TF-IDF + MultinomialNB or LogisticRegression pipeline."""
    models = {
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(models)}")
    return Pipeline([("tfidf", build_vectorizer()), ("classifier", models[model_name])])


def evaluate_with_oof(frame: pd.DataFrame, n_splits: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    """Create stratified out-of-fold predictions and compact metric benchmark."""
    binary = binary_training_data(frame)
    X = binary[TEXT_COLUMN].tolist()
    y = binary[TARGET_COLUMN].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metric_rows: list[dict[str, float | str]] = []
    prediction_frame = binary[[ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN]].copy()
    fitted_models: dict[str, Pipeline] = {}
    for model_name in ("multinomial_nb", "logistic_regression"):
        pipeline = build_pipeline(model_name)
        predictions = cross_val_predict(pipeline, X, y, cv=splitter, method="predict")
        probabilities = cross_val_predict(pipeline, X, y, cv=splitter, method="predict_proba")
        class_order = sorted(set(y))
        positive_index = class_order.index("positive")
        positive_prob = probabilities[:, positive_index]
        prediction_frame[f"{model_name}_prediction"] = predictions
        prediction_frame[f"{model_name}_positive_probability"] = positive_prob
        metric_rows.append({
            "model": model_name,
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, pos_label="positive", zero_division=0),
            "recall": recall_score(y, predictions, pos_label="positive", zero_division=0),
            "f1": f1_score(y, predictions, pos_label="positive", zero_division=0),
            "roc_auc": roc_auc_score((y == "positive").astype(int), positive_prob),
            "average_precision": average_precision_score((y == "positive").astype(int), positive_prob),
        })
        fitted = build_pipeline(model_name)
        fitted.fit(X, y)
        fitted_models[model_name] = fitted
    return pd.DataFrame(metric_rows), prediction_frame, fitted_models


def save_models(models: dict[str, Pipeline], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")


def predict_sentiment(text: str | Iterable[str], model: Pipeline) -> pd.DataFrame:
    """Predict positive/negative sentiment and model confidence for custom text."""
    texts = [text] if isinstance(text, str) else list(text)
    probabilities = model.predict_proba(texts)
    classes = list(model.classes_)
    positive_index = classes.index("positive")
    predictions = model.predict(texts)
    confidence = probabilities.max(axis=1)
    return pd.DataFrame({
        "review_text": texts,
        "predicted_sentiment": predictions,
        "confidence": confidence,
        "positive_probability": probabilities[:, positive_index],
    })
