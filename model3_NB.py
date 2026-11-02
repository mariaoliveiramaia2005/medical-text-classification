import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

DATASETS = {
    "original": "train_cleaned.csv",
    "deletion": "train_augmented_d.csv",
    "swap": "train_augmented_s.csv",
}
MODELS_DIR, RESULTS_DIR = Path("models"), Path("results")
TARGET = "medical_specialty"
TEST_SIZE, RANDOM_STATE = 0.20, 42
VAL_FILE = "val_cleaned.csv"

# Hyperparameters to tune
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 1)
MIN_DF = 2
ALPHA = 0.1


def parse_args():
    p = argparse.ArgumentParser(
        description="Naive Bayes specialty classifier (runs without and with keywords).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-d", "--deletion", action="store_true")
    g.add_argument("-s", "--swap", action="store_true")
    p.add_argument("--keywords", action="store_true",
                    help="Include keywords alongside the text fields. If impty no keywords.")

    return p.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in (TARGET, "description", "transcription") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=[TARGET]).copy()
    for c in ("description", "transcription", "sample_name", "keywords"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[(df[TARGET] != "") & (df["description"].str.strip().ne("")
                                  | df["transcription"].str.strip().ne(""))]
    print(f"Loaded {len(df)} records, {df[TARGET].nunique()} specialties.")
    return df


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
                                  min_df=MIN_DF, sublinear_tf=True)),
        ("clf", ComplementNB(alpha=ALPHA)),
    ])

def prepare_features(df, use_keywords=False):
    """Combine columns."""
    cols = ["description", "transcription"]
    if "sample_name" in df.columns:
        cols.insert(0, "sample_name")
    if use_keywords and "keywords" in df.columns:
        cols.append("keywords")

    X = df[cols].agg(" ".join, axis=1).str.strip()
    y = df[TARGET]
    return X, y

def main():
    args = parse_args()
    dataset = "deletion" if args.deletion else "swap" if args.swap else "original"
    print(f"Keywords: {'Yes' if args.keywords else 'No'}")

    df_train = load_data(DATASETS[dataset])
    df_val = load_data(VAL_FILE)

    X_train, y_train = prepare_features(df_train, use_keywords=args.keywords)
    X_val, y_val = prepare_features(df_val, use_keywords=args.keywords)

    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_val)
    classes = model.classes_

    metrics = {
        "model": "NaiveBayes", "dataset": dataset, "keywords_enabled": args.keywords,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }
    print(pd.Series(metrics).to_string())
    print(classification_report(y_val, y_pred, labels=classes, zero_division=0))

if __name__ == "__main__":
    main()
