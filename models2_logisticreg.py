import argparse
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report



DATASETS = {
    "original": Path("train_cleaned.csv"),
    "deletion": Path("train_augmented_d.csv"),
    "swap": Path("train_augmented_s.csv"),
}



TARGET = "medical_specialty"
VAL_FILE =Path("val_cleaned.csv")

RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a TF-IDF + Logistic Regression classifier "
            "with optional data augmentation and keywords."
        )
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-d",
        "--deletion",
        action="store_true",
        help="Train using the random deletion augmented dataset.",
    )

    group.add_argument(
        "-s",
        "--swap",
        action="store_true",
        help="Train using the random swap augmented dataset.",
    )

    parser.add_argument(
        "--keywords",
        action="store_true",
        help="Include keywords alongside description and transcription.",
    )

    return parser.parse_args()


def load_data(data_path):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    required_columns = [
        TARGET,
        "description",
        "transcription",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if "keywords" not in df.columns:
        print("Warning: 'keywords' column not found.")

    # Remove records without a target label.
    df = df.dropna(subset=[TARGET]).copy()

    for col in ["description", "transcription"]:
        df[col] = df[col].fillna("").astype(str)

    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].fillna("").astype(str)

    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != ""].copy()

    # Require at least one non-empty primary text field.
    has_text = (
        df["description"].str.strip().ne("")
        | df["transcription"].str.strip().ne("")
    )

    df = df[has_text].copy()

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "At least two different specialties are required."
        )

    print(f"Loaded {len(df)} records.")
    print(f"Specialties: {df[TARGET].nunique()}")

    return df

def prepare_features(df, use_keywords=False):
    """Select text field"""
    columns = ["description", "transcription"]

    if use_keywords:
        if "keywords" not in df.columns:
            raise ValueError(
                "--keywords was specified, but the dataset has no keywords column."
            )
        columns.append("keywords")

    X = df[columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()

    y = df[TARGET]

    return X, y


def train_model(df_train, df_val, args):
    X_train, y_train = prepare_features(df_train, use_keywords=args.keywords)
    X_val, y_val = prepare_features(df_val, use_keywords=args.keywords)

    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_val)}")

    # TF-IDF + Logistic Regression.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_val)


    
    # Classification report: metrics for every specialty.
    report_text = classification_report(
        y_val,
        y_pred,
        digits=2,
        zero_division=0,
    )
    print("\nClassification report:")
    print(report_text)



def main():
    args = parse_args()

    # Select the dataset.
    if args.deletion:
        dataset_name = "deletion"
    elif args.swap:
        dataset_name = "swap"
    else:
        dataset_name = "original"

    data_path = DATASETS[dataset_name]

    print(f"Dataset: {dataset_name}")
    print(f"Dataset path: {data_path}")

    print("\nExperiment:")
    print(f"Keywords: {'Yes' if args.keywords else 'No'}")

    if dataset_name == "original":
        print("Augmentation: No")
    else:
        print(f"Augmentation: Yes ({dataset_name})")

    df_train = load_data(data_path)
    df_val = load_data(VAL_FILE)

    train_model(df_train, df_val, args)


if __name__ == "__main__":
    main()