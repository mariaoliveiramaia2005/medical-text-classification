import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


DATASETS = {
    "original": "train_cleaned.csv",
    "deletion": "train_augmented_d.csv",
    "swap": "train_augmented_s.csv",
}

VAL_FILE = "val_cleaned.csv"
TEST_FILE = "test_cleaned.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Bidirectional LSTM for medical specialty classification."
    )

    augmentation_group = parser.add_mutually_exclusive_group()

    augmentation_group.add_argument(
        "-d",
        "--deletion",
        action="store_true",
        help="Use the dataset augmented with random deletion.",
    )

    augmentation_group.add_argument(
        "-s",
        "--swap",
        action="store_true",
        help="Use the dataset augmented with random swap.",
    )

    parser.add_argument(
        "--keywords",
        action="store_true",
        help="Include the keywords column in the input text.",
    )

    return parser.parse_args()

def train_model(training_set, val_set, test_set, args):
    """
    Train a Bidirectional LSTM to classify medical specialties
    and predict labels for the unlabeled test dataset.
    """

    columns_keywords = [
        "description",
        "sample_name",
        "transcription",
        "keywords",
    ]

    columns = [
        "description",
        "sample_name",
        "transcription",
    ]

    def combine_text(data, args):
        if args.keywords:
            text_columns = columns_keywords
        else:
            text_columns = columns

        missing_columns = [
            column for column in text_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in dataset: {missing_columns}"
            )

        return (
            data[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )

    training_set = training_set.dropna(subset=["medical_specialty"]).copy()
    val_set = val_set.dropna(subset=["medical_specialty"]).copy()

    X_train = combine_text(training_set, args).to_numpy()
    X_val = combine_text(val_set, args).to_numpy()
    X_test_real = combine_text(test_set, args).to_numpy()


    label_encoder = LabelEncoder()

    label_encoder.fit(
        pd.concat([training_set["medical_specialty"], val_set["medical_specialty"]])
        .astype(str)
    )

    y_train = label_encoder.transform(training_set["medical_specialty"].astype(str))
    y_val = label_encoder.transform(val_set["medical_specialty"].astype(str))

    num_classes = len(label_encoder.classes_)

    print("Number of classes:", num_classes)
    print("Classes:", list(label_encoder.classes_))

    max_words = 30000
    max_len = 1000

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=max_words,
        oov_token="<OOV>",
    )

    # Fit only on the training partition to avoid vocabulary leakage.
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test_real)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )

    X_val_pad = pad_sequences(
        X_val_seq,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )

    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )

    vocab_size = min(
        max_words,
        len(tokenizer.word_index) + 1,
    )

    print("Vocabulary size:", vocab_size)
    print("Training shape:", X_train_pad.shape)
    print("Validation shape:", X_val_pad.shape)
    print("Test shape:", X_test_pad.shape)


    model = Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=128,
            mask_zero=True,
        ),
        layers.Bidirectional(
            layers.LSTM(64, dropout=0.2)
        ),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=5,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1,
    )

    val_loss, val_accuracy = model.evaluate(
        X_val_pad,
        y_val,
        verbose=0,
    )

    print(f"\nValidation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    y_val_pred = np.argmax(
        model.predict(X_val_pad, verbose=0),
        axis=1,
    )
    
    val_probs = model.predict(X_val_pad, verbose=0)

    errors = pd.DataFrame({
        "text": X_val,
        "dataset_label": label_encoder.inverse_transform(y_val),
        "predicted_label": label_encoder.inverse_transform(y_val_pred),
        "confidence": val_probs.max(axis=1),
        "has_keywords": val_set["keywords"].notna().to_numpy(),
    })

    errors = errors[
        errors["dataset_label"] != errors["predicted_label"]
    ].copy()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    errors.to_csv(f"{RESULTS_DIR}/val_errors.csv", index=False)
    print(f"Misclassified validation notes: {len(errors)}")

    # Confusion matrix on the validation set
    cm = confusion_matrix(
        y_val,
        y_val_pred,
        labels=np.arange(num_classes),
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_,
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )

    ax.set_title("Confusion Matrix - BiLSTM")
    plt.tight_layout()

    # Create results directory if it does not exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.savefig(
        f"{RESULTS_DIR}/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("Confusion matrix saved to results/confusion_matrix.png")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            y_val_pred,
            labels=np.arange(num_classes),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    # Predict labels for the test set
    test_probabilities = model.predict(
        X_test_pad,
        verbose=0,
    )

    predicted_codes = np.argmax(
        test_probabilities,
        axis=1,
    )

    predictions = label_encoder.inverse_transform(
        predicted_codes
    )

    # Save results.txt: one predicted label per line, no header
    with open("results.txt", "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    print(f"Predictions saved to results.txt")
    print(f"Number of predictions: {len(predictions)}")
    print("First predictions:")
    print(predictions[:5])

    results = test_set.copy()

    results["predicted_medical_specialty"] = predictions
    results["prediction_confidence"] = np.max(
        test_probabilities,
        axis=1,
    )

    return model, tokenizer, label_encoder, results

def main():
    args = parse_args()

    # Select dataset
    if args.deletion:
        dataset_name = "deletion"
    elif args.swap:
        dataset_name = "swap"
    else:
        dataset_name = "original"

    data_path = DATASETS[dataset_name]

    training_set = pd.read_csv(data_path)
    val_set = pd.read_csv(VAL_FILE)
    test_set = pd.read_csv(TEST_FILE)

    # Print experiment configuration
    print("\nExperiment:")
    print(f"Keywords: {'Yes' if args.keywords else 'No'}")

    if dataset_name == "original":
        print("Augmentation: No")
    else:
        print(f"Augmentation: Yes ({dataset_name})")

    print(f"Training dataset: {data_path}")

    train_model(training_set, val_set, test_set, args)  

if __name__ == "__main__":
    main()
