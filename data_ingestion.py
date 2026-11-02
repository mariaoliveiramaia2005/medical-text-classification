import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

LABELS = ["Surgery", "Cardiovascular-Pulmonary","Orthopedic","Radiology","General Medicine",
        "Gastroenterology", "Neurology", "Obstetrics-Gynecology", "Neurosurgery", "Ophthalmology",
        "Psychiatry-Psychology", "Dermatology",
]

SPECIALTY_WORDS = tuple(l.lower().replace('-', ' / ') for l in LABELS)

LABEL_STARTS = tuple(l + ';' for l in LABELS)


def clean(text):
    """Remove only what we can prove is not content."""
    # remove unecessary whitespaces and alone ';' at the end of lines
    text = text.strip().rstrip(";").strip()

    # field wrapped in quotes: remove ONE pair, unescape ""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].replace('""', '"')
    # opening quote that is never closed
    elif text.startswith('"'):
        text = text[1:]
    return text.strip()


def fix_junk(line):
    """Fix records ending with an extra ';;;'.  Keep a single ';' (= empty last field)."""
    return line[:-2] if line.endswith(";;;") else line


def split_fields(line):
    """
    One record -> description, sample_name, transcription, keywords.
    Function used on both training and testing data sets 
    """
    # medical_specialty; description ; sample_name ; everything else
    parts = line.split(";", 2)

    if len(parts) < 3:
        return clean(parts[0]), "", "", ""

    description, sample_name, rest = parts
    # keywords = after the LAST ';'
    transcription, _, keywords = rest.rpartition(";")

    # description missing -> fields shifted left
    if transcription.strip().lower().startswith(SPECIALTY_WORDS):
        return (
            "",
            clean(description),
            clean(sample_name),
            clean(transcription),
        )

    return (
        clean(description),
        clean(sample_name),
        clean(transcription),
        clean(keywords),
    )


def load_train(path):
    """
    Load and clean the labeled training dataset.
    """
    with open(path, encoding="utf-8", newline="") as file:
        lines = file.read().splitlines()

    # Skip the header.
    lines = lines[1:]

    records = []

    for line in lines:
        if line.startswith(LABEL_STARTS):
            records.append(fix_junk(line))
        elif line.strip() and records:
            # Join records that continue on the next line.
            records[-1] += " " + line

    rows = []
    for record in records:
        label, rest = record.split(";", 1)
        rows.append([label] + list(split_fields(rest)))

    return pd.DataFrame(
        rows,
        columns=["medical_specialty", "description", "sample_name",
            "transcription", "keywords"])


def load_test(path):
    """
    Load and clean the unlabeled test dataset.
    """
    with open(path, encoding="utf-8", newline="") as file:
        lines = file.read().splitlines()

    records = []

    for line in lines:
        if not line.strip():
            continue

        # Some keywords continue on the next line.
        if line[:1].islower() and records:
            records[-1] += line.rstrip(";")
        else:
            records.append(fix_junk(line))

    rows = []

    for record in records:
        # If a label appears accidentally in the test file,
        # remove it to avoid using it as model input.
        if record.startswith(LABEL_STARTS):
            record = record.split(";", 1)[1]

        rows.append(split_fields(record))

    return pd.DataFrame(
        rows,
        columns=[ "description", "sample_name", 
                 "transcription","keywords"]
    )


def main():
    """
    Import data and returns cleaned csv files test and training 
    """
    # Uncomment in case of errors bc can't find the file
    # print("Working directory:", os.getcwd())

    train_path = "train.csv"
    test_path = "test_no_labels.csv"

    df_train_full = load_train(train_path)
    df_test = load_test(test_path)


    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.10,
        random_state=SEED,
        stratify=df_train_full["medical_specialty"]
    )

    df_train.to_csv("train_cleaned.csv", index=False, encoding="utf-8")
    df_val.to_csv("val_cleaned.csv", index=False, encoding="utf-8")

    df_test.to_csv("test_cleaned.csv", index=False, encoding="utf-8")

    print("\nTraining dataset saved to train_cleaned.csv")
    print("\nValidation dataset saved to val_cleaned.csv")
    # print("Training shape:", df_train.shape)
    # print("Specialty distribution:")
    # print(df_train["medical_specialty"].value_counts())
    # print("Training preview:")
    # print(df_train.head())

    print("\nTest dataset saved to test_cleaned.csv")
    # print("Test shape:", df_test.shape)
    # print("Test preview:")
    # print(df_test.head())


if __name__ == "__main__":
    main()
