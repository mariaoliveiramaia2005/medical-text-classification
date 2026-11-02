import argparse
import random
import pandas as pd


RANDOM_STATE = 42
AUGMENTATIONS_PER_ROW = 2
AUGMENTATION_PROBABILITY = 0.05

INPUT_FILE = "train_cleaned.csv"

TEXT_COLUMNS = ["transcription", "description"]
TARGET_COLUMN = "medical_specialty"


parser = argparse.ArgumentParser(
    description=(
        "Generate augmented training data using "
        "random deletion or random swap."
    )
)

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument(
    "-d",
    "--deletion",
    action="store_const",
    const="deletion",
    dest="method",
    help="Use random deletion."
)

group.add_argument(
    "-s",
    "--swap",
    action="store_const",
    const="swap",
    dest="method",
    help="Use random swap."
)

args = parser.parse_args()

# Select the augmentation method
AUGMENTATION_METHOD = args.method

# Generate a different output filename for each method
OUTPUT_SUFFIX = {
    "deletion": "d",
    "swap": "s"
}[AUGMENTATION_METHOD]

OUTPUT_FILE = (
    f"train_augmented_{OUTPUT_SUFFIX}.csv"
)

random.seed(RANDOM_STATE)

df = pd.read_csv(INPUT_FILE)

required_columns = TEXT_COLUMNS + [TARGET_COLUMN]
missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Remove rows without a target label
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Replace missing text values with empty strings
for col in TEXT_COLUMNS:
    df[col] = df[col].fillna("").astype(str)

# Mark original examples
df["is_augmented"] = False


def random_deletion(text, probability=0.05):
    """
    Randomly removes words with the given probability.

    Texts shorter than 10 words remain unchanged.
    If more than half the words would be removed,
    the original text is returned.
    """
    words = text.split()

    if len(words) < 10:
        return text

    kept_words = [
        word for word in words
        if random.random() > probability
    ]

    # Avoid removing too many words
    if len(kept_words) < max(1, int(len(words) * 0.5)):
        return text

    return " ".join(kept_words)


def random_swap(text, n_swaps=1):
    """
    Randomly swaps pairs of words.

    Texts shorter than 10 words remain unchanged.
    """
    words = text.split()

    if len(words) < 10:
        return text

    for _ in range(n_swaps):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]

    return " ".join(words)


def augment_text(text):
    if not isinstance(text, str) or not text.strip():
        return text

    if AUGMENTATION_METHOD == "deletion":
        return random_deletion(
            text,
            probability=AUGMENTATION_PROBABILITY
        )

    elif AUGMENTATION_METHOD == "swap":
        return random_swap(
            text,
            n_swaps=1
        )

    raise ValueError(
        f"Unknown augmentation method: {AUGMENTATION_METHOD}"
    )

augmented_rows = []

for _, row in df.iterrows():

    for _ in range(AUGMENTATIONS_PER_ROW):

        new_row = row.copy()

        # Augment only the text columns
        for col in TEXT_COLUMNS:
            new_row[col] = augment_text(row[col])

        # Preserve the original label and other columns
        new_row["is_augmented"] = True

        augmented_rows.append(new_row)

augmented_df = pd.DataFrame(
    augmented_rows,
    columns=df.columns
)


final_df = pd.concat(
    [df, augmented_df],
    ignore_index=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Augmentation method:", AUGMENTATION_METHOD)
print("Original rows:", len(df))
print("Generated augmented rows:", len(augmented_df))
print("Total rows in final dataset:", len(final_df))
print("Saved to:", OUTPUT_FILE)
