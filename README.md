# Medical Text Classification

**Beyond the Shortcut: Medical Specialty Classification with and without Label-Revealing Keywords**

Academic NLP project at Instituto Superior Técnico, October 2026.

**Team:** Catalina Trajterman, Mathilde Saragoussi and Maria Oliveira Maia (Group 54; equal contributions as reported in the submission).

## Objective

Classify clinical notes into 12 medical specialties and investigate how models behave with and without keywords that explicitly reveal the target class.

The course data contains 2,669 labelled notes and 398 unlabelled test notes. A custom parser handles malformed records and creates a 90/10 stratified training/validation split (`random_state=42`): 2,402 training notes and 267 validation notes.

## Methods

- TF-IDF with Logistic Regression.
- TF-IDF with Complement Naive Bayes.
- A class-weighted Bidirectional LSTM.
- Random word deletion and random word swapping, evaluated separately.
- Experiments with and without the keywords field.

Augmentation creates two variants per training note, retaining the originals (7,206 training records). Validation notes are not augmented.

## Reported validation results

The selected submission model was **BiLSTM with random deletion and keywords**:

| Metric | Value |
| --- | ---: |
| Accuracy | 73.78% |
| Macro-F1 | 0.70 |
| Weighted F1 | 0.74 |
| Validation loss | 0.8279 |

These are the selected run's results recorded in the final report, not hidden-test scores. Neural training results may vary between runs and environments. Another swap run reached 74.16% accuracy, so the report does not establish consistent superiority of deletion over swapping.

### What we learned

- **Keywords reveal labels:** the first keyword matches the specialty in all 2,217 labelled notes containing keywords. Such features can encourage reliance on explicit label cues rather than clinical content.
- **Class imbalance matters:** Surgery accounts for 33.5% of notes, while Dermatology accounts for 0.9%. Macro-F1 and per-class scores complement accuracy.
- **Repeated text complicates evaluation:** many transcriptions occur more than once, often under different labels. The original split is note-level and does not group duplicates; performance should be interpreted with that limitation.
- **Missing keywords change the task:** the report finds substantial difficulty when these features are removed.
- The reported augmentation comparisons do not establish an improvement over an unaugmented baseline.

This is an academic classification experiment, not a clinically validated system.

Read the [original project report](reports/project-report.pdf) for the full analysis and comparison table.

## Files

| File | Purpose |
| --- | --- |
| `data_ingestion.py` | Parse raw records and create training/validation/test files |
| `data_analysis.py` | Explore class distribution, keywords and duplicate texts |
| `data_augmentation.py` | Generate deletion or swapping variants |
| `models2_logisticreg.py` | Train and evaluate TF-IDF Logistic Regression |
| `model3_NB.py` | Train and evaluate TF-IDF Complement Naive Bayes |
| `model1_BiLSTM.py` | Train/evaluate the neural model and generate test predictions |

The original Python filenames and modelling logic are preserved. Datasets, generated patient-text error tables, predictions and fitted models are excluded from this portfolio package.

## Setup

Use Python 3.10 or 3.11 and run all commands from the repository root. TensorFlow installation depends on the operating system and hardware; the ranges below are setup guidance, not a locked historical environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## Data and execution

The original course datasets are not redistributed. Obtain authorised copies of `train.csv` and `test_no_labels.csv` and place them in the repository root. The parser expects the original course format, with specialty labels on training records and fields for description, sample name, transcription and keywords. An arbitrary medical CSV is not a drop-in replacement.

Preprocess:

```bash
python data_ingestion.py
```

Create deletion augmentation and run the selected model:

```bash
python data_augmentation.py -d
python model1_BiLSTM.py -d --keywords
```

Compare the TF-IDF models with the same augmented training data:

```bash
python models2_logisticreg.py -d --keywords
python model3_NB.py -d --keywords
```

For swapping, first run `python data_augmentation.py -s`, then use `-s` instead of `-d` in a model command. Omit `--keywords` to exclude that field. Omit `-d`/`-s` to use original training records. Model runs may overwrite outputs from earlier runs; preserve results separately when comparing experiments.

## Portfolio verification

All six Python files were syntax-checked. Data ingestion and deletion augmentation were run locally using the supplied course files; they produced 2,402 training, 267 validation, 398 test and 7,206 augmented training records. A Complement Naive Bayes deletion/keywords run was also checked. The BiLSTM was not retrained for this portfolio package; its metrics above come from the final report.

## Contact

[Maria Oliveira Maia — LinkedIn](https://www.linkedin.com/in/mariaoliveiramaia/)
