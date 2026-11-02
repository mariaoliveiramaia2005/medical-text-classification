"""
Short data analysis
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ['Surgery', 'Cardiovascular-Pulmonary', 'Orthopedic', 'Radiology',
          'General Medicine', 'Gastroenterology', 'Neurology',
          'Obstetrics-Gynecology', 'Neurosurgery', 'Ophthalmology',
          'Psychiatry-Psychology', 'Dermatology']
SPECIALTY_WORDS = tuple(l.lower().replace('-', ' / ') for l in LABELS)
LABEL_STARTS = tuple(l + ';' for l in LABELS)

#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# uncomment in case you have an error when ingesting data and adapt the file path
print(os.getcwd())

def clean(text):
    """Remove only what we can prove is not content."""
    # remove unecessary whitespaces and alone ';' at the end of lines
    text = text.strip().rstrip(';').strip()

    # field wrapped in quotes: remove ONE pair, unescape ""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].replace('""', '"')
    # opening quote that is never closed
    elif text.startswith('"'):
        text = text[1:]
    return text.strip()


def fix_junk(line):
    """Some lines end with a junk ';;;'. Keep a single ';' (= empty last field)."""
    return line[:-2] if line.endswith(';;;') else line


def split_fields(line):
    """
    One record -> description, sample_name, transcription, keywords.
    Function used on both training and testing data sets 
    """
    parts = line.split(';', 2) # medical_specialty; description ; sample_name ; everything else
    if len(parts) < 3:# truncated record: only a description
        return clean(parts[0]), '', '', ''
    description, sample_name, rest = parts
    transcription, _, keywords = rest.rpartition(';')   # keywords = after the LAST ';'
    # description missing -> fields shifted left
    if transcription.strip().lower().startswith(SPECIALTY_WORDS): 
        return '', clean(description), clean(sample_name), clean(transcription)
    return clean(description), clean(sample_name), clean(transcription), clean(keywords)


def load_train(path):
    """
    Data ingestion of the traning set "train.csv"
    
    Input:
        path: path of the CSV file containing training data
    Output:
        pd.Dataframe with the formated and cleaned data for training
    """
    lines = open(path, encoding='utf-8', newline='').read().split('\r\n')[1:] # skip header
    records = []
    for line in lines:
        if line.startswith(LABEL_STARTS):# a record always starts with its label
            records.append(fix_junk(line))
        elif line: # else we continue the previous record
            records[-1] += ' ' + line
    rows = []
    for rec in records:
        label, rest = rec.split(';', 1)
        rows.append([label] + list(split_fields(rest)))
    return pd.DataFrame(rows, columns=['medical_specialty', 'description', 'sample_name',
                                       'transcription', 'keywords'])


def load_test(path):
    """
    Load testing data set that has no header and no medical_specialty column

    Input:
            path: path of the CSV file containing testing data
    Output:
        pd.Dataframe with the formated and cleaned data for testing
    """
    lines = open(path, encoding='utf-8', newline='').read().split('\r\n')
    records = []
    for line in lines:
        if line[:1].islower(): # keywords that wrapped onto a new line
            records[-1] += line.rstrip(';')
        else:
            records.append(fix_junk(line))
    rows = []
    for rec in records:
        if rec.startswith(LABEL_STARTS): # if a label is present -> drop it, we don't use it
            rec = rec.split(';', 1)[1]
        rows.append(split_fields(rec))
    return pd.DataFrame(rows, columns=['description', 'sample_name',
                                       'transcription', 'keywords'])

train = load_train('train.csv')
test = load_test('test_no_labels.csv')

# Class distribution
counts = train.medical_specialty.value_counts()
print("----------Class distribution---------")
print(counts.to_string())
print('ratio largest/smallest:', round(counts.max() / counts.min(), 1))
print('ratio Surgery/total:', round(100 * counts['Surgery'] / len(train), 1), '%')
print(len(train))
print()

# Keywords: how many notes have them, and what do they contain?
has_kw = train.keywords != ''
print("---------Keywords analysis----------")
print('notes with keywords: train', round(100 * has_kw.mean(), 1), '% | test',
      round(100 * (test.keywords != '').mean(), 1), '%')
first = train.keywords.str.split(',').str[0].str.strip().str.lower()
# "Obstetrics-Gynecology" -> "obstetrics / gynecology"
medical_specialty_as_text = train.medical_specialty.str.lower().str.replace('-', ' / ')
print('first keyword equals the medical_specialty (training set):',
       (first == medical_specialty_as_text)[has_kw].sum(), 'of', has_kw.sum())
# -> for every note that has a keyword, the first keyword is the exact medical speciality
# -> the model can predict the speciality using those first keywords
print('% of notes WITHOUT keywords, per class:')
print((100 * (~has_kw).groupby(train.medical_specialty).mean()).round().sort_values(ascending=False).to_string())
#  Surgery was the most represented class but here we see that it is not the class with the most important 
# % whitout keywords, in that class
print()

# Are notes without keywords different?
print("---------Notes whithout keywords----------")
words = train.transcription.str.split().str.len()
print('median words: without keywords', words[~has_kw].median(), 
      '| with keywords', words[has_kw].median())
# ->Notes without keywords are longer then the ones with keywords
print()

# text duplicates and medical speciality
print("---------Text duplicates----------")
#train
# Pre-processing to get transcriptions that are not empty + no whitespaces + lower case for comparaison
text = train.transcription.str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
t = train[text != ''].copy()
text_clean = text[text != '']
t['text'] = text_clean
# print(t)
notes_with_transcription = len(t)
print(f"Notes with transcription : {notes_with_transcription}")

n_medical_specialtys = t.groupby('text').medical_specialty.nunique()
n_copies = t.groupby('text').size()
print('Notes sharing their text with another note:',
      n_copies[n_copies > 1].sum(), 'of', notes_with_transcription)
print('repeated texts:', (n_copies > 1).sum(),
       '| with different medical_specialtys:', (n_medical_specialtys > 1).sum())

# test
test_text = test.transcription.str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
test_text = test_text[test_text != '']          # ignore empty transcriptions
print('test notes whose text is in train:',
       test_text.isin(set(t.text)).sum(), 'of', len(test)) # t.text: empty texts excluded
print()

# Figure: notes per class, split with / without keywords
order = counts.index[::-1]
with_kw = train[has_kw].medical_specialty.value_counts()[order]
no_kw = train[~has_kw].medical_specialty.value_counts()[order]
plt.figure(figsize=(3.3, 1.7))
plt.barh(order, with_kw, label='with keywords')
plt.barh(order, no_kw, left=with_kw, label='without keywords')
plt.yticks(fontsize=6); plt.xticks(fontsize=6); plt.legend(fontsize=6, frameon=False)
plt.xlabel('Training notes', fontsize=6); plt.tight_layout(pad=0.3)
plt.savefig('fig_class_distribution.pdf')


# print(len(test))
# print((test.keywords == '').sum())
# print(test.keywords)
