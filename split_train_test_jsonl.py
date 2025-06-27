import os
import jsonlines
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(__file__)
INPUT_PATH = os.path.join(ROOT, 'filtered_transcripts.jsonl')
DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
TRAIN_OUT = os.path.join(DATA_DIR, 'train.jsonl')
TEST_OUT = os.path.join(DATA_DIR, 'test.jsonl')

# Read all records
records = []
with jsonlines.open(INPUT_PATH) as reader:
    for record in reader:
        records.append(record)

# Stratify if label exists, else random split
label_field = None
if records and any('label' in r for r in records):
    label_field = 'label'
    y = [r.get('label', 0) for r in records]
else:
    y = None

train_recs, test_recs = train_test_split(records, test_size=0.25, random_state=42, stratify=y if y else None)

with jsonlines.open(TRAIN_OUT, mode='w') as writer:
    for rec in train_recs:
        writer.write(rec)

with jsonlines.open(TEST_OUT, mode='w') as writer:
    for rec in test_recs:
        writer.write(rec)

print(f"Wrote {len(train_recs)} train records to {TRAIN_OUT}")
print(f"Wrote {len(test_recs)} test records to {TEST_OUT}")
