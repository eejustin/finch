import os
import jsonlines
import re
import numpy as np
import pandas as pd
from lead_score import extract_key_elements_with_llm
from llm_logreg_features import process_llm_outputs, LLM_FIELDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Paths
ROOT = os.path.dirname(__file__)
INPUT_PATH = os.path.join(ROOT, 'filtered_transcripts.jsonl')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
LLM_OUT = os.path.join(RESULTS_DIR, 'llm_extract_results.jsonl')
FEAT_OUT = os.path.join(RESULTS_DIR, 'features.csv')
MODEL_OUT = os.path.join(RESULTS_DIR, 'logreg_model.joblib')

LABEL_FIELD = 'label'

def map_status_to_label(status):
    """
    Map outcome_status to binary label for ML.
    Returns 1 for positive outcomes ('settled', 'trial'), else 0.
    """
    return 1 if str(status).strip().lower() in ['settled', 'trial'] else 0

# Step 1: Extract LLM results and produce features
print('Extracting LLM features from transcripts...')
with jsonlines.open(INPUT_PATH) as reader, jsonlines.open(LLM_OUT, mode='w') as writer:
    for record in reader:
        try:
            llm_result_str = extract_key_elements_with_llm(record)
            # Remove markdown formatting if present
            cleaned = re.sub(r'^```json|```$', '', llm_result_str.strip(), flags=re.MULTILINE).strip()
            try:
                llm_result = json.loads(cleaned)
            except Exception:
                import ast
                try:
                    llm_result = ast.literal_eval(cleaned)
                except Exception:
                    llm_result = {"raw_output": llm_result_str}
            if 'id' in record:
                llm_result['id'] = record['id']
            if 'outcome_status' in record:
                llm_result[LABEL_FIELD] = map_status_to_label(record['outcome_status'])
            else:
                llm_result[LABEL_FIELD] = 0  # default to 0 if missing
            writer.write(llm_result)
        except Exception as e:
            writer.write({"error": str(e), "record": record.get('id', None)})

print('Producing features CSV...')
process_llm_outputs(LLM_OUT, FEAT_OUT, label_field=LABEL_FIELD)

# Step 2: Train/test split and model training
print('Splitting data into train/test and training model...')
df = pd.read_csv(FEAT_OUT)
X = df[LLM_FIELDS]
y = df[LABEL_FIELD]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
joblib.dump(model, MODEL_OUT)
print(f'Model saved to {MODEL_OUT}')

# Step 3: Evaluation
print('Evaluating model on test set...')
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f'Accuracy: {acc:.3f}')
print(f'AUC: {auc:.3f}')
print(classification_report(y_test, y_pred))
print('Feature weights:')
for fname, coef in zip(X.columns, model.coef_[0]):
    print(f'  {fname}: {coef:.3f}')
