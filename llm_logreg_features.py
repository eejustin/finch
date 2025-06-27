"""
llm_logreg_features.py
Feature engineering utilities for converting LLM-extracted results into features for logistic regression or other ML models.
"""

import csv
import jsonlines

LLM_FIELDS = [
    "admission_of_fault",
    "eyewitness_or_video",
    "police_report_in_favor",
    "client_shares_fault",
    "injury_seriousness",
    "medical_bills_and_lost_wages",
    "insurance_exchanged",
    "injury_permanency",
    "property_damage",
    "pre_existing_conditions",
    "story_inconsistency"
]

ANSWER_MAP = {
    'yes': 2,
    'no': 0,
    'uncertain': 1,
    'temporary': 1,
    'permanent': 2,
    'severe': 3,
    'medium': 2,
    'light': 1
}
REVERSE_MAP = {
    'client_shares_fault',
    'pre_existing_conditions',
    'story_inconsistency'
}

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def encode_answer(field, answer):
    answer = answer.lower()
    if field == 'injury_seriousness':
        return ANSWER_MAP.get(answer, 1)
    if field == 'injury_permanency':
        if answer in ['permanent', 'temporary', 'uncertain']:
            return ANSWER_MAP.get(answer, 1)
        return 1
    if field in REVERSE_MAP:
        if answer == 'no':
            return 2
        elif answer == 'yes':
            return 0
        elif answer == 'uncertain':
            return 1
    return ANSWER_MAP.get(answer, 1)

def llm_output_to_features(llm_output, record_idx=None):
    """
    Convert LLM output (dict) to a feature vector (dict of field: int value)
    Optionally logs extraction and scoring for each attribute.
    """
    features = {}
    logging.info("\n--- Extracted LLM Output for Record %s ---", record_idx if record_idx is not None else "?")
    for field in LLM_FIELDS:
        answer = llm_output.get(field, {}).get('answer', 'uncertain')
        features[field] = encode_answer(field, answer)
        logging.info("%s: answer='%s' -> score=%s", field, answer, features[field])
    return features

def process_llm_outputs(input_jsonl, output_csv, label_field=None):
    """
    Reads a JSONL file of LLM outputs, writes a CSV of features for ML.
    Logs extraction and scoring for each record.
    If label_field is provided, include it as the last column.
    """
    with jsonlines.open(input_jsonl) as reader, open(output_csv, 'w', newline='') as csvfile:
        fieldnames = LLM_FIELDS + ([label_field] if label_field else [])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, obj in enumerate(reader):
            logging.info("\n================ Record %d ================", idx+1)
            row = llm_output_to_features(obj, record_idx=idx+1)
            if label_field:
                row[label_field] = obj.get(label_field, 0)
            writer.writerow(row)

def train_logistic_regression(features_csv, label_col, save_model_path=None):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib

    df = pd.read_csv(features_csv)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy: {:.3f}".format(acc))
    print(classification_report(y_test, y_pred))
    print("Feature Weights:")
    for fname, coef in zip(X.columns, model.coef_[0]):
        print(f"  {fname}: {coef:.3f}")

    if save_model_path:
        joblib.dump(model, save_model_path)
        print(f"Model saved to {save_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM feature engineering and logistic regression training.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser for feature extraction
    parser_extract = subparsers.add_parser('extract', help='Convert LLM extracted results to ML features CSV')
    parser_extract.add_argument('--input', required=True, help='Input JSONL file with LLM outputs')
    parser_extract.add_argument('--output', required=True, help='Output CSV file for ML features')
    parser_extract.add_argument('--label', required=False, help='Optional label field to include')

    # Subparser for training
    parser_train = subparsers.add_parser('train', help='Train logistic regression model on features CSV')
    parser_train.add_argument('--features', required=True, help='CSV file with ML features (output of extract)')
    parser_train.add_argument('--label', required=True, help='Name of the label column')
    parser_train.add_argument('--save_model', required=False, help='Path to save trained model (joblib)')

    args = parser.parse_args()

    if args.command == 'extract':
        process_llm_outputs(args.input, args.output, label_field=args.label)
    elif args.command == 'train':
        train_logistic_regression(args.features, args.label, args.save_model)
