# Finch: Personal Injury Case Scoring & ML Pipeline

This repository provides a robust end-to-end pipeline for scoring personal injury case leads using a combination of LLM-based information extraction and logistic regression modeling. It is designed for law firms or legal tech teams to prioritize, score, and evaluate personal injury leads from call transcripts.

## Features
- **LLM Extraction**: Extracts structured key elements from call transcripts using OpenAI GPT models.
- **Feature Engineering**: Converts LLM outputs into numeric features for ML.
- **Model Training**: Trains a logistic regression model to predict case outcomes.
- **Batch Scoring**: Scores new or existing leads using the trained model.
- **Dataset Splitting**: Supports proper train/test splits for ML workflows.

## Installation
```bash
pip install -r requirements.txt
```

## Workflow Overview
1. **Prepare Data**: Place your filtered call transcripts in `filtered_transcripts.jsonl` (one JSON object per line, each with a transcript and an `outcome_status` field).
2. **Split Data**: Split your data into train/test sets:
   ```bash
   python split_train_test_jsonl.py
   ```
   This creates `data/train.jsonl` and `data/test.jsonl`.
3. **End-to-End Training & Evaluation**: Run the full pipeline:
   ```bash
   python train_and_eval_logreg.py
   ```
   This will:
   - Extract LLM features from transcripts
   - Generate intermediate LLM and feature files in `results/`
   - Train a logistic regression model
   - Print evaluation metrics (accuracy, AUC, classification report)
   - Save the trained model to `results/logreg_model.joblib`
4. **Score New Leads**: Use the trained model to score new transcripts:
   ```bash
   python lead_score.py --input new_transcripts.jsonl --score-with-model results/logreg_model.joblib --output scored_leads.jsonl
   ```

## Key Scripts
- `split_train_test_jsonl.py` — Splits your dataset into train/test JSONL files, stratified by outcome.
- `train_and_eval_logreg.py` — Runs LLM extraction, feature engineering, model training, and evaluation end-to-end.
- `llm_logreg_features.py` — Feature engineering utilities and CLI for extracting features and training models.
- `lead_score.py` — Scores transcripts using either the ML model or fallback heuristics.

## Project Structure
- `filtered_transcripts.jsonl` — Source data (call transcripts + outcome_status)
- `data/` — Contains train/test splits
- `results/` — Contains intermediate LLM outputs, feature CSV, and trained model
- `llm_logreg_features.py` — Feature engineering and model training utilities
- `train_and_eval_logreg.py` — Main pipeline script
- `split_train_test_jsonl.py` — Dataset splitting utility
- `lead_score.py` — Batch scoring utility
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## Labeling Convention
- The label for ML is derived from the `outcome_status` field in each transcript:
  - `settled` or `trial` → **1** (positive outcome)
  - All other statuses (e.g., `dropped`) → **0** (negative outcome)

## Example Data Record
```json
{
  "id": "123...",
  "transcript": [...],
  "outcome_status": "settled",
  "outcome_value": 15000
}
```

## Requirements
- Python 3.8+
- OpenAI API key (for LLM extraction; set via `config.json` or environment variable)

## Getting Help
If you have questions or want to extend the pipeline (e.g., add new features, try different ML models, or automate more steps), please open an issue or contact the maintainers.
