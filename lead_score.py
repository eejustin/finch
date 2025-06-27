"""
lead_score.py
Module for scoring personal injury case transcripts and extracting key elements using LLM.
"""

import json
import openai
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def extract_key_elements_with_llm(transcript, prompt_placeholder=None):
    """
    Uses OpenAI's ChatGPT to extract key elements from a transcript.
    The prompt can be customized; currently a placeholder is used.
    Returns structured output from LLM.
    """
    config = load_config()
    openai.api_key = config['openai_api_key']

    # Detailed prompt for extraction
    prompt = prompt_placeholder or (
        "You are an expert legal intake analyst. Read the following call transcript between a potential client and an intake specialist. Extract the following information and return your response as a structured JSON object with clear, concise answers for each item:\n\n"
        "1. Is there a clear admission of fault by the other party? (yes/no/uncertain, with explanation)\n"
        "2. Are there eyewitness accounts or video evidence? (yes/no/uncertain, with explanation)\n"
        "3. Is there a police report in the client's favor? (yes/no/uncertain, with explanation)\n"
        "4. Is there any suggestion that the client shares some fault? (yes/no/uncertain, with explanation)\n"
        "5. What is the level of seriousness of the injury? (light/medium/severe, with explanation)\n"
        "6. Are there any medical bills and loss of wages? (yes/no/uncertain, with explanation)\n"
        "7. Has the client exchanged insurance information with the party at fault? (yes/no/uncertain, with explanation)\n"
        "8. What is the permanency of the injury? (temporary/permanent/uncertain, with explanation)\n"
        "9. Was there any property damage? (yes/no/uncertain, with explanation)\n"
        "10. Are there any pre-existing conditions that could also contribute to the injury? (yes/no/uncertain, with explanation)\n"
        "11. Are there any inconsistencies in the client's story? (yes/no/uncertain, with explanation)\n\n"
        "Return your answer in the following JSON format:\n\n"
        "{\n"
        "  \"admission_of_fault\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"eyewitness_or_video\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"police_report_in_favor\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"client_shares_fault\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"injury_seriousness\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"medical_bills_and_lost_wages\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"insurance_exchanged\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"injury_permanency\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"property_damage\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"pre_existing_conditions\": {\"answer\": \"\", \"explanation\": \"\"},\n"
        "  \"story_inconsistency\": {\"answer\": \"\", \"explanation\": \"\"}\n"
        "}\n\n"
    )
    transcript_blocks = transcript.get('transcript', [])
    transcript_text = format_transcript_for_llm(transcript_blocks)

    client = openai.OpenAI(api_key=config['openai_api_key'])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert legal intake analyst."},
            {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript_text}"}
        ],
        max_tokens=512,
        temperature=0.2
    )
    return response.choices[0].message.content

def format_transcript_for_llm(transcript_blocks):
    """
    Accepts a list of {speaker, text} dicts and returns a readable string for LLM input.
    """
    return "\n".join(f"{block['speaker']}: {block['text']}" for block in transcript_blocks)

# Logistic regression feature engineering utilities are now in llm_logreg_features.py
from llm_logreg_features import (
    LLM_FIELDS, encode_answer, llm_output_to_features, process_llm_outputs
)

def score_lead(transcript):
    """
    Accepts a transcript object (dict) and returns a scalar or structured score.
    Default: Returns transcript length as score. Use score_lead_with_model for ML-based scoring.
    """
    transcript_blocks = transcript.get('transcript', [])
    transcript_text = format_transcript_for_llm(transcript_blocks)
    score = len(transcript_text)
    return {
        'score': score
    }

def score_lead_with_model(transcript, model_path):
    """
    Accepts a transcript object (dict) and a path to a trained logistic regression model.
    Extracts features using llm_output_to_features and returns the model's score/probability.
    """
    import joblib
    import numpy as np
    # Assume transcript already has extracted LLM output fields
    features = llm_output_to_features(transcript)
    X = np.array([features[field] for field in LLM_FIELDS]).reshape(1, -1)
    model = joblib.load(model_path)
    prob = model.predict_proba(X)[0,1] if hasattr(model, 'predict_proba') else model.decision_function(X)[0]
    return {
        'model_score': float(prob)
    }

# Optional: Scriptable entry point
if __name__ == "__main__":
    import argparse
    import jsonlines
    import ast
    parser = argparse.ArgumentParser(description="LLM extraction, ML feature conversion, and ML scoring.")
    parser.add_argument('--input', required=True, help='Input JSONL file with transcripts or LLM outputs')
    parser.add_argument('--output', required=True, help='Output CSV file for ML features or scored results')
    parser.add_argument('--label', required=False, help='Optional label field to include')
    parser.add_argument('--save-intermediate', required=False, help='Optional JSONL file to save LLM extraction results')
    parser.add_argument('--batch-extract', action='store_true', help='Run LLM extraction on input transcripts and save intermediate results')
    parser.add_argument('--score-with-model', required=False, help='Path to trained model to score cases (input must have LLM fields)')
    args = parser.parse_args()

    if args.score_with_model:
        # Score input JSONL (with LLM fields) using trained model, write results
        import joblib
        import numpy as np
        with jsonlines.open(args.input) as reader, open(args.output, 'w') as outfile:
            model = joblib.load(args.score_with_model)
            for record in reader:
                features = llm_output_to_features(record)
                X = np.array([features[field] for field in LLM_FIELDS]).reshape(1, -1)
                prob = model.predict_proba(X)[0,1] if hasattr(model, 'predict_proba') else model.decision_function(X)[0]
                scored = dict(record)
                scored['model_score'] = float(prob)
                outfile.write(json.dumps(scored) + '\n')
    elif args.batch_extract and args.save_intermediate:
        # Batch extract LLM results and save to JSONL
        with jsonlines.open(args.input) as reader, jsonlines.open(args.save_intermediate, mode='w') as writer:
            for record in reader:
                try:
                    llm_result_str = extract_key_elements_with_llm(record)
                    # Try to parse as dict, fallback to string if not JSON
                    try:
                        llm_result = json.loads(llm_result_str)
                    except Exception:
                        # Sometimes LLM returns single quoted or invalid JSON, try ast.literal_eval
                        try:
                            llm_result = ast.literal_eval(llm_result_str)
                        except Exception:
                            llm_result = {"raw_output": llm_result_str}
                    # Optionally keep transcript id or metadata
                    if 'id' in record:
                        llm_result['id'] = record['id']
                    writer.write(llm_result)
                except Exception as e:
                    writer.write({"error": str(e), "record": record.get('id', None)})
        # Now use intermediate file as input for feature extraction
        process_llm_outputs(args.save_intermediate, args.output, label_field=args.label)
    else:
        # Default: process input as LLM output JSONL
        process_llm_outputs(args.input, args.output, label_field=args.label)
