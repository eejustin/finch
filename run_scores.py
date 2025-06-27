import argparse
import jsonlines
import csv
from lead_score import score_lead, score_lead_with_model, extract_key_elements_with_llm
import json
import re

def main():
    parser = argparse.ArgumentParser(description="Run LLM extraction and rank leads from raw transcript JSONL. Optionally score with ML model.")
    parser.add_argument('--input', required=True, help='Path to raw transcript JSONL file (e.g. transcripts_50.jsonl)')
    parser.add_argument('--output', default='ranked_leads.csv', help='Path to output ranked CSV file')
    parser.add_argument('--model', required=False, help='Path to trained model.joblib for ML-based scoring')
    args = parser.parse_args()

    leads = []
    print(f"[INFO] Reading transcripts from {args.input}")
    with jsonlines.open(args.input) as reader:
        for idx, record in enumerate(reader, 1):
            transcript = dict(record)
            # Step 1: LLM extraction (in-memory, no intermediate file)
            try:
                llm_result_str = extract_key_elements_with_llm(transcript)
                cleaned = re.sub(r'^```json|```$', '', llm_result_str.strip(), flags=re.MULTILINE).strip()
                try:
                    llm_result = json.loads(cleaned)
                except Exception:
                    import ast
                    try:
                        llm_result = ast.literal_eval(cleaned)
                    except Exception:
                        llm_result = {"raw_output": llm_result_str}
                # Merge LLM fields into transcript
                transcript.update(llm_result)
            except Exception as e:
                print(f"[WARN] LLM extraction failed for record {transcript.get('id', idx)}: {e}")
                transcript['llm_extract_error'] = str(e)
            # Step 2: Scoring
            if args.model:
                try:
                    model_result = score_lead_with_model(transcript, args.model)
                    transcript.update(model_result)
                except Exception as e:
                    print(f"[WARN] Model scoring failed for record {transcript.get('id', idx)}: {e}")
                    transcript['model_score_error'] = str(e)
            try:
                base_result = score_lead(transcript)
                transcript.update(base_result)
            except Exception as e:
                print(f"[WARN] Heuristic scoring failed for record {transcript.get('id', idx)}: {e}")
                transcript['score_error'] = str(e)
            leads.append(transcript)
            if idx % 10 == 0:
                print(f"[INFO] Processed {idx} transcripts...")

    if not leads:
        print("[ERROR] No leads processed. Exiting.")
        return

    # Decide which score to use for ranking
    score_field = 'model_score' if args.model else 'score'
    leads.sort(key=lambda x: x.get(score_field, 0), reverse=True)

    # Assign overall_rank (1 = best)
    for idx, obj in enumerate(leads, start=1):
        obj['overall_rank'] = idx

    # Compose fieldnames for CSV
    base_fields = ['call_id', 'overall_rank']
    extra_fields = []
    if any('model_score' in x for x in leads):
        extra_fields.append('model_score')
    if any('score' in x for x in leads):
        extra_fields.append('score')
    rest = [k for k in leads[0].keys() if k not in (base_fields + extra_fields)]
    fieldnames = base_fields + extra_fields + rest

    print(f"[INFO] Writing ranked leads to {args.output}")
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for obj in leads:
            writer.writerow({k: obj.get(k, '') for k in fieldnames})
    print(f"[INFO] Done. {len(leads)} leads scored and ranked.")

if __name__ == '__main__':
    main()
