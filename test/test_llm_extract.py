import jsonlines
import os
from lead_score import extract_key_elements_with_llm

INPUT_PATH = os.path.join(os.path.dirname(__file__), '../filtered_transcripts.jsonl')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
OUTPUT_INTERMEDIATE = os.path.join(RESULTS_DIR, 'llm_extract_results.jsonl')
OUTPUT_FEATURES = os.path.join(RESULTS_DIR, 'features.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # Step 1: Read first 5 records
    records = []
    with jsonlines.open(INPUT_PATH) as reader:
        for idx, record in enumerate(reader):
            if idx >= 5:
                break
            records.append(record)
    import re
    # Step 2: Extract LLM results and save intermediate
    def parse_llm_json(raw_str):
        # Remove markdown formatting if present
        cleaned = re.sub(r'^```json|```$', '', raw_str.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            import ast
            try:
                return ast.literal_eval(cleaned)
            except Exception:
                return {"raw_output": raw_str}

    with jsonlines.open(OUTPUT_INTERMEDIATE, mode='w') as writer:
        for record in records:
            try:
                llm_result_str = extract_key_elements_with_llm(record)
                llm_result = parse_llm_json(llm_result_str)
                if 'id' in record:
                    llm_result['id'] = record['id']
                writer.write(llm_result)
            except Exception as e:
                writer.write({"error": str(e), "record": record.get('id', None)})
    # Step 3: Produce features.csv
    from llm_logreg_features import process_llm_outputs
    process_llm_outputs(OUTPUT_INTERMEDIATE, OUTPUT_FEATURES)
    print(f"Extracted results saved to: {OUTPUT_INTERMEDIATE}")
    print(f"Features saved to: {OUTPUT_FEATURES}")

if __name__ == "__main__":
    main()
