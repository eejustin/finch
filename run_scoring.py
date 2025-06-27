import argparse
import jsonlines


def score_case(transcript):
    """Basic scoring logic for demonstration. Replace with real logic."""
    # Example: score based on transcript length
    return len(transcript.get('transcript', ''))


def main():
    parser = argparse.ArgumentParser(description="Score personal injury case transcripts.")
    parser.add_argument('--input', required=True, help='Path to input JSONL file')
    parser.add_argument('--output', required=True, help='Path to output JSONL file with scores')
    args = parser.parse_args()

    results = []
    with jsonlines.open(args.input) as reader:
        for obj in reader:
            score = score_case(obj)
            obj['score'] = score
            results.append(obj)

    with jsonlines.open(args.output, mode='w') as writer:
        for item in results:
            writer.write(item)

if __name__ == '__main__':
    main()
