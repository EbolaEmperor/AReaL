"""
Filter cold-start SFT data: remove samples where qwen3.5-plus predicted wrong verdict.

Usage:
    python examples/matlab-ai-detect/filter_cold_start.py \
        --input  examples/matlab-ai-detect/cold_start_sft.jsonl \
        --output examples/matlab-ai-detect/cold_start_sft_filtered.jsonl
"""

import argparse
import json
import os


def filter_file(input_path: str, output_path: str):
    total = correct = skipped = 0
    label_dist: dict[str, int] = {}

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            sample = json.loads(line)

            if not sample.get("is_correct"):
                skipped += 1
                continue

            label = sample.get("label", "unknown")
            label_dist[label] = label_dist.get(label, 0) + 1
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            correct += 1

    print(f"Input:   {total} samples")
    print(f"Kept:    {correct} samples (is_correct=True)")
    print(f"Removed: {skipped} samples (wrong verdict or missing)")
    print(f"Label distribution: {label_dist}")
    print(f"Output:  {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "cold_start_sft.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "cold_start_sft_filtered.jsonl"),
    )
    args = parser.parse_args()
    filter_file(args.input, args.output)
