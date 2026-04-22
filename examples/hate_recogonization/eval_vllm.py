"""Evaluate a model served by vLLM via OpenAI-compatible API."""
import argparse
import json
import os
import sys
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from areal.dataset.hate_recognition import (
    HATE_RECOGNITION_PROMPT_PREFIX,
    HATE_RECOGNITION_PROMPT_PREFIX_FEWSHOT,
    _load_json_array,
)
from areal.reward.hate_recognition import (
    _extract_answer_region,
    _hard_match,
    _parse_quads,
    _soft_match,
)
from datasets import Dataset


def load_val_split(seed=42, train_ratio=0.9):
    records = _load_json_array(
        os.path.join(os.path.dirname(__file__), "train.json")
    )
    ds = Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(ds) * train_ratio)
    return ds.select(range(split_idx, len(ds)))


def aggregate_f1(preds, golds, matcher):
    tp = pred_total = gold_total = 0
    for pq, gq in zip(preds, golds):
        pred_total += len(pq)
        gold_total += len(gq)
        used = [False] * len(gq)
        for p in pq:
            for j, g in enumerate(gq):
                if not used[j] and matcher(p, g):
                    used[j] = True
                    tp += 1
                    break
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "pred_total": pred_total, "gold_total": gold_total,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    prefix = HATE_RECOGNITION_PROMPT_PREFIX_FEWSHOT if args.fewshot else HATE_RECOGNITION_PROMPT_PREFIX
    val = load_val_split()
    if args.limit:
        val = val.select(range(min(args.limit, len(val))))
    print(f"[eval-vllm] {len(val)} samples, {'3-shot' if args.fewshot else '0-shot'}")

    pred_quads_all, gold_quads_all = [], []
    empty_count = 0
    records = []

    for i, s in enumerate(val):
        prompt_text = prefix + s["content"]
        # Build chat-formatted prompt manually (Qwen3.5 chat template)
        full_prompt = (
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        resp = requests.post(
            f"{args.api_base}/completions",
            json={
                "model": args.model,
                "prompt": full_prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stop": ["<|im_end|>"],
            },
            timeout=120,
        )
        completion = resp.json()["choices"][0]["text"]

        pred_q = _parse_quads(_extract_answer_region(completion))
        gold_q = _parse_quads(_extract_answer_region(s["output"]))
        if pred_q is None:
            pred_q = []
        if gold_q is None:
            gold_q = []
        if len(pred_q) == 0:
            empty_count += 1
        pred_quads_all.append(pred_q)
        gold_quads_all.append(gold_q)
        records.append({
            "id": s.get("id"), "content": s["content"],
            "gold": s["output"], "completion": completion,
            "pred_count": len(pred_q), "gold_count": len(gold_q),
        })

        if (i + 1) % 20 == 0 or i == len(val) - 1:
            print(f"  ... {i + 1}/{len(val)} samples", flush=True)

    hard = aggregate_f1(pred_quads_all, gold_quads_all, _hard_match)
    soft = aggregate_f1(pred_quads_all, gold_quads_all, _soft_match)
    avg_f1 = (hard["f1"] + soft["f1"]) / 2

    print()
    print("=" * 70)
    print(f"Model:             {args.model}")
    print(f"Dev samples:       {len(val)}")
    print(f"Empty predictions: {empty_count} ({empty_count / len(val):.1%})")
    print(f"Predicted quads:   {hard['pred_total']}")
    print(f"Gold quads:        {hard['gold_total']}")
    print("-" * 70)
    print(f"HARD  P={hard['precision']:.4f}  R={hard['recall']:.4f}  F1={hard['f1']:.4f}  (TP={hard['tp']})")
    print(f"SOFT  P={soft['precision']:.4f}  R={soft['recall']:.4f}  F1={soft['f1']:.4f}  (TP={soft['tp']})")
    print("-" * 70)
    print(f"AVG F1 (官方排名分): {avg_f1:.4f}")
    print("=" * 70)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[eval-vllm] wrote {args.out}")


if __name__ == "__main__":
    main()
