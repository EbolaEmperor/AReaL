"""Evaluate a checkpoint on the dev split using the official CCL25-Eval Task 10 metric:
hard-match F1 and soft-match F1 (corpus-level, sklearn-style), plus their average.

Usage:
    PYTHONPATH=/home/wenchong/code/AReaL python examples/hate_recogonization/eval.py \\
        [--ckpt PATH_TO_CHECKPOINT] [--limit N] [--out preds.jsonl]

If --ckpt is omitted, the latest checkpoint under /tmp/areal/experiments/checkpoints/
wenchong/hate-recognition-sft/trial0/default/ is used. Pass `--ckpt base` to evaluate
the un-finetuned base model (Qwen3-1.7B) for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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

BASE_MODEL = "/home/wenchong/grpo-2048/modelscope_cache/Qwen/Qwen3-1___7B"
DATA_DIR = "/home/wenchong/code/AReaL/examples/hate_recogonization"
SFT_CKPT_ROOT = (
    "/tmp/areal/experiments/checkpoints/wenchong/hate-recognition-sft/trial0/default"
)


def find_latest_checkpoint(root: str) -> str:
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"No checkpoint root: {root}")
    candidates = [d for d in root_p.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints under {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def load_val_split(seed: int = 42, train_ratio: float = 0.9) -> Dataset:
    records = _load_json_array(os.path.join(DATA_DIR, "train.json"))
    ds = Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(ds) * train_ratio)
    return ds.select(range(split_idx, len(ds)))


def aggregate_f1(preds: list[list], golds: list[list], matcher) -> dict:
    tp = pred_total = gold_total = 0
    for pred_quads, gold_quads in zip(preds, golds):
        pred_total += len(pred_quads)
        gold_total += len(gold_quads)
        used = [False] * len(gold_quads)
        for p in pred_quads:
            for j, g in enumerate(gold_quads):
                if not used[j] and matcher(p, g):
                    used[j] = True
                    tp += 1
                    break
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to checkpoint dir, or 'base' to use the un-finetuned base model. "
        "Default: latest SFT checkpoint.",
    )
    parser.add_argument("--fewshot", action="store_true",
                        help="Use the 3-shot prompt prefix instead of zero-shot.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Eval only first N samples (smoke test).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (for large MoE models).")
    parser.add_argument("--out", default=None,
                        help="Write per-sample predictions to this JSONL.")
    args = parser.parse_args()

    if args.ckpt is None:
        ckpt_path = find_latest_checkpoint(SFT_CKPT_ROOT)
        print(f"[eval] auto-selected ckpt: {ckpt_path}")
    elif args.ckpt == "base":
        ckpt_path = BASE_MODEL
        print(f"[eval] using BASE model: {ckpt_path}")
    else:
        ckpt_path = args.ckpt
        print(f"[eval] using ckpt: {ckpt_path}")

    print("[eval] loading tokenizer + model ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.load_in_4bit:
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: "25GiB", 1: "25GiB", "cpu": "64GiB"}
        print("[eval] loading quantized model with balanced device_map")
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(ckpt_path, **load_kwargs)
    if not args.load_in_4bit:
        model = model.to(args.device)
    model.eval()

    prompt_prefix = (
        HATE_RECOGNITION_PROMPT_PREFIX_FEWSHOT if args.fewshot
        else HATE_RECOGNITION_PROMPT_PREFIX
    )
    print(f"[eval] prompt variant: {'3-shot' if args.fewshot else 'zero-shot'}")

    val = load_val_split()
    if args.limit:
        val = val.select(range(min(args.limit, len(val))))
    print(f"[eval] dev set size: {len(val)}")

    pred_quads_all: list[list] = []
    gold_quads_all: list[list] = []
    parse_failures = 0
    raw_records: list[dict] = []

    with torch.no_grad():
        for i in range(0, len(val), args.batch_size):
            batch = val.select(range(i, min(i + args.batch_size, len(val))))
            prompts = []
            for s in batch:
                messages = [
                    {
                        "role": "user",
                        "content": prompt_prefix + s["content"],
                    }
                ]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(args.device)
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            input_len = enc["input_ids"].shape[1]
            for j, s in enumerate(batch):
                completion = tokenizer.decode(
                    out[j][input_len:], skip_special_tokens=True
                )
                pred_q = _parse_quads(_extract_answer_region(completion))
                gold_q = _parse_quads(_extract_answer_region(s["output"]))
                if pred_q is None:
                    pred_q = []
                if gold_q is None:
                    gold_q = []
                # Count as "empty pred" when model produced 0 quads —
                # the robust parser never returns None, so this is the right metric.
                if len(pred_q) == 0:
                    parse_failures += 1
                pred_quads_all.append(pred_q)
                gold_quads_all.append(gold_q)
                raw_records.append(
                    {
                        "id": s.get("id"),
                        "content": s["content"],
                        "gold": s["output"],
                        "completion": completion,
                        "pred_count": len(pred_q),
                        "gold_count": len(gold_q),
                    }
                )

            done = i + len(batch)
            print(f"  ... {done}/{len(val)} samples", flush=True)

    hard = aggregate_f1(pred_quads_all, gold_quads_all, _hard_match)
    soft = aggregate_f1(pred_quads_all, gold_quads_all, _soft_match)
    avg_f1 = (hard["f1"] + soft["f1"]) / 2

    print()
    print("=" * 70)
    print(f"Checkpoint:        {ckpt_path}")
    print(f"Dev samples:       {len(val)}")
    print(f"Empty predictions: {parse_failures} ({parse_failures / len(val):.1%})")
    print(f"Predicted quads:   {hard['pred_total']}")
    print(f"Gold quads:        {hard['gold_total']}")
    print("-" * 70)
    print(
        f"HARD  P={hard['precision']:.4f}  R={hard['recall']:.4f}  "
        f"F1={hard['f1']:.4f}  (TP={hard['tp']})"
    )
    print(
        f"SOFT  P={soft['precision']:.4f}  R={soft['recall']:.4f}  "
        f"F1={soft['f1']:.4f}  (TP={soft['tp']})"
    )
    print("-" * 70)
    print(f"AVG F1 (官方排名分): {avg_f1:.4f}")
    print("=" * 70)

    if args.out:
        out_path = args.out
        with open(out_path, "w", encoding="utf-8") as f:
            for r in raw_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = {
            "ckpt": ckpt_path,
            "n_samples": len(val),
            "parse_failures": parse_failures,
            "hard": hard,
            "soft": soft,
            "avg_f1": avg_f1,
        }
        with open(out_path.replace(".jsonl", ".summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[eval] wrote per-sample predictions to {out_path}")


if __name__ == "__main__":
    main()
