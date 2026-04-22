"""Generate submission file for test1.json using a trained LoRA checkpoint.

Usage:
    PYTHONPATH=~/code/AReaL python examples/hate_recogonization/generate_submission.py \
        --base-model /path/to/Qwen3.5-9B \
        --lora-path /path/to/lora/final \
        --test-file examples/hate_recogonization/test1.json \
        --output submission.txt
"""

import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from areal.dataset.hate_recognition import HATE_RECOGNITION_PROMPT_PREFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", default="submission.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print("[gen] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    with open(args.test_file, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"[gen] {len(test_data)} test samples")

    # Generate predictions in order (must match test file order for submission)
    results = {}
    with torch.no_grad():
        for i in range(0, len(test_data), args.batch_size):
            batch = test_data[i: i + args.batch_size]
            prompts = []
            for s in batch:
                messages = [
                    {"role": "user",
                     "content": HATE_RECOGNITION_PROMPT_PREFIX + s["content"]}
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
            ).to(model.device)

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
                ).strip()
                results[s["id"]] = completion

            done = i + len(batch)
            if done % 100 == 0 or done == len(test_data):
                print(f"  ... {done}/{len(test_data)}", flush=True)

    # Write submission: one line per test sample, in original order
    with open(args.output, "w", encoding="utf-8") as f:
        for s in test_data:
            line = results.get(s["id"], "[END]")
            # Ensure single line
            line = line.replace("\n", " ").strip()
            f.write(line + "\n")

    print(f"[gen] Submission written to {args.output} ({len(test_data)} lines)")


if __name__ == "__main__":
    main()
