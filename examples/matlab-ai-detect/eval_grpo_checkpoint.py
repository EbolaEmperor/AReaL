"""
Evaluate a GRPO checkpoint on the held-out test split (last 20% of samples.jsonl).
Uses batched HuggingFace inference for compatibility.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 examples/matlab-ai-detect/eval_grpo_checkpoint.py
    CUDA_VISIBLE_DEVICES=0 python3 examples/matlab-ai-detect/eval_grpo_checkpoint.py --ckpt /path/to/checkpoint
"""

import argparse
import json
import os
import re

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# GRPO LoRA was trained on top of the SFT-merged model, so use that as base
BASE_MODEL = "/home/wenchong/code/AReaL/examples/matlab-ai-detect/sft-merged"
TOKENIZER_PATH = "/home/wenchong/grpo-2048/modelscope_cache/deepseek-ai/deepseek-coder-7b-instruct-v1___5"
GRPO_CKPT_DIR = "/tmp/areal/experiments/checkpoints/matlab-ai-detect-grpo"
DATA_PATH = os.path.join(os.path.dirname(__file__), "samples.jsonl")

SYSTEM_PROMPT = (
    "你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。\n\n"
    "请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。"
)
USER_TEMPLATE = (
    "请分析以下 MATLAB 代码，判断它是由 AI 生成的还是由人类手写的。\n"
    "请给出你的分析过程，并在最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "```matlab\n{code}\n```"
)

_VERDICT_RE = re.compile(r"<verdict>\s*(ai|human)\s*</verdict>", re.IGNORECASE)


def load_test_split(train_ratio: float = 0.8, seed: int = 42) -> list[dict]:
    records = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8") if l.strip()]
    ds = Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(ds) * train_ratio)
    test = [ds[i] for i in range(split_idx, len(ds))]
    print(f"Test split: {len(test)} samples (seed={seed}, last {100*(1-train_ratio):.0f}%)")
    return test


def evaluate(ckpt_path: str, tokenizer, test_samples: list[dict], batch_size: int = 8, max_new_tokens: int = 600):
    print(f"Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"Loading LoRA adapter from {ckpt_path}...", flush=True)
    model = PeftModel.from_pretrained(base, ckpt_path)
    model.eval()
    device = next(model.parameters()).device

    # Build all prompts
    prompts = []
    for sample in test_samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(code=sample["code"])},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    tokenizer.padding_side = "left"
    correct = parseable = 0
    per_label = {"ai": {"correct": 0, "total": 0}, "human": {"correct": 0, "total": 0}}

    with torch.no_grad():
        for i in range(0, len(test_samples), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_samples = test_samples[i:i + batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3500,
            ).to(device)

            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

            for j, (output_ids, sample) in enumerate(zip(out, batch_samples)):
                input_len = enc["input_ids"].shape[1]
                gen_ids = output_ids[input_len:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                label = sample["label"]
                per_label[label]["total"] += 1
                matches = _VERDICT_RE.findall(response)
                if len(matches) == 1:
                    verdict = matches[0].lower()
                    parseable += 1
                    if verdict == label:
                        correct += 1
                        per_label[label]["correct"] += 1

            if i % (batch_size * 10) == 0:
                print(f"  [{i}/{len(test_samples)}] correct={correct} parseable={parseable}", flush=True)

    del model, base
    torch.cuda.empty_cache()
    return correct, parseable, per_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Specific checkpoint path (default: latest)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    test_samples = load_test_split()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt_name = os.path.basename(ckpt_path)
    else:
        checkpoints = sorted(
            [d for d in os.listdir(GRPO_CKPT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        print(f"Found checkpoints: {checkpoints}")
        ckpt_name = checkpoints[-1]
        ckpt_path = os.path.join(GRPO_CKPT_DIR, ckpt_name)

    print(f"\n=== Evaluating {ckpt_name} ===", flush=True)
    correct, parseable, per_label = evaluate(ckpt_path, tokenizer, test_samples, batch_size=args.batch_size)
    n = len(test_samples)
    acc_str = f"{correct/parseable*100:.1f}%" if parseable else "N/A"
    print(
        f"\nRESULT [{ckpt_name}] accuracy={correct}/{parseable}({acc_str}) | "
        f"parseable={parseable}/{n} | "
        f"ai={per_label['ai']['correct']}/{per_label['ai']['total']} | "
        f"human={per_label['human']['correct']}/{per_label['human']['total']}"
    )


if __name__ == "__main__":
    main()
