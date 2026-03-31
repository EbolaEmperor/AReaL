"""
Evaluate each SFT checkpoint on 200 held-out samples.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 examples/matlab-ai-detect/eval_sft_checkpoints.py
"""

import json
import os
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "/home/wenchong/grpo-2048/modelscope_cache/deepseek-ai/deepseek-coder-7b-instruct-v1___5"
CHECKPOINT_DIR = "/tmp/areal/experiments/checkpoints/matlab-ai-detect-sft"
HELD_OUT_PATH = os.path.join(os.path.dirname(__file__), "held_out_200.jsonl")

SYSTEM_PROMPT = (
    "你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。\n\n"
    "请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>"
)
USER_TEMPLATE = (
    "请分析以下 MATLAB 代码，判断它是由 AI 生成的还是由人类手写的。\n"
    "请给出你的分析过程，并在最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "```matlab\n{code}\n```"
)

_VERDICT_RE = re.compile(r"<verdict>\s*(ai|human)\s*</verdict>", re.IGNORECASE)


def evaluate(ckpt_path, tokenizer, held_out, max_new_tokens=800):
    print(f"  Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"  Loading LoRA adapter from {ckpt_path}...", flush=True)
    model = PeftModel.from_pretrained(base, ckpt_path)
    model.eval()
    device = next(model.parameters()).device

    correct = parseable = 0
    per_label = {"ai": {"correct": 0, "total": 0}, "human": {"correct": 0, "total": 0}}

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for i, sample in enumerate(held_out):
            if i % 20 == 0:
                print(f"  [{i}/{len(held_out)}] correct={correct} parseable={parseable}", flush=True)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(code=sample["code"])},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(
                prompt,
                return_tensors="pt",
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

            gen_ids = out[0][enc["input_ids"].shape[1]:]
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

    tokenizer.padding_side = orig_side
    del model, base
    torch.cuda.empty_cache()
    return correct, parseable, per_label


def main():
    with open(HELD_OUT_PATH, encoding="utf-8") as f:
        held_out = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(held_out)} held-out samples")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoints = sorted([
        d for d in os.listdir(CHECKPOINT_DIR)
        if d.startswith("checkpoint-")
    ], key=lambda x: int(x.split("-")[1]))
    print(f"Checkpoints: {checkpoints}\n")

    # Only evaluate the last checkpoint
    checkpoints = checkpoints[-1:]

    for ckpt in checkpoints:
        epoch = int(ckpt.split("-")[1]) // (81 // 3)
        print(f"=== {ckpt} (epoch {epoch}) ===", flush=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
        correct, parseable, per_label = evaluate(ckpt_path, tokenizer, held_out)
        n = len(held_out)
        acc_str = f"{correct/parseable*100:.1f}%" if parseable else "N/A"
        print(
            f"RESULT [{ckpt}] accuracy={correct}/{parseable}({acc_str}) | "
            f"parseable={parseable}/{n} | "
            f"ai={per_label['ai']['correct']}/{per_label['ai']['total']} | "
            f"human={per_label['human']['correct']}/{per_label['human']['total']}\n"
        )

    print("Done.")


if __name__ == "__main__":
    main()
