"""
Generate cold-start SFT data for matlab-ai-detect using qwen3.5-plus (thinking mode).

Usage:
  # Test with 10 samples first:
  python examples/matlab-ai-detect/gen_cold_start.py --test

  # Generate full cold-start dataset:
  python examples/matlab-ai-detect/gen_cold_start.py --n 500 --output examples/matlab-ai-detect/cold_start_sft.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_URL = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = "sk-sp-20282b4e0dac480eb5ff908a0a49717e"
MODEL = "qwen3.5-plus"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "samples.jsonl")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。

请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：
<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>

注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。"""

USER_TEMPLATE = (
    "请分析以下 MATLAB 代码，判断它是由 AI 生成的还是由人类手写的。\n"
    "请给出你的分析过程，并在最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "```matlab\n{code}\n```"
)

_VERDICT_RE = re.compile(r"<verdict>\s*(ai|human)\s*</verdict>", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.IGNORECASE)

client = OpenAI(api_key=API_KEY, base_url=API_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_samples(data_path: str = DATA_PATH) -> list[dict]:
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_code(code: str) -> str:
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else code
        if code.endswith("```"):
            code = code[: -len("```")]
        code = code.strip()
    return code


def call_api(sample: dict) -> dict:
    code = clean_code(sample["code"])
    label = sample["label"].strip().lower()
    user_msg = USER_TEMPLATE.format(code=code)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            extra_body={"enable_thinking": True},
            max_tokens=900,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        thinking = getattr(msg, "reasoning_content", None) or ""

        verdict_m = _VERDICT_RE.findall(content)
        conf_m = _CONFIDENCE_RE.findall(content)

        if len(verdict_m) == 1 and len(conf_m) == 1:
            verdict = verdict_m[0].lower()
            confidence = max(0.0, min(1.0, float(conf_m[0])))
            is_correct = verdict == label
        else:
            verdict = None
            confidence = None
            is_correct = None

        return {
            "code": code,
            "label": label,
            "response": content,
            "thinking": thinking,
            "verdict": verdict,
            "confidence": confidence,
            "is_correct": is_correct,
            "error": None,
        }
    except Exception as e:
        return {
            "code": code,
            "label": label,
            "response": None,
            "thinking": None,
            "verdict": None,
            "confidence": None,
            "is_correct": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Test run: 10 samples in parallel
# ---------------------------------------------------------------------------
def run_test(n: int = 10, seed: int = 42, data_path: str = DATA_PATH):
    samples = load_samples(data_path)
    random.seed(seed)
    ai_samples = [s for s in samples if s["label"].strip().lower() == "ai"]
    human_samples = [s for s in samples if s["label"].strip().lower() == "human"]
    half = n // 2
    selected = random.sample(ai_samples, half) + random.sample(human_samples, n - half)
    random.shuffle(selected)

    print(f"Calling {MODEL} (thinking=True) on {n} samples in parallel ...\n")

    results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(call_api, s): i for i, s in enumerate(selected)}
        for future in as_completed(futures):
            results.append((futures[future], future.result()))
    results.sort(key=lambda x: x[0])

    correct = sum(1 for _, r in results if r["is_correct"])
    parseable = sum(1 for _, r in results if r["verdict"] is not None)

    print(f"{'#':>3}  {'label':6}  {'verdict':7}  {'conf':5}  {'ok':10}  response snippet")
    print("-" * 80)
    for i, r in results:
        label = r["label"]
        verdict = r["verdict"] or "FORMAT_ERR"
        conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "  -  "
        if r["error"]:
            status = "API_ERROR"
            snippet = r["error"][:50]
        elif r["is_correct"] is None:
            status = "FORMAT_ERR"
            snippet = (r["response"] or "")[:60].replace("\n", " ")
        else:
            status = "✓ correct" if r["is_correct"] else "✗ wrong"
            snippet = (r["response"] or "")[:60].replace("\n", " ")
        print(f"{i+1:>3}  {label:6}  {verdict:7}  {conf:5}  {status:10}  {snippet}")

    print("-" * 80)
    print(f"Parseable: {parseable}/{n}   Accuracy: {correct}/{parseable} = "
          f"{correct/parseable*100:.1f}%" if parseable else "Parseable: 0/10")

    # Show one full example
    good = [(i, r) for i, r in results if r["is_correct"]]
    if good:
        _, ex = good[0]
        print(f"\n--- Example response (label={ex['label']}, verdict={ex['verdict']}) ---")
        print(ex["response"][:800])
        if ex["thinking"]:
            print(f"\n--- Thinking (first 300 chars) ---")
            print(ex["thinking"][:300])

    return results


# ---------------------------------------------------------------------------
# Full generation: write SFT JSONL
# ---------------------------------------------------------------------------
def run_generate(n: int, output: str, seed: int = 42, workers: int = 16, data_path: str = DATA_PATH):
    samples = load_samples(data_path)
    random.seed(seed)
    ai_samples = [s for s in samples if s["label"].strip().lower() == "ai"]
    human_samples = [s for s in samples if s["label"].strip().lower() == "human"]
    if n <= 0 or n >= len(samples):
        selected = samples
        random.shuffle(selected)
    else:
        half = n // 2
        selected = random.sample(ai_samples, min(half, len(ai_samples))) + \
                   random.sample(human_samples, min(n - half, len(human_samples)))
        random.shuffle(selected)

    print(f"Generating {len(selected)} SFT examples with {workers} workers ...")

    written = 0
    correct = 0
    errors = 0

    with open(output, "w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_api, s): s for s in selected}
        for idx, future in enumerate(as_completed(futures), 1):
            r = future.result()
            if r["error"]:
                errors += 1
                print(f"[{idx}/{len(selected)}] ERROR: {r['error'][:80]}")
                continue
            if r["verdict"] is None:
                errors += 1
                print(f"[{idx}/{len(selected)}] FORMAT_ERR, skipping")
                continue

            # Save as SFT sample: messages format
            sft = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(code=r["code"])},
                    {"role": "assistant", "content": r["response"]},
                ],
                "label": r["label"],
                "verdict": r["verdict"],
                "is_correct": r["is_correct"],
                "thinking": r["thinking"],
            }
            fout.write(json.dumps(sft, ensure_ascii=False) + "\n")
            written += 1
            if r["is_correct"]:
                correct += 1

            if idx % 50 == 0:
                print(f"[{idx}/{len(selected)}] written={written}, acc={correct}/{written}")

    print(f"\nDone. Written: {written}, Correct: {correct}/{written} "
          f"({correct/written*100:.1f}%), Errors/skipped: {errors}")
    print(f"Output: {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run 10-sample test")
    parser.add_argument("--n", type=int, default=500, help="Number of SFT samples to generate")
    parser.add_argument("--data", default=DATA_PATH, help="Input samples JSONL path")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "cold_start_sft.jsonl"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.test:
        run_test(n=10, seed=args.seed, data_path=args.data)
    else:
        run_generate(n=args.n, output=args.output, seed=args.seed, workers=args.workers, data_path=args.data)
