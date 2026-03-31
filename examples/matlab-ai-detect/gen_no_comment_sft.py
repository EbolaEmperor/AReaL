"""
Generate SFT cold-start data for MATLAB code that has NO comments (no % symbols).

The teacher model is given a meta-instruction to handle no-comment code correctly:
acknowledge the absence of comments, analyze other features, and avoid hallucinating
comments. The stored SFT data uses the ORIGINAL system prompt (same as training),
so this requires no prompt change and no regeneration of existing data.

Usage:
    # Dry run (print samples, no API call):
    python examples/matlab-ai-detect/gen_no_comment_sft.py --dry-run

    # Test with 5 samples:
    python examples/matlab-ai-detect/gen_no_comment_sft.py --test

    # Full generation:
    python examples/matlab-ai-detect/gen_no_comment_sft.py --n 100
"""

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ---------------------------------------------------------------------------
# API config (same as gen_cold_start.py)
# ---------------------------------------------------------------------------
API_URL = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = "sk-sp-20282b4e0dac480eb5ff908a0a49717e"
MODEL = "qwen3.5-plus"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "samples.jsonl")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "no_comment_sft.jsonl")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# The STORED system prompt in SFT data — must match training exactly
STORED_SYSTEM_PROMPT = """你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。

请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：
<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>

注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。"""

# Meta-instruction given ONLY to the teacher — not stored in SFT data.
# Guides the teacher to produce a faithful response for no-comment code.
TEACHER_META_SYSTEM_PROMPT = """你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。

请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：
<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>

注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。

【重要】本次分析的代码没有任何注释（无 % 符号）。请务必：
1. 在分析开头明确说明"代码无注释"，不要声称代码有注释或引用任何不存在的注释内容。
2. 转而从变量命名风格、函数结构、算法实现习惯、魔法数字、代码冗余度等其他维度进行分析。
3. 由于缺少注释这一重要特征，适当降低置信度（建议不超过 0.8）。"""

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
def load_no_comment_samples(data_path: str = DATA_PATH) -> list[dict]:
    """Load all samples with no % in the code."""
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            code = r.get("code", "")
            # Strip markdown fences
            if code.startswith("```"):
                code = code.split("\n", 1)[1] if "\n" in code else code
                if code.endswith("```"):
                    code = code[:-3]
            code = code.strip()
            if "%" not in code and code:
                r["_clean_code"] = code
                records.append(r)
    return records


def response_hallucinated_comment(response: str, code: str) -> bool:
    """Return True if the response cites comments that don't exist in the code."""
    reasoning = re.split(r"<verdict>", response, maxsplit=1)[0]
    # If response mentions 注释 but code truly has no %
    if re.search(r"注释|comment", reasoning, re.IGNORECASE) and "%" not in code:
        # Check if it correctly says "无注释" vs claiming comments exist
        if re.search(r"无注释|没有注释|无.*%|缺少注释|不含注释", reasoning):
            return False  # Correctly acknowledging absence
        return True  # Hallucinating comments
    return False


def call_api(sample: dict) -> dict:
    code = sample["_clean_code"]
    label = sample["label"].strip().lower()
    user_msg = USER_TEMPLATE.format(code=code)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TEACHER_META_SYSTEM_PROMPT},
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
            verdict = confidence = is_correct = None

        hallucinated = response_hallucinated_comment(content, code) if content else False

        return {
            "code": code,
            "label": label,
            "response": content,
            "thinking": thinking,
            "verdict": verdict,
            "confidence": confidence,
            "is_correct": is_correct,
            "hallucinated": hallucinated,
            "error": None,
        }
    except Exception as e:
        return {
            "code": code, "label": label, "response": None, "thinking": None,
            "verdict": None, "confidence": None, "is_correct": None,
            "hallucinated": None, "error": str(e),
        }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def run_dry(n: int = 5, seed: int = 42):
    samples = load_no_comment_samples()
    random.seed(seed)
    selected = random.sample(samples, min(n, len(samples)))
    print(f"Available no-comment samples: {len(load_no_comment_samples())}")
    print(f"Showing {len(selected)} samples:\n")
    for i, s in enumerate(selected, 1):
        code = s["_clean_code"]
        lines = code.split("\n")
        print(f"[{i}] label={s['label']}  lines={len(lines)}")
        print(f"     {lines[0][:80]}")
    print(f"\nTeacher meta-prompt (not stored in SFT data):")
    print(TEACHER_META_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Test run
# ---------------------------------------------------------------------------
def run_test(n: int = 5, seed: int = 42):
    samples = load_no_comment_samples()
    random.seed(seed)
    selected = random.sample(samples, min(n, len(samples)))

    print(f"Calling {MODEL} (thinking=True) on {n} no-comment samples ...\n")

    results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(call_api, s): i for i, s in enumerate(selected)}
        for future in as_completed(futures):
            results.append((futures[future], future.result()))
    results.sort(key=lambda x: x[0])

    for i, r in results:
        status = "✓" if r["is_correct"] else ("FORMAT_ERR" if r["verdict"] is None else "✗")
        halluc = "HALLUC!" if r["hallucinated"] else "ok"
        conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else " - "
        print(f"[{i+1}] label={r['label']} verdict={r['verdict'] or '?'} conf={conf} {status} {halluc}")
        if r["response"]:
            print(f"     {r['response'][:120].replace(chr(10), ' ')}")
        print()


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------
def run_generate(n: int, output: str, seed: int = 42, workers: int = 16):
    samples = load_no_comment_samples()
    random.seed(seed)
    selected = random.sample(samples, min(n, len(samples)))

    print(f"Generating SFT data for {len(selected)} no-comment samples ...")
    print(f"Output: {output}\n")

    written = hallucinated_skipped = error_skipped = 0

    with open(output, "w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_api, s): s for s in selected}
        for idx, future in enumerate(as_completed(futures), 1):
            r = future.result()

            if r["error"]:
                error_skipped += 1
                print(f"[{idx}/{len(selected)}] ERROR: {r['error'][:80]}")
                continue
            if r["verdict"] is None:
                error_skipped += 1
                print(f"[{idx}/{len(selected)}] FORMAT_ERR, skipping")
                continue
            if r["hallucinated"]:
                hallucinated_skipped += 1
                print(f"[{idx}/{len(selected)}] HALLUCINATION detected, skipping")
                continue

            sft = {
                "messages": [
                    {"role": "system", "content": STORED_SYSTEM_PROMPT},
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

            if idx % 20 == 0:
                print(f"[{idx}/{len(selected)}] written={written}, "
                      f"halluc_skip={hallucinated_skipped}, err_skip={error_skipped}")

    print(f"\nDone.")
    print(f"  Written:             {written}")
    print(f"  Skipped (halluc):    {hallucinated_skipped}")
    print(f"  Skipped (error/fmt): {error_skipped}")
    print(f"  Output: {output}")
    print(f"\nNext: append to cold_start_sft_filtered.jsonl:")
    print(f"  cat {output} >> examples/matlab-ai-detect/cold_start_sft_filtered.jsonl")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview samples, no API call")
    parser.add_argument("--test", action="store_true", help="Test with 5 samples")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        run_dry(seed=args.seed)
    elif args.test:
        run_test(seed=args.seed)
    else:
        run_generate(n=args.n, output=args.output, seed=args.seed, workers=args.workers)
