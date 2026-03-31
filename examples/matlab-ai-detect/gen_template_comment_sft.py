"""
Generate SFT cold-start data for MATLAB code that contains template comments —
boilerplate scaffolding written by course instructors, not by students or AI.

Examples of template comments (appear ≥10 times across OJ data):
  % 在这里实现题目要求的函数
  % 在这里开始写你的代码
  % 下面的代码用于输出答案，请勿修改
  % 请修改下面的代码，以实现作业要求
  %% 你可以在这里自定义一些需要用到的函数

The teacher is instructed to:
  1. Identify and explicitly list any template comments found
  2. Ignore them as evidence
  3. Base the verdict solely on code the student/AI actually wrote

The stored SFT data uses the ORIGINAL system prompt (unchanged), so existing
SFT data does not need to be regenerated.

Usage:
    python examples/matlab-ai-detect/gen_template_comment_sft.py --dry-run
    python examples/matlab-ai-detect/gen_template_comment_sft.py --test
    python examples/matlab-ai-detect/gen_template_comment_sft.py --n 232
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
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
DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), "samples.jsonl"),
    os.path.join(os.path.dirname(__file__), "oj_samples.jsonl"),
]
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "template_comment_sft.jsonl")

# ---------------------------------------------------------------------------
# Template comment detection
# ---------------------------------------------------------------------------
TEMPLATE_PATTERNS = [
    "你可以在这里自定义一些需要用到的函数",
    "在这里实现题目要求的函数",
    "在这里开始写你的代码",
    "下面的代码用于输出答案",
    "下面的代码用于输入数据",
    "请修改下面的代码，以实现作业要求",
    "在这里写你的代码",
    "注意：您不能修改下面这个函数",
    "接下来，系统首先检测您的",
    "随后，系统检查您的",
    "只有当 LDLT 求解成功",
    "下面四行代码用于输入数据",
    "下面一行代码用于输出答案",
    "在这里实现你的代码",
    "您需要根据题目要求完善这个函数",
    "在这里完成你的代码",
    "在这里实现您的函数",
    "在这里完成您的程序",
    "第一个函数",
    "第二个函数",
]


def find_template_comments(code: str) -> list[str]:
    found = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("%") and any(p in stripped for p in TEMPLATE_PATTERNS):
            found.append(stripped)
    return found


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
STORED_SYSTEM_PROMPT = """你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。

请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：
<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>

注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。"""

TEACHER_META_SYSTEM_PROMPT = """你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。

请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：
<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>

注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。

【关于模板注释】
题目初始代码骨架中常包含"模板注释"，例如：
  % 在这里实现您的函数
  % 在这里开始写你的代码
  % 下面的代码用于输出答案，请勿修改
  % 请修改下面的代码，以实现作业要求
  %% 你可以在这里自定义一些需要用到的函数
这类注释是课程出题方预先写入的，不是学生或 AI 的创作，不能作为判断依据。
请在分析开头说明你识别出了哪些模板注释（如有），然后忽略它们，仅依据学生
实际编写的代码内容（逻辑实现、变量命名、算法选择等）进行分析。"""

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
# Data loading
# ---------------------------------------------------------------------------
def load_template_comment_samples() -> list[dict]:
    seen = set()
    results = []
    for path in DATA_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                code = r.get("code", "")
                if code.startswith("```"):
                    code = code.split("\n", 1)[1] if "\n" in code else code
                    if code.endswith("```"):
                        code = code[:-3]
                code = code.strip()
                label = r.get("label", "").strip().lower()
                if not code or not label:
                    continue
                templates = find_template_comments(code)
                if not templates:
                    continue
                fp = hashlib.md5(code.encode()).hexdigest()
                if fp in seen:
                    continue
                seen.add(fp)
                r["_clean_code"] = code
                r["_template_comments"] = templates
                results.append(r)
    return results


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
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

        return {
            "code": code,
            "label": label,
            "template_comments": sample["_template_comments"],
            "response": content,
            "thinking": thinking,
            "verdict": verdict,
            "confidence": confidence,
            "is_correct": is_correct,
            "error": None,
        }
    except Exception as e:
        return {
            "code": code, "label": label,
            "template_comments": sample["_template_comments"],
            "response": None, "thinking": None,
            "verdict": None, "confidence": None, "is_correct": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def run_dry():
    samples = load_template_comment_samples()
    label_dist = Counter(s["label"] for s in samples)
    print(f"Found {len(samples)} unique samples with template comments")
    print(f"Label distribution: {dict(label_dist)}\n")
    print("Sample preview:")
    for s in samples[:5]:
        code_lines = s["_clean_code"].split("\n")
        print(f"  label={s['label']}  lines={len(code_lines)}")
        print(f"  template comments: {s['_template_comments'][:2]}")
        print(f"  first line: {code_lines[0][:70]}")
        print()


# ---------------------------------------------------------------------------
# Test run
# ---------------------------------------------------------------------------
def run_test(n: int = 5, seed: int = 42):
    import random
    samples = load_template_comment_samples()
    random.seed(seed)
    selected = random.sample(samples, min(n, len(samples)))

    print(f"Calling {MODEL} on {len(selected)} template-comment samples ...\n")

    results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(call_api, s): i for i, s in enumerate(selected)}
        for future in as_completed(futures):
            results.append((futures[future], future.result()))
    results.sort(key=lambda x: x[0])

    for i, r in results:
        status = "✓" if r["is_correct"] else ("FORMAT_ERR" if r["verdict"] is None else "✗")
        conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else " - "
        print(f"[{i+1}] label={r['label']} verdict={r['verdict'] or '?'} conf={conf} {status}")
        print(f"     templates: {r['template_comments'][:2]}")
        if r["response"]:
            print(f"     response: {r['response'][:150].replace(chr(10), ' ')}")
        print()


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------
def run_generate(n: int, output: str, seed: int = 42, workers: int = 16):
    import random
    samples = load_template_comment_samples()
    random.seed(seed)
    ai = [s for s in samples if s["label"] == "ai"]
    human = [s for s in samples if s["label"] == "human"]
    # Balance: equal samples from each class, capped by the minority class (AI)
    n_each = min(len(ai), len(human), n // 2)
    selected = random.sample(ai, n_each) + random.sample(human, n_each)
    random.shuffle(selected)

    print(f"Generating SFT data for {len(selected)} template-comment samples ...")
    label_dist = Counter(s["label"] for s in selected)
    print(f"Label distribution: {dict(label_dist)}")
    print(f"Output: {output}\n")

    written = error_skipped = 0

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
                correct = sum(1 for s in [r] if s.get("is_correct"))
                print(f"[{idx}/{len(selected)}] written={written}, err_skip={error_skipped}")

    print(f"\nDone.")
    print(f"  Written:             {written}")
    print(f"  Skipped (error/fmt): {error_skipped}")
    print(f"  Output: {output}")
    print(f"\nNext: append to cold_start_sft_filtered.jsonl:")
    print(f"  cat {output} >> examples/matlab-ai-detect/cold_start_sft_filtered.jsonl")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n", type=int, default=52)  # 26 AI + 26 human (1:1 balance)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
    elif args.test:
        run_test(seed=args.seed)
    else:
        run_generate(n=args.n, output=args.output, seed=args.seed, workers=args.workers)
