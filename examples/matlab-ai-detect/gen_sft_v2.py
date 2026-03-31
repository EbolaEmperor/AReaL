"""
Regenerate ALL SFT cold-start data using the unified teacher meta-prompt.

Reads codes + labels from the existing cold_start_sft_filtered.jsonl,
calls the teacher API with the new unified prompt, and writes fresh responses.

Usage:
    python examples/matlab-ai-detect/gen_sft_v2.py --dry-run   # preview counts
    python examples/matlab-ai-detect/gen_sft_v2.py --test      # 5 samples
    python examples/matlab-ai-detect/gen_sft_v2.py             # full regeneration
"""

import argparse
import json
import os
import random
import re
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
DIR = os.path.dirname(__file__)
SOURCE = os.path.join(DIR, "cold_start_sft_filtered.jsonl")
DEFAULT_OUTPUT = os.path.join(DIR, "cold_start_sft_v2.jsonl")

# ---------------------------------------------------------------------------
# Stored system prompt — unchanged, matches training exactly
# ---------------------------------------------------------------------------
STORED_SYSTEM_PROMPT = """你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。

请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：
<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>

注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。"""

# ---------------------------------------------------------------------------
# Teacher meta-prompt — unified, covers all comment scenarios
# ---------------------------------------------------------------------------
TEACHER_META_SYSTEM_PROMPT = """你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。

请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：
<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>

注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。

【分析流程】

第一步：识别并排除模板注释。
模板注释是课程出题方预先写入代码骨架的指令性文字，不代表学生或 AI 的创作，
不能作为判断依据。常见形式包括：
  % 在这里实现你的代码 / 在这里完成你的代码
  % 在这里实现题目要求的函数
  %% 你可以在这里自定义一些需要用到的函数
  % 下面的代码用于输出答案，请勿修改
  % TODO: implement / Your code here
排除上述内容后，判断代码中是否还存在实质性注释（即描述逻辑、算法或变量的注释）。

第二步：根据注释情况选择分析路径。

▶ 情况 A：排除模板注释后，代码无实质性注释
  不依据注释做任何判断。改从以下维度综合分析：
  - 变量命名：是否过于规范通用（如 matrix、result、idx），还是随意、个性化
  - 缩进与格式：是否机械整齐，还是存在人类习惯的细微不一致
  - 算法实现：是否采用教科书标准写法，还是带有个人风格的非最优实现
  - 代码完整度：是否存在未完成占位符、魔法数字、调试残留、冗余逻辑等人类痕迹
  由于缺少注释这一重要特征，适当降低置信度（建议不超过 0.8）。

▶ 情况 B：排除模板注释后，代码仍有实质性注释
  综合注释风格与代码本身进行分析：
  - 注释是否过于详尽、解释显而易见的操作（AI 典型特征）
  - 注释是否简洁、只记录意图而非逐行解释（人类典型特征）
  - 注释语言风格与代码命名是否一致
  - 结合变量命名、代码结构、算法选择等其他维度综合判断"""

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
# Load source samples (code + label) from existing filtered data
# ---------------------------------------------------------------------------
def load_source_samples() -> list[dict]:
    samples = []
    with open(SOURCE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Extract code from user message
            user_content = r["messages"][1]["content"]
            m = re.search(r"```matlab\n(.*?)\n```", user_content, re.DOTALL)
            code = m.group(1).strip() if m else ""
            label = r.get("label", "").strip().lower()
            if code and label:
                samples.append({"code": code, "label": label})
    return samples


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_api(sample: dict) -> dict:
    code = sample["code"]
    label = sample["label"]
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
            "code": code, "label": label,
            "response": content, "thinking": thinking,
            "verdict": verdict, "confidence": confidence,
            "is_correct": is_correct, "error": None,
        }
    except Exception as e:
        return {
            "code": code, "label": label,
            "response": None, "thinking": None,
            "verdict": None, "confidence": None,
            "is_correct": None, "error": str(e),
        }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def run_dry():
    samples = load_source_samples()
    from collections import Counter
    dist = Counter(s["label"] for s in samples)
    print(f"Source samples: {len(samples)}  {dict(dist)}")
    print(f"Output: {DEFAULT_OUTPUT}")
    print(f"\nTeacher meta-prompt:\n{TEACHER_META_SYSTEM_PROMPT}")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def run_test(n: int = 5, seed: int = 42):
    samples = load_source_samples()
    random.seed(seed)
    selected = random.sample(samples, min(n, len(samples)))

    print(f"Testing {len(selected)} samples with new unified prompt ...\n")
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
        if r["response"]:
            print(f"     {r['response'][:200].replace(chr(10), ' ')}")
        print()


# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------
def run_generate(output: str, workers: int = 16):
    samples = load_source_samples()
    print(f"Regenerating {len(samples)} samples with unified teacher prompt ...")
    print(f"Output: {output}\n")

    written = correct = error_skipped = 0

    with open(output, "w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_api, s): s for s in samples}
        for idx, future in enumerate(as_completed(futures), 1):
            r = future.result()

            if r["error"]:
                error_skipped += 1
                print(f"[{idx}/{len(samples)}] ERROR: {r['error'][:80]}")
                continue
            if r["verdict"] is None:
                error_skipped += 1
                print(f"[{idx}/{len(samples)}] FORMAT_ERR, skipping")
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
            if r["is_correct"]:
                correct += 1

            if idx % 50 == 0:
                print(f"[{idx}/{len(samples)}] written={written} "
                      f"acc={correct}/{written} ({correct/written*100:.1f}%) "
                      f"err={error_skipped}")

    print(f"\nDone.")
    print(f"  Written:  {written}  Accuracy: {correct}/{written} ({correct/written*100:.1f}%)")
    print(f"  Skipped:  {error_skipped}")
    print(f"\nTo replace existing data:")
    print(f"  cp {output} {SOURCE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
    elif args.test:
        run_test(seed=args.seed)
    else:
        run_generate(output=args.output, workers=args.workers)
