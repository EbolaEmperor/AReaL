import json
import os

from datasets import Dataset

MATLAB_AI_DETECT_PROMPT_TEMPLATE = (
    "请分析以下 MATLAB 代码，判断它是由 AI 生成的还是由人类手写的。\n"
    "请给出你的分析过程，并在最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "```matlab\n{code}\n```"
)


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_matlab_ai_detect_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
):
    jsonl_path = os.path.join(path, "samples.jsonl")
    if not os.path.exists(jsonl_path):
        jsonl_path = path

    records = _load_jsonl(jsonl_path)

    dataset = Dataset.from_list(records)
    dataset = dataset.shuffle(seed=seed)

    split_idx = int(len(dataset) * train_ratio)
    if split == "train":
        dataset = dataset.select(range(split_idx))
    else:
        dataset = dataset.select(range(split_idx, len(dataset)))

    def process(sample):
        code = sample["code"]
        # Remove markdown fences if already present in the code
        if code.startswith("```"):
            code = code.split("\n", 1)[1] if "\n" in code else code
            if code.endswith("```"):
                code = code[: -len("```")]
            code = code.strip()

        prompt_text = MATLAB_AI_DETECT_PROMPT_TEMPLATE.format(code=code)
        messages = [{"role": "user", "content": prompt_text}]
        return {"messages": messages, "answer": sample["label"]}

    dataset = dataset.map(process)

    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
