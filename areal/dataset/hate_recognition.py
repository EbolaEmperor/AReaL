import json
import os

from datasets import Dataset

_TASK_DESC = (
    "任务：细粒度中文仇恨言论识别（四元组抽取）\n\n"
    "请阅读以下中文文本，抽取其中的仇恨言论四元组。每个四元组由 4 个字段构成，"
    "字段之间用 ` | ` 分隔（即空格-竖线-空格），4 个字段依次为：\n"
    "1. 评论对象 (Target)：从原文中摘出被评论的具体对象、个体或群体名称；"
    "若文本无明确评论对象，填字符串 NULL。\n"
    "2. 论点 (Argument)：从原文中摘出针对该对象的关键论点或言论片段（一般是原文片段）。\n"
    "3. 仇恨群体 (Targeted Group)：从下列 6 个英文标签中选择——"
    "Racism, Region, Sexism, LGBTQ, others, non-hate；"
    "若同时涉及多个群体，用英文逗号加空格连接（例如 Region, Racism）；"
    "若文本不构成仇恨言论，填 non-hate。\n"
    "4. 是否仇恨 (Hateful)：英文小写，仅取 hate 或 non-hate。\n\n"
    "输出协议：\n"
    "- 结尾必须以空格加 [END] 终止；多个四元组之间用空格加 [SEP] 加空格分隔。\n"
    "- 只输出该结构本身，不要写任何分析、解释、前缀、标点或 Markdown。\n"
    "- Target 与 Argument 必须来源于待分析文本，不要使用占位符。\n"
    "- **无论文本是否构成仇恨，都必须至少输出一个完整的四元组**；"
    "若文本不含仇恨，仍需摘出主要评论对象与论点，并将 Group 与 Hateful 都填为 non-hate。\n\n"
)

HATE_RECOGNITION_PROMPT_PREFIX = _TASK_DESC + "待分析文本："

# 3-shot examples drawn from the TRAIN split (ids 2591, 7696, 5001) —
# never present in the held-out dev set.
HATE_RECOGNITION_PROMPT_PREFIX_FEWSHOT = (
    _TASK_DESC
    + "以下是三个标注好的示例，最后一条请你补全输出：\n\n"
    + "文本：是呗，正常女哪有喜欢黑人的啊😅\n"
    + "输出：黑人 | 正常女哪有喜欢黑人的啊😅 | Racism | hate [END]\n\n"
    + "文本：跳梁小丑寻组织来了lz说的是你\n"
    + "输出：你 | 跳梁小丑 | non-hate | non-hate [END]\n\n"
    + "文本：看网上广州好像老黑比较多🤔 我在南京除了英语外教就没见过黑人😅\n"
    + "输出：广州 | 好像老黑比较多 | Region, Racism | hate [SEP] 南京 | 除了英语外教就没见过黑人 | Region, Racism | hate [END]\n\n"
    + "文本："
)


def _load_json_array(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _split_records(path, split, seed, train_ratio):
    json_path = os.path.join(path, "train.json") if os.path.isdir(path) else path
    records = _load_json_array(json_path)
    dataset = Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(dataset) * train_ratio)
    if split == "train":
        return dataset.select(range(split_idx))
    return dataset.select(range(split_idx, len(dataset)))


def get_hate_recognition_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.9,
    fewshot: bool = True,
):
    dataset = _split_records(path, split, seed, train_ratio)

    def process(sample):
        prefix = (
            HATE_RECOGNITION_PROMPT_PREFIX_FEWSHOT
            if fewshot
            else HATE_RECOGNITION_PROMPT_PREFIX
        )
        prompt_text = prefix + sample["content"]
        messages = [{"role": "user", "content": prompt_text}]
        return {"messages": messages, "answer": sample["output"]}

    dataset = dataset.map(process, remove_columns=["id", "content", "output"])

    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_hate_recognition_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.9,
):
    """SFT dataset: tokenize chat-formatted (prompt, gold) pairs with loss only
    on the assistant tokens (loss_mask = 0 over prompt, 1 over response)."""
    dataset = _split_records(path, split, seed, train_ratio)

    def process(sample):
        prompt_text = HATE_RECOGNITION_PROMPT_PREFIX + sample["content"]
        gold = sample["output"]

        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": gold},
            ],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        return {"input_ids": full_ids, "loss_mask": loss_mask}

    dataset = dataset.map(process, remove_columns=["id", "content", "output"])

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset
