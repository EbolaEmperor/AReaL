"""
GRPO fine-tuning of DeepSeek-Coder-7B-Instruct-v1.5 with LoRA via TRL.

Launch (dual GPU):
    torchrun --nproc_per_node=2 examples/matlab-ai-detect/matlab_ai_detect_grpo_trl.py
"""

import json
import os
import sys

# Must be set before vLLM is imported (inherited by all subprocesses)
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import torch

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from areal.reward.matlab_ai_detect import matlab_ai_detect_reward_fn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_MODEL = "/home/wenchong/grpo-2048/modelscope_cache/deepseek-ai/deepseek-coder-7b-instruct-v1___5"
_SFT_CHECKPOINT = "/home/wenchong/code/AReaL/examples/matlab-ai-detect/sft-merged"
# Use SFT checkpoint as starting point if it exists, else fall back to base model
MODEL_PATH = _SFT_CHECKPOINT if os.path.isdir(_SFT_CHECKPOINT) else _BASE_MODEL
DATA_PATH = os.path.join(os.path.dirname(__file__), "samples.jsonl")
OUTPUT_DIR = "/tmp/areal/experiments/checkpoints/matlab-ai-detect-grpo"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。\n\n"
    "请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "注意：你的整个回复（分析 + 结论）不得超过 600 字，请保持简洁。"
)

PROMPT_TEMPLATE = (
    "请分析以下 MATLAB 代码，判断它是由 AI 生成的还是由人类手写的。\n"
    "请给出你的分析过程，并在最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "```matlab\n{code}\n```"
)


def _load_dataset(tokenizer, split: str, train_ratio: float = 0.8, seed: int = 42,
                  max_prompt_tokens: int = 3500, max_train_samples: int | None = None) -> Dataset:
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    dataset = Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(dataset) * train_ratio)
    dataset = dataset.select(range(split_idx) if split == "train" else range(split_idx, len(dataset)))

    def process(sample):
        code = sample["code"]
        if code.startswith("```"):
            code = code.split("\n", 1)[1] if "\n" in code else code
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
        prompt_text = PROMPT_TEMPLATE.format(code=code)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        return {"prompt": messages, "answer": sample["label"]}

    dataset = dataset.map(process, remove_columns=dataset.column_names)

    # Filter by tokenized prompt length
    def filter_length(sample):
        text = tokenizer.apply_chat_template(
            sample["prompt"], tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer.encode(text)) <= max_prompt_tokens

    dataset = dataset.filter(filter_length)

    # Randomly subsample training set if requested
    if split == "train" and max_train_samples is not None and len(dataset) > max_train_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_train_samples))

    return dataset


# ---------------------------------------------------------------------------
# Reward function (TRL signature: completions, **dataset_columns)
# ---------------------------------------------------------------------------
def reward_fn(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    return [
        matlab_ai_detect_reward_fn(
            prompt=None, completions=c, prompt_ids=None, completion_ids=None, answer=a
        )
        for c, a in zip(completions, answer)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    train_dataset = _load_dataset(tokenizer, split="train", max_train_samples=2000)
    eval_dataset = _load_dataset(tokenizer, split="test")
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # Training
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,    # effective = 1 GPU × 1 × 8 = 8 prompts/step
        learning_rate=5e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.02,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        # GRPO
        num_generations=4,
        max_completion_length=1000,
        temperature=1.0,
        epsilon=0.2,
        beta=0.0,               # no KL penalty
        scale_rewards="group",  # group-level reward normalisation
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="localhost",
        vllm_server_port=8000,
        # Eval & save
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        logging_steps=1,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
