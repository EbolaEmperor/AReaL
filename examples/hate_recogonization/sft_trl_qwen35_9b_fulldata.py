"""
LoRA SFT of Qwen3.5-9B using ALL 4000 training samples (train+dev).
Same config as the best run: LoRA r=32 alpha=64, lr 2e-4, cosine, 3 epochs.

Launch:
    CUDA_VISIBLE_DEVICES=0,1 python examples/hate_recogonization/sft_trl_qwen35_9b_fulldata.py
"""

import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = "/home/wenchong/grpo-2048/modelscope_cache/Qwen/Qwen3.5-9B"
DATA_DIR = "/home/wenchong/code/AReaL/examples/hate_recogonization"
OUTPUT_DIR = "/tmp/areal/experiments/checkpoints/wenchong/hate-recognition-sft-qwen35-9b-fulldata"

from areal.dataset.hate_recognition import HATE_RECOGNITION_PROMPT_PREFIX


def load_full_dataset():
    """Load ALL 4000 samples as training data (no held-out split)."""
    with open(os.path.join(DATA_DIR, "train.json"), encoding="utf-8") as f:
        records = json.load(f)
    ds = Dataset.from_list(records)

    def format_messages(sample):
        prompt = HATE_RECOGNITION_PROMPT_PREFIX + sample["content"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sample["output"]},
        ]
        return {"messages": messages}

    ds = ds.map(format_messages, remove_columns=["id", "content", "output"])
    return ds


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[sft] Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft] Loading model from {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    train_dataset = load_full_dataset()
    print(f"[sft] Train: {len(train_dataset)} (full data, no held-out)")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 2*8 = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        packing=False,
        dataset_text_field=None,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"[sft] Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
