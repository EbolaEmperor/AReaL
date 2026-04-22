"""
Full fine-tune of Qwen3.5-4B using TRL SFTTrainer + DeepSpeed ZeRO-3 on 2 GPUs.
lr 1e-5, cosine, 3 epochs, effective batch 16.

Launch:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        examples/hate_recogonization/sft_trl_qwen35_4b_full.py
"""

import json
import os
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = "/home/wenchong/grpo-2048/modelscope_cache/Qwen/Qwen3.5-4B"
DATA_DIR = "/home/wenchong/code/AReaL/examples/hate_recogonization"
OUTPUT_DIR = "/tmp/areal/experiments/checkpoints/wenchong/hate-recognition-sft-qwen35-4b-full"

from areal.dataset.hate_recognition import HATE_RECOGNITION_PROMPT_PREFIX


def load_dataset_splits(seed=42, train_ratio=0.9):
    with open(os.path.join(DATA_DIR, "train.json"), encoding="utf-8") as f:
        records = json.load(f)
    ds = Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(ds) * train_ratio)
    train_ds = ds.select(range(split_idx))
    val_ds = ds.select(range(split_idx, len(ds)))

    def format_messages(sample):
        prompt = HATE_RECOGNITION_PROMPT_PREFIX + sample["content"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sample["output"]},
        ]
        return {"messages": messages}

    train_ds = train_ds.map(format_messages, remove_columns=["id", "content", "output"])
    val_ds = val_ds.map(format_messages, remove_columns=["id", "content", "output"])
    return train_ds, val_ds


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
    )

    train_dataset, eval_dataset = load_dataset_splits()
    print(f"[sft] Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # effective batch = 1*8*2gpus = 16
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=512,
        packing=False,
        dataset_text_field=None,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        deepspeed=os.path.join(os.path.dirname(__file__), "ds_zero3_config.json"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"[sft] Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
