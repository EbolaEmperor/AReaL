"""
Continue LoRA SFT of Qwen3.5-9B for 1 more epoch (epoch 4) from the 3-epoch checkpoint.
"""

import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

BASE_MODEL = "/home/wenchong/grpo-2048/modelscope_cache/Qwen/Qwen3.5-9B"
LORA_3EP = "/tmp/areal/experiments/checkpoints/wenchong/hate-recognition-sft-qwen35-9b-fulldata/final"
DATA_DIR = "/home/wenchong/code/AReaL/examples/hate_recogonization"
OUTPUT_DIR = "/tmp/areal/experiments/checkpoints/wenchong/hate-recognition-sft-qwen35-9b-fulldata-4ep"

from areal.dataset.hate_recognition import HATE_RECOGNITION_PROMPT_PREFIX


def load_full_dataset():
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

    print("[sft] Loading base model + 3-epoch LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_3EP, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    # Load and merge the 3-epoch LoRA, then add fresh LoRA for continued training
    model = PeftModel.from_pretrained(model, LORA_3EP)
    model = model.merge_and_unload()
    print("[sft] Merged 3-epoch LoRA into base")

    # Fresh LoRA for the 4th epoch
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
    print(f"[sft] Train: {len(train_dataset)}")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,  # lower lr for continued training
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        packing=False,
        dataset_text_field=None,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
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
    print(f"[sft] Epoch 4 model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
