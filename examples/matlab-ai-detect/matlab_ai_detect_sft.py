"""
SFT cold-start fine-tuning of DeepSeek-Coder-7B-Instruct-v1.5 with LoRA via TRL.

Launch (dual GPU):
    CUDA_VISIBLE_DEVICES=1,0 ~/.local/bin/torchrun --nproc_per_node=2 \\
        examples/matlab-ai-detect/matlab_ai_detect_sft.py

Launch (single GPU):
    CUDA_VISIBLE_DEVICES=1 python3 examples/matlab-ai-detect/matlab_ai_detect_sft.py
"""

import hashlib
import json
import os
import random
import re
import sys

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/wenchong/grpo-2048/modelscope_cache/deepseek-ai/deepseek-coder-7b-instruct-v1___5"
SFT_DATA_PATH = os.path.join(os.path.dirname(__file__), "cold_start_sft_filtered.jsonl")
ALL_DATA_PATH = os.path.join(os.path.dirname(__file__), "samples.jsonl")
OUTPUT_DIR = "/tmp/areal/experiments/checkpoints/matlab-ai-detect-sft"
EVAL_LOG_PATH = os.path.join(OUTPUT_DIR, "eval_accuracy.log")
HELD_OUT_PATH = os.path.join(os.path.dirname(__file__), "held_out_200.jsonl")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一位 MATLAB 代码风格专家，擅长区分 AI 生成代码和人类手写代码。\n\n"
    "请对代码进行深入分析，给出你的判断依据，最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>"
)

USER_TEMPLATE = (
    "请分析以下 MATLAB 代码，判断它是由 AI 生成的还是由人类手写的。\n"
    "请给出你的分析过程，并在最后以如下格式输出结论：\n"
    "<verdict>ai 或 human</verdict><confidence>0.0 到 1.0 之间的置信度数字</confidence>\n\n"
    "```matlab\n{code}\n```"
)

_VERDICT_RE = re.compile(r"<verdict>\s*(ai|human)\s*</verdict>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Build 200 held-out samples (not in SFT training set)
# ---------------------------------------------------------------------------
def _code_fingerprint(code: str) -> str:
    return hashlib.md5(code.strip().encode()).hexdigest()


def _extract_code_from_user_msg(content: str) -> str:
    """Extract matlab code from user message content."""
    m = re.search(r"```matlab\n(.*?)\n```", content, re.DOTALL)
    return m.group(1).strip() if m else content


def build_held_out_samples(n: int = 200, seed: int = 42) -> list[dict]:
    """Pick n samples from samples.jsonl that are NOT in the SFT training set."""
    if os.path.exists(HELD_OUT_PATH):
        with open(HELD_OUT_PATH, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f if l.strip()]
        print(f"Loaded {len(samples)} held-out samples from {HELD_OUT_PATH}")
        return samples

    # Collect fingerprints of SFT training codes
    sft_fps: set[str] = set()
    with open(SFT_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            user_content = rec["messages"][1]["content"]
            code = _extract_code_from_user_msg(user_content)
            sft_fps.add(_code_fingerprint(code))

    # Load all samples and filter out SFT ones
    all_samples = []
    with open(ALL_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            code = rec["code"]
            if code.startswith("```"):
                code = code.split("\n", 1)[1] if "\n" in code else code
                if code.endswith("```"):
                    code = code[:-3]
                code = code.strip()
            fp = _code_fingerprint(code)
            if fp not in sft_fps:
                all_samples.append({"code": code, "label": rec["label"].strip().lower()})

    # Balanced sample
    random.seed(seed)
    ai = [s for s in all_samples if s["label"] == "ai"]
    human = [s for s in all_samples if s["label"] == "human"]
    half = n // 2
    selected = random.sample(ai, min(half, len(ai))) + random.sample(human, min(n - half, len(human)))
    random.shuffle(selected)

    os.makedirs(os.path.dirname(HELD_OUT_PATH), exist_ok=True)
    with open(HELD_OUT_PATH, "w", encoding="utf-8") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Built {len(selected)} held-out samples "
          f"(ai={sum(s['label']=='ai' for s in selected)}, "
          f"human={sum(s['label']=='human' for s in selected)}), "
          f"saved to {HELD_OUT_PATH}")
    return selected


# ---------------------------------------------------------------------------
# Accuracy eval callback
# ---------------------------------------------------------------------------
class AccuracyEvalCallback(TrainerCallback):
    def __init__(self, eval_samples: list[dict], tokenizer, log_path: str,
                 max_new_tokens: int = 400, batch_size: int = 8):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.log_path = log_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, model=None, **kwargs):
        is_main = args.local_rank in (-1, 0)

        if is_main:
            epoch = round(state.epoch)
            print(f"\n[AccuracyEval] Epoch {epoch}: evaluating {len(self.eval_samples)} held-out samples ...")
            raw_model = model.module if hasattr(model, "module") else model
            device = next(raw_model.parameters()).device
            raw_model.eval()

            # Left-pad for batched generation
            orig_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"

            correct = 0
            parseable = 0
            per_label: dict[str, dict] = {
                "ai": {"correct": 0, "total": 0},
                "human": {"correct": 0, "total": 0},
            }

            with torch.no_grad():
                for i in range(0, len(self.eval_samples), self.batch_size):
                    batch = self.eval_samples[i: i + self.batch_size]
                    prompts = []
                    for s in batch:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_TEMPLATE.format(code=s["code"])},
                        ]
                        prompts.append(
                            self.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        )

                    enc = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=3800,
                    ).to(device)

                    out = raw_model.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    input_len = enc["input_ids"].shape[1]
                    for j, sample in enumerate(batch):
                        gen_ids = out[j][input_len:]
                        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                        label = sample["label"]
                        per_label[label]["total"] += 1
                        matches = _VERDICT_RE.findall(response)
                        if len(matches) == 1:
                            verdict = matches[0].lower()
                            parseable += 1
                            if verdict == label:
                                correct += 1
                                per_label[label]["correct"] += 1

            self.tokenizer.padding_side = orig_padding_side
            raw_model.train()
            torch.cuda.empty_cache()

            n = len(self.eval_samples)
            acc_str = f"{correct/parseable*100:.1f}%" if parseable else "N/A"
            log_line = (
                f"Epoch {epoch} | "
                f"accuracy={correct}/{parseable}({acc_str}) | "
                f"parseable={parseable}/{n} | "
                f"ai={per_label['ai']['correct']}/{per_label['ai']['total']} | "
                f"human={per_label['human']['correct']}/{per_label['human']['total']}"
            )
            print(f"[AccuracyEval] {log_line}")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

        if dist.is_initialized():
            dist.barrier()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_sft_dataset(train_ratio: float = 0.9, seed: int = 42) -> tuple[Dataset, Dataset]:
    records = []
    with open(SFT_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    dataset = Dataset.from_list(records).shuffle(seed=seed)
    split_idx = int(len(dataset) * train_ratio)
    train_ds = dataset.select(range(split_idx)).select_columns(["messages"])
    eval_ds = dataset.select(range(split_idx, len(dataset))).select_columns(["messages"])
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    train_dataset, eval_dataset = load_sft_dataset()
    print(f"SFT train: {len(train_dataset)}, SFT eval: {len(eval_dataset)}")

    # Build held-out accuracy eval set (only on main process before DDP init)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in (-1, 0):
        held_out = build_held_out_samples(n=200)
    else:
        # Wait for rank 0 to write the file, then load
        import time
        for _ in range(30):
            if os.path.exists(HELD_OUT_PATH):
                break
            time.sleep(1)
        with open(HELD_OUT_PATH, encoding="utf-8") as f:
            held_out = [json.loads(l) for l in f if l.strip()]

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        max_length=4096,
        packing=False,
        dataset_text_field=None,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=5,
        logging_steps=5,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"SFT model saved to {OUTPUT_DIR}")
    print(f"Accuracy log: {EVAL_LOG_PATH}")
    print("Next: run matlab_ai_detect_grpo_trl.py (will auto-load this checkpoint)")


if __name__ == "__main__":
    main()
