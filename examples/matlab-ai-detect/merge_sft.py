"""Merge SFT LoRA adapter into base model and save as full model."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/wenchong/grpo-2048/modelscope_cache/deepseek-ai/deepseek-coder-7b-instruct-v1___5"
LORA = "/tmp/areal/experiments/checkpoints/matlab-ai-detect-sft/checkpoint-210"
OUT  = "/home/wenchong/code/AReaL/examples/matlab-ai-detect/sft-merged"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, LORA)
print("Merging and saving...")
merged = model.merge_and_unload()
merged.save_pretrained(OUT)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print(f"Saved merged model to {OUT}")
