#!/usr/bin/env python3
"""Unsloth fine-tuning template wired to the generated dataset.

Copy to your working directory and adjust the env vars, e.g.

  MODEL_ID=Qwen/Qwen3-1.7B  EPOCHS=2  python finetune.py

Notes
-----
- QLoRA (4-bit) is the right default for 16GB cards (T4/A100 16GB/Colab T4).
- max_seq_length defaults to 512 - raise it only if your rows are long.
- 4-bit + RS-LoRA is supported by Unsloth and is stable on Turing (T4).
"""
import os

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.environ.get("TRAIN_FILE", os.path.join(BASE, "train.jsonl"))
EVAL_FILE = os.environ.get("EVAL_FILE", os.path.join(BASE, "eval.jsonl"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
OUT_DIR = os.environ.get("OUT_DIR", "model-out")
MAX_SEQ = int(os.environ.get("MAX_SEQ", "512"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "2e-4"))
RANK = int(os.environ.get("RANK", "32"))
BATCH = int(os.environ.get("BATCH", "4"))
ACCUM = int(os.environ.get("ACCUM", "8"))
MERGE = os.environ.get("MERGE", "1") == "1"

use_bf16 = torch.cuda.is_bf16_supported()
print(f"model={MODEL_ID} seq={MAX_SEQ} epochs={EPOCHS} lr={LR} rank={RANK} "
      f"batch={BATCH}x{ACCUM} bf16={use_bf16}")

train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
eval_ds = load_dataset("json", data_files=EVAL_FILE, split="train")
print(f"train rows: {len(train_ds)}   eval rows: {len(eval_ds)}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    lora_alpha=RANK * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        warmup_steps=20,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUT_DIR,
        report_to="none",
    ),
)

trainer.train()

if MERGE:
    merged = f"{OUT_DIR}-merged"
    model.save_pretrained_merged(merged, tokenizer, save_method="merged_pretrained")
    tokenizer.save_pretrained(merged)
    print(f"merged model -> {merged}")
else:
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"adapter -> {OUT_DIR}")
