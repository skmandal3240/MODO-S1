#!/usr/bin/env python3
"""
MODO S1 — LoRA fine-tuning for Nemotron 3 Ultra (53B, Apache 2.0)
Run on H100/A100. Uses Unsloth for 2x speed, 70% less VRAM.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ponytail: minimal config, no bloat
MAX_SEQ_LEN = 8192
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4

def load_data(data_path: str):
    """Load and format training data from JSONL."""
    if data_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=data_path, split="train")
    else:
        ds = load_dataset(data_path, split="train")

    def format(example):
        # Handle various formats: messages, prompt/completion, instruction/input/output
        if "messages" in example:
            return {"text": example["messages"]}
        if "prompt" in example and "completion" in example:
            return {"text": [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["completion"]}]}
        if "instruction" in example:
            out = example.get("output") or example.get("response") or ""
            return {"text": [{"role": "user", "content": example["instruction"] + (" " + example.get("input", ""))}, {"role": "assistant", "content": out}]}
        return {"text": []}

    return ds.map(format, remove_columns=ds.column_names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="nvidia/Nemotron-3-Ultra", help="Base model (HF repo)")
    ap.add_argument("--data", required=True, help="Training data JSONL or HF dataset")
    ap.add_argument("--out", default="adapters/modo-s1", help="Output adapter path")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    print(f"[MODO S1] Loading base: {args.base}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # 4-bit for 53B on H100 80GB
    )

    # LoRA config — targets all linear layers for max adaptation
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"[MODO S1] Loading data: {args.data}")
    train_data = load_data(args.data)
    print(f"[MODO S1] Samples: {len(train_data)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=4,
        packing=True,
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
        ),
    )

    print("[MODO S1] Training...")
    trainer.train()

    print(f"[MODO S1] Saving adapter to {args.out}")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # Also save merged model for easy vLLM loading
    merged_path = f"{args.out}-merged"
    print(f"[MODO S1] Merging and saving to {merged_path}")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print("[MODO S1] Done.")

if __name__ == "__main__":
    main()
