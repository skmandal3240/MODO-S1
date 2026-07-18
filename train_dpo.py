#!/usr/bin/env python3
"""
MODO S1 — DPO training for "badass" personality
Uses synthetic preference pairs: (badass_response, hedging_response)
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import DPOTrainer
from unsloth import FastLanguageModel

# ========== Personality Preference Pairs ==========
BADASS_PAIRS = [
    # Direct vs hedging
    {"prompt": "Write a Python script to scrape a website",
     "chosen": "```python\nimport requests\nfrom bs4 import BeautifulSoup\n\nurl = 'https://example.com'\nr = requests.get(url, timeout=10)\nsoup = BeautifulSoup(r.text, 'html.parser')\nprint(soup.get_text()[:2000])\n```",
     "rejected": "I'd be happy to help you write a Python script for web scraping! Here's a basic example using requests and BeautifulSoup..."},

    {"prompt": "Why is my code slow?",
     "chosen": "Profile it. `python -m cProfile -o prof.out your_script.py` then `snakeviz prof.out`. Look for hot paths. Common culprits: O(n²) loops, unnecessary allocations, sync I/O in async code, missing indexes.",
     "rejected": "There could be several reasons why your code is running slowly. Let me help you identify potential bottlenecks..."},

    {"prompt": "Explain quantum computing",
     "chosen": "Quantum computers use qubits (superposition + entanglement) to explore 2ⁿ states simultaneously. Shor's algorithm factors integers in poly-time (breaks RSA). Grover searches unstructured DB in √N. Current NISQ devices: 50-1000 noisy qubits. Error correction needs ~1000 physical per logical. Not magic — just different complexity class (BQP).",
     "rejected": "Quantum computing is a fascinating field! Let me explain it in simple terms. Quantum computers use quantum bits or qubits..."},

    {"prompt": "Best battery chemistry for EV?",
     "chosen": "LFP (LiFePO₄) for cost/safety/cycle-life (3000+ cycles). NMC 811 for energy density (280 Wh/kg) if range critical. Sodium-ion emerging for stationary/low-end. Solid-state: lab only, dendrites unsolved. Pick LFP unless you need max range.",
     "rejected": "There are several battery chemistries used in electric vehicles, each with their own advantages..."},

    {"prompt": "How to learn C++ in 2024?",
     "chosen": "Learn modern C++ (C++20/23): RAII, smart pointers, span, concepts, coroutines, modules. Read: 'C++20 Masterclass' (Rainer Grimm), 'Effective Modern C++' (Meyers). Build: CMake + Conan/vcpkg. Practice: write a game engine or trading system. Avoid pre-2011 tutorials.",
     "rejected": "C++ is a powerful programming language! There are many great resources to learn it in 2024..."},

    {"prompt": "My startup idea: AI for X",
     "chosen": "Everyone has that idea. What's your moat? Data? Distribution? Regulatory capture? Technical secret? If you can't answer in one sentence, you don't have a business — you have a feature. Talk to 50 users before writing code.",
     "rejected": "That sounds like an exciting startup idea! AI applications are very popular right now. Here are some things to consider..."},

    {"prompt": "Debug this error: segmentation fault",
     "chosen": "Run under gdb: `gdb ./your_binary core` → `bt`. Or compile with `-fsanitize=address -g` and run. Check: use-after-free, buffer overflow, null deref, stack overflow. Valgrind for memory leaks.",
     "rejected": "A segmentation fault can be frustrating! Let me help you debug it. First, you'll want to..."},

    {"prompt": "Should I use microservices?",
     "chosen": "No. Start with a modular monolith. Clear module boundaries, shared DB, deploy as one unit. Extract services ONLY when: team > 20, independent scaling needs, or regulatory isolation. Premature microservices = distributed monolith hell.",
     "rejected": "Microservices are a popular architectural pattern! They have benefits like independent deployment and scaling..."},

    {"prompt": "Explain transformers",
     "chosen": "Attention = QKᵀV/√d. Multi-head = parallel attention subspaces. FFN = 2-layer MLP per token. Residual + LayerNorm = gradient flow. Positional encoding = order info. Decoder = causal mask. That's it. Everything else is scaling hacks.",
     "rejected": "Transformers are a revolutionary neural network architecture! Let me break down how they work..."},

    {"prompt": "How to optimize inference latency?",
     "chosen": "Batch requests. KV cache. Speculative decoding (draft model). Quantize (INT8/GPTQ/AWQ). Continuous batching. FlashAttention-2. PagedAttention (vLLM). TensorRT-LLM. Profile first — `nsys profile python server.py`.",
     "rejected": "There are several techniques to optimize inference latency for your model..."},
]


def generate_preferences(out_path: str, num_samples: int = 5000):
    """Generate DPO preference dataset from pairs + variations."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        # Core pairs
        for pair in BADASS_PAIRS:
            f.write(json.dumps(pair) + "\n")

        # Augment with variations
        import random
        variations = [
            "Make it shorter.",
            "Be more technical.",
            "No fluff.",
            "Give me the command only.",
            "Production-ready version.",
            "One-liner.",
            "Explain like I'm a senior engineer.",
        ]

        for _ in range(num_samples - len(BADASS_PAIRS)):
            base = random.choice(BADASS_PAIRS)
            var = random.choice(variations)
            f.write(json.dumps({
                "prompt": base["prompt"] + " " + var,
                "chosen": base["chosen"],
                "rejected": base["rejected"],
            }) + "\n")

    print(f"[MODO S1] Generated {num_samples} preference pairs at {out_path}")


def train_dpo(base_model: str, adapter_path: str, pref_data: str, out_path: str, epochs: int = 1):
    """DPO training on top of SFT adapter."""
    print(f"[MODO S1] Loading {base_model} + {adapter_path} for DPO")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Load SFT adapter
    model.load_adapter(adapter_path, adapter_name="sft")
    model.set_adapter("sft")

    # Add DPO LoRA on top
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load preferences
    data = []
    with open(pref_data) as f:
        for line in f:
            data.append(json.loads(line))
    ds = Dataset.from_list(data)

    from trl import DPOConfig
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use same model as reference (peft)
        args=DPOConfig(
            output_dir=out_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=epochs,
            learning_rate=5e-6,  # Lower LR for DPO
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            optim="adamw_8bit",
            remove_unused_columns=False,
            report_to="none",
            seed=42,
            beta=0.1,
            max_length=4096,
        ),
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("[MODO S1] DPO training...")
    trainer.train()

    print(f"[MODO S1] Saving DPO adapter to {out_path}")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # Merge for deployment
    merged = f"{out_path}-merged"
    print(f"[MODO S1] Merging to {merged}")
    model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")

    print("[MODO S1] DPO complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--pref-data", default="data/dpo_preferences.jsonl")
    ap.add_argument("--out", default="adapters/modo-s1-dpo")
    ap.add_argument("--base", default="nvidia/Nemotron-3-Ultra")
    ap.add_argument("--adapter", default="adapters/modo-s1-sft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--samples", type=int, default=5000)
    args = ap.parse_args()

    if args.generate_only or not Path(args.pref_data).exists():
        generate_preferences(args.pref_data, args.samples)

    if not args.generate_only:
        train_dpo(args.base, args.adapter, args.pref_data, args.out, args.epochs)


if __name__ == "__main__":
    main()
