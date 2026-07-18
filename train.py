#!/usr/bin/env python3
"""
MODO S1 — Master training script
Runs: data prep → LoRA SFT → DPO personality → merge → deploy prep
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, desc, check=True):
    print(f"\n{'='*60}")
    print(f"[MODO S1] {desc}")
    print(f"[MODO S1] $ {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"[MODO S1] ✗ FAILED: {desc}")
        sys.exit(result.returncode)
    print(f"[MODO S1] ✓ {desc} complete\n")

def main():
    ap = argparse.ArgumentParser(description="MODO S1 full training pipeline")
    ap.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    ap.add_argument("--skip-sft", action="store_true", help="Skip LoRA SFT")
    ap.add_argument("--skip-dpo", action="store_true", help="Skip DPO personality")
    ap.add_argument("--skip-merge", action="store_true", help="Skip model merge")
    ap.add_argument("--base", default="nvidia/Nemotron-3-Ultra", help="Base model")
    ap.add_argument("--epochs", type=int, default=3, help="SFT epochs")
    ap.add_argument("--dpo-epochs", type=int, default=1, help="DPO epochs")
    ap.add_argument("--max-samples", type=int, default=200000, help="Max training samples")
    ap.add_argument("--scientific-ratio", type=float, default=0.3, help="Scientific data ratio")
    args = ap.parse_args()

    # Ensure directories
    Path("data").mkdir(exist_ok=True)
    Path("adapters").mkdir(exist_ok=True)

    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    MODO S1 TRAINING PIPELINE                     ║
║  Base: {args.base:<50} ║
║  Scientific ratio: {args.scientific_ratio}                                        ║
╚═══════════════════════════════════════════════════════════════════╝
""")

    # 1. Data preparation
    if not args.skip_data:
        run(
            f"python prepare_data.py --out data/train.jsonl "
            f"--max-samples {args.max_samples} "
            f"--scientific-ratio {args.scientific_ratio}",
            "Preparing training data (scientific + general + code + Indian languages)"
        )

        run(
            "python generate_dpo.py --out data/dpo_preferences.jsonl --samples 10000",
            "Generating DPO preference data for badass personality"
        )
    else:
        print("[MODO S1] Skipping data prep (using existing data/)")

    # 2. LoRA SFT
    if not args.skip_sft:
        run(
            f"python train_lora.py --base {args.base} "
            f"--data data/train.jsonl "
            f"--out adapters/modo-s1-sft "
            f"--epochs {args.epochs}",
            f"LoRA SFT on Nemotron 3 Ultra ({args.epochs} epochs)"
        )
    else:
        print("[MODO S1] Skipping SFT")

    # 3. DPO Personality
    if not args.skip_dpo:
        sft_adapter = "adapters/modo-s1-sft"
        if not Path(f"{sft_adapter}/adapter_config.json").exists():
            sft_adapter = "adapters/modo-s1-sft-merged"

        run(
            f"python train_dpo.py --base {args.base} "
            f"--adapter {sft_adapter} "
            f"--pref-data data/dpo_preferences.jsonl "
            f"--out adapters/modo-s1-dpo "
            f"--epochs {args.dpo_epochs}",
            f"DPO personality alignment ({args.dpo_epochs} epoch)"
        )
    else:
        print("[MODO S1] Skipping DPO")

    # 4. Merge final model
    if not args.skip_merge:
        final_adapter = "adapters/modo-s1-dpo"
        if not Path(f"{final_adapter}/adapter_config.json").exists():
            final_adapter = "adapters/modo-s1-sft"

        run(
            f"python -c \""
            f"from unsloth import FastLanguageModel; "
            f"m,t=FastLanguageModel.from_pretrained(model_name='{args.base}', max_seq_length=8192, dtype='bfloat16'); "
            f"m.load_adapter('{final_adapter}', adapter_name='final'); "
            f"m.set_adapter('final'); "
            f"m.save_pretrained_merged('adapters/modo-s1-final-merged', t, save_method='merged_16bit')"
            f"\"",
            "Merging final model (SFT + DPO) for vLLM deployment"
        )
    else:
        print("[MODO S1] Skipping merge")

    # Summary
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    MODO S1 TRAINING COMPLETE                     ║
║                                                                  ║
║  Final model: adapters/modo-s1-final-merged/                     ║
║  Deploy with:                                                    ║
║    docker build -t modo-s1 .                                     ║
║    docker run -p 8000:8000 -e MODO_ADAPTER=/app/adapters/modo-  ║
║      s1-final-merged modo-s1                                     ║
║                                                                  ║
║  Or deploy to GPU cloud:                                         ║
║    ./deploy.sh runpod H100                                       ║
║    ./deploy.sh vast H100                                         ║
╚═══════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
