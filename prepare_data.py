#!/usr/bin/env python3
"""
MODO S1 — Scientific data preparation for physics/engineering understanding
Downloads and formats open scientific datasets for training.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ========== Scientific Datasets (all open/permissive) ==========
SCIENTIFIC_SOURCES = {
    # Physics / Engineering papers
    "arxiv_physics": {
        "dataset": "arxiv-abstracts/physics",
        "split": "train",
        "filter": lambda x: any(c in x["categories"] for c in ["physics.", "cond-mat.", "quant-ph", "hep-", "astro-ph", "gr-qc"]),
    },
    "arxiv_engineering": {
        "dataset": "arxiv-abstracts/engineering",
        "split": "train",
        "filter": lambda x: any(c in x["categories"] for c in ["eess.", "cs.RO", "cs.SY", "cs.CE", "cs.ET"]),
    },

    # Textbooks / Educational
    "openstax_physics": {
        "dataset": "openstax/physics",
        "split": "train",
    },
    "khan_academy_physics": {
        "dataset": "khan-academy/physics",
        "split": "train",
    },
    "physionet": {
        "dataset": "physionet/physionet2023",
        "split": "train",
    },

    # Materials science / Chemistry (for batteries, semiconductors)
    "materials_project": {
        "dataset": "materialsproject/mp-2024",
        "split": "train",
    },
    "pubchem_bioassays": {
        "dataset": "pubchem/bioassays",
        "split": "train",
    },

    # Scientific code
    "scientific_code": {
        "dataset": "bigcode/scientific-code",
        "split": "train",
        "filter": lambda x: x["language"] in ["python", "cpp", "fortran", "julia", "matlab", "cuda"],
    },

    # Engineering Q&A
    "engineering_se": {
        "dataset": "stackexchange/engineering",
        "split": "train",
    },
    "physics_se": {
        "dataset": "stackexchange/physics",
        "split": "train",
    },
    "electronics_se": {
        "dataset": "stackexchange/electronics",
        "split": "train",
    },

    # Datasheets / Component specs
    "datasheets": {
        "dataset": "hf-internal-testing/datasheets",
        "split": "train",
    },
}

# ========== General + Code + Instruction + Indian Languages ==========
GENERAL_SOURCES = {
    "dolma": {"dataset": "allenai/dolma", "split": "train", "subset": "v1_5", "filter": lambda x: len(x["text"]) > 500},
    "fineweb_edu": {"dataset": "HuggingFaceFW/fineweb-edu", "split": "train", "filter": lambda x: x["score"] > 3.0},
    "codeparrot": {"dataset": "codeparrot/github-code", "split": "train", "filter": lambda x: x["language"] in ["python", "cpp", "rust", "c", "cuda", "go"]},
    "ultrachat": {"dataset": "HuggingFaceH4/ultrachat_200k", "split": "train_sft"},
    "function_calling": {"dataset": "glaiveai/glaive-function-calling-v2", "split": "train"},
    "indic_trans2": {"dataset": "ai4bharat/indictrans2", "split": "train"},
    "samanantar": {"dataset": "ai4bharat/samanantar", "split": "train", "filter": lambda x: x["src_lang"] == "eng" and x["tgt_lang"] in ["hin", "tam", "tel", "kan", "mal", "ben", "guj", "mar", "ori", "pun", "asm"]},
    "bhashini": {"dataset": "ai4bharat/bhashini", "split": "train"},
}


def format_scientific(example, source_name):
    """Convert scientific data to chat format."""
    if source_name in ["arxiv_physics", "arxiv_engineering"]:
        title = example.get("title", "")
        abstract = example.get("abstract", "")
        cats = example.get("categories", "")
        text = f"Title: {title}\nAbstract: {abstract}\nCategories: {cats}"
        return {"messages": [
            {"role": "user", "content": f"Explain this research paper:\n{text}"},
            {"role": "assistant", "content": f"This {cats.split()[0] if cats else 'physics/engineering'} paper covers: {abstract[:2000]}"}
        ]}

    if source_name in ["engineering_se", "physics_se", "electronics_se"]:
        q = example.get("question", "")
        a = example.get("accepted_answer", "") or example.get("top_answer", "")
        if q and a:
            return {"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a[:4000]}
            ]}

    if source_name == "scientific_code":
        lang = example.get("language", "Python")
        doc = example.get("docstring", "")
        code = example.get("code", "")
        if code:
            return {"messages": [
                {"role": "user", "content": f"Write {lang} code for: {doc[:500]}"},
                {"role": "assistant", "content": code[:8000]}
            ]}

    if source_name in ["materials_project", "pubchem_bioassays"]:
        # Structured data - create analysis prompt
        return {"messages": [
            {"role": "user", "content": f"Analyze this scientific data: {json.dumps(example)[:3000]}"},
            {"role": "assistant", "content": f"Key properties: {list(example.keys())[:10]}. This requires domain-specific computational analysis."}
        ]}

    if source_name in ["openstax_physics", "khan_academy_physics", "physionet"]:
        content = example.get("text") or example.get("content") or str(example)[:2000]
        return {"messages": [
            {"role": "user", "content": f"Teach me: {content[:1500]}"},
            {"role": "assistant", "content": content[1500:3000] if len(content) > 1500 else "This covers fundamental physics concepts."}
        ]}

    if source_name == "datasheets":
        return {"messages": [
            {"role": "user", "content": f"Extract key specifications from this datasheet: {str(example)[:3000]}"},
            {"role": "assistant", "content": "Key specs extracted: voltage, current, timing, thermal, package. Ready for component selection."}
        ]}

    # Default
    text = example.get("text") or example.get("content") or str(example)[:2000]
    return {"messages": [{"role": "user", "content": text[:1000]}, {"role": "assistant", "content": text[1000:2000]}]}


def format_general(example, source_name):
    """Convert general data to chat format."""
    if source_name in ["dolma", "fineweb_edu"]:
        text = example.get("text", "")
        if len(text) < 200:
            return None
        # Split into chunks
        chunks = [text[i:i+4000] for i in range(0, min(len(text), 16000), 4000)]
        msgs = []
        for c in chunks:
            msgs.append({"role": "user", "content": c[:2000]})
            msgs.append({"role": "assistant", "content": c[2000:]})
        return {"messages": msgs}

    if source_name == "codeparrot":
        return {"messages": [
            {"role": "user", "content": f"Write {example.get('language', 'Python')} code: {example.get('docstring', '')[:500]}"},
            {"role": "assistant", "content": example.get("code", "")[:8000]}
        ]}

    if source_name in ["ultrachat", "function_calling"]:
        if "messages" in example:
            return {"messages": example["messages"]}
        return {"messages": [
            {"role": "user", "content": example.get("prompt", "")},
            {"role": "assistant", "content": example.get("completion", "") or example.get("response", "")}
        ]}

    if source_name in ["indic_trans2", "samanantar", "bhashini"]:
        src = example.get("src_text") or example.get("source", "")
        tgt = example.get("tgt_text") or example.get("target", "")
        lang = example.get("tgt_lang") or example.get("target_lang", "Hindi")
        if src and tgt:
            return {"messages": [
                {"role": "user", "content": f"Translate to {lang}: {src}"},
                {"role": "assistant", "content": tgt}
            ]}

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train.jsonl", help="Output JSONL path")
    ap.add_argument("--max-samples", type=int, default=500000, help="Max total samples")
    ap.add_argument("--scientific-ratio", type=float, default=0.3, help="Ratio of scientific data")
    ap.add_argument("--sources", nargs="+", help="Specific sources to include")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    all_sources = {}
    all_sources.update(SCIENTIFIC_SOURCES)
    all_sources.update(GENERAL_SOURCES)

    if args.sources:
        all_sources = {k: v for k, v in all_sources.items() if k in args.sources}

    scientific_keys = set(SCIENTIFIC_SOURCES.keys())
    scientific_target = int(args.max_samples * args.scientific_ratio)
    general_target = args.max_samples - scientific_target
    print(f"[MODO S1] Target: {scientific_target} scientific + {general_target} general = {args.max_samples} total")

    with open(args.out, "w") as f:
        # Scientific first
        sci_count = 0
        gen_count = 0

        for name, cfg in tqdm(all_sources.items(), desc="Loading sources"):
            is_sci = name in scientific_keys
            if is_sci and sci_count >= scientific_target:
                continue
            if not is_sci and gen_count >= general_target:
                continue

            try:
                print(f"  Loading {name}...")
                ds_kwargs = {"split": cfg.get("split", "train")}
                if "subset" in cfg:
                    ds = load_dataset(cfg["dataset"], cfg["subset"], **ds_kwargs)
                else:
                    ds = load_dataset(cfg["dataset"], **ds_kwargs)

                if "filter" in cfg:
                    ds = ds.filter(cfg["filter"])

                formatter = format_scientific if is_sci else format_general

                for ex in ds:
                    if is_sci and sci_count >= scientific_target:
                        break
                    if not is_sci and gen_count >= general_target:
                        break

                    formatted = formatter(ex, name)
                    if formatted and formatted.get("messages"):
                        f.write(json.dumps(formatted) + "\n")
                        if is_sci:
                            sci_count += 1
                        else:
                            gen_count += 1

            except Exception as e:
                print(f"  [SKIP] {name}: {e}")
                continue

    total = sci_count + gen_count
    print(f"[MODO S1] Done. Total samples: {total} (scientific: {sci_count}, general: {gen_count})")


if __name__ == "__main__":
    main()
