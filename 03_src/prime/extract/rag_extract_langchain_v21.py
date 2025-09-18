#!/usr/bin/env python3
"""
v21: Heuristic post‑RAG cleaner / filler with optional seeding from a prior CSV.

Usage (standalone):
  python rag_extract_langchain_v21.py \
    --snippets-dir gold_snippets \
    --out-csv out.csv \
    --out-md  out.md \
    --evidence-mode strict \
    --in-csv v18.csv   # optional

What it does
- Reads text snippets (*.txt) under --snippets-dir (as produced by the parallel runner sharding).
- Optionally loads a prior CSV (--in-csv) and uses those rows as the starting point per file.
- Extracts fields using deterministic regex/keyword heuristics.
- Merges: prior → evidence found in snippets; gate by --evidence-mode:
    strict : only “strong” (exact-keyword) evidence may overwrite non-empty prior.
    hybrid : strong + soft evidence allowed (i.e., contextual / synonym hits).
- Writes a single CSV and a single Markdown table for the shard.

Columns (kept identical to earlier versions):
  File, Modality, Datasets (train), Datasets (eval), Paired, VLM?, Model, Class,
  Task, Vision Enc, Lang Dec, Fusion, Objectives, Family, RAG, Metrics(primary)
"""

import argparse
import csv
import os
import re
from collections import defaultdict, Counter

# ----------------------
# Field schema & helpers
# ----------------------
COLUMNS = [
    "File", "Modality", "Datasets (train)", "Datasets (eval)", "Paired", "VLM?",
    "Model", "Class", "Task", "Vision Enc", "Lang Dec", "Fusion",
    "Objectives", "Family", "RAG", "Metrics(primary)"
]

def blank_row(fname: str):
    return {
        "File": fname,
        "Modality": "",
        "Datasets (train)": "",
        "Datasets (eval)": "",
        "Paired": "",
        "VLM?": "",
        "Model": "",
        "Class": "",
        "Task": "",
        "Vision Enc": "",
        "Lang Dec": "",
        "Fusion": "",
        "Objectives": "",
        "Family": "",
        "RAG": "",
        "Metrics(primary)": "",
    }

# Canonical keyword banks (extend as needed)
DATASETS = {
    "iu x-ray": r"\b(IU[-\s]?X[-\s]?ray|Open[-\s]?i)\b",
    "mimic-cxr": r"\bMIMIC[-\s]?CXR\b",
    "chexpert": r"\bCheXpert\b",
    "nih chestx-ray14": r"\b(NIH[-\s]?ChestX[-\s]?ray14|ChestX[-\s]?ray14)\b",
    "rsna pneumonia": r"\bRSNA(?:\sPneumonia)?\b",
    "cxr-repair": r"\bCXR[-\s]?RePaiR\b",
}
MODALITIES = {
    "X-Ray": r"\b(X[-\s]?ray|radiograph|chest[-\s]?x[-\s]?ray)\b",
    "CT": r"\bCT\b",
    "MRI": r"\bMRI\b",
    "Ultrasound": r"\bultrasound\b",
}
MODELS = {
    "CheXNet": r"\bCheXNet\b",
    "ALBEF": r"\bALBEF\b",
    "BLIP-2": r"\bBLIP[-\s]?2\b",
    "BLIP": r"\bBLIP\b",
    "Flamingo": r"\bFlamingo\b",
    "LLaVA": r"\bLLaVA\b",
    "MiniGPT-4": r"\bMiniGPT[-\s]?4\b",
    "XrayGPT": r"\bXrayGPT\b",
    "TrMRG": r"\bTrMRG\b",
}
VLM_HINTS = r"\b(CLIP|ViT|vision[-\s]?language|multimodal|VLM)\b"

VISION_ENC = {
    "CNN": r"\bCNN\b|\bResNet\b",
    "ViT": r"\bViT\b|\bVision Transformer\b",
    "CLIP": r"\bCLIP\b",
    "Swin": r"\bSwin\b",
    "ResNet50": r"\bResNet[-\s]?50\b"
}
LANG_DEC = {
    "BERT": r"\bBERT\b|bioclinicalBERT",
    "GPT-2": r"\bGPT[-\s]?2\b",
    "GPT-3": r"\bGPT[-\s]?3\b",
    "LLaMA": r"\bLLaMA\b",
    "T5": r"\bT5\b",
    "Transformer": r"\bTransformer(s)?\b",
    "LSTM": r"\bLSTM\b",
    "GRU": r"\bGRU\b",
    "Vicuna": r"\bVicuna\b"
}
FUSION = {
    "cross-attention": r"\bcross[-\s]?attention\b",
    "co-attention": r"\bco[-\s]?attention\b",
    "conditioned": r"\bconditioned\b|\bconditioning\b"
}
OBJECTIVES = {
    "contrastive": r"\bcontrastive\b",
    "ITM": r"\bimage[-\s]?text matching\b|\bITM\b",
    "coverage": r"\bcoverage\b"
}
TASKS = {
    "report-generation": r"\breport (?:generation|writing)\b|\bradiology report generation\b",
    "zero-shot classification": r"\bzero[-\s]?shot classification\b",
}
METRICS = {
    "Accuracy": r"\baccuracy\b",
    "AUC": r"\bAUC\b",
    "BLEU": r"\bBLEU\b",
    "CIDEr": r"\bCIDEr\b",
    "METEOR": r"\bMETEOR\b",
    "ROUGE": r"\bROUGE\b",
    "F1": r"\bF1\b|\bF1[-\s]?score\b",
    "RadGraph": r"\bRadGraph\b",
    "RadCliQ": r"\bRadCliQ\b",
    "GLEU": r"\bGLEU\b",
    "BERTScore": r"\bBERTScore\b",
}

# “Strong” vs “soft” patterns (strict mode requires strong)
STRONG_KEYS = [
    DATASETS, MODELS, VISION_ENC, LANG_DEC, FUSION, OBJECTIVES, TASKS, METRICS
]
SOFT_KEYS = [
    MODALITIES
]

def compile_bank(bank):
    return {k: re.compile(v, flags=re.I) for k, v in bank.items()}

DATASETS_RX = compile_bank(DATASETS)
MODALITIES_RX = compile_bank(MODALITIES)
MODELS_RX = compile_bank(MODELS)
VISION_RX = compile_bank(VISION_ENC)
LANG_RX = compile_bank(LANG_DEC)
FUSION_RX = compile_bank(FUSION)
OBJ_RX = compile_bank(OBJECTIVES)
TASK_RX = compile_bank(TASKS)
METRICS_RX = compile_bank(METRICS)
VLM_RX = re.compile(VLM_HINTS, re.I)

def find_hits(text, rx_bank):
    hits = []
    for name, rx in rx_bank.items():
        if rx.search(text):
            hits.append(name)
    return hits

def uniq_join(items):
    items = [x for x in items if x]
    seen = set()
    out = []
    for x in items:
        xl = x.lower()
        if xl not in seen:
            out.append(x)
            seen.add(xl)
    return ", ".join(out)

def load_seed_csv(path):
    seed = {}
    if not path or not os.path.exists(path):
        return seed
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize name like earlier runs (text files often end with .txt)
            fname = row.get("File", "").strip()
            seed[fname] = {k: row.get(k, "") for k in COLUMNS}
    return seed

def choose_fill(prior_val, new_val, mode, is_strong):
    """
    strict: only overwrite non-empty prior if strong AND new_val not empty.
    hybrid: overwrite if new_val not empty; strong evidence preferred but not required.
    """
    new_val = (new_val or "").strip()
    prior_val = (prior_val or "").strip()
    if mode == "strict":
        if not new_val:
            return prior_val
        if not prior_val:
            return new_val if is_strong else prior_val
        # both present → allow overwrite only if strong
        return new_val if is_strong else prior_val
    else:  # hybrid
        return new_val if new_val else prior_val

def main():
    ap = argparse.ArgumentParser(description="v21 post-RAG denoiser/filler (with optional seed CSV).")
    ap.add_argument("--snippets-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--evidence-mode", choices=["strict","hybrid"], default="strict")
    ap.add_argument("--in-csv", default=None, help="Optional seed CSV (e.g., v18.csv)")
    args = ap.parse_args()

    # Seed
    seed = load_seed_csv(args.in_csv)

    # Aggregate hits per file
    per_file_hits = defaultdict(lambda: {
        "modality": Counter(),
        "datasets": Counter(),
        "models": Counter(),
        "vision": Counter(),
        "lang": Counter(),
        "fusion": Counter(),
        "objectives": Counter(),
        "tasks": Counter(),
        "metrics": Counter(),
        "vlm": False,
        "paired": False,
        "family": Counter(),   # keep for future family normalization if needed
        "rag": Counter(),      # keep for future if you tag RAG presence in text
        "class": Counter(),    # lightweight “Class” notion if text hints appear
    })

    # Walk snippets
    for fname in sorted(os.listdir(args.snippets_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(args.snippets_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue

        hits = per_file_hits[fname]

        # Strong banks
        for k in find_hits(text, DATASETS_RX): hits["datasets"][k] += 1
        for k in find_hits(text, MODELS_RX): hits["models"][k] += 1
        for k in find_hits(text, VISION_RX): hits["vision"][k] += 1
        for k in find_hits(text, LANG_RX): hits["lang"][k] += 1
        for k in find_hits(text, FUSION_RX): hits["fusion"][k] += 1
        for k in find_hits(text, OBJ_RX): hits["objectives"][k] += 1
        for k in find_hits(text, TASK_RX): hits["tasks"][k] += 1
        for k in find_hits(text, METRICS_RX): hits["metrics"][k] += 1

        # Soft banks
        for k in find_hits(text, MODALITIES_RX): hits["modality"][k] += 1

        # Booleans
        if VLM_RX.search(text):
            hits["vlm"] = True

        # Paired? crude heuristic
        if re.search(r"\bpaired (image[-\s]?text|data)\b", text, re.I) or re.search(r"\bpaired\b", text, re.I):
            hits["paired"] = True

        # Class (very soft): VLM (multimodal) vs CNN+Transformer
        if hits["vlm"]:
            hits["class"]["VLM (multimodal)"] += 1
        elif hits["vision"] or hits["lang"]:
            hits["class"]["CNN+Transformer"] += 1

    # Build rows with merging
    rows = []
    for fname in sorted(per_file_hits.keys()):
        base = seed.get(fname, blank_row(fname))

        h = per_file_hits[fname]
        # Compose candidates (strings)
        modality = uniq_join([k for k,_ in h["modality"].most_common()])
        datasets = uniq_join([k for k,_ in h["datasets"].most_common()])
        models = uniq_join([k for k,_ in h["models"].most_common()])
        vision_enc = uniq_join([k for k,_ in h["vision"].most_common()])
        lang_dec = uniq_join([k for k,_ in h["lang"].most_common()])
        fusion = uniq_join([k for k,_ in h["fusion"].most_common()])
        objectives = uniq_join([k for k,_ in h["objectives"].most_common()])
        tasks = uniq_join([k for k,_ in h["tasks"].most_common()])
        metrics = uniq_join([k for k,_ in h["metrics"].most_common()])
        klass = uniq_join([k for k,_ in h["class"].most_common()])

        vlm_val = "Yes" if h["vlm"] else ""
        paired_val = "Yes" if h["paired"] else ""

        # Strong signals if the corresponding counter is non-empty
        is_strong_models = bool(h["models"])
        is_strong_datasets = bool(h["datasets"])
        is_strong_vision = bool(h["vision"])
        is_strong_lang = bool(h["lang"])
        is_strong_fusion = bool(h["fusion"])
        is_strong_obj = bool(h["objectives"])
        is_strong_tasks = bool(h["tasks"])
        is_strong_metrics = bool(h["metrics"])
        is_strong_modality = bool(h["modality"])  # treated as soft in strict

        # Merge strategy
        base["Modality"] = choose_fill(base.get("Modality",""), modality, args.evidence_mode, is_strong_modality)
        # We do not attempt to split train/eval reliably → place into both if we have any
        base["Datasets (train)"] = choose_fill(base.get("Datasets (train)",""), datasets, args.evidence_mode, is_strong_datasets)
        base["Datasets (eval)"]  = choose_fill(base.get("Datasets (eval)",""),  datasets, args.evidence_mode, is_strong_datasets)

        base["Paired"] = choose_fill(base.get("Paired",""), paired_val, args.evidence_mode, True if paired_val else False)
        base["VLM?"]   = choose_fill(base.get("VLM?",""), vlm_val, args.evidence_mode, True if vlm_val else False)

        base["Model"]      = choose_fill(base.get("Model",""), models, args.evidence_mode, is_strong_models)
        base["Class"]      = choose_fill(base.get("Class",""), klass, args.evidence_mode, bool(klass))
        base["Task"]       = choose_fill(base.get("Task",""), tasks, args.evidence_mode, is_strong_tasks)
        base["Vision Enc"] = choose_fill(base.get("Vision Enc",""), vision_enc, args.evidence_mode, is_strong_vision)
        base["Lang Dec"]   = choose_fill(base.get("Lang Dec",""), lang_dec, args.evidence_mode, is_strong_lang)
        base["Fusion"]     = choose_fill(base.get("Fusion",""), fusion, args.evidence_mode, is_strong_fusion)
        base["Objectives"] = choose_fill(base.get("Objectives",""), objectives, args.evidence_mode, is_strong_obj)
        # Family & RAG left as-is unless prior had values; keep blank otherwise
        base["Metrics(primary)"] = choose_fill(base.get("Metrics(primary)",""), metrics, args.evidence_mode, is_strong_metrics)

        rows.append(base)

    # Also include any seed rows that didn’t appear in this shard (rare)
    for fname, prior in seed.items():
        if not any(r["File"] == fname for r in rows):
            rows.append(prior)

    # Write CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write Markdown (compact: print two blocks to keep table readable in merges)
    def md_header():
        return "| " + " | ".join(COLUMNS) + " |\n" + "|" + "---|"*len(COLUMNS) + "\n"

    with open(args.out_md, "w") as f:
        f.write(md_header())
        for r in rows:
            f.write("| " + " | ".join([r.get(c, "") or "" for c in COLUMNS]) + " |\n")

    print(f"✅ Wrote {args.out_csv} and {args.out_md}")

if __name__ == "__main__":
    main()
