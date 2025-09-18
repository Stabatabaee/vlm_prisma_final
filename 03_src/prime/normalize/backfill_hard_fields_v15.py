#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backfill_hard_fields_v15.py

Lightweight post-processor to reduce "Not reported" by mining snippet text
with regex/keyword rules for the hardest columns:
  - Model
  - Class
  - Objectives
  - Family

It NEVER overwrites a non-empty cell. It only fills when current value is
"Not reported" (case-insensitive) or empty.

Usage:
  python backfill_hard_fields_v15.py \
    --in-csv gold_seed_merged.csv \
    --snippets-dir gold_snippets \
    --out-csv gold_seed_backfilled.csv
"""

import argparse, re, sys
from pathlib import Path
import pandas as pd

NR = "Not reported"

# --------------------------
# Canonical dictionaries
# --------------------------

# Known model names that often appear verbatim
KNOWN_MODELS = [
    "ALBEF", "TrMRG", "CXR-IRGen", "EGGCA-Net", "METransformer", "MATNet",
    "S3-Net", "MEDIFICS", "PairAug", "IRGen", "CoVT", "InVERGe",
]

# Family cues -> canonical family
FAMILY_MAP = {
    # exact / contains
    "transformer": "Transformer",
    "diffusion": "Diffusion Model",
    "latent diffusion": "Diffusion Model",
    "cnn-rnn": "CNN-RNN",
    "cnn": "CNN",
    "rnn": "RNN",
    "gpt-2": "Transformer",
    "t5": "Transformer",
    "bert": "Transformer",
    "llama": "Transformer",
    "vit": "Transformer",
    "clip": "Transformer",  # CLIP is transformer-based
}

# Class detection from phrases
CLASS_RULES = [
    (r"\bradiology report generation\b", "Radiology Report Generation"),
    (r"\breport generation\b", "Report Generation"),
    (r"\bimage[- ]text matching\b", "Image-Text Matching"),
    (r"\bimage[- ]to[- ]sequence\b", "Image-to-Sequence"),
    (r"\bimage[- ]report\b", "Image-Report"),
]

# Objectives cues (training losses/objectives—not evaluation metrics!)
OBJECTIVE_SYNONYMS = {
    r"\b(itm|image[- ]text matching)\b": "ITM",
    r"\bmlm|masked[- ]language[- ]model(ing)?\b": "MLM",
    r"\bnsp|next sentence prediction\b": "NSP",
    r"\bcontrastive\b": "contrastive",
    r"\bcross[- ]entropy\b": "cross-entropy",
    r"\breinforcement learning\b": "RL",
    r"\badversarial\b": "adversarial",
    r"\bcoverage\b": "coverage",
}

# Things we DON'T want to mistake as objectives (evaluation metrics)
METRIC_WORDS = set([
    "bleu", "rouge", "meteor", "cider", "radgraph", "bertscore",
    "accuracy", "f1", "gleu", "fid", "auroc", "radcliq", "chexbert",
])


def is_missing(x: str) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return (s == "" or s.lower() == NR.lower())


def read_snippet(snippets_dir: Path, base: str) -> str:
    # try .txt, .md, .pdf.txt in that order
    for ext in [".txt", ".md", ".pdf.txt"]:
        p = (snippets_dir / f"{base}{ext}")
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    # fallback: also look in top-level "snippets" if user pointed to split dirs
    fallback = Path("snippets")
    if fallback.exists():
        for ext in [".txt", ".md", ".pdf.txt"]:
            p = (fallback / f"{base}{ext}")
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
    return ""


def canonicalize_list(strings):
    """Lowercase, strip, dedupe while preserving order; return pretty-joined string or NR."""
    out = []
    seen = set()
    for s in strings:
        t = re.sub(r"\s+", " ", s.strip())
        if not t:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return ", ".join(out) if out else NR


def mine_model(text: str) -> str:
    hits = []
    for m in KNOWN_MODELS:
        # word boundary or underscores/hyphens
        if re.search(rf"(?i)\b{re.escape(m)}\b", text):
            hits.append(m)
    return canonicalize_list(hits)


def mine_family(text: str) -> str:
    hits = []
    tl = text.lower()
    for k, canon in FAMILY_MAP.items():
        if k in tl:
            hits.append(canon)
    return canonicalize_list(hits)


def mine_class(text: str) -> str:
    hits = []
    for pat, canon in CLASS_RULES:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(canon)
    return canonicalize_list(hits)


def mine_objectives(text: str) -> str:
    # Avoid picking up metrics: filter sentences that look like "We report BLEU/ROUGE..."
    # Still, this is heuristic: collect matched objective tokens then drop metric words.
    candidates = []
    for pat, canon in OBJECTIVE_SYNONYMS.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            candidates.append(canon)

    # also scan for generic phrases that often indicate training objectives
    extra = []
    for m in re.finditer(r"(?i)\b(loss|objective|optimiz(e|ation)|train(ed|ing))\b.*?(?:(?:\w+[- ]?){1,5})", text):
        span = m.group(0).lower()
        # if span contains metric words, skip
        if any(w in span for w in METRIC_WORDS):
            continue
        # pick a few common objective nouns if present
        if "contrastive" in span:
            extra.append("contrastive")
        if "cross-entropy" in span or "cross entropy" in span:
            extra.append("cross-entropy")
        if "adversarial" in span:
            extra.append("adversarial")

    return canonicalize_list(candidates + extra)


def main():
    ap = argparse.ArgumentParser(description="Regex/keyword backfill for hard fields.")
    ap.add_argument("--in-csv", required=True, help="Input CSV to backfill (e.g., gold_seed_merged.csv)")
    ap.add_argument("--snippets-dir", required=True, help="Directory with snippets for these File IDs.")
    ap.add_argument("--out-csv", required=True, help="Output CSV with backfilled values.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "File" not in df.columns:
        sys.exit("Input CSV missing 'File' column.")

    targets = ["Model", "Class", "Objectives", "Family"]
    for c in targets:
        if c not in df.columns:
            sys.exit(f"Input CSV missing '{c}' column.")

    snip_dir = Path(args.snippets_dir)
    if not snip_dir.exists():
        sys.exit(f"Snippets dir not found: {snip_dir}")

    before_total_nr = int(df.drop(columns=["File"]).fillna(NR).eq(NR).values.sum())

    filled_count = 0
    for idx, row in df.iterrows():
        fid = str(row["File"]).strip()
        text = ""
        # only load text if we need it
        if any(is_missing(row[c]) for c in targets):
            text = read_snippet(snip_dir, fid)

        # Model
        if is_missing(row["Model"]):
            val = mine_model(text)
            if val != NR:
                df.at[idx, "Model"] = val
                filled_count += 1

        # Class
        if is_missing(row["Class"]):
            val = mine_class(text)
            if val != NR:
                df.at[idx, "Class"] = val
                filled_count += 1

        # Objectives
        if is_missing(row["Objectives"]):
            val = mine_objectives(text)
            if val != NR:
                df.at[idx, "Objectives"] = val
                filled_count += 1

        # Family
        if is_missing(row["Family"]):
            val = mine_family(text)
            if val != NR:
                df.at[idx, "Family"] = val
                filled_count += 1

    after_total_nr = int(df.drop(columns=["File"]).fillna(NR).eq(NR).values.sum())
    df.to_csv(args.out_csv, index=False)

    print(f"✅ Backfilled CSV -> {args.out_csv}")
    print(f"Cells total: {df.shape[0]*(df.shape[1]-1)}")
    print(f"  'Not reported' before: {before_total_nr}")
    print(f"  Filled cells from snippets: {filled_count}")
    print(f"  'Not reported' after:  {after_total_nr}")


if __name__ == "__main__":
    main()
