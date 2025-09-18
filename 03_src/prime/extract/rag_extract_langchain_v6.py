#!/usr/bin/env python3
"""
rag_extract_langchain_v6.py

Goal: Deterministic PRISMA field extraction that avoids LLM hallucinations.
- Parses fields directly from <snippets-dir>/<paper>.txt using regex + lexical sweeps
- Uses a curated datasets/architectures vocabulary
- Produces filled_papers_llama_v6.{md,csv} by default
- Keeps Chroma around (optional) but does not depend on it
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict

# (Optional) Keep these to match your previous environment; not strictly required for v6 logic.
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    HAVE_CHROMA = True
except Exception:
    HAVE_CHROMA = False

DB_DIR = "chroma_db"

COLUMNS = [
    "File", "Title", "Authors", "Modality", "Datasets",
    "Model Name", "Vision Encoder", "Language Decoder", "Fusion Strategy"
]

DATASET_LIST = [
    "MIMIC-CXR", "MIMIC CXR", "IU X-ray", "Indiana University Chest X-rays",
    "Open-i", "Open-I", "CheXpert", "PadChest", "VinDr-CXR",
    "NIH ChestX-ray14", "ChestX-ray14", "COVIDx"
]

VISION_ENCODERS = [
    "ViT-B/16", "ViT-L/14", "ViT", "CLIP", "ResNet", "DenseNet", "CNN",
    "Swin", "EfficientNet", "DINO", "DeiT"
]
LANG_DECODERS = [
    "LLaMA", "LLaMa", "LLaMA-2", "LLaMA-3", "BERT", "RoBERTa", "GPT-2",
    "DistilGPT-2", "T5", "Transformer", "LSTM", "GRU"
]
FUSION_STRATEGIES = [
    "Cross-Attention", "Co-Attention", "Early Fusion", "Late Fusion",
    "Concatenation", "Multimodal Encoder", "Gated Fusion", "Attention Fusion"
]

def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        k = x.strip()
        if not k: 
            continue
        if k not in seen:
            out.append(k); seen.add(k)
    return out

def sanitize_cell(s: str) -> str:
    if not s:
        return "Not reported"
    s = s.replace("|", "/").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    # keep fields short-ish to avoid spillover
    if len(s) > 140:
        s = s[:140].rstrip(",;/ ") + "…"
    return s if s else "Not reported"

def row_to_markdown(cells: List[str]) -> str:
    cells = [sanitize_cell(x) for x in cells]
    if len(cells) < len(COLUMNS):
        cells += ["Not reported"] * (len(COLUMNS) - len(cells))
    return "| " + " | ".join(cells) + " |"

# -------------------- Parsing helpers --------------------

def load_snippet(snippets_dir: str, pid: str) -> str:
    fp = Path(snippets_dir) / f"{pid}.txt"
    if fp.exists():
        return fp.read_text(encoding="utf-8", errors="ignore")
    return ""

def guess_title(text: str, pid: str) -> str:
    if not text:
        # prettify filename as last resort
        t = re.sub(r'[_\-]+', ' ', pid).strip()
        return t.title() if t else "Not reported"
    # consider only the first chunk before ABSTRACT
    head = text[:3000]
    abs_m = re.search(r"\babstract\b", head, re.I)
    if abs_m:
        head = head[:abs_m.start()]
    # pick a candidate line that looks like a title
    lines = [l.strip() for l in head.splitlines() if l.strip()]
    candidates = []
    for l in lines[:20]:
        if len(l) < 8 or len(l) > 200: 
            continue
        if re.search(r"@|doi|arxiv|http|www\.", l, re.I):
            continue
        letters = re.sub(r"[^A-Za-z]", "", l)
        if not letters:
            continue
        # avoid ALL CAPS or affiliation-like lines
        if letters.isupper():
            continue
        # should have at least 4 word-like tokens
        if len(re.findall(r"[A-Za-z][A-Za-z\-]+", l)) < 4:
            continue
        candidates.append(l.rstrip(" .:-"))
    if candidates:
        # choose the longest (often closest to real title)
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    # fallback: prettify filename
    t = re.sub(r'[_\-]+', ' ', pid).strip()
    return t.title() if t else "Not reported"

def guess_authors(text: str) -> str:
    if not text:
        return "Not reported"
    head = text[:4000]
    abs_m = re.search(r"\babstract\b", head, re.I)
    if abs_m:
        head = head[:abs_m.start()]
    # strip emails, footnote markers
    head = re.sub(r"\S+@\S+\.\S+", " ", head)
    head = re.sub(r"[\*\d\^\[\]\(\)º°†‡§•]+", " ", head)
    # collapse spaces
    head = re.sub(r"\s+", " ", head)
    # heuristic: cut after a phrase like "Abstract", "Keywords" if present
    head = re.split(r"\bkeywords\b|\bindex terms\b", head, flags=re.I)[0]
    # Find personal names: sequences of 2–4 capitalized tokens
    name_pat = r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+){0,2}(?:\s+[A-Z][a-z]+(?:[- ][A-Z][a-z]+){0,2})\b"
    names = re.findall(name_pat, head)
    # Filter out likely false positives
    bad = {"Abstract", "Introduction", "Results", "Methods", "Conclusion"}
    names = [n for n in names if n not in bad and len(n.split()) <= 4]
    names = uniq_preserve(names)
    # If we captured *a lot*, it's probably mixing affiliations—cut to first 12
    if names:
        return ", ".join(names[:12])
    return "Not reported"

def guess_modality(text: str) -> str:
    if not text:
        return "Not reported"
    if re.search(r"\bcxr\b|\bx[- ]?ray\b", text, re.I):
        return "X-Ray"
    if re.search(r"\bmri\b", text, re.I):
        return "MRI"
    if re.search(r"\bcomputed tomography\b|\bct\b", text, re.I):
        return "CT"
    if re.search(r"\bultrasound\b|\bus\b", text, re.I):
        return "Ultrasound"
    return "Not reported"

def guess_datasets(text: str) -> str:
    if not text:
        return "Not reported"
    hits = []
    tl = text.lower()
    for ds in DATASET_LIST:
        if ds.lower() in tl:
            hits.append(ds)
    hits = uniq_preserve(hits)
    return ", ".join(hits) if hits else "Not reported"

def guess_vision_encoder(text: str) -> str:
    if not text: return "Not reported"
    tl = text.lower()
    # prefer specific variants before generic
    if "vit-b/16" in tl: return "ViT-B/16"
    if re.search(r"\b(deit)\b", tl): return "DeiT"
    if re.search(r"\bclip\b", tl): return "CLIP"
    if re.search(r"\bdensenet\b", tl): return "DenseNet"
    if re.search(r"\bswin\b", tl): return "Swin"
    if re.search(r"\befficientnet\b", tl): return "EfficientNet"
    if re.search(r"\bdino\b", tl): return "DINO"
    if re.search(r"\bresnet\b", tl): return "ResNet"
    if re.search(r"\bcnn\b", tl): return "CNN"
    if re.search(r"\bvision transformer\b|\bvit\b", tl): return "ViT"
    return "Not reported"

def guess_language_decoder(text: str) -> str:
    if not text: return "Not reported"
    tl = text.lower()
    if re.search(r"\bllama[- ]?2\b|\bllama[- ]?3\b|\bllama\b", tl): return "LLaMA"
    if re.search(r"\broberta\b", tl): return "RoBERTa"
    if re.search(r"\bbert\b", tl): return "BERT"
    if re.search(r"\bgpt-2\b|\bdistilgpt-2\b", tl): return "GPT-2"
    if re.search(r"\bt5\b", tl): return "T5"
    if re.search(r"\blstm\b", tl): return "LSTM"
    if re.search(r"\bgru\b", tl): return "GRU"
    if re.search(r"\btransformer\b", tl): return "Transformer"
    return "Not reported"

def guess_fusion_strategy(text: str) -> str:
    if not text: return "Not reported"
    tl = text.lower()
    if "cross-attention" in tl or "cross attention" in tl: return "Cross-Attention"
    if "co-attention" in tl or "coattention" in tl: return "Co-Attention"
    if "early fusion" in tl: return "Early Fusion"
    if "late fusion" in tl: return "Late Fusion"
    if "concatenation" in tl: return "Concatenation"
    if "multimodal encoder" in tl or "multi-modal encoder" in tl: return "Multimodal Encoder"
    if "gated fusion" in tl: return "Gated Fusion"
    if "attention fusion" in tl: return "Attention Fusion"
    return "Not reported"

def guess_model_name(text: str) -> str:
    """Try to capture distinctive model names like ALBEF, CXR-IRGen, etc."""
    if not text:
        return "Not reported"
    # Look near verbs: propose/present/introduce/call
    windows = []
    for m in re.finditer(r"(propose|present|introduce|develop|call(?:ed)?)\b.{0,120}", text, flags=re.I):
        s = max(0, m.start() - 80); e = min(len(text), m.end() + 80)
        windows.append(text[s:e])
    if not windows:
        windows = [text[:4000]]
    name_pat = r"\b([A-Z][A-Z0-9]+(?:[-/][A-Z0-9]+){0,3})\b"
    # whitelist some known names too
    known = ["ALBEF", "BLIP", "CXR-IRGen", "R2Gen", "KERP", "PPKED", "HRGR", "M2KT", "RoentGen", "LDM"]
    hits = []
    for w in windows:
        hits += re.findall(name_pat, w)
        for kn in known:
            if re.search(rf"\b{re.escape(kn)}\b", w):
                hits.append(kn)
    hits = [h for h in hits if len(h) >= 3 and not h.isdigit()]
    hits = uniq_preserve(hits)
    # prune overly generic tokens
    bad = {"MRI", "CT", "CNN", "LSTM", "T5", "BERT", "GPT", "VIT"}
    hits = [h for h in hits if h.upper() not in bad]
    return ", ".join(hits[:4]) if hits else "Not reported"

def extract_row(snippets_dir: str, pid: str) -> List[str]:
    text = load_snippet(snippets_dir, pid)

    title = guess_title(text, pid)
    authors = guess_authors(text)
    modality = guess_modality(text)
    datasets = guess_datasets(text)
    model_name = guess_model_name(text)
    venc = guess_vision_encoder(text)
    ldec = guess_language_decoder(text)
    fusion = guess_fusion_strategy(text)

    cells = [
        f"{pid}.pdf",
        title if title else "Not reported",
        authors if authors else "Not reported",
        modality if modality else "Not reported",
        datasets if datasets else "Not reported",
        model_name if model_name else "Not reported",
        venc if venc else "Not reported",
        ldec if ldec else "Not reported",
        fusion if fusion else "Not reported",
    ]
    return cells

# -------------------- Main --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True, help="Directory with <paper>.txt snippets")
    ap.add_argument("--out-md", default="filled_papers_llama_v6.md")
    ap.add_argument("--out-csv", default="filled_papers_llama_v6.csv")
    ap.add_argument("--use-chroma", action="store_true",
                    help="Optional: open Chroma DB (not required for v6 extraction)")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.use_chroma and HAVE_CHROMA:
        try:
            _embed = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
            _ = Chroma(persist_directory=DB_DIR, embedding_function=_embed, collection_name="papers")
        except Exception:
            pass  # purely optional

    papers = sorted([p.stem for p in Path(args.snippets_dir).glob("*.txt")])
    print(f"▶ Extracting {len(papers)} papers…")

    with open(args.out_md, "w", encoding="utf-8") as fmd, \
         open(args.out_csv, "w", encoding="utf-8", newline="") as fcsv:

        # headers
        fmd.write("| " + " | ".join(COLUMNS) + " |\n")
        fmd.write("|" + "|".join(["---"] * len(COLUMNS)) + "|\n")
        cw = csv.writer(fcsv); cw.writerow(COLUMNS)

        for pid in papers:
            try:
                row = extract_row(args.snippets_dir, pid)
                fmd.write(row_to_markdown(row) + "\n")
                cw.writerow(row)
                print(f"✓ {pid}")
            except Exception as e:
                fallback = [f"{pid}.pdf"] + ["Not reported"] * (len(COLUMNS) - 1)
                fmd.write(row_to_markdown(fallback) + "\n")
                cw.writerow(fallback)
                print(f"[WARN] {pid}: {e}")

    print(f"✅ Wrote {args.out_md} and {args.out_csv}")

if __name__ == "__main__":
    main()
