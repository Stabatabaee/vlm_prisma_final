#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_extract_langchain_v7.py

Deterministic PRISMA metadata extractor from snippet text files.
Key improvements over v6:
- Robust title detection with line scoring + stopword/affiliation/venue filters
- Authors parsed only from a tight window under the title; filters IEEE roles/affiliations
- Stricter vocab matching + precedence for encoders/decoders
- Model-name detection whitelisted to common RRG names and proposal windows
- Safer sanitization (no code leaking into cells)
No LLM required.

Usage:
  python rag_extract_langchain_v7.py \
    --snippets-dir snippets \
    --out-md filled_papers_llama_v7.md \
    --out-csv filled_papers_llama_v7.csv
"""

import re
import csv
import argparse
from pathlib import Path
from typing import List, Tuple

COLUMNS = [
    "File", "Title", "Authors", "Modality", "Datasets",
    "Model Name", "Vision Encoder", "Language Decoder", "Fusion Strategy"
]

DATASET_LIST = [
    "MIMIC-CXR", "MIMIC CXR", "IU X-ray", "IU X-Ray",
    "Indiana University Chest X-rays", "Open-i", "OpenI", "Open-I",
    "CheXpert", "PadChest", "VinDr-CXR", "NIH ChestX-ray14",
    "ChestX-ray14", "COVIDx"
]

# Prioritized encoders (more specific first)
ENCODER_PATTERNS = [
    (r"\bvit[- ]?b/?16\b", "ViT-B/16"),
    (r"\bvit[- ]?l/?14\b", "ViT-L/14"),
    (r"\bclip[- ]?vit\b|\bclip\b", "CLIP"),
    (r"\bdeit\b", "DeiT"),
    (r"\bdino\b", "DINO"),
    (r"\bswin\b", "Swin"),
    (r"\befficientnet\b", "EfficientNet"),
    (r"\bdensenet\b", "DenseNet"),
    (r"\bresnet\b", "ResNet"),
    (r"\bcnn\b", "CNN"),
    (r"\bvision transformer\b|\bvit\b", "ViT"),
]

# Prioritized decoders (more specific first)
DECODER_PATTERNS = [
    (r"\bllama[- ]?3\b|\bllama[- ]?2\b|\bllama\b", "LLaMA"),
    (r"\broberta\b", "RoBERTa"),
    (r"\bbert\b", "BERT"),
    (r"\bdistilgpt-?2\b|\bgpt-?2\b", "GPT-2"),
    (r"\bt5\b", "T5"),
    (r"\blstm\b", "LSTM"),
    (r"\bgru\b", "GRU"),
    (r"\btransformer\b", "Transformer"),
]

# Common model names seen in radiology report generation (RRG)
KNOWN_MODELS = {
    "ALBEF", "BLIP", "CXR-IRGen", "R2Gen", "KERP", "PPKED", "HRGR",
    "M2KT", "RoentGen", "LDM", "CoAtt", "CMN", "KERP++", "PMA",
    "EALBEF", "LLM", "ViT-MAE", "MiniLM"
}
# Filter acronyms we don't want to misreport as model names
MODEL_BANLIST = {"MRI", "CT", "CXR", "CNN", "LSTM", "BERT", "VIT", "T5", "GPT", "IEEE"}

VENUE_AFFIL_STOP = {
    # venues/publishers/boilerplate
    "open access", "digital object identifier", "doi", "arxiv", "springer",
    "elsevier", "proceedings", "journal", "transactions", "conference",
    "computer vision foundation", "wacv", "cvpr", "iccv", "eccv", "isbi",
    "ieee", "acm", "kdd", "neurips", "iclr",
    # headers/process
    "received", "accepted", "published", "copyright",
    # affiliation words
    "university", "department", "school", "college", "institute", "hospital",
    "laboratory", "centre", "center", "faculty", "research group"
}

ROLE_TOKENS = {
    "member", "senior member", "student member", "fellow", "ieee", "acm",
    "phd", "md", "msc", "bsc"
}

FUSION_PATTERNS = [
    (r"\bcross[- ]?attention\b", "Cross-Attention"),
    (r"\bco[- ]?attention\b|\bcoattention\b", "Co-Attention"),
    (r"\bearly fusion\b", "Early Fusion"),
    (r"\blate fusion\b", "Late Fusion"),
    (r"\bconcatenation\b", "Concatenation"),
    (r"\bgated fusion\b", "Gated Fusion"),
    (r"\battention fusion\b", "Attention Fusion"),
    (r"\bmultimodal encoder\b|\bmulti[- ]?modal encoder\b", "Multimodal Encoder"),
]

def sanitize_cell(s: str) -> str:
    if not s:
        return "Not reported"
    s = s.replace("|", "/")
    s = re.sub(r"```.*?```", " ", s, flags=re.S)  # strip code fences if present
    s = re.sub(r"`+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "Not reported"

def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        k = x.strip()
        if not k:
            continue
        if k not in seen:
            out.append(k); seen.add(k)
    return out

def load_text(snippets_dir: str, paper_id: str) -> str:
    fp = Path(snippets_dir) / f"{paper_id}.txt"
    if not fp.exists():
        return ""
    return fp.read_text(encoding="utf-8", errors="ignore")

# ---------------- Title detection ---------------- #

def _looks_like_title(line: str) -> bool:
    s = line.strip()
    if len(s) < 8 or len(s) > 180:
        return False
    if re.search(r"http[s]?://|www\.", s, re.I):
        return False
    # avoid lines with lots of commas (often affiliations)
    if s.count(",") > 3:
        return False
    # reject lines dominated by caps (affiliations/venues)
    letters = re.sub(r"[^A-Za-z]", "", s)
    if not letters:
        return False
    lower_ratio = sum(1 for c in s if c.islower()) / max(1, sum(1 for c in s if c.isalpha()))
    if lower_ratio < 0.25:  # mostly caps -> likely NOT a title
        return False
    # reject if venue/affiliation stopwords present
    low = s.lower()
    if any(tok in low for tok in VENUE_AFFIL_STOP):
        return False
    # reasonable tokenization
    if len(re.findall(r"[A-Za-z][A-Za-z\-]+", s)) < 4:
        return False
    return True

def guess_title(text: str, paper_id: str) -> Tuple[str, int]:
    """
    Return (title, title_line_index). If not found, prettify filename.
    """
    if not text:
        pretty = re.sub(r"[_\-]+", " ", paper_id).strip().title()
        return (pretty or "Not reported", -1)

    # Work within head region
    head = text[:4000]
    abs_m = re.search(r"\babstract\b", head, re.I)
    if abs_m:
        head = head[:abs_m.start()]

    lines = [l.strip() for l in head.splitlines()]
    candidates = []
    for i, l in enumerate(lines[:40]):  # only early lines
        if not l.strip():
            continue
        if _looks_like_title(l):
            # score: prefer longer, fewer commas, and earlier lines
            length_score = min(len(l), 160) / 160.0
            comma_pen = 1.0 / (1 + l.count(","))
            pos_bonus = 1.0 / (1 + i)  # earlier is better
            score = 0.55 * length_score + 0.25 * comma_pen + 0.20 * pos_bonus
            candidates.append((score, i, l.rstrip(" .:-")))
    if candidates:
        candidates.sort(reverse=True)
        best = candidates[0]
        return best[2], best[1]

    # fallback: filename prettified
    pretty = re.sub(r"[_\-]+", " ", paper_id).strip().title()
    return (pretty or "Not reported", -1)

# ---------------- Authors detection ---------------- #

NAME_PATTERN = r"\b[A-Z][a-z]+(?:[-' ][A-Z][a-z]+){0,2}\b"  # up to 3 tokens
ROLE_CLEAN = re.compile(r"\b(Member|Senior Member|Student Member|Fellow|IEEE|ACM)\b", re.I)

def clean_author_line(s: str) -> str:
    s = re.sub(r"\S+@\S+\.\S+", " ", s)                # remove emails
    s = ROLE_CLEAN.sub(" ", s)                         # remove roles
    s = re.sub(r"[\*\d\^\[\]\(\)º°†‡§•]+", " ", s)     # remove markers
    s = re.sub(r"\s+", " ", s).strip()
    return s

def guess_authors(text: str, title_idx: int) -> str:
    if not text:
        return "Not reported"
    lines = [l.strip() for l in text[:6000].splitlines()]

    # search window: a few non-empty lines after title until Abstract/Keywords
    start = 0 if title_idx < 0 else max(0, title_idx + 1)
    end = min(len(lines), start + 12)
    block = []
    for i in range(start, end):
        l = lines[i]
        if not l:
            if block:
                break
            else:
                continue
        if re.search(r"\babstract\b|\bkeywords\b|\bindex terms\b", l, re.I):
            break
        block.append(l)

    if not block:
        # fallback: look for a "by ..." line near top
        for i, l in enumerate(lines[:30]):
            if re.search(r"^\s*by\s+.+", l, re.I):
                block = [re.sub(r"^\s*by\s+", "", l, flags=re.I)]
                break

    if not block:
        return "Not reported"

    # join small block and clean
    joined = clean_author_line(" ".join(block))
    # extract names
    names = re.findall(NAME_PATTERN, joined)
    # Filter out affiliation-like tokens
    names = [n for n in names if n.lower() not in VENUE_AFFIL_STOP and n.lower() not in ROLE_TOKENS]
    # Keep 2–4 token names only
    names = [n for n in names if 2 <= len(n.split()) <= 4]
    names = uniq_preserve(names)
    if not names:
        return "Not reported"
    return ", ".join(names[:15])

# ---------------- Other fields ---------------- #

def guess_modality(text: str) -> str:
    if not text:
        return "Not reported"
    t = text.lower()
    if re.search(r"\bx[- ]?ray\b|\bcxr\b", t):
        return "X-Ray"
    if re.search(r"\bmri\b", t):
        return "MRI"
    if re.search(r"\bcomputed tomography\b|\bct\b", t):
        return "CT"
    if re.search(r"\bultrasound\b|\bsonography\b|\bus\b", t):
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

def guess_encoder(text: str) -> str:
    if not text:
        return "Not reported"
    tl = text.lower()
    for pat, label in ENCODER_PATTERNS:
        if re.search(pat, tl):
            return label
    return "Not reported"

def guess_decoder(text: str) -> str:
    if not text:
        return "Not reported"
    tl = text.lower()
    for pat, label in DECODER_PATTERNS:
        if re.search(pat, tl):
            return label
    return "Not reported"

def guess_fusion(text: str) -> str:
    if not text:
        return "Not reported"
    tl = text.lower()
    for pat, label in FUSION_PATTERNS:
        if re.search(pat, tl):
            return label
    return "Not reported"

def guess_model_name(text: str) -> str:
    if not text:
        return "Not reported"

    # prefer windows around propose/present/introduce/develop/call
    windows = []
    for m in re.finditer(r"(propose|present|introduce|develop|call(?:ed)?)\b.{0,160}", text, flags=re.I):
        s = max(0, m.start() - 100); e = min(len(text), m.end() + 100)
        windows.append(text[s:e])
    if not windows:
        windows = [text[:5000]]

    hits = []
    # all-caps-ish acronyms w/ optional hyphens
    acronym_pat = r"\b([A-Z][A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,}){0,3})\b"

    for w in windows:
        # known models first
        for km in KNOWN_MODELS:
            if re.search(rf"\b{re.escape(km)}\b", w):
                hits.append(km)
        # capture other acronyms
        for a in re.findall(acronym_pat, w):
            if a.upper() in MODEL_BANLIST:
                continue
            # avoid very generic short tokens
            if len(a) < 3:
                continue
            hits.append(a)

    hits = uniq_preserve(hits)
    # prune obvious datasets/encoders that may sneak in
    bad_like = {"CXR", "VI", "VIT", "BERT", "GPT", "MRI", "CT"}
    hits = [h for h in hits if h.upper() not in bad_like]
    return ", ".join(hits[:4]) if hits else "Not reported"

# ---------------- IO helpers ---------------- #

def row_to_markdown(cells: List[str]) -> str:
    cells = [sanitize_cell(x) for x in cells]
    return "| " + " | ".join(cells) + " |"

def extract_for_paper(snippets_dir: str, paper_id: str) -> List[str]:
    text = load_text(snippets_dir, paper_id)

    title, t_idx = guess_title(text, paper_id)
    authors = guess_authors(text, t_idx)
    modality = guess_modality(text)
    datasets = guess_datasets(text)
    model_name = guess_model_name(text)
    vision = guess_encoder(text)
    decoder = guess_decoder(text)
    fusion = guess_fusion(text)

    return [
        f"{paper_id}.pdf",
        title or "Not reported",
        authors or "Not reported",
        modality or "Not reported",
        datasets or "Not reported",
        model_name or "Not reported",
        vision or "Not reported",
        decoder or "Not reported",
        fusion or "Not reported",
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True, help="Directory containing <paper>.txt snippet files")
    ap.add_argument("--out-md", default="filled_papers_llama_v7.md")
    ap.add_argument("--out-csv", default="filled_papers_llama_v7.csv")
    args = ap.parse_args()

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
                row = extract_for_paper(args.snippets_dir, pid)
            except Exception as e:
                row = [f"{pid}.pdf"] + ["Not reported"] * (len(COLUMNS) - 1)
                print(f"[WARN] {pid}: {e}")
            fmd.write(row_to_markdown(row) + "\n")
            cw.writerow(row)
            print(f"✓ {pid}")

    print(f"✅ Wrote {args.out_md} and {args.out_csv}")

if __name__ == "__main__":
    main()
