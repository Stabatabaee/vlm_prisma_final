#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_extract_langchain_v8.py

Deterministic PRISMA metadata extractor from snippet text files.
Key fixes vs v7:
- Title wrapping across lines + strong rejection of email/venue/affiliation lines
- Authors parsed from a tight post-title window using stricter name regex
- Canonical dataset mapping (IU X-ray/Open-i/MIMIC-CXR variants unified)
- Model names restricted to a curated whitelist (adds EGGCA-Net, EALBEF, CXR-IRGen, RoentGen…)
- Safer sanitization

Usage:
  python rag_extract_langchain_v8.py \
    --snippets-dir snippets \
    --out-md filled_papers_llama_v8.md \
    --out-csv filled_papers_llama_v8.csv
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

# -------- Canonical datasets (with aliases) -------- #
DATASET_CANON = {
    "mimic-cxr": "MIMIC-CXR",
    "mimic cxr": "MIMIC-CXR",
    "iu x-ray": "IU X-ray",
    "iu x ray": "IU X-ray",
    "indiana university chest x-rays": "Indiana University Chest X-rays",
    "open-i": "Open-i",
    "open i": "Open-i",
    "openi": "Open-i",
    "chexpert": "CheXpert",
    "padchest": "PadChest",
    "vindr-cxr": "VinDr-CXR",
    "vindr cxr": "VinDr-CXR",
    "nih chestx-ray14": "NIH ChestX-ray14",
    "chestx-ray14": "ChestX-ray14",
    "covidx": "COVIDx",
}

# -------- Encoders/Decoders (prioritized patterns) -------- #
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

# -------- Model whitelist (RRG domain) -------- #
KNOWN_MODELS = {
    "ALBEF", "BLIP", "CXR-IRGen", "R2Gen", "KERP", "PPKED", "HRGR",
    "M2KT", "RoentGen", "LDM", "CoAtt", "CMN", "PMA",
    "EALBEF", "EGGCA-Net", "EGGCA", "KERP++"
}
MODEL_BANLIST = {"MRI", "CT", "CXR", "CNN", "LSTM", "BERT", "VIT", "T5", "GPT", "IEEE", "LLM", "BEF"}

VENUE_AFFIL_STOP = {
    # venues/publishers/boilerplate
    "open access", "digital object identifier", "doi", "arxiv", "springer",
    "elsevier", "proceedings", "journal", "transactions", "conference",
    "computer vision foundation", "wacv", "cvpr", "iccv", "eccv", "isbi",
    "ieee", "acm", "kdd", "neurips", "iclr",
    # headers/process
    "received", "accepted", "published", "copyright",
    "corresponding author", "corresponding authors",
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

NAME_PATTERN = r"\b[A-Z][a-z]+(?:[-' ][A-Z][a-z]+){1,2}\b"  # 2–3 tokens

# ---------------- Utilities ---------------- #

def sanitize_cell(s: str) -> str:
    if not s:
        return "Not reported"
    s = s.replace("|", "/")
    s = re.sub(r"```.*?```", " ", s, flags=re.S)  # strip fenced code
    s = re.sub(r"`+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
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

def _bad_title_line(s: str) -> bool:
    low = s.lower()
    if "@" in s:                                  # emails
        return True
    if any(tok in low for tok in VENUE_AFFIL_STOP):
        return True
    # too many commas (affiliations) or ends with email/URL-like
    if s.count(",") > 3:
        return True
    if re.search(r"http[s]?://|www\.", s, re.I):
        return True
    # mostly caps (affiliations/headers)
    letters = re.sub(r"[^A-Za-z]", "", s)
    if not letters:
        return True
    alpha = sum(1 for c in s if c.isalpha())
    lower_ratio = sum(1 for c in s if c.islower()) / max(1, alpha)
    if lower_ratio < 0.25:
        return True
    return False

def _looks_like_title(line: str) -> bool:
    s = line.strip()
    if len(s) < 8 or len(s) > 180:
        return False
    if _bad_title_line(s):
        return False
    # reasonable wordiness
    if len(re.findall(r"[A-Za-z][A-Za-z\-]+", s)) < 4:
        return False
    return True

def _title_continuation(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 160:
        return False
    if _bad_title_line(s):
        return False
    if re.search(r"\babstract\b|\bkeywords\b|\bindex terms\b", s, re.I):
        return False
    # continuation often starts lowercase or not a full sentence
    if s and s[0].islower():
        return True
    # lines without terminal period/colon can be continuations
    if not re.search(r"[.:!?]\s*$", s):
        # avoid obvious affiliation hints
        if not any(tok in s.lower() for tok in ("university", "department", "institute")):
            return True
    return False

def guess_title(text: str, paper_id: str) -> Tuple[str, int]:
    if not text:
        pretty = re.sub(r"[_\-]+", " ", paper_id).strip().title()
        return (pretty or "Not reported", -1)

    head = text[:4500]
    abs_m = re.search(r"\babstract\b", head, re.I)
    if abs_m:
        head = head[:abs_m.start()]

    lines = [l.strip() for l in head.splitlines()]
    candidates = []
    for i, l in enumerate(lines[:50]):  # scan a bit deeper
        if _looks_like_title(l):
            length_score = min(len(l), 160) / 160.0
            comma_pen = 1.0 / (1 + l.count(","))
            pos_bonus = 1.0 / (1 + i)
            score = 0.55 * length_score + 0.25 * comma_pen + 0.20 * pos_bonus
            candidates.append((score, i, l.rstrip(" .:-")))
    if not candidates:
        pretty = re.sub(r"[_\-]+", " ", paper_id).strip().title()
        return (pretty or "Not reported", -1)

    candidates.sort(reverse=True)
    best_score, idx, base = candidates[0]

    # Try to join wrapped continuation lines (up to next 2 non-empty lines)
    title = base
    j = idx + 1; appended = 0
    while j < len(lines) and appended < 2:
        if _title_continuation(lines[j]):
            title = f"{title} {lines[j].rstrip(' .:-')}"
            appended += 1
            j += 1
        else:
            break

    return title, idx

# ---------------- Authors detection ---------------- #

ROLE_CLEAN = re.compile(r"\b(Member|Senior Member|Student Member|Fellow|IEEE|ACM)\b", re.I)

def _clean_author_block(s: str) -> str:
    s = re.sub(r"\S+@\S+\.\S+", " ", s)                # remove emails
    s = ROLE_CLEAN.sub(" ", s)                         # remove roles
    s = re.sub(r"[\*\d\^\[\]\(\)º°†‡§•]+", " ", s)     # remove markers
    s = re.sub(r"\bcorresponding authors?\b.*?$", " ", s, flags=re.I)
    s = re.sub(r"\b(received|accepted|published)\b.*?$", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" ,;.")
    return s

def guess_authors(text: str, title_idx: int) -> str:
    if not text:
        return "Not reported"

    lines = [l.strip() for l in text[:6000].splitlines()]

    # Window just under the title
    start = 0 if title_idx < 0 else max(0, title_idx + 1)
    end = min(len(lines), start + 12)

    block_lines = []
    for i in range(start, end):
        l = lines[i]
        if not l:
            if block_lines:
                break
            else:
                continue
        if re.search(r"\babstract\b|\bkeywords\b|\bindex terms\b", l, re.I):
            break
        # skip affiliation/venue/email lines
        if _bad_title_line(l):
            continue
        block_lines.append(l)

    if not block_lines:
        # Fallback: "by ..." line
        for i, l in enumerate(lines[:40]):
            m = re.search(r"^\s*by\s+(.+)$", l, re.I)
            if m:
                block_lines = [m.group(1)]
                break

    if not block_lines:
        return "Not reported"

    joined = _clean_author_block(" ".join(block_lines))

    # Split by common separators first to isolate names
    prelim = re.split(r"\s+(?:and|&)\s+|[,;]", joined)
    candidates = []
    for chunk in prelim:
        chunk = chunk.strip(" -·")
        # collect name-like spans
        for nm in re.findall(NAME_PATTERN, chunk):
            if nm.lower() in VENUE_AFFIL_STOP or nm.lower() in ROLE_TOKENS:
                continue
            # basic sanity (avoid words like "Report Generation")
            if nm.split()[0] in {"Report", "Computer", "Vision", "Smart", "Systems"}:
                continue
            candidates.append(nm)

    names = uniq_preserve(candidates)
    # Keep 2–4 token names only
    names = [n for n in names if 2 <= len(n.split()) <= 4]
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
    tl = text.lower()
    hits = []
    for alias, canon in DATASET_CANON.items():
        if alias in tl:
            hits.append(canon)
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

    # Look near "propose/present/introduce/…" windows first
    windows = []
    for m in re.finditer(r"(propose|present|introduce|develop|call(?:ed)?)\b.{0,240}", text, flags=re.I):
        s = max(0, m.start() - 120); e = min(len(text), m.end() + 120)
        windows.append(text[s:e])
    if not windows:
        windows = [text[:6000]]

    hits = []
    for w in windows:
        # direct whitelist
        for km in KNOWN_MODELS:
            if re.search(rf"\b{re.escape(km)}\b", w):
                hits.append(km)
        # EGGCA variants spelled without hyphen
        if re.search(r"\beggca\s*-?\s*net\b", w, re.I):
            hits.append("EGGCA-Net")
        # ALBEF/EALBEF exact
        if re.search(r"\bealbef\b", w, re.I):
            hits.append("EALBEF")
        if re.search(r"\balbef\b", w, re.I):
            hits.append("ALBEF")

    hits = [h for h in uniq_preserve(hits) if h.upper() not in MODEL_BANLIST]
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
    ap.add_argument("--out-md", default="filled_papers_llama_v8.md")
    ap.add_argument("--out-csv", default="filled_papers_llama_v8.csv")
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
