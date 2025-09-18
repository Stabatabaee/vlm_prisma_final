#!/usr/bin/env python3
# availability_mask.py
# Build an availability matrix A[file, field] in {0,1} by combining:
#   (A) evidence-first: any non-null value in v22_strict.csv => available
#   (B) regex cues in snippet texts => available
#
# Usage:
#   python availability_mask.py \
#     --snippets-dir gold_snippets \
#     --v22-csv gold_pred_v22_strict.csv \
#     --out-csv availability_mask.csv

import argparse, os, re, glob
import pandas as pd

FIELDS_DEFAULT = [
    "Modality", "Datasets (train)", "Datasets (eval)", "Paired", "VLM?",
    "Model", "Class", "Task", "Vision Enc", "Lang Dec", "Fusion",
    "Objectives", "Family", "RAG", "Metrics(primary)"
]

# Null-ish tokens (lowercased/stripped) that we treat as not-filled
NULL_TOKENS = {"", "not reported", "n/a", "na", "none", "null", "not applicable", "—", "-"}

# Section-aware-ish regex cues (lowercased); expand as needed
CUES = {
    "Datasets (train)": r"(mimic[- ]?cxr|chexpert|open[- ]?i|padchest|nih chestx[- ]?ray(?:14)?|cxr[- ]?repair)",
    "Datasets (eval)" : r"(mimic[- ]?cxr|chexpert|open[- ]?i|padchest|nih chestx[- ]?ray(?:14)?|cxr[- ]?repair)",
    "Vision Enc"      : r"\b(vit|swin|resnet|clip|cnn)\b",
    "Lang Dec"        : r"\b(bert|gpt[- ]?2|llama|t5|vicuna|transformer)\b",
    "Fusion"          : r"(cross[- ]?attention|co[- ]?attention|early fusion|late fusion|gated|conditional|conditioned)",
    "Paired"          : r"\bpaired\b|\bpaired image[- ]?text\b|\baligned report\b|\bradiology report pairs?\b",
    "RAG"             : r"(retrieval[- ]?augmented|rag\b|retrieve|bm25|faiss|chroma)",
    "Family"          : r"(albef|blip[- ]?2|flamingo|llava|minigpt[- ]?4|otter|xraygpt|trmrg|cxr[- ]?irgen)",
    "Metrics(primary)": r"(bleu|rouge|cid(er)?|meteor|f1\b|accuracy|radgraph|chexbert|bertScore|spice|radcliq|gleu)",
    "Modality"        : r"(x[- ]?ray|cxr|rad(iology)?|imaging|ct|mri|ultrasound|fundus|patholog(y|ical))",
    "Model"           : r"(albef|blip[- ]?2|flamingo|llava|minigpt[- ]?4|chexnet|chexpert|trmrg|cxr[- ]?irgen|gpt[- ]?4v?)",
    "Class"           : r"\b(classification|caption|retrieval|segmentation|localization|vqa|report[- ]?generation)\b",
    "Task"            : r"\b(vqa|report[- ]?generation|retrieval|caption|classification|segmentation|localization)\b",
    "Objectives"      : r"(itm|itc|contrastive|cross[- ]?entropy|rl|reinforcement|coverage)",
    "VLM?"            : r"\b(vlm|vision[- ]?language model|multimodal (model|transformer))\b",
}

def is_filled(x: str) -> bool:
    if x is None:
        return False
    x = str(x).strip().lower()
    return x not in NULL_TOKENS

def load_snippets(snippets_dir: str):
    texts = {}
    for p in glob.glob(os.path.join(snippets_dir, "*.txt")):
        fid = os.path.basename(p)[:-4]  # strip .txt
        try:
            with open(p, "r", errors="ignore") as f:
                texts[fid] = f.read().lower()
        except Exception:
            texts[fid] = ""
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True)
    ap.add_argument("--v22-csv", required=False, help="gold_pred_v22_strict.csv (for evidence-first OR signal)")
    ap.add_argument("--fields", nargs="*", default=FIELDS_DEFAULT)
    ap.add_argument("--out-csv", default="availability_mask.csv")
    args = ap.parse_args()

    # (B) Regex miner over snippets
    texts = load_snippets(args.snippets_dir)
    files = sorted(texts.keys())

    # Prepare frame with zeros
    A = pd.DataFrame(0, index=files, columns=args.fields, dtype=int)

    # Compile regex once
    comp = {k: re.compile(v, flags=re.IGNORECASE) for k, v in CUES.items()}

    for fid, txt in texts.items():
        for field in args.fields:
            pat = comp.get(field, None)
            if pat is None:
                continue
            if pat.search(txt) is not None:
                A.at[fid, field] = 1

    # (A) Evidence-first: if v22-strict has a filled value for a field, mark available=1
    if args.v22_csv and os.path.exists(args.v22_csv):
        v22 = pd.read_csv(args.v22_csv)
        if "File" not in v22.columns:
            raise SystemExit("ERROR: 'File' column not found in {}".format(args.v22_csv))
        v22 = v22.set_index("File")
        # Align to our fields only
        for field in args.fields:
            if field in v22.columns:
                mask = v22[field].apply(is_filled).astype(int)
                # union with existing availability
                A[field] = A[field].reindex(A.index).fillna(0).astype(int)
                # Align index union (in case some Files exist only in CSV)
                for fid in mask.index:
                    if fid not in A.index:
                        # add missing row (e.g., missing snippet) with zeros first
                        A.loc[fid, :] = 0
                A.loc[mask.index, field] = (A.loc[mask.index, field].astype(int) | mask.astype(int)).astype(int)

    A.sort_index(inplace=True)
    A.to_csv(args.out_csv)
    print(f"✓ Wrote availability mask -> {args.out_csv}")
    print(f"Rows (files): {A.shape[0]}  |  Fields: {A.shape[1]}")

if __name__ == "__main__":
    main()
