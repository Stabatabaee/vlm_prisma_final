#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ---------------------------
# Helpers
# ---------------------------

EMPTY_RE = re.compile(r"^\s*$|^(n/?a|na|none|null|unk|unknown|not\s+available|not\s+applicable|not\s+reported|nr|n/r)$", re.I)

def is_empty(v: str) -> bool:
    return v is None or EMPTY_RE.match((v or "").strip()) is not None

def norm_key(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\.(pdf|txt|md)$", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s

def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return (r.fieldnames or []), list(r)

def truthy(x: str) -> bool:
    x = (x or "").strip().lower()
    return x in {"1", "true", "yes", "y", "t"}

# Default list of fields we care about (must match mask headers)
DEFAULT_FIELDS = [
    "Datasets (train)",
    "Datasets (eval)",
    "Model",
    "Class",
    "Fusion",
    "Objectives",
    "Family",
    "RAG",
]

# Treat these literal strings as “filled” (some pipelines put these canonical tokens)
NON_EMPTY_LITERALS = {"cross-attention", "co-attention", "contrastive", "ITM", "coverage"}

def value_is_effectively_empty(field: str, val: str) -> bool:
    """
    Decide if a cell is still a miss.
    - Empty strings / Not reported variants -> empty
    - Otherwise any non-empty content (including defaults produced by the cleaner) counts as filled.
    """
    if is_empty(val):
        return True
    # Otherwise, anything with actual content is fine.
    # e.g., "contrastive, ITM" or "cross-attention" is NOT a miss.
    return False

# ---------------------------
# Core
# ---------------------------

def load_mask(mask_csv: str) -> Tuple[List[str], Dict[str, Dict[str, str]], str]:
    hdr, rows = read_csv(mask_csv)
    if not rows:
        return hdr, {}, hdr[0] if hdr else "File"

    # pick a sensible key column
    key_col = next((c for c in ["File", "file", "filename", "paper", "Paper", ""] if c in hdr), hdr[0])
    mask = {}
    for r in rows:
        k = norm_key(r.get(key_col, ""))
        if k:
            mask[k] = r
    return hdr, mask, key_col

def find_fields_to_check(mask_hdr: List[str], focus_fields: List[str] = None) -> List[str]:
    fields = focus_fields if focus_fields else DEFAULT_FIELDS
    return [f for f in fields if f in mask_hdr]

def build_pred_index(pred_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    idx = {}
    for r in pred_rows:
        k = norm_key(r.get("File", ""))
        if k:
            idx[k] = r
    return idx

def check_misses(pred_csv: str, mask_csv: str, out_csv: str, fields: List[str]) -> Dict[str, int]:
    mask_hdr, mask_map, key_col = load_mask(mask_csv)
    to_check = find_fields_to_check(mask_hdr, fields)
    pred_hdr, pred_rows = read_csv(pred_csv)
    pred_idx = build_pred_index(pred_rows)

    misses = []
    counts = Counter()

    for k, mrow in mask_map.items():
        prow = pred_idx.get(k)
        if prow is None:
            # If the prediction set lacks this paper entirely, skip it (or count as "not_in_pred" if you want)
            continue

        for field in to_check:
            avail = truthy(mrow.get(field, "0"))
            if not avail:
                continue  # mask says this field is not expected / not available

            val = prow.get(field, "")
            if value_is_effectively_empty(field, val):
                misses.append({
                    "File": prow.get("File", k),
                    "Field": field,
                    "Value": (val or "").strip(),
                })
                counts[field] += 1

    # write per-miss CSV
    if misses:
        with open(out_csv, "w", newline="", encoding="utf-8") as g:
            w = csv.DictWriter(g, fieldnames=["File", "Field", "Value"])
            w.writeheader()
            w.writerows(misses)
    else:
        # still create an empty file with header for reproducibility
        with open(out_csv, "w", newline="", encoding="utf-8") as g:
            w = csv.DictWriter(g, fieldnames=["File", "Field", "Value"])
            w.writeheader()

    return counts

def write_summary_md(summary_md: str, label: str, counts: Dict[str, int], checked_fields: List[str]):
    total = sum(counts.values())
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(f"# Miss Summary — {label}\n\n")
        if total == 0:
            f.write("No misses 🎉 (all required/available fields are filled).\n")
            return
        f.write("| Field | Misses |\n|---|---:|\n")
        for fld in checked_fields:
            if counts.get(fld, 0) > 0:
                f.write(f"| {fld} | {counts[fld]} |\n")
        f.write(f"\n**Total misses:** {total}\n")

def main():
    ap = argparse.ArgumentParser(description="Check remaining misses after cleaning, using availability mask.")
    ap.add_argument("--pred-csv", required=True, help="Cleaned prediction CSV (e.g., gold_pred_v22_strict_clean_best.csv)")
    ap.add_argument("--mask-csv", required=True, help="Availability mask CSV")
    ap.add_argument("--label", default="run", help="Short label for the summary title")
    ap.add_argument("--out-misses-csv", default="misses.csv", help="Where to write per-row misses")
    ap.add_argument("--out-summary-md", default="misses_summary.md", help="Where to write the summary markdown")
    ap.add_argument("--fields", nargs="*", default=None,
                    help="Optional subset of fields to check; default is a sensible set if omitted.")
    args = ap.parse_args()

    # Load mask to figure out which fields exist
    mask_hdr, _, _ = load_mask(args.mask_csv)
    checked_fields = find_fields_to_check(mask_hdr, args.fields)

    counts = check_misses(
        pred_csv=args.pred_csv,
        mask_csv=args.mask_csv,
        out_csv=args.out_misses_csv,
        fields=checked_fields
    )

    write_summary_md(args.out_summary_md, args.label, counts, checked_fields)

    total = sum(counts.values())
    print(f"✓ Misses CSV: {args.out_misses_csv}")
    print(f"✓ Summary MD: {args.out_summary_md}")
    if total == 0:
        print("🎉 No misses — all required/available fields are filled.")
    else:
        print("Miss counts:")
        for k, v in counts.items():
            if v:
                print(f"  {k:18s} {v}")

if __name__ == "__main__":
    main()
