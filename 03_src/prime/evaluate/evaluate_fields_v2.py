#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import pandas as pd

MISSING = {"", "n/a", "na", "none", "null", "unk", "unknown",
           "not available", "not applicable", "not reported", "nr", "n/r"}

TASK_TERMS = [
    r"report generation", r"radiology report generation",
    r"image[-\s]?text matching", r"image[-\s]?report", r"image[-\s]?to[-\s]?sequence",
    r"classification", r"segmentation", r"retrieval", r"caption", r"captioning",
    r"vqa", r"question[-\s]?answer(ing)?", r"detection"
]

def is_missing(x):
    x = (str(x) if pd.notna(x) else "").strip().lower()
    return x in MISSING

def tokenize_cell(x: str):
    if pd.isna(x):
        return []
    s = str(x)
    s = re.sub(r"[\uFF0C；;|/]", ",", s)
    s = re.sub(r"\s*\+\s*", ",", s)
    s = re.sub(r"\s*&\s*", ",", s)
    s = re.sub(r"\s+and\s+", ",", s, flags=re.I)
    toks = [t.strip().lower() for t in s.split(",") if t.strip()]
    return sorted(set(toks))

def equal_multi(a, b):
    # order-insensitive, case-insensitive exact set match
    return set(tokenize_cell(a)) == set(tokenize_cell(b))

def looks_like_task(token: str) -> bool:
    t = token.strip().lower()
    for pat in TASK_TERMS:
        if re.search(pat, t, flags=re.I):
            return True
    return False

def fix_gold_class(gold: pd.DataFrame) -> pd.DataFrame:
    gold = gold.copy()
    if "Class" not in gold.columns or "Task" not in gold.columns:
        return gold
    new_tasks = []
    new_class = []
    for _, row in gold.iterrows():
        cls = str(row.get("Class", "") or "")
        tsk = str(row.get("Task", "") or "")
        cls_tokens = tokenize_cell(cls)
        tsk_tokens = tokenize_cell(tsk)
        moved = [tok for tok in cls_tokens if looks_like_task(tok)]
        keep  = [tok for tok in cls_tokens if tok not in moved]
        merged_tasks = sorted(set(tsk_tokens + moved))
        new_tasks.append(", ".join(merged_tasks))
        new_class.append(", ".join(sorted(set(keep))))
    gold["Task"] = new_tasks
    gold["Class"] = new_class
    return gold

def eval_field(clean_path, gold_path, outdir, field, fix_gold=False):
    clean = pd.read_csv(clean_path)
    gold  = pd.read_csv(gold_path)

    # optional gold Class fix
    if fix_gold:
        gold = fix_gold_class(gold)

    # inner merge on File + field
    col_pred = field
    if field not in clean.columns:
        raise ValueError(f"Field '{field}' not found in CLEAN.")
    if field not in gold.columns:
        raise ValueError(f"Field '{field}' not found in GOLD.")

    df = gold.merge(clean[["File", col_pred]], on="File", how="left", suffixes=("_gold", "_pred"))

    # Stage 1: availability detection (does gold say it exists?)
    df["available"] = ~df[f"{field}_gold"].apply(is_missing)

    # Stage 1 metrics (treat "pred_present" = prediction is non-missing)
    df["pred_present"] = ~df[f"{field}_pred"].apply(is_missing)
    tp_avail = int(((df["available"] == True) & (df["pred_present"] == True)).sum())
    fp_avail = int(((df["available"] == False) & (df["pred_present"] == True)).sum())
    fn_avail = int(((df["available"] == True) & (df["pred_present"] == False)).sum())
    prec_avail = tp_avail / (tp_avail + fp_avail) if (tp_avail + fp_avail) else 0.0
    rec_avail  = tp_avail / (tp_avail + fn_avail) if (tp_avail + fn_avail) else 0.0
    f1_avail   = 2*prec_avail*rec_avail/(prec_avail+rec_avail) if (prec_avail+rec_avail) else 0.0

    # Stage 2: correctness conditioned on availability (order-insensitive)
    df["correct"] = df["available"] & df.apply(
        lambda r: equal_multi(r[f"{field}_gold"], r[f"{field}_pred"]), axis=1
    )

    available = int(df["available"].sum())
    correct   = int(df["correct"].sum())
    total     = len(df)

    coverage_ceiling = correct / available if available else 0.0
    yield_end_to_end = correct / total if total else 0.0

    # write diagnostics
    os.makedirs(outdir, exist_ok=True)
    mismatches = df[(df["available"]) & (~df["correct"]) & (df["pred_present"])][["File", f"{field}_gold", f"{field}_pred"]]
    missingpred = df[(df["available"]) & (~df["pred_present"])][["File", f"{field}_gold", f"{field}_pred"]]
    mismatches.to_csv(os.path.join(outdir, f"mismatches_{field.replace(' ','_').lower()}.csv"), index=False)
    missingpred.to_csv(os.path.join(outdir, f"missing_preds_{field.replace(' ','_').lower()}.csv"), index=False)

    return {
        "field": field,
        "total_rows": total,
        "available_gold": available,
        "pred_present": int(df["pred_present"].sum()),
        "tp_available_detect": tp_avail,
        "fp_available_detect": fp_avail,
        "fn_available_detect": fn_avail,
        "availability_precision": prec_avail,
        "availability_recall": rec_avail,
        "availability_f1": f1_avail,
        "correct_given_available": correct,
        "coverage_ceiling": coverage_ceiling,
        "yield_end_to_end": yield_end_to_end,
    }

def main():
    ap = argparse.ArgumentParser(description="Two-stage evaluation (availability + conditional correctness) for multiple fields.")
    ap.add_argument("--clean", required=True, help="Path to cleaned predictions CSV (e.g., output/extractions_clean.csv)")
    ap.add_argument("--gold",  required=True, help="Path to gold CSV (e.g., gold_seed_final.csv)")
    ap.add_argument("--outdir", default="output/eval", help="Directory for diagnostics/summary")
    ap.add_argument("--fields", default="Model,Family,RAG,Fusion,Objectives,Vision Enc,Lang Dec,Task",
                    help="Comma-separated list of fields to evaluate. Use quotes if fields contain spaces.")
    ap.add_argument("--exclude-class", action="store_true", help="Exclude 'Class' field from evaluation.")
    ap.add_argument("--fix-gold-class", action="store_true",
                    help="Move task-like tokens out of Class_gold into Task_gold before evaluation.")
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if args.exclude_class and "Class" in fields:
        fields = [f for f in fields if f != "Class"]

    os.makedirs(args.outdir, exist_ok=True)
    summary = []
    for f in fields:
        res = eval_field(args.clean, args.gold, args.outdir, f, fix_gold=args.fix_gold_class)
        summary.append(res)
        print(f"[{f}] available={res['available_gold']}/{res['total_rows']}, "
              f"cov_ceiling={res['coverage_ceiling']:.3f}, yield={res['yield_end_to_end']:.3f}, "
              f"avail_P/R/F1={res['availability_precision']:.3f}/{res['availability_recall']:.3f}/{res['availability_f1']:.3f}")

    pd.DataFrame(summary).to_csv(os.path.join(args.outdir, "summary_two_stage_eval.csv"), index=False)
    print(f"\n✓ Wrote summary: {os.path.join(args.outdir, 'summary_two_stage_eval.csv')}")
    print("Diagnostics per field saved as missing_preds_*.csv and mismatches_*.csv in", args.outdir)

if __name__ == "__main__":
    main()
