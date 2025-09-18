#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd

MISSING_TOKENS = {
    "", "n/a", "na", "none", "null", "unk", "unknown",
    "not available", "not applicable", "not reported", "nr", "n/r"
}

DEFAULT_FIELDS = [
    "Model", "Family", "RAG", "Fusion", "Objectives",
    "Vision Enc", "Lang Dec", "Class", "Task"
]

def is_missing(x) -> bool:
    x = (str(x) if pd.notna(x) else "").strip().lower()
    return x in MISSING_TOKENS

def norm(x) -> str:
    return (str(x) if pd.notna(x) else "").strip().lower()

def safe_name(field: str) -> str:
    return field.strip().lower().replace(" ", "_").replace("/", "_")

def evaluate_field(df_gold: pd.DataFrame, df_clean: pd.DataFrame, field: str, outdir: str):
    # Ensure columns exist
    if "File" not in df_gold.columns or "File" not in df_clean.columns:
        raise ValueError("Both CSVs must contain a 'File' column.")

    if field not in df_gold.columns:
        raise ValueError(f"Gold file is missing column: {field}")

    if field not in df_clean.columns:
        raise ValueError(f"Clean file is missing column: {field}")

    # Join on File to align rows
    df = df_gold[["File", field]].merge(
        df_clean[["File", field]],
        on="File", how="left", suffixes=("_gold", "_pred")
    )

    # Stage 1: availability (gold says present)
    df["available"] = ~df[f"{field}_gold"].apply(is_missing)

    # Model predicted "present" if clean has non-missing
    df["pred_present"] = ~df[f"{field}_pred"].apply(is_missing)

    # Stage 2: correctness conditioned on availability (exact string match after normalization)
    df["correct"] = df["available"] & (
        df[f"{field}_pred"].apply(norm) == df[f"{field}_gold"].apply(norm)
    )

    total = len(df)
    available = int(df["available"].sum())
    correct = int(df["correct"].sum())

    # Availability detection metrics (Stage 1 classification)
    tp = int((df["available"] & df["pred_present"]).sum())
    fp = int((~df["available"] & df["pred_present"]).sum())
    fn = int((df["available"] & ~df["pred_present"]).sum())

    availability_precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    availability_recall    = (tp / (tp + fn)) if (tp + fn) else 0.0
    if availability_precision + availability_recall == 0:
        availability_f1 = 0.0
    else:
        availability_f1 = 2 * availability_precision * availability_recall / (availability_precision + availability_recall)

    # Stage 2 report
    coverage_ceiling = (correct / available) if available else 0.0  # availability-adjusted coverage
    yield_end_to_end = (correct / total) if total else 0.0          # end-to-end yield

    # Diagnostics: where gold is available but prediction is missing / incorrect
    fn_avail = df[df["available"] & ~df["pred_present"]]  # missing predictions
    wrong    = df[df["available"] & ~df["correct"]]       # available but incorrect (includes missing too)
    base = ["File", f"{field}_gold", f"{field}_pred"]

    os.makedirs(outdir, exist_ok=True)
    fn_avail[base].to_csv(os.path.join(outdir, f"missing_preds_{safe_name(field)}.csv"), index=False)
    wrong[base].to_csv(os.path.join(outdir, f"mismatches_{safe_name(field)}.csv"), index=False)

    # Summary row
    return {
        "field": field,
        "total_rows": total,
        "available_gold": available,
        "pred_present": int(df["pred_present"].sum()),
        "tp_available_detect": tp,
        "fp_available_detect": fp,
        "fn_available_detect": fn,
        "availability_precision": round(availability_precision, 6),
        "availability_recall": round(availability_recall, 6),
        "availability_f1": round(availability_f1, 6),
        "correct_given_available": correct,
        "coverage_ceiling": round(coverage_ceiling, 6),
        "yield_end_to_end": round(yield_end_to_end, 6),
    }

def main():
    ap = argparse.ArgumentParser(
        description="Two-stage evaluation (availability detection + conditional accuracy) across multiple fields."
    )
    ap.add_argument("--clean", default="output/extractions_clean.csv", help="Path to cleaned predictions CSV")
    ap.add_argument("--gold",  default="gold_seed_final.csv", help="Path to gold CSV")
    ap.add_argument("--outdir", default="output/eval", help="Directory to save summary and diagnostics")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                    help=f"Comma-separated list of fields. Default: {', '.join(DEFAULT_FIELDS)}")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    clean = pd.read_csv(args.clean)
    gold  = pd.read_csv(args.gold)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    results = []
    for fld in fields:
        res = evaluate_field(gold, clean, fld, args.outdir)
        results.append(res)
        print(
            f"[{fld}] available={res['available_gold']}/{res['total_rows']}, "
            f"cov_ceiling={res['coverage_ceiling']:.3f}, yield={res['yield_end_to_end']:.3f}, "
            f"avail_P/R/F1={res['availability_precision']:.3f}/{res['availability_recall']:.3f}/{res['availability_f1']:.3f}"
        )

    summary = pd.DataFrame(results)
    summary_path = os.path.join(args.outdir, "summary_two_stage_eval.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n✓ Wrote summary: {summary_path}")
    print(f"Diagnostics per field saved as missing_preds_*.csv and mismatches_*.csv in {args.outdir}/")

if __name__ == "__main__":
    main()
