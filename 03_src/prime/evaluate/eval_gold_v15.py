#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, pandas as pd, re
from pathlib import Path

NR = "Not reported"

# columns we score (all except File)
def get_cols(df): return [c for c in df.columns if c != "File"]

# normalize cell into a canonical set of tokens (for set-equality matching)
def norm_cell(x: str) -> frozenset:
    if pd.isna(x) or not str(x).strip() or str(x).strip().lower() == NR.lower():
        return frozenset()
    s = str(x).strip().lower()
    # unify separators
    s = re.sub(r"[;|/]", ",", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return frozenset(parts)

def load_csv(p):
    df = pd.read_csv(p)
    if "File" not in df.columns:
        raise SystemExit(f"ERROR: missing 'File' column in {p}")
    return df

def main():
    ap = argparse.ArgumentParser(description="Evaluate extractor CSV vs gold CSV (exact set match, case-insensitive).")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="coverage_summary_v15.csv", help="CSV with per-field metrics")
    args = ap.parse_args()

    gold = load_csv(args.gold)
    pred = load_csv(args.pred)

    # align on File
    common = sorted(set(gold["File"]) & set(pred["File"]))
    if not common:
        raise SystemExit("ERROR: no overlapping File ids between gold and pred.")
    gold = gold.set_index("File").loc[common]
    pred = pred.set_index("File").loc[common]

    cols = get_cols(gold)
    # ensure same columns
    if set(cols) - set(get_cols(pred)):
        missing = sorted(set(cols) - set(get_cols(pred)))
        raise SystemExit(f"ERROR: pred missing columns: {missing}")

    rows = []
    micro_tp = micro_fp = micro_fn = 0

    for col in cols:
        tp = fp = fn = 0
        for fid in common:
            g = norm_cell(gold.at[fid, col])
            p = norm_cell(pred.at[fid, col])
            if len(g) == 0 and len(p) == 0:
                # neither reported -> neither counts toward precision/recall
                continue
            if len(p) == 0 and len(g) > 0:
                fn += 1
            elif len(p) > 0 and len(g) == 0:
                fp += 1
            else:
                # both non-empty: exact set match to count as TP
                if p == g:
                    tp += 1
                else:
                    # mismatch -> count as FN+FP (strict)
                    fn += 1
                    fp += 1

        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec  = tp / (tp + fn) if (tp + fn) else 1.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 1.0

        rows.append({"Field": col, "TP": tp, "FP": fp, "FN": fn,
                     "Precision": round(prec, 4),
                     "Recall": round(rec, 4),
                     "F1": round(f1, 4)})

        micro_tp += tp; micro_fp += fp; micro_fn += fn

    # micro-averaged
    m_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 1.0
    m_rec  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 1.0
    m_f1   = (2*m_prec*m_rec)/(m_prec+m_rec) if (m_prec+m_rec) else 1.0

    df_out = pd.DataFrame(rows).sort_values("Field")
    df_out.loc[len(df_out)] = ["__MICRO__", micro_tp, micro_fp, micro_fn,
                               round(m_prec, 4), round(m_rec, 4), round(m_f1, 4)]
    Path(args.out).write_text(df_out.to_csv(index=False), encoding="utf-8")

    print(f"✅ Wrote metrics -> {args.out}")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()
