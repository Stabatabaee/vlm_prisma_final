#!/usr/bin/env python3
import argparse
import pandas as pd
from collections import Counter, defaultdict

FIELDS = [
    "File","Modality","Datasets (train)","Datasets (eval)","Paired","VLM?","Model",
    "Class","Task","Vision Enc","Lang Dec","Fusion","Objectives","Family","RAG",
    "Metrics(primary)"
]

def norm(x):
    if x is None: return ""
    s = str(x).strip()
    if s.lower() in {"", "n/a", "na", "none"}: return ""
    return s

def tokenize_set(val):
    """
    For multi-value fields, split on commas/semicolons and normalize.
    """
    s = norm(val)
    if not s: return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    return sorted(set(parts))

def join_set(vals):
    return ", ".join(sorted(set([v for v in vals if v])))

def fuse_group(df_group):
    """
    Collapse multiple rows for a single File by:
      - voting (most frequent non-empty) for single-label-ish fields
      - union for list-like fields
    Also compute a simple 'strength' = frequency of the chosen value.
    """
    out = {c: "" for c in FIELDS}
    strengths = {}
    out["File"] = df_group["File"].iloc[0]

    # Heuristic: which are list-like
    list_like = {"Datasets (train)","Datasets (eval)","Metrics(primary)","Vision Enc","Lang Dec","Objectives","Family","Model"}
    # Single-label-ish
    single_like = set(FIELDS) - list_like - {"File"}

    for col in single_like:
        vals = [norm(v) for v in df_group[col].tolist() if not norm(v) in {"", "not reported", "unknown"}]
        if not vals:
            out[col] = ""
            strengths[col] = 0
        else:
            c = Counter(vals)
            best, cnt = c.most_common(1)[0]
            out[col] = best
            strengths[col] = cnt

    for col in list_like:
        bags = []
        for v in df_group[col].tolist():
            bags.extend(tokenize_set(v))
        if not bags:
            out[col] = ""
            strengths[col] = 0
        else:
            # choose union, and strength = max frequency of any element
            c = Counter(bags)
            out[col] = ", ".join(sorted(c.keys()))
            strengths[col] = max(c.values())

    # attach strength columns so v22 can use them for overwrite policy
    for col, s in strengths.items():
        out[f"{col}__strength"] = s

    return out

def main():
    ap = argparse.ArgumentParser(description="Fuse duplicate rows per file via vote/union.")
    ap.add_argument("--in-csv", required=True, help="CSV to fuse (typically v20 output).")
    ap.add_argument("--aux-csv", default=None, help="Optional auxiliary CSV (e.g., v18) for more rows.")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    df_main = pd.read_csv(args.in_csv)
    if args.aux_csv and args.aux_csv != args.in_csv:
        try:
            df_aux = pd.read_csv(args.aux_csv)
            df = pd.concat([df_main, df_aux], ignore_index=True, sort=False)
        except Exception:
            df = df_main
    else:
        df = df_main

    # ensure expected columns
    for col in FIELDS:
        if col not in df.columns:
            df[col] = ""

    # group by File
    fused_rows = []
    for file_val, grp in df.groupby("File"):
        fused_rows.append(fuse_group(grp))

    out_df = pd.DataFrame(fused_rows)
    out_df.to_csv(args.out_csv, index=False)

    # simple MD view
    with open(args.out_md, "w") as f:
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("|" + "|".join(["---"]*len(FIELDS)) + "|\n")
        for _, r in out_df.iterrows():
            f.write("| " + " | ".join([str(r.get(col,"")) for col in FIELDS]) + " |\n")

    print(f"✅ Fused -> {args.out_csv}")
    print(f"✅ Fused MD -> {args.out_md}")

if __name__ == "__main__":
    main()
