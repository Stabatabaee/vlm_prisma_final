#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
results_summary_availability.py
Compare two extraction runs (e.g., v18 vs v22) with availability-adjusted metrics.

Inputs:
  --base-csv <csv>        (e.g., v18_full.csv)
  --base-label <name>     (e.g., v18)
  --target-csv <csv>      (e.g., gold_pred_v22_strict_norm.csv)
  --target-label <name>   (e.g., v22_strict)
  --availability-csv <csv> (mask with 0/1 per field indicating presence in each paper)
  --out-md <file>         Markdown summary
  --out-csv <file>        CSV summary

Notes:
- The first column is assumed to be the file key (often "File"). We auto-detect its name.
- Availability CSV may contain rows with ".txt" suffix; we normalize keys by stripping ".txt".
- Missing values = "", "not reported", "n/a", "na" (case-insensitive) are considered unfilled.
- Wilson CIs are computed with safe guards; when denominator is 0, we print "NA".
- McNemar exact 2-sided test on available subset; when no discordant pairs (b+c==0), we print "NA".
"""

import argparse
import csv
import math
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Set

MISSING_TOKENS = {"", "not reported", "n/a", "na", "none", "null", "-"}

def norm_key(s: str) -> str:
    """Normalize file keys (strip trailing .txt if present)."""
    s = s.strip()
    if s.endswith(".txt"):
        return s[:-4]
    return s

def is_filled(val: str) -> bool:
    if val is None:
        return False
    v = str(val).strip()
    if v.lower() in MISSING_TOKENS:
        return False
    return len(v) > 0

def load_csv_as_dict(path: str) -> Tuple[Dict[str, Dict[str, str]], List[str], str]:
    """
    Load CSV. Return:
      rows_by_key: dict[file_key] -> {col: value}
      columns: list of column names (preserves order)
      key_col: name of the first column used as key
    """
    rows_by_key: Dict[str, Dict[str, str]] = {}
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not cols:
            raise ValueError(f"No columns detected in {path}")
        key_col = cols[0]
        for row in reader:
            key = norm_key(row[key_col])
            rows_by_key[key] = row
    return rows_by_key, cols, cols[0]

def load_availability_mask(path: str) -> Tuple[Dict[str, Dict[str, int]], List[str], str]:
    """
    Load availability CSV (0/1 per field).
    Returns rows_by_key with int values (0/1), list of columns, and key_col.
    Automatically converts values to 0/1 (non-numeric -> 0).
    """
    rows_by_key: Dict[str, Dict[str, int]] = {}
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not cols:
            raise ValueError(f"No columns detected in {path}")
        key_col = cols[0]
        for row in reader:
            key = norm_key(row[key_col])
            conv: Dict[str, int] = {}
            for c in cols[1:]:
                v = row.get(c, "")
                try:
                    conv[c] = int(float(v))
                except Exception:
                    # treat anything non-numeric as 0
                    conv[c] = 0
            rows_by_key[key] = conv
    return rows_by_key, cols, cols[0]

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Safe Wilson score interval for a binomial proportion.
    Returns (lo, hi) in [0,1]. If n<=0, returns (0,0).
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2) / n
    term = (p * (1 - p) / n) + ((z**2) / (4 * n * n))
    if term < 0:  # numerical guard
        term = 0.0
    half = (z * math.sqrt(term)) / denom
    center = (p + (z**2) / (2 * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

def mcnemar_exact_two_sided(b: int, c: int) -> float:
    """
    Exact two-sided McNemar p-value using binomial distribution on discordant pairs.
    n = b + c; X ~ Binom(n, 0.5); two-sided p = 2 * min(P[X<=min(b,c)], P[X>=max(b,c)]).
    If n==0, return NaN-like indicator (we'll print 'NA').
    """
    n = b + c
    if n == 0:
        return float("nan")
    x = min(b, c)
    # Compute cumulative probability P[X <= x] under p=0.5
    # With n <= ~100, direct sum is fine.
    def comb(n_, k_):
        return math.comb(n_, k_)
    p_le_x = sum(comb(n, k) for k in range(0, x + 1)) * (0.5 ** n)
    p_ge_x = sum(comb(n, k) for k in range(max(b, c), n + 1)) * (0.5 ** n)
    p_two = 2.0 * min(p_le_x, p_ge_x)
    return min(1.0, p_two)

def fmt_prop(k: int, n: int) -> Tuple[str, float]:
    """Return pretty proportion string with Wilson CI and the raw proportion as float (or 'NA')."""
    if n <= 0:
        return "NA", float("nan")
    p = k / n
    lo, hi = wilson_ci(k, n)
    return f"{p:.3f} [{lo:.2f},{hi:.2f}]", p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--base-label", required=True)
    ap.add_argument("--target-csv", required=True)
    ap.add_argument("--target-label", required=True)
    ap.add_argument("--availability-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    base_rows, base_cols, base_key = load_csv_as_dict(args.base_csv)
    tgt_rows,  tgt_cols,  tgt_key  = load_csv_as_dict(args.target_csv)
    avail_rows, avail_cols, avail_key = load_availability_mask(args.availability_csv)

    # Identify shared files
    files_intersection: List[str] = sorted(set(base_rows.keys()) & set(tgt_rows.keys()) & set(avail_rows.keys()))
    N_global = len(files_intersection)
    if N_global == 0:
        raise SystemExit("No overlapping files between base, target, and availability mask.")

    # Identify fields to evaluate = intersection of columns present in all three inputs
    # (skip the first key column from each)
    base_fields = set(base_cols[1:])
    tgt_fields  = set(tgt_cols[1:])
    avail_fields = set(avail_cols[1:])
    fields = [c for c in base_cols[1:] if c in tgt_fields and c in avail_fields]

    # Prepare output rows
    out_rows = []
    md_lines: List[str] = []
    md_lines.append(f"# Results: {args.base_label} vs {args.target_label}\n")
    md_lines.append(f"**Paired files (N)**: {N_global}\n")
    md_lines.append("| Field | N | Avail N | "
                    f"{args.base_label} ApparentCov | {args.target_label} ApparentCov | Δ Apparent | "
                    f"{args.base_label} AdjCov | {args.target_label} AdjCov | Δ AdjCov | "
                    f"MissRate({args.target_label}) | McNemar p (available) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for field in fields:
        # Apparent coverage counts (denominator = N_global)
        base_filled_all = sum(1 for fn in files_intersection if is_filled(base_rows[fn].get(field, "")))
        tgt_filled_all  = sum(1 for fn in files_intersection if is_filled(tgt_rows[fn].get(field, "")))

        base_app_str, base_app = fmt_prop(base_filled_all, N_global)
        tgt_app_str, tgt_app   = fmt_prop(tgt_filled_all,  N_global)
        delta_app = float("nan") if any(math.isnan(x) for x in [base_app, tgt_app]) else (tgt_app - base_app)

        # Availability subset
        available_files = [fn for fn in files_intersection if int(avail_rows[fn].get(field, 0)) == 1]
        Avail_N = len(available_files)

        # Adjusted coverage: among available only
        if Avail_N > 0:
            base_filled_avail = sum(1 for fn in available_files if is_filled(base_rows[fn].get(field, "")))
            tgt_filled_avail  = sum(1 for fn in available_files if is_filled(tgt_rows[fn].get(field, "")))
            base_adj_str, base_adj = fmt_prop(base_filled_avail, Avail_N)
            tgt_adj_str,  tgt_adj  = fmt_prop(tgt_filled_avail,  Avail_N)
            delta_adj = tgt_adj - base_adj
            miss_rate = 1.0 - tgt_adj  # among available
            # McNemar on available subset: discordant pairs
            b = sum(1 for fn in available_files if is_filled(base_rows[fn].get(field, "")) and not is_filled(tgt_rows[fn].get(field, "")))
            c = sum(1 for fn in available_files if not is_filled(base_rows[fn].get(field, "")) and is_filled(tgt_rows[fn].get(field, "")))
            p_mcnemar = mcnemar_exact_two_sided(b, c)
            p_str = "NA" if math.isnan(p_mcnemar) else f"{p_mcnemar:.4f}"
        else:
            base_adj_str, tgt_adj_str = "NA", "NA"
            delta_adj = float("nan")
            miss_rate = float("nan")
            p_str = "NA"

        # Markdown row
        md_lines.append(
            f"| {field} | {N_global} | {Avail_N} | "
            f"{base_app_str} | {tgt_app_str} | "
            f"{'NA' if math.isnan(delta_app) else f'{delta_app:+.3f}'} | "
            f"{base_adj_str} | {tgt_adj_str} | "
            f"{'NA' if math.isnan(delta_adj) else f'{delta_adj:+.3f}'} | "
            f"{'NA' if math.isnan(miss_rate) else f'{miss_rate:.3f}'} | {p_str} |"
        )

        # CSV row (raw numbers as well)
        out_rows.append(OrderedDict([
            ("Field", field),
            ("N", N_global),
            ("Avail_N", Avail_N),
            (f"{args.base_label}_Apparent_k", base_filled_all),
            (f"{args.base_label}_Apparent_n", N_global),
            (f"{args.target_label}_Apparent_k", tgt_filled_all),
            (f"{args.target_label}_Apparent_n", N_global),
            ("Delta_Apparent", delta_app if not math.isnan(delta_app) else ""),
            (f"{args.base_label}_Adjusted_k", base_filled_avail if Avail_N > 0 else ""),
            (f"{args.base_label}_Adjusted_n", Avail_N if Avail_N > 0 else ""),
            (f"{args.target_label}_Adjusted_k", tgt_filled_avail if Avail_N > 0 else ""),
            (f"{args.target_label}_Adjusted_n", Avail_N if Avail_N > 0 else ""),
            ("Delta_Adjusted", delta_adj if Avail_N > 0 else ""),
            (f"MissRate_{args.target_label}", miss_rate if Avail_N > 0 else ""),
            ("McNemar_p", p_str),
        ]))

    # Write Markdown
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"✓ Wrote Markdown -> {args.out_md}")

    # Write CSV
    csv_cols = list(out_rows[0].keys()) if out_rows else []
    with open(args.out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    print(f"✓ Wrote CSV      -> {args.out_csv}")

if __name__ == "__main__":
    main()
