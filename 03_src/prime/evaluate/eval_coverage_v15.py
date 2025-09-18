#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_coverage_v15.py

Compare coverage (non–"Not reported") across multiple extractor outputs.

Input format:
  --inputs "Label A:/path/to/a.csv" "Label B:/path/to/b.csv" ...

Outputs:
  - Pretty printed summary to stdout
  - Optional CSV/MD reports via --out-csv / --out-md

Example:
  python eval_coverage_v15.py \
    --inputs \
      "MMR strict:filled_papers_vlm_v15_mmr_strict.csv" \
      "SIM strict:filled_papers_vlm_v15_sim_strict.csv" \
      "MMR lenient:filled_papers_vlm_v15_mmr_lenient.csv" \
      "SIM lenient:filled_papers_vlm_v15_sim_lenient.csv" \
    --out-csv coverage_summary_v15.csv \
    --out-md coverage_report_v15.md
"""

import argparse
import csv
import os
from collections import OrderedDict

NR = "Not reported"

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows

def is_reported(v: str) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return bool(s) and s != NR

def summarize(rows):
    """Return (overall_cov, per_col_cov, per_row_cov_mean) and some metadata."""
    if not rows:
        return 0.0, {}, 0.0, [], []

    cols = [c for c in rows[0].keys() if c != "File"]
    total_cells = len(rows) * len(cols)
    reported_cells = 0

    col_counts = OrderedDict((c, 0) for c in cols)
    row_fracs = []

    for r in rows:
        row_rep = 0
        for c in cols:
            if is_reported(r.get(c, "")):
                reported_cells += 1
                col_counts[c] += 1
                row_rep += 1
        row_fracs.append(row_rep / len(cols) if cols else 0.0)

    overall = reported_cells / total_cells if total_cells else 0.0
    per_col = OrderedDict((c, col_counts[c] / len(rows) if rows else 0.0) for c in cols)
    per_row_mean = sum(row_fracs) / len(row_fracs) if row_fracs else 0.0
    return overall, per_col, per_row_mean, cols, [r.get("File", "") for r in rows]

def fmt_pct(x):
    return f"{x*100:.1f}%"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help='List like "Label:/path/to/file.csv" (quote each item).')
    ap.add_argument("--out-csv", default=None, help="Write summary CSV here (optional).")
    ap.add_argument("--out-md", default=None, help="Write markdown report here (optional).")
    args = ap.parse_args()

    datasets = []
    for item in args.inputs:
        if ":" not in item:
            raise SystemExit(f'Bad --inputs item: "{item}". Use "Label:/path/to/file.csv"')
        label, path = item.split(":", 1)
        path = path.strip()
        if not os.path.isfile(path):
            raise SystemExit(f'File not found for label "{label}": {path}')
        datasets.append((label.strip(), path))

    # Load and summarize
    summaries = {}
    cols_master = None
    for label, path in datasets:
        rows = load_csv(path)
        overall, per_col, per_row_mean, cols, files = summarize(rows)
        summaries[label] = {
            "overall": overall,
            "per_col": per_col,
            "per_row_mean": per_row_mean,
            "n_rows": len(rows),
            "files": files,
        }
        if cols_master is None:
            cols_master = list(per_col.keys())

    # Print summary
    print("\n=== Coverage Summary (non–\"Not reported\") ===\n")
    for label in summaries:
        s = summaries[label]
        print(f"[{label}]  rows={s['n_rows']}")
        print(f"  Overall coverage:   {fmt_pct(s['overall'])}")
        print(f"  Per-row avg cover:  {fmt_pct(s['per_row_mean'])}")
        print("  Per-column coverage:")
        for c in cols_master:
            print(f"    - {c:18s} {fmt_pct(s['per_col'].get(c, 0.0))}")
        print("")

    # Pairwise MMR - SIM deltas (if both exist)
    def find(lbls, key):
        return next((l for l in lbls if key.lower() in l.lower()), None)

    labels = list(summaries.keys())
    pairs = [
        ("MMR strict", "SIM strict"),
        ("MMR lenient", "SIM lenient"),
        ("MMR final", "SIM final"),
    ]
    print("=== Pairwise Delta (MMR - SIM) ===\n")
    for mmr_key, sim_key in pairs:
        mmr = find(labels, mmr_key)
        sim = find(labels, sim_key)
        if not (mmr and sim):
            print(f"(skipping delta: need both '{mmr_key}' and '{sim_key}')\n")
            continue
        sm = summaries[mmr]; ss = summaries[sim]
        print(f"[{mmr_key} vs {sim_key}]")
        print(f"  Overall Δ:          {fmt_pct(sm['overall'] - ss['overall'])}")
        print(f"  Per-row avg Δ:      {fmt_pct(sm['per_row_mean'] - ss['per_row_mean'])}")
        print(f"  Per-column Δ:")
        for c in cols_master:
            dv = sm['per_col'].get(c, 0.0) - ss['per_col'].get(c, 0.0)
            sign = "+" if dv >= 0 else ""
            print(f"    - {c:18s} {sign}{fmt_pct(dv)}")
        print("")

    # Optional CSV summary
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            cols = ["Label", "Rows", "Overall", "PerRowAvg"] + cols_master
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for label, s in summaries.items():
                row = {
                    "Label": label,
                    "Rows": s["n_rows"],
                    "Overall": f"{s['overall']:.6f}",
                    "PerRowAvg": f"{s['per_row_mean']:.6f}",
                }
                for c in cols_master:
                    row[c] = f"{s['per_col'].get(c, 0.0):.6f}"
                w.writerow(row)
        print(f"✅ Wrote summary CSV: {args.out_csv}")

    # Optional Markdown report
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("# Coverage Report (non–\"Not reported\")\n\n")
            for label, s in summaries.items():
                f.write(f"## {label}\n\n")
                f.write(f"- Rows: **{s['n_rows']}**\n")
                f.write(f"- Overall coverage: **{fmt_pct(s['overall'])}**\n")
                f.write(f"- Per-row avg coverage: **{fmt_pct(s['per_row_mean'])}**\n\n")
                f.write("| Column | Coverage |\n|---|---:|\n")
                for c in cols_master:
                    f.write(f"| {c} | {fmt_pct(s['per_col'].get(c, 0.0))} |\n")
                f.write("\n")
            # Deltas
            f.write("## Pairwise Delta (MMR - SIM)\n\n")
            for mmr_key, sim_key in pairs:
                mmr = find(labels, mmr_key)
                sim = find(labels, sim_key)
                if not (mmr and sim):
                    continue
                sm = summaries[mmr]; ss = summaries[sim]
                f.write(f"### {mmr_key} vs {sim_key}\n\n")
                f.write(f"- Overall Δ: **{fmt_pct(sm['overall'] - ss['overall'])}**\n")
                f.write(f"- Per-row avg Δ: **{fmt_pct(sm['per_row_mean'] - ss['per_row_mean'])}**\n\n")
                f.write("| Column | Δ Coverage |\n|---|---:|\n")
                for c in cols_master:
                    dv = sm['per_col'].get(c, 0.0) - ss['per_col'].get(c, 0.0)
                    f.write(f"| {c} | {fmt_pct(dv)} |\n")
                f.write("\n")
        print(f"✅ Wrote markdown report: {args.out_md}")

if __name__ == "__main__":
    main()
