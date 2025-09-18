#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd

# Force non-interactive backend (safe on headless servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS_ORDER = [
    "Model", "Family", "RAG", "Fusion", "Objectives",
    "Vision Enc", "Lang Dec", "Task",
]

def read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: field, coverage_ceiling  (from your evaluate_fields_v2.py)
    need = {"field", "coverage_ceiling"}
    if not need.issubset(set(df.columns)):
        raise ValueError(
            f"{path} must have at least columns: {need}. Found: {list(df.columns)}"
        )
    # keep only the fields we care about, preserve a consistent order
    df = df[df["field"].isin(FIELDS_ORDER)].copy()
    df["field"] = pd.Categorical(df["field"], categories=FIELDS_ORDER, ordered=True)
    df.sort_values("field", inplace=True)
    return df

def main():
    ap = argparse.ArgumentParser(
        description="Grouped bar chart of availability-adjusted coverage across versions."
    )
    ap.add_argument(
        "--csv", nargs="+", required=True,
        help="One or more summary CSV paths (e.g., v18, v21, v22)."
    )
    ap.add_argument(
        "--labels", nargs="+", required=False,
        help="Labels for each CSV (same length as --csv). Defaults to basenames."
    )
    ap.add_argument(
        "--out", default="fig3_grouped_bar.png",
        help="Output figure path (png/svg/pdf supported)."
    )
    ap.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for raster outputs (png)."
    )
    args = ap.parse_args()

    csvs = args.csv
    labels = args.labels if args.labels else [os.path.splitext(os.path.basename(x))[0] for x in csvs]
    if len(labels) != len(csvs):
        raise SystemExit("❌ --labels must have the same length as --csv")

    # Load and align all frames on 'field'
    frames = [read_summary(p)[["field", "coverage_ceiling"]].rename(columns={"coverage_ceiling": lab})
              for p, lab in zip(csvs, labels)]

    # Outer-join on 'field' to handle missing fields gracefully
    from functools import reduce
    M = reduce(lambda left, right: pd.merge(left, right, on="field", how="outer"), frames)
    # Ensure field order
    M["field"] = pd.Categorical(M["field"], categories=FIELDS_ORDER, ordered=True)
    M.sort_values("field", inplace=True)
    M.set_index("field", inplace=True)

    # Plot
    fig_w = 10
    fig_h = 4.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n_groups = M.shape[0]
    n_series = len(labels)
    x = range(n_groups)
    total_width = 0.8
    bar_width = total_width / max(n_series, 1)
    offsets = [(-total_width/2 + (i + 0.5)*bar_width) for i in range(n_series)]

    for i, lab in enumerate(labels):
        y = M[lab].values
        # Handle potential NaNs
        y = [v if pd.notna(v) else 0.0 for v in y]
        ax.bar(
            [xi + offsets[i] for xi in x], y,
            width=bar_width, label=lab
        )
        # Label bars with percentages
        for xi, yi in zip([xi + offsets[i] for xi in x], y):
            ax.text(xi, yi + 0.02, f"{yi*100:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(list(M.index))
    ax.set_ylabel("Availability-adjusted coverage")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(title="Version", ncol=min(n_series, 3), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.15))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Save in requested format
    ext = os.path.splitext(args.out)[1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    elif ext in [".svg", ".pdf"]:
        fig.savefig(args.out, bbox_inches="tight")
    else:
        # default to png if extension unrecognized
        fig.savefig(args.out if args.out.endswith(".png") else args.out + ".png",
                    dpi=args.dpi, bbox_inches="tight")

    print(f"✓ Saved {args.out}")

if __name__ == "__main__":
    main()
