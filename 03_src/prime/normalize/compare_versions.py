import argparse
import pandas as pd
from pathlib import Path

MISSING_TOKENS = {"", "not reported", "nan", "none", "null", "na", "n/a", "-"}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize to strings for uniform checks; keep original for uniqueness
    return df

def is_missing(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isna() | s.isin(MISSING_TOKENS)

def coverage_stats(df: pd.DataFrame, label: str) -> pd.Series:
    total_fields = df.shape[0] * df.shape[1]
    miss_mask = df.apply(is_missing)
    missing = int(miss_mask.sum().sum())
    filled = total_fields - missing
    coverage = 100.0 * filled / total_fields if total_fields else 0.0
    return pd.Series({
        "rows": df.shape[0],
        "cols": df.shape[1],
        "total_fields": total_fields,
        "missing_fields": missing,
        "filled_fields": filled,
        "coverage_%": round(coverage, 2),
    }, name=label)

def per_column_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    miss = df.apply(is_missing).sum()
    uniq = df.nunique(dropna=True)
    return pd.DataFrame({
        f"missing_{label}": miss,
        f"unique_{label}": uniq
    })

def compare_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str):
    # Align columns/rows by name & order to avoid false diffs
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    a = df_a[common_cols].copy()
    b = df_b[common_cols].copy()

    # Field-level improvements: A missing -> B filled
    a_miss = a.apply(is_missing)
    b_miss = b.apply(is_missing)
    improved_mask = a_miss & ~b_miss
    degraded_mask = ~a_miss & b_miss

    improved_fields = int(improved_mask.sum().sum())
    degraded_fields = int(degraded_mask.sum().sum())

    # Per-column diffs
    col_a = per_column_stats(a, label_a)
    col_b = per_column_stats(b, label_b)
    col_merged = col_a.join(col_b, how="outer")

    # Add deltas
    col_merged[f"missing_drop({label_a}->{label_b})"] = (
        col_merged[f"missing_{label_a}"] - col_merged[f"missing_{label_b}"]
    )
    col_merged[f"unique_gain({label_b}-{label_a})"] = (
        col_merged[f"unique_{label_b}"] - col_merged[f"unique_{label_a}"]
    )

    return improved_fields, degraded_fields, col_merged

def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=True)

def main():
    ap = argparse.ArgumentParser(description="Compare coverage/uniqueness between versions.")
    ap.add_argument("--base-csv", required=True, help="Path to baseline CSV (e.g., v18 stage CSV).")
    ap.add_argument("--base-label", default="v18")
    ap.add_argument("--target-csv", required=True, help="Path to target CSV (e.g., v22 final CSV).")
    ap.add_argument("--target-label", default="v22")
    ap.add_argument("--out-md", default="compare_report.md", help="Markdown report output.")
    args = ap.parse_args()

    base = load_csv(args.base_csv)
    targ = load_csv(args.target_csv)

    # Overall stats
    summ = pd.concat([
        coverage_stats(base, args.base_label),
        coverage_stats(targ, args.target_label)
    ], axis=1)

    # Pairwise improvements
    improved_fields, degraded_fields, col_cmp = compare_pair(base, targ, args.base_label, args.target_label)

    print("=== Overall Coverage ===")
    print(summ.to_string())
    print("\n=== Field-level Changes ===")
    print(f"Improved fields ({args.base_label} missing -> {args.target_label} filled): {improved_fields}")
    print(f"Degraded fields ({args.base_label} filled -> {args.target_label} missing): {degraded_fields}")

    print("\n=== Per-column Comparison ===")
    # sort by biggest missing drop then by unique gain
    sort_cols = [f"missing_drop({args.base_label}->{args.target_label})", f"unique_gain({args.target_label}-{args.base_label})"]
    col_cmp_sorted = col_cmp.sort_values(by=sort_cols, ascending=[False, False])
    print(col_cmp_sorted.to_string())

    # Write Markdown report
    md_lines = []
    md_lines.append("# Version Comparison Report")
    md_lines.append("")
    md_lines.append("## Overall Coverage")
    md_lines.append(md_table(summ))
    md_lines.append("")
    md_lines.append("## Field-level Changes")
    md_lines.append(f"- Improved fields ({args.base_label} missing → {args.target_label} filled): **{improved_fields}**")
    md_lines.append(f"- Degraded fields ({args.base_label} filled → {args.target_label} missing): **{degraded_fields}**")
    md_lines.append("")
    md_lines.append("## Per-column Comparison")
    md_lines.append(md_table(col_cmp_sorted))
    md = "\n".join(md_lines)

    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"\n✅ Wrote Markdown report -> {args.out_md}")

if __name__ == "__main__":
    main()
