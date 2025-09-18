#!/usr/bin/env python3
import argparse, csv, math, sys, re
from collections import defaultdict, Counter

def parse_args():
    p = argparse.ArgumentParser(description="Availability-adjusted comparison: base vs target, with availability mask and applicability policy.")
    p.add_argument("--base-csv", required=True, help="CSV for baseline (e.g., v18_full.csv)")
    p.add_argument("--base-label", required=True, help="Short label for baseline (e.g., v18)")
    p.add_argument("--target-csv", required=True, help="CSV for target (e.g., v22_strict_norm.csv)")
    p.add_argument("--target-label", required=True, help="Short label for target (e.g., v22_strict)")
    p.add_argument("--availability-csv", required=True, help="Availability mask CSV (rows=files, cols=fields; 0/1 flags)")
    p.add_argument("--out-md", required=True, help="Output markdown file")
    p.add_argument("--out-csv", required=True, help="Output CSV file")
    p.add_argument("--key-col", default=None, help="Name of key column (default: first col in each CSV)")
    p.add_argument("--assume-applicable", choices=["MASK","ALL","LIST"], default="MASK",
                   help="How to decide applicability: MASK=use availability CSV; ALL=assume every field applies to every paper; LIST=apply to only fields listed in --fields-apply")
    p.add_argument("--fields-apply", default="", help="Comma-separated field names used when --assume-applicable LIST. Others treated as not applicable.")
    return p.parse_args()

def normalize_key(s: str) -> str:
    s = s.strip()
    # strip common suffixes
    s = re.sub(r"\.(txt|pdf|md)$", "", s, flags=re.IGNORECASE)
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def load_csv_as_rows(path, key_col_name=None):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{path}: empty CSV")

    header = rows[0]
    if key_col_name is None:
        key_idx = 0
    else:
        if key_col_name not in header:
            raise RuntimeError(f"{path}: key column '{key_col_name}' not found. Available: {header}")
        key_idx = header.index(key_col_name)

    data = []
    for r in rows[1:]:
        if not r: 
            continue
        # pad short rows
        if len(r) < len(header):
            r = r + [""]*(len(header)-len(r))
        rec = {header[i]: r[i] for i in range(len(header))}
        rec["_key_raw"] = r[key_idx]
        rec["_key"] = normalize_key(r[key_idx])
        data.append(rec)
    return header, data

def extract_fields(header):
    # All fields except key column
    fields = [c for c in header[1:]]
    return fields

def is_filled(value: str) -> bool:
    v = (value or "").strip().lower()
    if v in ("", "not reported", "n/a", "na", "none", "null", "-", "—", "–"):
        return False
    return True

def wilson_ci(k, n, z=1.96):
    # Safe Wilson CI for proportions; handle n=0.
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    term = (p*(1-p)/n) + (z**2)/(4*n*n)
    if term < 0:   # numerical safety
        term = 0.0
    half = (z * math.sqrt(term)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

def fmt_prop(p):
    return f"{p:.3f}"

def fmt_ci(lo, hi):
    return f"[{lo:.2f},{hi:.2f}]"

def main():
    args = parse_args()

    # Load base, target, availability
    h_base, base_rows = load_csv_as_rows(args.base_csv, key_col_name=args.key_col)
    h_tgt,  tgt_rows  = load_csv_as_rows(args.target_csv, key_col_name=args.key_col)
    h_av,   av_rows   = load_csv_as_rows(args.availability_csv, key_col_name=args.key_col)

    # Derive fields
    base_fields = extract_fields(h_base)
    tgt_fields  = extract_fields(h_tgt)
    av_fields   = extract_fields(h_av)

    # Ensure field sets line up (we only compare common fields)
    common_fields = [f for f in base_fields if f in tgt_fields and f in av_fields]
    if not common_fields:
        raise RuntimeError("No common fields across CSVs. Check headers.")

    # Index by normalized key
    def index_by_key(rows):
        m = {}
        for r in rows:
            k = r["_key"]
            if k in m:
                # keep first; warn but proceed
                pass
            m[k] = r
        return m

    base_map = index_by_key(base_rows)
    tgt_map  = index_by_key(tgt_rows)
    av_map   = index_by_key(av_rows)

    keys_base = set(base_map.keys())
    keys_tgt  = set(tgt_map.keys())
    keys_av   = set(av_map.keys())

    inter = keys_base & keys_tgt & keys_av
    if not inter:
        # print a helpful diff
        only_b = sorted(list(keys_base - keys_tgt))
        only_t = sorted(list(keys_tgt - keys_base))
        only_a = sorted(list(keys_av - (keys_base & keys_tgt)))
        msg = ["ERROR: First column (key) does not match across CSVs."]
        if only_b:
            msg.append(f"- In base only (first 10): {only_b[:10]}")
        if only_t:
            msg.append(f"- In target only (first 10): {only_t[:10]}")
        if only_a:
            msg.append(f"- In availability only (first 10): {only_a[:10]}")
        print("\n".join(msg))
        sys.exit(1)

    dropped = len(keys_base | keys_tgt | keys_av) - len(inter)
    if dropped > 0:
        print(f"⚠️  Aligning on intersection of keys. Dropped {dropped} rows not present in all three CSVs.")
    keys = sorted(list(inter))
    N = len(keys)

    # Applicability policy
    apply_mode = args.assume_applicable
    list_fields = [s.strip() for s in args.fields-apply.split(",") if s.strip()] if apply_mode=="LIST" else []

    # Prepare tallies
    out_rows = []
    # For McNemar: per-field discordant counts among applicable set
    def mcnemar_p(b01, b10):
        n = b01 + b10
        if n == 0:
            return "NA"
        # Exact binomial (two-sided) with p=0.5
        # For small n, compute 2*sum_{k<=min(b01,b10)} C(n,k)/2^n truncated at 1.0
        from math import comb
        k = min(b01, b10)
        tail = sum(comb(n, i) for i in range(0, k+1)) / (2**n)
        pval = 2 * tail
        if pval > 1.0:
            pval = 1.0
        return f"{pval:.4f}"

    for field in common_fields:
        # Applicability mask Af(i)
        A = []
        if apply_mode == "ALL":
            A = [1]*N
        elif apply_mode == "LIST":
            A = [1 if field in list_fields else 0 for _ in range(N)]
        else:  # MASK (default)
            # Availability from mask CSV; non-numeric treated as 0
            for k in keys:
                cell = av_map[k].get(field, "").strip()
                v = 0
                if cell != "":
                    try:
                        v = 1 if float(cell) > 0 else 0
                    except:
                        v = 0
                A.append(v)

        availN = sum(A)

        # Count baseline and target filled (apparent, on all N)
        b_filled_all = 0
        t_filled_all = 0

        # Count baseline and target filled on available set (adjusted)
        b_filled_av = 0
        t_filled_av = 0

        # For MissRate on target among available, we need to classify Not-reported vs Missing.
        # Here we consider: among available, "not filled" (empty/NR) → a miss (Ø or NR). We can't
        # fully distinguish Ø vs NR without provenance; so we report MissRate as 1 - AdjCov_target.
        # (If you want to split Ø vs NR, we can add a second CSV with gold availability/reportedness.)
        b_app_vec = []
        t_app_vec = []
        b_adj_vec = []
        t_adj_vec = []

        # McNemar discordant counts among available set
        b01 = 0  # base=0, tgt=1
        b10 = 0  # base=1, tgt=0

        for i, k in enumerate(keys):
            bval = base_map[k].get(field, "")
            tval = tgt_map[k].get(field, "")
            bfilled = is_filled(bval)
            tfilled = is_filled(tval)

            if bfilled: b_filled_all += 1
            if tfilled: t_filled_all += 1
            b_app_vec.append(1 if bfilled else 0)
            t_app_vec.append(1 if tfilled else 0)

            if A[i] == 1:
                if bfilled: b_filled_av += 1
                if tfilled: t_filled_av += 1
                b_adj_vec.append(1 if bfilled else 0)
                t_adj_vec.append(1 if tfilled else 0)
                # discordant?
                if not bfilled and tfilled:
                    b01 += 1
                elif bfilled and not tfilled:
                    b10 += 1

        # Apparent coverage (on N)
        b_app = b_filled_all / N
        t_app = t_filled_all / N
        b_app_ci = wilson_ci(b_filled_all, N)
        t_app_ci = wilson_ci(t_filled_all, N)

        # Adjusted coverage (on Avail N)
        if availN > 0:
            b_adj = b_filled_av / availN
            t_adj = t_filled_av / availN
            b_adj_ci = wilson_ci(b_filled_av, availN)
            t_adj_ci = wilson_ci(t_filled_av, availN)
            miss_rate_t = 1.0 - t_adj
            p_mcn = mcnemar_p(b01, b10)
        else:
            b_adj = t_adj = 0.0
            b_adj_ci = t_adj_ci = (0.0, 0.0)
            miss_rate_t = 0.0
            p_mcn = "NA"

        out_rows.append({
            "Field": field,
            "N": N,
            "Avail N": availN,
            f"{args.base_label} ApparentCov": f"{b_app:.3f} {fmt_ci(*b_app_ci)}",
            f"{args.target_label} ApparentCov": f"{t_app:.3f} {fmt_ci(*t_app_ci)}",
            "Δ Apparent": f"{(t_app - b_app):+.3f}",
            f"{args.base_label} AdjCov": f"{b_adj:.3f} {fmt_ci(*b_adj_ci)}",
            f"{args.target_label} AdjCov": f"{t_adj:.3f} {fmt_ci(*t_adj_ci)}",
            "Δ AdjCov": f"{(t_adj - b_adj):+.3f}",
            f"MissRate({args.target_label})": f"{miss_rate_t:.3f}",
            "McNemar p (available)": p_mcn,
        })

    # Write CSV
    csv_fields = ["Field","N","Avail N",
                  f"{args.base_label} ApparentCov", f"{args.target_label} ApparentCov","Δ Apparent",
                  f"{args.base_label} AdjCov", f"{args.target_label} AdjCov","Δ AdjCov",
                  f"MissRate({args.target_label})","McNemar p (available)"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Write Markdown
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(f"# Results: {args.base_label} vs {args.target_label}\n\n")
        f.write(f"**Paired files (N)**: {N}\n\n")
        # table header
        f.write("| Field | N | Avail N | "
                f"{args.base_label} ApparentCov | {args.target_label} ApparentCov | Δ Apparent | "
                f"{args.base_label} AdjCov | {args.target_label} AdjCov | Δ AdjCov | "
                f"MissRate({args.target_label}) | McNemar p (available) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in out_rows:
            f.write("| {Field} | {N} | {Avail N} | {bapp} | {tapp} | {dapp} | {badj} | {tadj} | {dadj} | {miss} | {pmcn} |\n".format(
                Field=r["Field"], N=r["N"], **{
                    "Avail N": r["Avail N"],
                    "bapp": r[f"{args.base_label} ApparentCov"],
                    "tapp": r[f"{args.target_label} ApparentCov"],
                    "dapp": r["Δ Apparent"],
                    "badj": r[f"{args.base_label} AdjCov"],
                    "tadj": r[f"{args.target_label} AdjCov"],
                    "dadj": r["Δ AdjCov"],
                    "miss": r[f"MissRate({args.target_label})"],
                    "pmcn": r["McNemar p (available)"],
                }
            ))
    print(f"✓ Wrote Markdown -> {args.out_md}")
    print(f"✓ Wrote CSV      -> {args.out_csv}")

if __name__ == "__main__":
    main()
