# fig5_wilson_cis.py
import argparse, os, math
import pandas as pd
import matplotlib.pyplot as plt

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2/(2*n)
    margin = z * math.sqrt(phat*(1-phat)/n + z**2/(4*n*n))
    low  = (centre - margin) / denom
    high = (centre + margin) / denom
    return phat, max(0.0, low), min(1.0, high)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="output/eval_v22/summary_two_stage_eval.csv")
    ap.add_argument("--label", default="v22")
    ap.add_argument("--out", default="figs/fig5_wilson_cis_v22.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.summary)
    # Must have: field, available_gold, correct_given_available
    fields = df["field"].tolist()
    k = df["correct_given_available"].tolist()
    n = df["available_gold"].tolist()

    pts, los, his = [], [], []
    for ki, ni in zip(k, n):
        p, lo, hi = wilson_ci(int(ki), int(ni))
        pts.append(p)
        los.append(p - lo)
        his.append(hi - p)

    # order fields by point estimate (optional)
    order = sorted(range(len(fields)), key=lambda i: pts[i])
    fields = [fields[i] for i in order]
    pts    = [pts[i] for i in order]
    los    = [los[i] for i in order]
    his    = [his[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    y = range(len(fields))
    ax.errorbar(pts, y, xerr=[los, his], fmt='o', capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(fields)
    ax.set_xlabel("Availability-adjusted coverage (Wilson 95% CI)")
    ax.set_title(f"Figure 5. Coverage with Wilson CIs — {args.label}")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"✓ Saved {args.out}")

if __name__ == "__main__":
    main()
