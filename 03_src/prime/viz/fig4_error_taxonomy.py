# fig4_error_taxonomy.py
import argparse, os, re
import pandas as pd
import matplotlib.pyplot as plt

def has_multiple_items(x: str) -> bool:
    if not isinstance(x, str):
        return False
    s = x.strip()
    return bool(re.search(r",|\+|\band\b", s, flags=re.I))

def pct(n, d): 
    return 0.0 if d == 0 else 100.0 * n / d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaldir", default="output/eval_v22", help="Directory with mismatches_*.csv")
    ap.add_argument("--out", default="figs/fig4_error_taxonomy.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Collect mismatches
    buckets = {"multi-objective ambiguity": 0, "multi-model reporting": 0}
    total_err = 0

    # Objectives-based ambiguity
    obj_path = os.path.join(args.evaldir, "mismatches_Objectives.csv")
    if os.path.exists(obj_path):
        df = pd.read_csv(obj_path)
        for _, r in df.iterrows():
            g = str(r.get("Objectives_gold", "") or "")
            p = str(r.get("Objectives_pred", "") or "")
            if has_multiple_items(g) or has_multiple_items(p):
                buckets["multi-objective ambiguity"] += 1
            total_err += 1

    # Model-based multi-model reporting
    model_path = os.path.join(args.evaldir, "mismatches_Model.csv")
    if os.path.exists(model_path):
        df = pd.read_csv(model_path)
        for _, r in df.iterrows():
            g = str(r.get("Model_gold", "") or "")
            p = str(r.get("Model_pred", "") or "")
            if has_multiple_items(g) or has_multiple_items(p):
                buckets["multi-model reporting"] += 1
            total_err += 1

    # Avoid zero division; show an empty pie if no mismatches found
    labels = []
    sizes  = []
    for k, v in buckets.items():
        if total_err > 0 and v > 0:
            labels.append(f"{k} ({v})")
            sizes.append(v)
    if not sizes:
        labels = ["No residual mismatches"]
        sizes  = [1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda x: f"{x:.1f}%",
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 10}
    )
    ax.set_title("Figure 4. Error taxonomy in v22 diagnostics", fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"✓ Saved {args.out}")

if __name__ == "__main__":
    main()
