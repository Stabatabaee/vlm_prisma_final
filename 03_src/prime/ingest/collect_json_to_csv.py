mkdir -p scripts

cat > scripts/collect_json_to_csv.py <<'PY'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# Unified schema (no "Paired")
SCHEMA = [
    "File","Modality","Datasets (train)","Datasets (eval)","VLM?","Model","Class","Task",
    "Vision Enc","Lang Dec","Fusion","Objectives","RAG","Metrics(primary)","Family"
]

def normalize(val):
    if val is None: return np.nan
    if isinstance(val, (list, tuple)): return ", ".join(map(str, val))
    s = str(val).strip()
    return np.nan if s == "" else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True, help="Directory containing one JSON per File")
    ap.add_argument("--manifest", required=True, help="outputs/manifest_84.csv to enforce order and coverage")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    rows = []
    for _, r in man.iterrows():
        fid = r["File"]
        p = Path(args.json_dir) / f"{fid}.json"
        rec = {k: np.nan for k in SCHEMA}
        rec["File"] = fid
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for k in SCHEMA:
                if k in data:
                    rec[k] = normalize(data[k])
        rows.append(rec)

    df = pd.DataFrame(rows, columns=SCHEMA)
    df.columns = [c.strip() for c in df.columns]  # trim header whitespace
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Wrote {args.out} rows={len(df)}")

if __name__ == "__main__":
    main()
PY

chmod +x scripts/collect_json_to_csv.py
