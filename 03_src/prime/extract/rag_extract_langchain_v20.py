#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heuristic post-RAG cleaner/filler for radiology VLM papers (no LLMs).

Run example:
  python rag_extract_langchain_v20.py \
    --snippets-dir gold_snippets \
    --out-csv gold_pred_v20_strict.csv \
    --out-md  gold_pred_v20_strict.md \
    --evidence-mode strict

Evidence modes:
  - strict : high precision (tight windows, exact-ish cues)
  - hybrid : balanced (broader synonyms, larger windows)
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Tuple

COLUMNS = [
    "File", "Modality", "Datasets (train)", "Datasets (eval)", "Paired",
    "VLM?", "Model", "Class", "Task", "Vision Enc", "Lang Dec",
    "Fusion", "Objectives", "Family", "RAG", "Metrics(primary)"
]

# -----------------------
# helpers
# -----------------------

def uniq_join(values: List[str]) -> str:
    seen = set()
    out = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            out.append(v)
            seen.add(key)
    return ", ".join(out)

def truthy(x: str) -> bool:
    return bool(x and x.strip())

def find_all_with_pos(text: str, pats: List[str]) -> List[int]:
    pos = []
    for pat in pats:
        for m in re.finditer(pat, text, re.I):
            pos.append(m.start())
    return pos

def window(text: str, pos: int, w: int) -> str:
    L = max(0, pos - w)
    R = min(len(text), pos + w)
    return text[L:R]

def any_re(text: str, pats: List[str]) -> bool:
    return any(re.search(p, text, re.I) for p in pats)

# -----------------------
# cues and dictionaries
# -----------------------

# Datasets & aliases (lowercased keys)
DATASET_MAP = {
    "mimic-cxr": [r"\bmimic[- ]?cxr\b"],
    "iu x-ray": [r"\biu[- ]?x[- ]?ray\b", r"\bopen[- ]?i\b"],
    "chexpert": [r"\bchexpert\b"],
    "nih chestx-ray14": [r"\b(chestx[- ]?ray14|nih[- ]?14|nih chestx[- ]?ray14)\b"],
    "cxr-repair": [r"\bcxr[- ]?repair\b"],
    "rsna pneumonia": [r"\brsna pneumonia\b"],
}

# Pretty names
DS_CANON = {
    "mimic-cxr": "MIMIC-CXR",
    "iu x-ray": "IU X-ray",
    "chexpert": "CheXpert",
    "nih chestx-ray14": "NIH ChestX-ray14",
    "cxr-repair": "CXR-RePaiR",
    "rsna pneumonia": "RSNA Pneumonia",
}

# datasets that almost surely imply image–report pairs
PAIRED_STRONG = {"mimic-cxr", "iu x-ray", "cxr-repair", "chexpert"}

PAIRED_PHRASES = [
    r"\bpaired (image[- ]text|image[- ]report|image[- ]caption)\b",
    r"\bimage[- ]report pairs?\b",
    r"\bpaired data(set)?\b",
]

# Known families/models
FAMILY_CUES = {
    "ALBEF": [r"\balbef\b"],
    "BLIP": [r"\bblip\b(?!-?2)"],
    "BLIP-2": [r"\bblip[- ]?2\b", r"\binstruct[- ]?blip\b"],
    "Flamingo": [r"\bflamingo\b"],
    "LLaVA": [r"\bllava(\s*[-]?\s*med|\s*1\.5)?\b"],
    "MiniGPT-4": [r"\bminigpt[- ]?4\b"],
    "XrayGPT": [r"\bxraygpt\b"],
    "Qwen-VL": [r"\bqwen[- ]?vl\b"],
    "CheXNet": [r"\bchexnet\b"],
    "TrMRG": [r"\btrmrg\b", r"\bme[- ]?transformer\b"],
}
VLM_FAMILIES = {"ALBEF","BLIP","BLIP-2","Flamingo","LLaVA","MiniGPT-4","XrayGPT","Qwen-VL"}

# Vision enc / Language dec
VENC_CUES = {
    "ViT": [r"\bvit\b", r"\bvision transformer\b", r"\bvit[- ]?(b|l|h)/?(14|16)?\b"],
    "CLIP": [r"\bclip\b", r"\bclip (vit|resnet)\b"],
    "ResNet50": [r"\bresnet[- ]?50\b"],
    "CNN": [r"\bcnn\b", r"\bconvolution(al)?\b"],
    "Swin": [r"\bswin[- ]?transformer\b"],
    "ConvNeXt": [r"\bconvnext\b"],
}
LDEC_CUES = {
    "LLaMA": [r"\bllama(-\d+b)?\b", r"\bmeta[- ]?llama\b", r"\bllama[- ]?(2|3)\b"],
    "Vicuna": [r"\bvicuna\b"],
    "GPT-2": [r"\bgpt[- ]?2\b"],
    "GPT-3": [r"\bgpt[- ]?3(\.5)?\b"],
    "BioGPT": [r"\bbiogpt\b"],
    "T5": [r"\bt5\b"],
    "BERT": [r"\bbert\b"],
    "Transformer": [r"\btransformer\b(?! with)"],
    "LSTM": [r"\blstm\b"],
    "GRU": [r"\bgru\b"],
}

# Fusion / Objectives / Metrics
FUSION_CUES = {
    "cross-attention": [r"\bcross[- ]attention\b"],
    "conditioned": [r"\bcondition(ed|ing)\b"],
    "co-attention": [r"\bco[- ]attention\b"],
}
OBJ_CUES = {
    "ITM": [r"\bimage[- ]?text matching\b", r"\bitm\b"],
    "contrastive": [r"\bcontrastive\b"],
    "coverage": [r"\bcoverage\b"],
}
METRIC_CUES = {
    "BLEU": [r"\bbleu(\d*)?\b", r"\bbleu[- ]?(1|2|3|4)\b"],
    "CIDEr": [r"\bcider\b"],
    "METEOR": [r"\bmeteor\b"],
    "ROUGE": [r"\brouge(-[l1-9])?\b", r"\brouge[- ]?l\b"],
    "BERTScore": [r"\bbertscore\b"],
    "BLEURT": [r"\bbleurt\b"],
    "SPICE": [r"\bspice\b"],
    "RadGraph": [r"\bradgraph\b"],
    "RadCliQ": [r"\bradcliq\b"],
    "GLEU": [r"\bgleu\b"],
    "Accuracy": [r"\baccuracy\b"],
    "F1": [r"\bf1\b"],
    "AUC": [r"\bauc\b"],
}

RAG_CUES = [
    r"\bretrieval[- ]?augmented\b",
    r"\bRAG\b",
    r"\bretriever\b",
    r"\bfaiss\b",
    r"\bBM25\b",
    r"\bvector (index|store)\b",
    r"\bdpr\b",
    r"\bmemory (bank|module)\b",
]

# Modality cues (we’ll override with dataset logic first)
MOD_STRICT = {
    "X-Ray": [r"\bchest x-?ray(s)?\b", r"\bcxr\b", r"\bradiograph(s)?\b"],
    "CT": [r"\bct\b", r"\bcomputed tomography\b"],
    "MRI": [r"\bmri\b", r"\bmagnetic resonance\b"],
    "Ultrasound": [r"\bultrasound\b", r"\bus\b"],
}
MOD_HYBRID = {
    "X-Ray": [r"\bx[- ]?ray(s)?\b"],
    "CT": [r"\bct[- ]?scan\b"],
    "MRI": [r"\bmr[- ]?imaging\b"],
    "Ultrasound": [r"\bsono(graphy|gram)\b"],
}

TASK_STRICT = {
    "report-generation": [
        r"\bradiology report generation\b",
        r"\breport generation\b",
        r"\bfree[- ]?text report\b",
    ],
}
TASK_HYBRID = {
    "report-generation": [
        r"\breport(s)?\b",
        r"\bfindings\b",
        r"\bimpression(s)?\b",
        r"\bnatural[- ]?language\b",
    ],
}

# -----------------------
# extraction routines
# -----------------------

def nice_ds(name: str) -> str:
    return DS_CANON.get(name, name.title())

def extract_datasets(text: str, w_train: int, w_eval: int) -> Tuple[str, str, bool]:
    hits = []
    for ds, pats in DATASET_MAP.items():
        for pat in pats:
            for m in re.finditer(pat, text, re.I):
                hits.append((ds, m.start()))
    if not hits:
        return "", "", False

    train_keys = [r"\btrain(ing)?\b", r"\bpretrain(ing)?\b", r"\bfine[- ]?tune(d|ing)?\b"]
    eval_keys  = [r"\b(eval(uate|uation)?|test|validation|val)\b"]

    train, eval_ = set(), set()
    paired_flag = False

    for ds, pos in hits:
        tw = window(text, pos, w_train)
        ew = window(text, pos, w_eval)
        is_tr = any_re(tw, train_keys)
        is_ev = any_re(ew, eval_keys)

        if is_tr: train.add(ds)
        if is_ev: eval_.add(ds)
        if not is_tr and not is_ev:
            # unknown side: put in both (keeps recall)
            train.add(ds); eval_.add(ds)

        # paired signals in local window
        loc = window(text, pos, max(w_train, w_eval))
        if ds in PAIRED_STRONG or any_re(loc, PAIRED_PHRASES):
            # also ask for "report" or "caption" around it
            if any_re(loc, [r"\breport(s)?\b", r"\bcaption(s)?\b", r"\btext\b"]):
                paired_flag = True

    tr = ", ".join(sorted(nice_ds(d) for d in train))
    ev = ", ".join(sorted(nice_ds(d) for d in eval_))
    return tr, ev, paired_flag

def extract_modality(text: str, dtrain: str, deval: str, evidence_mode: str) -> str:
    # 1) dataset‑driven modality lock (CXR datasets → X-Ray)
    cxr_sets = {"MIMIC-CXR", "IU X-ray", "CheXpert", "NIH ChestX-ray14", "CXR-RePaiR", "RSNA Pneumonia"}
    any_cxr = any(ds.strip() in cxr_sets for ds in (dtrain + "," + deval).split(",") if ds.strip())
    if any_cxr:
        # Only add CT/MRI/US if explicit strong cues are present
        add = []
        strong_ct = any_re(text, [r"\bcomputed tomography\b"])
        strong_mri = any_re(text, [r"\bmagnetic resonance\b"])
        strong_us  = any_re(text, [r"\bultrasound\b"])
        base = ["X-Ray"]
        if evidence_mode == "hybrid":
            # in hybrid allow lenient extras but still require clear words
            if strong_ct: add.append("CT")
            if strong_mri: add.append("MRI")
            if strong_us: add.append("Ultrasound")
        else:
            # strict: only if super explicit and not just a survey mention
            if strong_ct and any_re(text, [r"\bct[- ]?scan\b"]): add.append("CT")
            if strong_mri and any_re(text, [r"\bmri\b"]): add.append("MRI")
            if strong_us and any_re(text, [r"\bsonograph[y|ic]\b", r"\bpoint[- ]of[- ]care ultrasound\b"]): add.append("Ultrasound")
        return uniq_join(base + add)

    # 2) if no dataset lock, fall back to text cues
    cue_map = MOD_STRICT if evidence_mode == "strict" else {**MOD_STRICT, **MOD_HYBRID}
    hits = []
    for mod, pats in cue_map.items():
        if any_re(text, pats):
            hits.append(mod)
    return uniq_join(hits)

def scan_scoped(text: str, cue_map: Dict[str, List[str]], anchors: List[str], win: int) -> List[str]:
    """Return labels whose cue AND an anchor appear within `win` chars."""
    found = []
    for label, pats in cue_map.items():
        pos = find_all_with_pos(text, pats)
        ok = False
        for p in pos:
            if any_re(window(text, p, win), anchors):
                ok = True
                break
        if ok:
            found.append(label)
    return found

def extract_venc(text: str, evidence_mode: str) -> str:
    # stronger if near encoder/backbone words
    win = 80 if evidence_mode == "strict" else 140
    anchors = [r"\bencoder\b", r"\bbackbone\b", r"\bvision\b", r"\bimage\b", r"\bpatch(es)?\b"]
    hits = scan_scoped(text, VENC_CUES, anchors, win)
    return uniq_join(sorted(set(hits), key=str.lower))

def extract_ldec(text: str, evidence_mode: str) -> str:
    win = 80 if evidence_mode == "strict" else 140
    anchors = [r"\bdecoder\b", r"\blanguage (model|decoder)\b", r"\bgenerator\b", r"\btext\b"]
    hits = scan_scoped(text, LDEC_CUES, anchors, win)
    return uniq_join(sorted(set(hits), key=str.lower))

def extract_fusion(text: str, evidence_mode: str) -> str:
    win = 80 if evidence_mode == "strict" else 140
    anchors = [r"\battention\b", r"\bfusion\b", r"\balign(ment|ing|ed)\b"]
    hits = scan_scoped(text, FUSION_CUES, anchors, win)
    return uniq_join(sorted(set(hits), key=str.lower))

def extract_objectives(text: str, evidence_mode: str) -> str:
    win = 80 if evidence_mode == "strict" else 140
    anchors = [r"\bloss\b", r"\bobjective\b", r"\boptimi[sz]e(d|r|ation)\b", r"\btraining\b"]
    hits = scan_scoped(text, OBJ_CUES, anchors, win)
    return uniq_join(sorted(set(hits), key=str.lower))

def extract_metrics(text: str) -> str:
    hits = []
    for lab, pats in METRIC_CUES.items():
        if any_re(text, pats):
            hits.append(lab)
    return uniq_join(sorted(set(hits), key=str.lower))

def extract_family_and_model(text: str) -> Tuple[str, str]:
    fams = []
    for fam, pats in FAMILY_CUES.items():
        if any_re(text, pats):
            fams.append(fam)
    fams = sorted(set(fams), key=str.lower)
    family = uniq_join(fams)
    model = family  # acceptable proxy
    return model, family

def infer_vlm(text: str, family: str, venc: str, ldec: str) -> str:
    # Prefer family whitelist
    if truthy(family):
        if any(fam in VLM_FAMILIES for fam in [s.strip() for s in family.split(",")]):
            return "Yes"
    # Generic claim must be backed by both sides present
    if any_re(text, [r"\bvision[- ]language\b", r"\bmulti[- ]modal\b"]) and truthy(venc) and truthy(ldec):
        return "Yes"
    return ""

def extract_task(text: str, evidence_mode: str) -> str:
    cue_map = TASK_STRICT if evidence_mode == "strict" else {**TASK_STRICT, **TASK_HYBRID}
    hits = []
    for lab, pats in cue_map.items():
        if any_re(text, pats):
            hits.append(lab)
    return uniq_join(hits)

def extract_rag(text: str) -> str:
    return "Yes" if any_re(text, RAG_CUES) else ""

def infer_class(vlm: str, venc: str, ldec: str) -> str:
    if vlm == "Yes":
        return "VLM (multimodal)"
    if ("CNN" in venc) and (("Transformer" in ldec) or ("BERT" in ldec)):
        return "CNN+Transformer"
    if (("ViT" in venc) or ("CLIP" in venc)) and ("Transformer" in ldec):
        return "Encoder–Decoder"
    return ""

# -----------------------
# main record builder
# -----------------------

def process_file(path: str, evidence_mode: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    rec = {c: "" for c in COLUMNS}
    rec["File"] = os.path.basename(path)

    # window sizes by evidence mode
    w_train = 80 if evidence_mode == "strict" else 140
    w_eval  = 80 if evidence_mode == "strict" else 140

    dtrain, deval, paired = extract_datasets(text, w_train, w_eval)
    modality = extract_modality(text, dtrain, deval, evidence_mode)
    task = extract_task(text, evidence_mode)
    venc = extract_venc(text, evidence_mode)
    ldec = extract_ldec(text, evidence_mode)
    fusion = extract_fusion(text, evidence_mode)
    objs = extract_objectives(text, evidence_mode)
    metrics = extract_metrics(text)
    model, family = extract_family_and_model(text)
    vlm = infer_vlm(text, family, venc, ldec)
    rag = extract_rag(text)
    clazz = infer_class(vlm, venc, ldec)

    if truthy(modality): rec["Modality"] = modality
    if truthy(dtrain):   rec["Datasets (train)"] = dtrain
    if truthy(deval):    rec["Datasets (eval)"] = deval
    if paired:           rec["Paired"] = "Yes"
    if truthy(vlm):      rec["VLM?"] = vlm
    if truthy(model):    rec["Model"] = model
    if truthy(clazz):    rec["Class"] = clazz
    if truthy(task):     rec["Task"] = task
    if truthy(venc):     rec["Vision Enc"] = venc
    if truthy(ldec):     rec["Lang Dec"] = ldec
    if truthy(fusion):   rec["Fusion"] = fusion
    if truthy(objs):     rec["Objectives"] = objs
    if truthy(family):   rec["Family"] = family
    if truthy(rag):      rec["RAG"] = rag
    if truthy(metrics):  rec["Metrics(primary)"] = metrics

    return rec

def write_csv(path: str, rows: List[Dict[str, str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})

def write_md(path: str, rows: List[Dict[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(COLUMNS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r.get(c, "") for c in COLUMNS) + " |\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True, help="Directory of *.txt snippets")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--evidence-mode", choices=["strict","hybrid"], default="strict")
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.snippets_dir, x)
        for x in os.listdir(args.snippets_dir)
        if x.lower().endswith(".txt")
    )

    rows = []
    for p in files:
        try:
            rec = process_file(p, args.evidence_mode)
            rows.append(rec)
            print(f"✓ {os.path.basename(p)}")
        except Exception as e:
            print(f"⚠️  Failed {os.path.basename(p)}: {e}")

    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    print(f"✅ Wrote {args.out_csv} and {args.out_md}")

if __name__ == "__main__":
    main()
