#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal CLI extractor for PRISMA VLM fields.

Usage:
  python tools/extract_cli.py --version v18 --in <input.txt> --out <output.json>

Notes:
- Emits a dict with keys matching our SCHEMA (collector fills "File" later).
- Heuristic rules: looks for common datasets, models, encoders/decoders, etc.
- Version flag is accepted (v18/v21) so you can later plug different behavior.
"""

import argparse, json, re, sys
from pathlib import Path

SCHEMA_KEYS = [
    "Modality","Datasets (train)","Datasets (eval)","VLM?","Model","Class","Task",
    "Vision Enc","Lang Dec","Fusion","Objectives","RAG","Metrics(primary)","Family"
]

DATASETS = [
    "MIMIC-CXR", "CheXpert", "IU X-ray", "Open-i", "NIH ChestX-ray14",
    "CXR-RePaiR", "RadGraph", "PadChest", "COCO", "ImageNet"
]
MODELS = [
    "BLIP-2","BLIP2","BLIP","ALBEF","LLaVA","LLaMA","GPT-4V","GPT4V",
    "Flamingo","CXR-IRGen","TrMRG","ViLT","MiniGPT-4","mPLUG","Q-Former"
]
VISION_ENCS = ["ViT","CLIP","ResNet50","CNN","Swin","ConvNeXt","EfficientNet"]
LANG_DECS = ["LLaMA","GPT-2","GPT2","GPT-4V","BERT","T5","Flan-T5","Transformer","LSTM","GRU"]
FUSIONS = ["cross-attention","cross attention","q-former","co-attention","late fusion","early fusion","single-stream"]
METRICS = ["BLEU","CIDEr","METEOR","ROUGE","Accuracy","F1","BERTScore","CheXbert","RadGraph","RadCliQ","GLEU"]

def find_any(text, terms):
    hits = []
    low = text.lower()
    for t in terms:
        if t.lower() in low:
            hits.append(t)
    return sorted(set(hits), key=lambda x: low.find(x.lower()))

def guess_modality(text):
    low = text.lower()
    for k in ["x-ray","xray","x ray","chest x-ray","cxr","mammography","ct","mri","ultrasound"]:
        if k in low:
            return "X-Ray" if "x" in k else k.upper()
    return "Not reported"

def guess_class(text):
    # classification/regression/detection/segmentation etc… but your corpus is mostly report-generation
    if re.search(r"\breport[- ]?generation\b", text, re.I):
        return "Not reported"  # keep Class separate from Task
    for k in ["classification","detection","segmentation","retrieval","captioning"]:
        if re.search(rf"\b{k}\b", text, re.I):
            return k
    return "Not reported"

def guess_task(text):
    if re.search(r"\breport[- ]?generation\b", text, re.I):
        return "report-generation"
    for k in ["captioning","retrieval","qa","question answering","triage","classification"]:
        if re.search(rf"\b{k}\b", text, re.I):
            return k
    return "report-generation"  # sensible default for your set

def guess_vlm(text):
    low = text.lower()
    if any(word in low for word in ["vlm","vision-language","vision language","multimodal","vqg","report generation"]):
        return "Yes"
    return "Not reported"

def guess_fusion(text):
    low = text.lower()
    if "cross-attention" in low or "cross attention" in low:
        return "cross-attention"
    if "q-former" in low or "qformer" in low:
        return "q-former"
    if "single-stream" in low or "single stream" in low:
        return "single-stream"
    if "late fusion" in low:
        return "late fusion"
    if "early fusion" in low:
        return "early fusion"
    return "Not reported"

def guess_objectives(text):
    low = text.lower()
    toks = []
    if any(k in low for k in ["itm", "image-text matching", "itc", "contrastive"]):
        toks.append("ITM/ITC")
    if any(k in low for k in ["coverage", "coverage reward"]):
        toks.append("coverage")
    return ", ".join(sorted(set(toks))) if toks else "Not reported"

def guess_rag(text):
    low = text.lower()
    return "Yes" if any(k in low for k in ["retrieval-augmented","retrieval augmented","rag","retrieve-then-generate"]) else "Not reported"

def guess_family(text, model_hits, fusion_guess):
    low = text.lower()
    if any(k in low for k in ["clip","siglip","contrast"]):
        return "Contrastive"
    if any(m.lower() in ["blip","blip2","blip-2","albef","flan-t5","t5"] for m in model_hits) or "cross-attention" in low:
        return "Encoder–Decoder"
    if any(m.lower() in ["llava","llama","q-former","mplug","minigpt","vision-instruct"] for m in model_hits) or "q-former" in fusion_guess:
        return "Adaptered LLM (V+L)"
    if "single-stream" in fusion_guess:
        return "Single-stream Transformer"
    if any(k in low for k in ["knowledge","retrieval","rag"]):
        return "Knowledge-augmented"
    return "Not reported"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v18","v21"])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    txt = Path(args.inp).read_text(encoding="utf-8", errors="ignore")

    ds_hits = find_any(txt, DATASETS)
    model_hits = find_any(txt, MODELS)
    venc_hits = find_any(txt, VISION_ENCS)
    ldec_hits = find_any(txt, LANG_DECS)
    fusion_guess = guess_fusion(txt)
    metrics_hits = find_any(txt, METRICS)

    rec = {
        "Modality": guess_modality(txt),
        "Datasets (train)": ", ".join(ds_hits) if ds_hits else "Not reported",
        "Datasets (eval)":  ", ".join(ds_hits) if ds_hits else "Not reported",
        "VLM?": guess_vlm(txt),
        "Model": ", ".join(model_hits) if model_hits else "Not reported",
        "Class": guess_class(txt),
        "Task": guess_task(txt),
        "Vision Enc": ", ".join(venc_hits) if venc_hits else ("CLIP" if "clip" in txt.lower() else "Not reported"),
        "Lang Dec": ", ".join(ldec_hits) if ldec_hits else "Not reported",
        "Fusion": fusion_guess,
        "Objectives": guess_objectives(txt),
        "RAG": guess_rag(txt),
        "Metrics(primary)": ", ".join(metrics_hits) if metrics_hits else "Not reported",
        "Family": guess_family(txt, model_hits, fusion_guess),
    }

    # Slightly stricter mode for v18 (optional): if you want fewer false positives
    if args.version == "v18":
        # Example: if model not detected, don’t backfill Vision Enc from CLIP
        if rec["Model"] == "Not reported" and rec["Vision Enc"] == "CLIP":
            rec["Vision Enc"] = "Not reported"

    Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
