#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V10 — VLM-centric extractor for systematic review tables (context-validated).

Key changes vs v9:
- Aggregate doc-level context across prompts and post-validate LLM outputs.
- Gate booleans (is_vlm, rag_used, paired_reports) by actual keyword presence.
- Filter lists (metrics/objectives/etc.) to tokens found in context.
- Keep imports resilient across LangChain versions.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# -----------------------------
# Resilient imports (LangChain)
# -----------------------------
Chroma = None
try:
    from langchain_chroma import Chroma  # new
except Exception:
    try:
        from langchain_community.vectorstores import Chroma  # mid
    except Exception:
        try:
            from langchain.vectorstores import Chroma  # legacy
        except Exception as e:
            print("[FATAL] Could not import Chroma.", file=sys.stderr)
            raise e

SentenceTransformerEmbeddings = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
        except Exception as e:
            print("[FATAL] Could not import HuggingFaceEmbeddings.", file=sys.stderr)
            raise e

from langchain.docstore.document import Document

DB_DIR = "chroma_db"

# -----------------------------
# Optional LLM backends
# -----------------------------
USE_OPENAI = False
try:
    import openai
    if os.getenv("OPENAI_API_KEY"):
        USE_OPENAI = True
except Exception:
    pass

HF_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextGenerationPipeline
    import torch
    HF_AVAILABLE = True
except Exception:
    pass

# -----------------------------
# Small utilities
# -----------------------------
def die(msg: str, code: int = 1):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def first_scalar(x: Any) -> str:
    if x is None:
        return "Not reported"
    if isinstance(x, list) and x:
        return norm_space(x[0])
    if isinstance(x, (str, int, float, bool)):
        return norm_space(x)
    return "Not reported"

def to_list(x: Any) -> List[str]:
    if x is None:
        return ["Not reported"]
    if isinstance(x, list):
        vals = [norm_space(i) for i in x if norm_space(i)]
        return vals if vals else ["Not reported"]
    s = norm_space(x)
    return [s] if s else ["Not reported"]

def to_bool(x: Any, default=False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"true","yes","y","1"}:
            return True
        if t in {"false","no","n","0"}:
            return False
    return default

def canon_modality(x: Any) -> str:
    s = first_scalar(x)
    if s == "Not reported":
        return s
    sl = s.lower()
    if "x-ray" in sl or "cxr" in sl or "chest" in sl:
        return "X-Ray"
    if "ct" in sl:
        return "CT"
    if "mri" in sl:
        return "MRI"
    if "ultra" in sl or sl == "us":
        return "Ultrasound"
    return "Other"

def load_vectordb() -> Tuple[Any, Any]:
    embed = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embed, collection_name="papers")
    return vectordb, embed

def make_retriever(vectordb: Any, k: int = 10, mmr: bool = True, lambda_mult: float = 0.45):
    search_type = "mmr" if mmr else "similarity"
    kwargs = {"k": k}
    if mmr:
        kwargs["lambda_mult"] = lambda_mult
    return vectordb.as_retriever(search_type=search_type, search_kwargs=kwargs)

def read_snippet_text(snippets_dir: Path, doc_id: str) -> str:
    txt = []
    for p in sorted(snippets_dir.glob("*.txt")):
        if p.stem.startswith(doc_id) or doc_id in p.stem:
            try:
                txt.append(p.read_text(errors="ignore"))
            except Exception:
                pass
    if not txt:
        for p in sorted(snippets_dir.glob("*.txt")):
            try:
                txt.append(p.read_text(errors="ignore"))
            except Exception:
                pass
    return "\n".join(txt)

# -----------------------------
# Lexicons for validation
# -----------------------------
DATASET_TOKENS = [
    "mimic-cxr","iu x-ray","iu xray","open-i","open i","chexpert","rsna","padchest","vindr","brax","nih","nih-chestxray","chestx-ray8"
]
ENCODER_TOKENS = ["vit","clip","swin","resnet","deit","densenet","efficientnet","dino"]
DECODER_TOKENS = ["llama","gpt-2","t5","bart","transformer","lstm","gpt2","llama-2","llama-3"]
FUSION_TOKENS = ["cross-attention","co-attention","q-former","concatenation","early","late","gated"]
OBJECTIVE_TOKENS = ["itc","contrastive","itm","cross-entropy","captioning","rl","cider","scst","coverage","alignment","radgraph"]
METRIC_TOKENS = ["cider","bleu","meteor","rouge","bertscore","chexbert","radgraph","clinical efficacy"]
FAMILY_TOKENS = ["blip","blip-2","albef","llava","instructblip","roentgen","cxr-irgen"]
RAG_TOKENS = ["rag","retrieval","faiss","bm25","top-k","nearest neighbor"]
PAIRED_TOKENS_TRUE = ["image-report pair","image–report pair","image text pair","paired reports","paired image","paired data"]
PAIRED_TOKENS_FALSE = ["unpaired","no paired","pairing not","without paired"]

def contains_any(text: str, tokens: List[str]) -> bool:
    tl = text.lower()
    return any(t in tl for t in tokens)

def present_tokens(text: str, tokens: List[str]) -> List[str]:
    tl = text.lower()
    found = []
    for t in tokens:
        if t in tl:
            found.append(t)
    return found

# -----------------------------
# Prompt pack (same as your v9 intent)
# -----------------------------
PREAMBLE = (
    "You are an information-extraction assistant. Read ONLY the provided context (snippets).\n"
    "Return STRICT JSON matching the requested schema.\n"
    '- If a value is absent, use "Not reported".\n'
    "- Do NOT guess; do NOT infer beyond the text.\n"
    "- Use canonical names where possible (ViT-B/16, CLIP-ViT-L/14, Swin-T, ResNet50, BERT, T5, GPT-2, LLaMA-2/3).\n"
)

PROMPTS: Dict[str, Dict[str, str]] = {
    "datasets_modalities": {
        "schema": json.dumps({
            "modality": ["X-Ray","CXR","CT","MRI","Ultrasound","Other","Not reported"],
            "datasets_train": ["..."],
            "datasets_eval": ["..."],
            "paired_reports": "true/false/Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Extract imaging modality and datasets (train/eval). State whether image–report pairs are used.",
        "query_hint": "modality dataset MIMIC IU Open-i CheXpert RSNA paired reports CXR chest X-Ray CT MRI Ultrasound"
    },
    "vlm_presence": {
        "schema": json.dumps({
            "is_vlm": "true/false",
            "model_name": "Not reported or specific",
            "model_class": "dual-encoder|single-stream|encoder-decoder|Q-Former-bridge|other|Not reported",
            "task": "report-generation|captioning|retrieval|other|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Decide whether a VLM is used; classify architecture & primary task.",
        "query_hint": "vision-language BLIP ALBEF LLaVA InstructBLIP encoder-decoder dual-encoder report generation captioning retrieval"
    },
    "vision_encoder": {
        "schema": json.dumps({
            "vision_encoder": ["ViT-B/16","CLIP-ViT-L/14","Swin-T","ResNet50","DeiT","DenseNet121","Not reported"],
            "pretrained_from": "ImageNet|CLIP|medical|scratch|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Identify visual backbone(s) + pretraining source.",
        "query_hint": "vision encoder backbone ViT Swin ResNet CLIP DeiT DenseNet pretrained ImageNet"
    },
    "language_decoder": {
        "schema": json.dumps({
            "language_decoder": ["LLaMA-2","LLaMA-3","GPT-2","T5","BART","Transformer","LSTM","Not reported"],
            "decoder_size_params": "e.g., 7B|770M|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Return the decoder used for generation.",
        "query_hint": "decoder LLaMA GPT-2 T5 BART Transformer LSTM parameter size"
    },
    "fusion_alignment": {
        "schema": json.dumps({
            "fusion": "cross-attention|co-attention|concatenation|early|late|gated|Q-Former|other|Not reported",
            "where": "encoder|decoder|both|Not reported",
            "notes": "brief or Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Describe vision–language fusion and where it occurs.",
        "query_hint": "fusion alignment cross-attention co-attention Q-Former decoder encoder"
    },
    "pretraining_objectives": {
        "schema": json.dumps({
            "objectives": ["ITC(contrastive)","ITM","captioning CE/NLL","RL(CIDEr/SCST)","alignment loss","coverage","other","Not reported"],
            "evidence": "..."
        }, indent=2),
        "instruction": "List pretraining/finetuning objectives.",
        "query_hint": "contrastive ITC ITM captioning cross-entropy RL CIDEr SCST RadGraph loss"
    },
    "vlm_family_or_adaptation": {
        "schema": json.dumps({
            "family": "BLIP|BLIP-2|ALBEF|LLaVA|InstructBLIP|custom-medical|other|Not reported",
            "adaptation": "from-general|from-medical-scratch|finetune-only|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Map to a known VLM family or bespoke; how adapted.",
        "query_hint": "BLIP BLIP-2 ALBEF LLaVA InstructBLIP medical adaptation finetune"
    },
    "retrieval_rag": {
        "schema": json.dumps({
            "rag_used": "true/false",
            "retriever": "FAISS|BM25|dual-encoder|other|Not reported",
            "top_k": "int or Not reported",
            "usage": "retrieval-before-generation|in-context|memory-bank|training-only|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Identify RAG component and integration.",
        "query_hint": "retrieval RAG FAISS BM25 top-k in-context memory"
    },
    "inputs_and_layout": {
        "schema": json.dumps({
            "views": "single|multi-view|temporal|Not reported",
            "image_resolution": "e.g., 224x224|1024x1024|Not reported",
            "visual_tokens": "patch|region|heatmap|report-attn|other|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Capture inputs (views), resolution, tokenization style.",
        "query_hint": "PA view lateral multi-view 224x224 512 tokens patch region"
    },
    "metrics_key_for_vlm": {
        "schema": json.dumps({
            "metrics": ["CIDEr","BERTScore","ROUGE-L","BLEU-4","METEOR","CheXbert-F1","RadGraph-F1","Clinical Efficacy","Not reported"],
            "primary_metric": "CIDEr|BERTScore|ROUGE-L|CheXbert-F1|RadGraph-F1|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "List metrics; mark primary if stated.",
        "query_hint": "CIDEr BLEU METEOR ROUGE BERTScore CheXbert RadGraph clinical efficacy"
    },
    "training_and_compute": {
        "schema": json.dumps({
            "train_images": "int or Not reported",
            "train_reports": "int or Not reported",
            "epochs": "int or Not reported",
            "batch_size": "int or Not reported",
            "optimizer": "Adam|AdamW|LAMB|SGD|Not reported",
            "learning_rate": "value or Not reported",
            "hardware": "GPUs/TPUs and count|Not reported",
            "evidence": "..."
        }, indent=2),
        "instruction": "Extract training setup & compute if available.",
        "query_hint": "epochs batch size optimizer learning rate GPU TPU"
    },
}

CSV_COLUMNS = [
    "doc_id",
    "modality","datasets_train","datasets_eval","paired_reports",
    "is_vlm","model_name","model_class","task",
    "vision_encoder","pretrained_from",
    "language_decoder","decoder_size_params",
    "fusion","where","notes",
    "objectives",
    "family","adaptation",
    "rag_used","retriever","top_k","usage",
    "views","image_resolution","visual_tokens",
    "metrics","primary_metric",
    "train_images","train_reports","epochs","batch_size","optimizer","learning_rate","hardware",
]

# -----------------------------
# LLM helpers
# -----------------------------
def load_local_llm(model_name: str, max_new_tokens: int = 512, temperature: float = 0.0):
    if not HF_AVAILABLE:
        die("Transformers not available.")
    print("Loading local model…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature == 0 else True,
        temperature=temperature,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tok, **gen_kwargs)
    return pipe

def llm_json(pipe, prompt: str, debug: bool = False) -> Dict[str, Any]:
    raw = ""
    if USE_OPENAI and not isinstance(pipe, (TextGenerationPipeline, type(None))):
        client = openai
        resp = client.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"system","content":"Return ONLY JSON."},{"role":"user","content": prompt}],
            temperature=0, max_tokens=800,
        )
        raw = resp["choices"][0]["message"]["content"]
    else:
        out = pipe(prompt)[0]["generated_text"]
        raw = out[len(prompt):].strip() if out.startswith(prompt) else out
    if debug:
        print("\n--- LLM RAW ---\n" + raw + "\n--------------\n")
    m = re.search(r"\{.*\}", raw, flags=re.S)
    text = m.group(0) if m else raw
    text = text.strip().rstrip("```").lstrip("```json").lstrip("```")
    try:
        return json.loads(text)
    except Exception:
        return {"error":"json_parse_error","raw": raw}

# -----------------------------
# Retrieval
# -----------------------------
def get_context_for_prompt(retriever, doc_id: str, query_hint: str, k: int = 10) -> str:
    query = f"{doc_id} {query_hint}"
    docs: List[Document] = []
    try:
        docs = retriever.invoke(query)  # preferred
    except Exception:
        try:
            docs = retriever.get_relevant_documents(query)  # legacy
        except Exception:
            docs = []
    seen = set()
    chunks = []
    for d in docs[:k]:
        txt = d.page_content.strip()
        if txt and txt not in seen:
            chunks.append(txt)
            seen.add(txt)
    return "\n---\n".join(chunks[:k]) if chunks else ""

# -----------------------------
# Regex baseline (unchanged)
# -----------------------------
DATASET_PAT = re.compile(r"\b(MIMIC[- ]?CXR|IU ?X[- ]?Ray|Open[- ]?i|Open[- ]?I|CheXpert|RSNA|PadChest|VinDr|BRAX|NIH-?ChestXray)\b", re.I)
MOD_PAT = re.compile(r"\b(CXR|X[- ]?Ray|CT|MRI|Ultrasound|US)\b", re.I)
ENC_PAT = re.compile(r"\b(ViT(?:-[BL]/?\d+)?|CLIP[- ]?ViT[- ]?[BL]/?\d+|Swin[- ]?T|ResNet(?:\d+)?|DeiT|DenseNet\d{3})\b", re.I)
DEC_PAT = re.compile(r"\b(LLaMA[- ]?\d?|GPT[- ]?2|T5|BART|Transformer|LSTM)\b", re.I)
FUS_PAT = re.compile(r"\b(cross[- ]?attention|co[- ]?attention|Q[- ]?Former|concatenation|early|late|gated)\b", re.I)
MET_PAT = re.compile(r"\b(CIDEr|BLEU(?:[- ]?4)?|METEOR|ROUGE[- ]?L|BERTScore|CheXbert|RadGraph)\b", re.I)

def regex_extract(all_text: str) -> Dict[str, Any]:
    datasets = sorted(set(m.group(1).strip() for m in DATASET_PAT.finditer(all_text)))
    mods = [m.group(1) for m in MOD_PAT.finditer(all_text)]
    encs = [m.group(1) for m in ENC_PAT.finditer(all_text)]
    decs = [m.group(1) for m in DEC_PAT.finditer(all_text)]
    fus = [m.group(1) for m in FUS_PAT.finditer(all_text)]
    mets = [m.group(1) for m in MET_PAT.finditer(all_text)]

    def any_or_not(x): return list(sorted(set(x))) if x else ["Not reported"]

    return {
        "modality": canon_modality(mods[0] if mods else "Not reported"),
        "datasets_train": datasets if datasets else ["Not reported"],
        "datasets_eval": ["Not reported"],
        "paired_reports": "Not reported",
        "is_vlm": True if (encs and decs) else False,
        "model_name": "Not reported",
        "model_class": "Not reported",
        "task": "report-generation" if "report" in all_text.lower() else "Not reported",
        "vision_encoder": any_or_not(encs)[0],
        "pretrained_from": "Not reported",
        "language_decoder": any_or_not(decs)[0],
        "decoder_size_params": "Not reported",
        "fusion": any_or_not(fus)[0],
        "where": "Not reported",
        "notes": "Not reported",
        "objectives": ["Not reported"],
        "family": "Not reported",
        "adaptation": "Not reported",
        "rag_used": True if ("retrieval" in all_text.lower() or "rag" in all_text.lower()) else False,
        "retriever": "Not reported",
        "top_k": "Not reported",
        "usage": "Not reported",
        "views": "Not reported",
        "image_resolution": "Not reported",
        "visual_tokens": "Not reported",
        "metrics": any_or_not(mets),
        "primary_metric": mets[0] if mets else "Not reported",
        "train_images": "Not reported",
        "train_reports": "Not reported",
        "epochs": "Not reported",
        "batch_size": "Not reported",
        "optimizer": "Not reported",
        "learning_rate": "Not reported",
        "hardware": "Not reported",
    }

# -----------------------------
# Build prompts / merge
# -----------------------------
def build_prompt(section_key: str, context: str) -> str:
    p = PROMPTS[section_key]
    return (
        f"{PREAMBLE}\n\n"
        f"Schema:\n{p['schema']}\n\n"
        f"Instruction:\n{p['instruction']}\n\n"
        f"Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
        f"Return ONLY JSON."
    )

def merge_results(acc: Dict[str, Any], section_key: str, obj: Dict[str, Any]):
    if section_key == "datasets_modalities":
        acc["modality"] = canon_modality(obj.get("modality"))
        acc["datasets_train"] = to_list(obj.get("datasets_train"))
        acc["datasets_eval"] = to_list(obj.get("datasets_eval"))
        acc["paired_reports"] = first_scalar(obj.get("paired_reports"))
    elif section_key == "vlm_presence":
        acc["is_vlm"] = to_bool(obj.get("is_vlm"), default=False)
        acc["model_name"] = first_scalar(obj.get("model_name"))
        acc["model_class"] = first_scalar(obj.get("model_class"))
        acc["task"] = first_scalar(obj.get("task"))
    elif section_key == "vision_encoder":
        acc["vision_encoder"] = first_scalar(obj.get("vision_encoder"))
        acc["pretrained_from"] = first_scalar(obj.get("pretrained_from"))
    elif section_key == "language_decoder":
        acc["language_decoder"] = first_scalar(obj.get("language_decoder"))
        acc["decoder_size_params"] = first_scalar(obj.get("decoder_size_params"))
    elif section_key == "fusion_alignment":
        acc["fusion"] = first_scalar(obj.get("fusion"))
        acc["where"] = first_scalar(obj.get("where"))
        acc["notes"] = first_scalar(obj.get("notes"))
    elif section_key == "pretraining_objectives":
        acc["objectives"] = to_list(obj.get("objectives"))
    elif section_key == "vlm_family_or_adaptation":
        acc["family"] = first_scalar(obj.get("family"))
        acc["adaptation"] = first_scalar(obj.get("adaptation"))
    elif section_key == "retrieval_rag":
        acc["rag_used"] = to_bool(obj.get("rag_used"), default=False)
        acc["retriever"] = first_scalar(obj.get("retriever"))
        acc["top_k"] = first_scalar(obj.get("top_k"))
        acc["usage"] = first_scalar(obj.get("usage"))
    elif section_key == "inputs_and_layout":
        acc["views"] = first_scalar(obj.get("views"))
        acc["image_resolution"] = first_scalar(obj.get("image_resolution"))
        acc["visual_tokens"] = first_scalar(obj.get("visual_tokens"))
    elif section_key == "metrics_key_for_vlm":
        acc["metrics"] = to_list(obj.get("metrics"))
        acc["primary_metric"] = first_scalar(obj.get("primary_metric"))
    elif section_key == "training_and_compute":
        acc["train_images"] = first_scalar(obj.get("train_images"))
        acc["train_reports"] = first_scalar(obj.get("train_reports"))
        acc["epochs"] = first_scalar(obj.get("epochs"))
        acc["batch_size"] = first_scalar(obj.get("batch_size"))
        acc["optimizer"] = first_scalar(obj.get("optimizer"))
        acc["learning_rate"] = first_scalar(obj.get("learning_rate"))
        acc["hardware"] = first_scalar(obj.get("hardware"))

def default_record(doc_id: str) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "modality":"Not reported","datasets_train":["Not reported"],"datasets_eval":["Not reported"],"paired_reports":"Not reported",
        "is_vlm":False,"model_name":"Not reported","model_class":"Not reported","task":"Not reported",
        "vision_encoder":"Not reported","pretrained_from":"Not reported",
        "language_decoder":"Not reported","decoder_size_params":"Not reported",
        "fusion":"Not reported","where":"Not reported","notes":"Not reported",
        "objectives":["Not reported"],
        "family":"Not reported","adaptation":"Not reported",
        "rag_used":False,"retriever":"Not reported","top_k":"Not reported","usage":"Not reported",
        "views":"Not reported","image_resolution":"Not reported","visual_tokens":"Not reported",
        "metrics":["Not reported"],"primary_metric":"Not reported",
        "train_images":"Not reported","train_reports":"Not reported","epochs":"Not reported","batch_size":"Not reported",
        "optimizer":"Not reported","learning_rate":"Not reported","hardware":"Not reported",
    }

# -----------------------------
# Post-validation against context
# -----------------------------
def post_validate(rec: Dict[str, Any], ctx_text: str) -> Dict[str, Any]:
    tl = ctx_text.lower()

    # Datasets: keep only those present in context (if any hit exists)
    if rec.get("datasets_train"):
        hits = []
        for ds in to_list(rec["datasets_train"]):
            dsl = ds.lower()
            if any(tok in tl for tok in DATASET_TOKENS if tok in dsl or dsl in tok or tok in tl):
                hits.append(ds)
        if hits:
            rec["datasets_train"] = sorted(set(hits))
        else:
            rec["datasets_train"] = ["Not reported"]

    if rec.get("datasets_eval"):
        hits = []
        for ds in to_list(rec["datasets_eval"]):
            dsl = ds.lower()
            if any(tok in tl for tok in DATASET_TOKENS if tok in dsl or dsl in tok or tok in tl):
                hits.append(ds)
        rec["datasets_eval"] = sorted(set(hits)) if hits else ["Not reported"]

    # Vision encoder
    ve = first_scalar(rec.get("vision_encoder"))
    if ve != "Not reported":
        if not contains_any(tl, ENCODER_TOKENS) or ve.lower().split("-")[0] not in "vitclipswinresnetdeitdensenetefficientnetdino":
            rec["vision_encoder"] = "Not reported"

    # Language decoder
    ld = first_scalar(rec.get("language_decoder"))
    if ld != "Not reported":
        if not contains_any(tl, DECODER_TOKENS):
            rec["language_decoder"] = "Not reported"

    # Fusion
    fu = first_scalar(rec.get("fusion"))
    if fu != "Not reported":
        if not contains_any(tl, FUSION_TOKENS):
            rec["fusion"] = "Not reported"

    # Objectives (filter to present ones)
    objs = [o.lower() for o in to_list(rec.get("objectives")) if o != "Not reported"]
    if objs:
        keep = []
        for o in objs:
            if "itc" in o or "contrastive" in o:
                if contains_any(tl, ["itc","contrastive"]): keep.append("ITC(contrastive)")
            if "itm" in o and contains_any(tl, ["itm","image-text matching"]): keep.append("ITM")
            if ("caption" in o or "cross-entropy" in o or "nll" in o) and contains_any(tl, ["caption","cross-entropy","nll"]):
                keep.append("captioning CE/NLL")
            if ("rl" in o or "cider" in o or "scst" in o) and contains_any(tl, ["rl","cider","scst"]):
                keep.append("RL(CIDEr/SCST)")
            if "coverage" in o and "coverage" in tl: keep.append("coverage")
            if "align" in o and "align" in tl: keep.append("alignment loss")
        rec["objectives"] = sorted(set(keep)) if keep else ["Not reported"]

    # Metrics (filter to present ones)
    mets = [m.lower() for m in to_list(rec.get("metrics")) if m != "Not reported"]
    if mets:
        keep = []
        if "cider" in tl: keep.append("CIDEr")
        if "bleu" in tl: keep.append("BLEU-4")
        if "meteor" in tl: keep.append("METEOR")
        if "rouge" in tl: keep.append("ROUGE-L")
        if "bertscore" in tl: keep.append("BERTScore")
        if "chexbert" in tl: keep.append("CheXbert-F1")
        if "radgraph" in tl: keep.append("RadGraph-F1")
        rec["metrics"] = keep if keep else ["Not reported"]
        if rec["primary_metric"] not in rec["metrics"]:
            rec["primary_metric"] = "Not reported"

    # Family
    fam = first_scalar(rec.get("family")).lower()
    if fam != "not reported":
        if not contains_any(tl, FAMILY_TOKENS):
            rec["family"] = "Not reported"

    # RAG used (override by presence)
    rec["rag_used"] = True if contains_any(tl, RAG_TOKENS) else False

    # Paired reports
    if contains_any(tl, PAIRED_TOKENS_TRUE):
        rec["paired_reports"] = "true"
    elif contains_any(tl, PAIRED_TOKENS_FALSE):
        rec["paired_reports"] = "false"
    elif rec.get("paired_reports","").lower() not in {"true","false"}:
        rec["paired_reports"] = "Not reported"

    # Is VLM? require at least one family token OR both encoder+decoder present
    is_vlm_rule = contains_any(tl, FAMILY_TOKENS) or (
        contains_any(tl, ENCODER_TOKENS) and contains_any(tl, DECODER_TOKENS)
    )
    rec["is_vlm"] = bool(is_vlm_rule)

    # Modality canonicalization
    rec["modality"] = canon_modality(rec.get("modality"))

    return rec

# -----------------------------
# Writers
# -----------------------------
def write_csv(out_csv: Path, rows: List[Dict[str, Any]]):
    import csv
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow([
                r.get("doc_id",""),
                r.get("modality","Not reported"),
                "; ".join(r.get("datasets_train",["Not reported"])) if isinstance(r.get("datasets_train"), list) else r.get("datasets_train","Not reported"),
                "; ".join(r.get("datasets_eval",["Not reported"])) if isinstance(r.get("datasets_eval"), list) else r.get("datasets_eval","Not reported"),
                r.get("paired_reports","Not reported"),
                r.get("is_vlm",False),
                r.get("model_name","Not reported"),
                r.get("model_class","Not reported"),
                r.get("task","Not reported"),
                r.get("vision_encoder","Not reported"),
                r.get("pretrained_from","Not reported"),
                r.get("language_decoder","Not reported"),
                r.get("decoder_size_params","Not reported"),
                r.get("fusion","Not reported"),
                r.get("where","Not reported"),
                r.get("notes","Not reported"),
                "; ".join(r.get("objectives",["Not reported"])) if isinstance(r.get("objectives"), list) else r.get("objectives","Not reported"),
                r.get("family","Not reported"),
                r.get("adaptation","Not reported"),
                r.get("rag_used",False),
                r.get("retriever","Not reported"),
                r.get("top_k","Not reported"),
                r.get("usage","Not reported"),
                r.get("views","Not reported"),
                r.get("image_resolution","Not reported"),
                r.get("visual_tokens","Not reported"),
                "; ".join(r.get("metrics",["Not reported"])) if isinstance(r.get("metrics"), list) else r.get("metrics","Not reported"),
                r.get("primary_metric","Not reported"),
                r.get("train_images","Not reported"),
                r.get("train_reports","Not reported"),
                r.get("epochs","Not reported"),
                r.get("batch_size","Not reported"),
                r.get("optimizer","Not reported"),
                r.get("learning_rate","Not reported"),
                r.get("hardware","Not reported"),
            ])

def write_md(out_md: Path, rows: List[Dict[str, Any]]):
    headers = [
        "File","Modality","Datasets (train)","Datasets (eval)","Paired",
        "VLM?","Model","Class","Task",
        "Vision Enc","Lang Dec","Fusion",
        "Objectives","Family","RAG",
        "Metrics(primary)"
    ]
    with out_md.open("w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"]*len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join([
                r.get("doc_id",""),
                r.get("modality","Not reported"),
                ", ".join(r.get("datasets_train",["Not reported"])) if isinstance(r.get("datasets_train"), list) else r.get("datasets_train","Not reported"),
                ", ".join(r.get("datasets_eval",["Not reported"])) if isinstance(r.get("datasets_eval"), list) else r.get("datasets_eval","Not reported"),
                str(r.get("paired_reports","Not reported")),
                "Yes" if r.get("is_vlm",False) else "No",
                r.get("model_name","Not reported"),
                r.get("model_class","Not reported"),
                r.get("task","Not reported"),
                r.get("vision_encoder","Not reported"),
                r.get("language_decoder","Not reported"),
                r.get("fusion","Not reported"),
                ", ".join(r.get("objectives",["Not reported"])) if isinstance(r.get("objectives"), list) else r.get("objectives","Not reported"),
                r.get("family","Not reported"),
                "Yes" if r.get("rag_used",False) else "No",
                f"{', '.join(r.get('metrics',['Not reported'])) if isinstance(r.get('metrics'), list) else r.get('metrics','Not reported')} "
                f"({r.get('primary_metric','Not reported')})",
            ]) + " |\n")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", type=str, required=True)
    ap.add_argument("--out-md", type=str, default="filled_papers_vlm_v10.md")
    ap.add_argument("--out-csv", type=str, default="filled_papers_vlm_v10.csv")

    # Retrieval / RAG
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-mmr", action="store_true")

    # Backend
    ap.add_argument("--backend", choices=["regex","llm"], default="regex")
    ap.add_argument("--local-model", type=str, default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--debug-prompts", action="store_true")

    args = ap.parse_args()
    snippets_dir = Path(args.snippets_dir)
    if not snippets_dir.exists():
        die(f"Snippets dir not found: {snippets_dir}")

    paper_ids = sorted({p.stem for p in snippets_dir.glob("*.txt")})
    if not paper_ids:
        die(f"No .txt files in {snippets_dir}")

    vectordb, _ = load_vectordb()
    retriever = make_retriever(vectordb, k=args.k, mmr=(not args.no_mmr))

    pipe = None
    if args.backend == "llm":
        if args.local_model:
            pipe = load_local_llm(args.local_model, max_new_tokens=900, temperature=args.temperature)
        elif USE_OPENAI:
            pipe = "openai"
        else:
            die("LLM backend requested but neither --local-model nor OPENAI_API_KEY set.")

    rows: List[Dict[str, Any]] = []
    print(f"▶ Extracting {len(paper_ids)} papers…", flush=True)

    for doc_id in paper_ids:
        rec = default_record(doc_id)
        doc_ctx_parts: List[str] = []

        if args.backend == "regex":
            text = read_snippet_text(snippets_dir, doc_id)
            rec.update(regex_extract(text))
            doc_ctx_parts.append(text)
        else:
            for section_key, meta in PROMPTS.items():
                ctx = get_context_for_prompt(retriever, doc_id, meta["query_hint"], k=args.k)
                if not ctx:
                    continue
                doc_ctx_parts.append(ctx)
                prompt = build_prompt(section_key, ctx)
                if args.debug_prompts:
                    print(f"\n--- PROMPT [{doc_id} :: {section_key}] ---\n{prompt}\n-------------------------------\n")
                ans = llm_json(pipe, prompt, debug=args.debug_prompts)
                if isinstance(ans, dict) and "error" in ans:
                    continue
                merge_results(rec, section_key, ans)

            # Normalize simple types before validation
            rec["modality"] = canon_modality(rec.get("modality"))
            rec["datasets_train"] = to_list(rec.get("datasets_train"))
            rec["datasets_eval"] = to_list(rec.get("datasets_eval"))
            rec["metrics"] = to_list(rec.get("metrics"))
            rec["objectives"] = to_list(rec.get("objectives"))
            rec["is_vlm"] = to_bool(rec.get("is_vlm"), default=False)
            rec["rag_used"] = to_bool(rec.get("rag_used"), default=False)

            # Context-validated corrections
            full_ctx = "\n".join(doc_ctx_parts)
            rec = post_validate(rec, full_ctx)

        print(f"✓ {doc_id}", flush=True)
        rows.append(rec)

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    write_md(out_md, rows)
    write_csv(out_csv, rows)
    print(f"✅ Wrote {out_md.name} and {out_csv.name}")

if __name__ == "__main__":
    main()
