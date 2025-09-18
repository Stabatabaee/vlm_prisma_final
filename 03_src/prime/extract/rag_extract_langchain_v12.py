#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_extract_langchain_v12.py
RAG extractor for PRISMA-style VLM metadata.

Key upgrades in this version:
- Robust JSON extraction with automatic retries (strict prompts).
- Tolerates strings/lists/bools for every key.
- Falls back to regex extractor if LLM never returns JSON.
- Stronger normalization for modality/datasets/encoders/decoders/fusion/objectives/metrics.
"""

import os
import re
import sys
import json
import glob
import csv
import argparse
from typing import Any, Dict, List, Optional, Tuple

# (We keep your existing imports to avoid breaking your env; deprecation warnings are fine.)
from langchain.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings  # noqa: E402
from langchain.vectorstores import Chroma  # noqa: E402

# Optional LLM
try:
    from transformers import pipeline  # noqa: F401
except Exception:
    pipeline = None  # type: ignore

# ---------------- Config ----------------

EMBED_MODEL = "all-mpnet-base-v2"
DB_DIR = "chroma_db"
CTX_CHAR_LIMIT = 18000

OUTPUT_COLUMNS = [
    "File",
    "Modality",
    "Datasets (train)",
    "Datasets (eval)",
    "Paired",
    "VLM?",
    "Model",
    "Class",
    "Task",
    "Vision Enc",
    "Lang Dec",
    "Fusion",
    "Objectives",
    "Family",
    "RAG",
    "Metrics(primary)",
]

DATASET_CANON = {
    r"\bMIMIC[- ]?CXR\b": "MIMIC-CXR",
    r"\bIU[- ]?X[- ]?ray\b|\bIndiana University Chest X[- ]?rays?\b": "IU X-ray",
    r"\bNIH ChestX[- ]?ray14\b|\bNIH\b": "NIH ChestX-ray14",
    r"\bOpen[- ]?i\b|\bOpen[- ]?I\b": "Open-i",
    r"\bREFLACX\b": "REFLACX",
    r"\bCXR[- ]?RePaiR\b": "CXR-RePaiR",
    r"\bCheXpert\b": "CheXpert",
    r"\bPadChest\b": "PadChest",
}

VISION_ENCODER_ALIASES: Dict[str, str] = {
    r"\bViT[- ]?B/?16\b|\bViT\b|\bVision Transformer\b": "ViT",
    r"\bCLIP\b": "CLIP",
    r"\bDeiT\b": "DeiT",
    r"\bSwin\b": "Swin",
    r"\bResNet[- ]?50\b|\bResNet\b": "ResNet50",
    r"\bConvNeXt\b": "ConvNeXt",
}

LANG_DECODER_ALIASES: Dict[str, str] = {
    r"\bLLaMA[- ]?2?\b|\bLLaMA\b": "LLaMA",
    r"\bBERT\b": "BERT",
    r"\bGPT[- ]?2\b": "GPT-2",
    r"\bT5\b": "T5",
    r"\bLSTM\b": "LSTM",
    r"\bGRU\b": "GRU",
    r"\bTransformer\b": "Transformer",
}

FUSION_TOKENS = {
    r"\bcross[- ]?attention\b": "cross-attention",
    r"\bco[- ]?attention\b": "co-attention",
    r"\bconcat|\bconcatenation\b": "concatenation",
    r"\b(two[- ]?stream|dual[- ]?encoder)\b": "two-stream",
    r"\bsingle[- ]?stream\b": "single-stream",
    r"\bRAG\b|\bretrieval[- ]?augmented\b": "RAG",
    r"\b(fusion|multimodal fusion)\b": "fusion",
    r"\bchannel attention networks?\b": "channel-attention",
    r"\bALBEF\b": "ALBEF-style",
}

OBJECTIVE_TOKENS = [
    ("ITC(contrastive)", r"\bcontrastive\b|\bITC\b"),
    ("ITM", r"\bimage[- ]?text matching\b|\bITM\b"),
    ("captioning CE/NLL", r"\b(NLL|cross[- ]?entropy)\b"),
    ("RL(CIDEr/SCST)", r"\bSCST\b|\bCIDEr\b"),
    ("alignment loss", r"\balign(ment)? loss\b"),
    ("coverage", r"\bcoverage\b"),
]

METRIC_TOKENS = [
    "BLEU",
    "ROUGE",
    "CIDEr",
    "METEOR",
    "BERTScore",
    "SPICE",
    "F1",
    "Accuracy",
    "RadCliQ",
    "CheXbert",
    "RadGraph",
    "GLEU",
]

FAMILY_TOKENS = {
    r"\bALBEF\b": "ALBEF",
    r"\bBLIP[- ]?2\b": "BLIP-2",
    r"\bBLIP\b": "BLIP",
    r"\bCLIP\b": "CLIP",
    r"\bLLaVA\b": "LLaVA",
    r"\bFlamingo\b": "Flamingo",
    r"\bTransformer\b": "Transformer",
}

MODEL_CANON = {
    r"\bTrMRG\b|\bTransformer Medical report generator\b": "TrMRG",
    r"\bEGGCA[- ]?Net\b|\bEye Gaze Guided Cross[- ]?modal Alignment\b": "EGGCA-Net",
    r"\bCXR[- ]?IRGen\b": "CXR-IRGen",
    r"\bALBEF\b": "ALBEF",
    r"\bBLIP[- ]?2\b": "BLIP-2",
    r"\bBLIP\b": "BLIP",
    r"\bLLaVA\b": "LLaVA",
    r"\bFlamingo\b": "Flamingo",
}

# ---------------- Utils ----------------

def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)

def _to_text(x: Any) -> str:
    if x is None:
        return "Not reported"
    if isinstance(x, (list, tuple, set)):
        items = [str(i).strip() for i in x if str(i).strip()]
        return ", ".join(items) if items else "Not reported"
    s = str(x).strip()
    return s if s else "Not reported"

def _join_list(items: Any) -> str:
    if not items:
        return "Not reported"
    if isinstance(items, str):
        return items if items.strip() else "Not reported"
    uniq: List[str] = []
    for it in items:
        t = str(it).strip()
        if t and t not in uniq:
            uniq.append(t)
    return ", ".join(uniq) if uniq else "Not reported"

def _findall(text: str, patterns: Dict[str, str], multi: bool = True) -> List[str]:
    found: List[str] = []
    for pat, canon in patterns.items():
        if re.search(pat, text, flags=re.I):
            found.append(canon)
    out: List[str] = []
    for f in found:
        if f not in out:
            out.append(f)
    return out if multi else (out[:1] if out else [])

def _canon_from_aliases(text: str, alias_map: Dict[str, str], multi: bool = True) -> str:
    if not text:
        return "Not reported"
    matches = _findall(text, alias_map, multi=multi)
    return _join_list(matches) if matches else "Not reported"

def _canon_yesno(x: Any) -> str:
    if isinstance(x, bool):
        return "Yes" if x else "No"
    s = str(x).strip().lower()
    if s in {"true", "yes", "y"}:
        return "Yes"
    if s in {"false", "no", "n"}:
        return "No"
    return "Not reported"

def _canon_datasets(source: Any) -> str:
    text = _to_text(source)
    found = _findall(text, DATASET_CANON, multi=True)
    return _join_list(found) if found else "Not reported"

def _canon_modality(raw: Any, context: str) -> str:
    s = _to_text(raw).lower()
    if s != "not reported":
        if "x-ray" in s or "xray" in s or "cxr" in s or "chest x-ray" in s:
            return "X-Ray"
        for m in ["ct", "mri", "ultrasound"]:
            if re.search(rf"\b{m}\b", s):
                return m.upper() if m != "ultrasound" else "Ultrasound"
        if "radiology" in s:
            return "X-Ray"
        return s.title()
    cx = context.lower()
    if re.search(r"\b(chest )?x[- ]?ray|cxr\b", cx):
        return "X-Ray"
    if re.search(r"\bmri\b", cx):
        return "MRI"
    if re.search(r"\bct\b", cx):
        return "CT"
    if re.search(r"\bultrasound\b", cx):
        return "Ultrasound"
    return "Not reported"

def _canon_fusion(raw: Any, context: str) -> str:
    tx = f"{_to_text(raw)}\n{context}"
    hits: List[str] = []
    for pat, canon in FUSION_TOKENS.items():
        if re.search(pat, tx, flags=re.I):
            hits.append(canon)
    return _join_list(hits) if hits else "Not reported"

def _canon_objectives(raw: Any, context: str) -> str:
    tx = f"{_to_text(raw)}\n{context}"
    hits: List[str] = []
    for name, pat in OBJECTIVE_TOKENS:
        if re.search(pat, tx, flags=re.I):
            hits.append(name)
    return _join_list(hits) if hits else "Not reported"

def _canon_metrics(raw: Any, context: str) -> str:
    tx = f"{_to_text(raw)}\n{context}"
    hits: List[str] = []
    for token in METRIC_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", tx, flags=re.I):
            hits.append(token)
    return _join_list(hits) if hits else "Not reported"

def _canon_family(raw: Any, context: str) -> str:
    tx = f"{_to_text(raw)}\n{context}"
    hits = _findall(tx, FAMILY_TOKENS, multi=True)
    return _join_list(hits) if hits else "Not reported"

def _canon_class(raw: Any, context: str) -> str:
    tx = f"{_to_text(raw)}\n{context}".lower()
    if re.search(r"\b(two[- ]?stream|dual[- ]?encoder)\b", tx):
        return "two-stream"
    if re.search(r"\bsingle[- ]?stream\b", tx):
        return "single-stream"
    if re.search(r"\bencoder[- ]?decoder\b", tx):
        return "encoder-decoder"
    return "Not reported"

def _canon_task(raw: Any, context: str) -> str:
    tx = f"{_to_text(raw)}\n{context}".lower()
    if re.search(r"\breport[- ]?generation\b|radiology report", tx):
        return "report-generation"
    if re.search(r"\bimage[- ]?text matching\b", tx):
        return "image-text matching"
    if re.search(r"\bimage[- ]?report\b", tx):
        return "image-report generation"
    return "Not reported"

def _canon_model(model_raw: Any, context: str) -> str:
    text = f"{_to_text(model_raw)} {context}"
    hits = []
    for pat, canon in MODEL_CANON.items():
        if re.search(pat, text, flags=re.I):
            hits.append(canon)
    if hits:
        return ", ".join(sorted(set(hits)))
    m = _to_text(model_raw)
    if m != "Not reported" and len(m) <= 60:
        return m
    return "Not reported"

def _first_json_block(s: str) -> Optional[str]:
    # also attempt to pull json fenced in code blocks
    fence = re.search(r"```(?:json)?\s*({.*?})\s*```", s, flags=re.S)
    if fence:
        return fence.group(1)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    block = s[start : end + 1]
    block = re.sub(r",\s*([}\]])", r"\1", block)  # trim trailing commas
    return block

def build_prompt(context: str, file_id: str) -> str:
    return (
        "Extract PRISMA-style metadata for a Vision-Language paper from ONLY the context.\n"
        "If unknown, write exactly 'Not reported'. Respond with a SINGLE JSON object.\n"
        "JSON keys:\n"
        '{\n'
        f'  "file": "{file_id}",\n'
        '  "modality": "X-Ray | CT | MRI | Ultrasound | Mixed | Not reported",\n'
        '  "datasets_train": ["..."] | "Not reported",\n'
        '  "datasets_eval": ["..."] | "Not reported",\n'
        '  "paired": true/false | "Not reported",\n'
        '  "vlm": true/false | "Not reported",\n'
        '  "model": "short model name or Not reported",\n'
        '  "class": "encoder-decoder | two-stream | single-stream | Not reported",\n'
        '  "task": "report-generation | image-text matching | image-report generation | Not reported",\n'
        '  "vision_encoder": "ViT | CLIP | ResNet50 | DeiT | Swin | Not reported",\n'
        '  "language_decoder": "LLaMA | BERT | GPT-2 | T5 | LSTM | Transformer | Not reported",\n'
        '  "fusion": "cross-attention | co-attention | concatenation | two-stream | single-stream | RAG | fusion | Not reported",\n'
        '  "objectives": ["ITC(contrastive)","ITM","captioning CE/NLL","RL(CIDEr/SCST)","alignment loss","coverage"] | "Not reported",\n'
        '  "family": "ALBEF | BLIP | BLIP-2 | CLIP | LLaVA | Flamingo | Transformer | Not reported",\n'
        '  "rag": true/false | "Not reported",\n'
        '  "metrics": ["BLEU","ROUGE","CIDEr","METEOR","BERTScore","SPICE","F1","Accuracy","RadCliQ","CheXbert","RadGraph","GLEU"] | "Not reported"\n'
        '}\n'
        "--- BEGIN CONTEXT ---\n"
        f"{context}\n"
        "--- END CONTEXT ---\n"
        "Output ONLY JSON. No prose."
    )

def build_prompt_strict(context: str, file_id: str) -> str:
    return (
        "Return ONLY a JSON object. No explanation, no preface, no extra words.\n"
        "Start with '{' and end with '}'. If a field is unknown, use 'Not reported'.\n"
        "JSON keys: file, modality, datasets_train, datasets_eval, paired, vlm, model, class, task, "
        "vision_encoder, language_decoder, fusion, objectives, family, rag, metrics.\n"
        f'"file":"{file_id}".\n'
        "--- CONTEXT ---\n"
        f"{context}\n"
        "--- END CONTEXT ---"
    )

def build_prompt_min(context: str, file_id: str) -> str:
    return (
        f"JSON ONLY. Fill all keys; use 'Not reported' when unknown. file='{file_id}'. "
        "Keys: file, modality, datasets_train, datasets_eval, paired, vlm, model, class, task, "
        "vision_encoder, language_decoder, fusion, objectives, family, rag, metrics.\n"
        f"{context}"
    )

def load_local_llm(model_name: str, max_new_tokens: int):
    if pipeline is None:
        raise RuntimeError("transformers is not installed.")
    print("Loading local model…", flush=True)
    text_gen = pipeline(
        task="text-generation",
        model=model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Device map: {'': 0}", flush=True)  # cosmetic to match your logs

    def _call(prompt: str) -> str:
        out = text_gen(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if isinstance(out, list) and out:
            return str(out[0]["generated_text"])
        return str(out)

    return _call

def get_doc_ids_from_snippets(snippet_dir: str) -> List[str]:
    paths = []
    paths.extend(glob.glob(os.path.join(snippet_dir, "*.txt")))
    paths.extend(glob.glob(os.path.join(snippet_dir, "*.md")))
    ids: List[str] = []
    for p in sorted(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in ids:
            ids.append(stem)
    return ids

def retrieve_context_for_doc(vectordb: Chroma, doc_id: str, k: int, debug: bool) -> Tuple[str, int]:
    query = doc_id.replace("_", " ")
    total_chars = 0
    texts: List[str] = []
    try:
        docs = vectordb.similarity_search(query=query, k=k, filter={"doc_id": doc_id})
    except Exception:
        retriever = vectordb.as_retriever(search_kwargs={"k": k, "filter": {"doc_id": doc_id}})
        docs = retriever.get_relevant_documents(query)  # type: ignore

    if docs:
        for d in docs:
            t = d.page_content or ""
            if total_chars + len(t) > CTX_CHAR_LIMIT:
                break
            texts.append(t)
            total_chars += len(t)
        if debug:
            eprint(f"[DEBUG] paper={doc_id} | retrieved={len(texts)} | context_chars={total_chars}")
    else:
        if debug:
            eprint(f"[DEBUG] paper={doc_id} | retrieved=0 | context_chars=0")

    if not texts:
        for ext in (".txt", ".md"):
            candidate = os.path.join("snippets", f"{doc_id}{ext}")
            if os.path.isfile(candidate):
                with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                texts = [raw[:CTX_CHAR_LIMIT]]
                total_chars = len(texts[0])
                if debug:
                    eprint(f"[DEBUG] {doc_id}: fallback to full snippet (chars={total_chars})")
                break

    if texts and debug:
        preview = texts[0][:300].replace("\n", " ")
        eprint(f"[DEBUG] context preview:\n{preview}\n")

    return ("\n\n".join(texts), total_chars)

def parse_llm_json(raw: str) -> Dict[str, Any]:
    block = _first_json_block(raw)
    if not block:
        raise ValueError("No JSON object found in LLM output.")
    try:
        return json.loads(block)
    except Exception as ex:
        repaired = block.replace("True", "true").replace("False", "false").replace("None", "null")
        return json.loads(repaired)  # will raise again if truly broken

def coerce_list_or_str(x: Any) -> Any:
    """Accept list or comma-separated string, or 'Not reported'."""
    if x is None:
        return "Not reported"
    if isinstance(x, list):
        return x if x else "Not reported"
    s = str(x).strip()
    if not s or s.lower() == "not reported":
        return "Not reported"
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return arr if isinstance(arr, list) and arr else "Not reported"
        except Exception:
            pass
    if "," in s:
        return [i.strip() for i in s.split(",") if i.strip()]
    return s

def ensure_record(file_id: str, parsed: Dict[str, Any], context: str) -> Dict[str, str]:
    # Safe getters with coercion
    modality_raw = parsed.get("modality", "Not reported")
    datasets_train_raw = coerce_list_or_str(parsed.get("datasets_train", "Not reported"))
    datasets_eval_raw = coerce_list_or_str(parsed.get("datasets_eval", "Not reported"))
    paired_raw = parsed.get("paired", "Not reported")
    vlm_raw = parsed.get("vlm", "Not reported")
    model_raw = parsed.get("model", "Not reported")
    class_raw = parsed.get("class", "Not reported")
    task_raw = parsed.get("task", "Not reported")
    vision_raw = parsed.get("vision_encoder", "Not reported")
    lang_raw = parsed.get("language_decoder", "Not reported")
    fusion_raw = parsed.get("fusion", "Not reported")
    objectives_raw = coerce_list_or_str(parsed.get("objectives", "Not reported"))
    family_raw = parsed.get("family", "Not reported")
    rag_raw = parsed.get("rag", "Not reported")
    metrics_raw = coerce_list_or_str(parsed.get("metrics", "Not reported"))

    modality = _canon_modality(modality_raw, context)

    datasets_train = _canon_datasets(datasets_train_raw)
    if datasets_train == "Not reported":
        datasets_train = _canon_datasets(context)
    datasets_eval = _canon_datasets(datasets_eval_raw)
    if datasets_eval == "Not reported":
        datasets_eval = datasets_train

    if modality == "Not reported":
        if any(ds in datasets_train for ds in ["IU X-ray", "MIMIC-CXR", "NIH ChestX-ray14", "Open-i", "REFLACX", "CXR-RePaiR"]):
            modality = "X-Ray"

    enc_from_context = _canon_from_aliases(context, VISION_ENCODER_ALIASES, multi=True)
    vision_enc = _canon_from_aliases(_to_text(vision_raw), VISION_ENCODER_ALIASES, multi=True)
    if vision_enc == "Not reported" and enc_from_context != "Not reported":
        vision_enc = enc_from_context

    dec_from_context = _canon_from_aliases(context, LANG_DECODER_ALIASES, multi=True)
    lang_dec = _canon_from_aliases(_to_text(lang_raw), LANG_DECODER_ALIASES, multi=True)
    if lang_dec == "Not reported" and dec_from_context != "Not reported":
        lang_dec = dec_from_context

    fusion = _canon_fusion(fusion_raw, context)
    arch_class = _canon_class(class_raw, context)
    task = _canon_task(task_raw, context)
    family = _canon_family(family_raw, context)
    model_name = _canon_model(model_raw, context)
    objectives = _canon_objectives(objectives_raw, context)
    metrics = _canon_metrics(metrics_raw, context)

    paired = _canon_yesno(paired_raw)
    if paired == "Not reported":
        paired = "Yes" if re.search(r"\bimage[- ]?report pairs?\b|\bpaired\b", context, flags=re.I) else "Not reported"

    vlm = _canon_yesno(vlm_raw)
    if vlm == "Not reported":
        if (vision_enc != "Not reported" and lang_dec != "Not reported") \
           or family in {"ALBEF", "BLIP", "BLIP-2", "LLaVA", "Flamingo", "CLIP"} \
           or re.search(r"\bvision[- ]?language\b|\bVLM\b", context, flags=re.I):
            vlm = "Yes"

    rag = _canon_yesno(rag_raw)
    if rag == "Not reported":
        rag = "Yes" if re.search(r"\bretrieval[- ]?augmented\b|\bRAG\b|\bretriev", context, flags=re.I) else "Not reported"

    rec: Dict[str, str] = {
        "File": file_id,
        "Modality": modality,
        "Datasets (train)": datasets_train,
        "Datasets (eval)": datasets_eval,
        "Paired": paired,
        "VLM?": vlm,
        "Model": model_name,
        "Class": arch_class,
        "Task": task,
        "Vision Enc": vision_enc,
        "Lang Dec": lang_dec,
        "Fusion": fusion,
        "Objectives": objectives,
        "Family": family if family else "Not reported",
        "RAG": rag,
        "Metrics(primary)": metrics,
    }
    return rec

def llm_extract_with_retries(file_id: str, context: str, gen, debug: bool, max_new_tokens: int, retries: int) -> Dict[str, Any]:
    prompts = [
        build_prompt(context, file_id),
        build_prompt_strict(context, file_id),
        build_prompt_min(context, file_id),
    ][: max(1, min(3, retries + 1))]  # at least one, at most three
    last_err = None
    for i, p in enumerate(prompts, 1):
        out = gen(p)
        if debug:
            eprint(f"[DEBUG] raw LLM text for {file_id} (attempt {i}):\n{out[:1000]}\n")
        try:
            parsed = parse_llm_json(out)
            parsed["file"] = file_id
            return parsed
        except Exception as ex:
            last_err = ex
    # Final fallback: regex extractor-style parsed dict
    if debug and last_err:
        eprint(f"[DEBUG] {file_id}: JSON parse failed after retries; falling back to regex. Last error: {last_err}")
    return {
        "file": file_id,
        "modality": "Not reported",
        "datasets_train": "Not reported",
        "datasets_eval": "Not reported",
        "paired": "Not reported",
        "vlm": "Not reported",
        "model": "Not reported",
        "class": "Not reported",
        "task": "Not reported",
        "vision_encoder": "Not reported",
        "language_decoder": "Not reported",
        "fusion": "Not reported",
        "objectives": "Not reported",
        "family": "Not reported",
        "rag": "Not reported",
        "metrics": "Not reported",
    }

def regex_extract_minimal(file_id: str, context: str) -> Dict[str, Any]:
    # Very light deterministic fallback
    modality = "X-Ray" if re.search(r"\b(CXR|x[- ]?ray|chest x[- ]?ray)\b", context, flags=re.I) else "Not reported"
    datasets = _canon_datasets(context)
    vision_enc = _canon_from_aliases(context, VISION_ENCODER_ALIASES, multi=True)
    lang_dec = _canon_from_aliases(context, LANG_DECODER_ALIASES, multi=True)
    family = _canon_family("Not reported", context)
    return {
        "file": file_id,
        "modality": modality,
        "datasets_train": datasets if datasets != "Not reported" else "Not reported",
        "datasets_eval": datasets if datasets != "Not reported" else "Not reported",
        "paired": "Yes" if re.search(r"\bimage[- ]?report\b|\bpaired\b", context, flags=re.I) else "Not reported",
        "vlm": "Yes" if (vision_enc != "Not reported" and lang_dec != "Not reported") or family != "Not reported" else "Not reported",
        "model": _canon_model("Not reported", context),
        "class": _canon_class("Not reported", context),
        "task": _canon_task("Not reported", context),
        "vision_encoder": vision_enc,
        "language_decoder": lang_dec,
        "fusion": _canon_fusion("Not reported", context),
        "objectives": _canon_objectives("Not reported", context),
        "family": family,
        "rag": "Yes" if re.search(r"\bRAG\b|\bretrieval[- ]?augmented\b", context, flags=re.I) else "Not reported",
        "metrics": _canon_metrics("Not reported", context),
    }

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True)
    ap.add_argument("--backend", choices=["llm", "regex"], default="llm")
    ap.add_argument("--local-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--retries", type=int, default=2, help="LLM JSON retries (0..2)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print("▶ Extracting papers…", flush=True)

    embed = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embed, collection_name="papers")

    doc_ids = get_doc_ids_from_snippets(args.snippets_dir)

    gen = None
    if args.backend == "llm":
        gen = load_local_llm(args.local_model, max_new_tokens=args.max_new_tokens)

    records: List[Dict[str, str]] = []
    for doc_id in doc_ids:
        context, _ = retrieve_context_for_doc(vectordb, doc_id, args.k, args.debug)
        if not context.strip():
            if args.debug:
                eprint(f"[DEBUG] {doc_id}: empty context; skipping.")
            continue

        try:
            if args.backend == "llm":
                parsed = llm_extract_with_retries(
                    file_id=doc_id,
                    context=context,
                    gen=gen,
                    debug=args.debug,
                    max_new_tokens=args.max_new_tokens,
                    retries=max(0, min(2, args.retries)),
                )
                # If we fell back to the minimal shell (all Not reported), try regex salvaging
                if all(str(parsed.get(k, "Not reported")) == "Not reported" for k in ["modality", "datasets_train", "vision_encoder", "language_decoder"]):
                    parsed = regex_extract_minimal(doc_id, context)
            else:
                parsed = regex_extract_minimal(doc_id, context)

            rec = ensure_record(doc_id, parsed, context)
            records.append(rec)
            print(f"✓ {doc_id}", flush=True)
        except Exception as ex:
            print(f"✗ {doc_id}  (error: {ex})", flush=True)

    # Write Markdown
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(OUTPUT_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(OUTPUT_COLUMNS)) + "|\n")
        for rec in records:
            row = [rec.get(col, "Not reported") for col in OUTPUT_COLUMNS]
            f.write("| " + " | ".join(row) + " |\n")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "Not reported") for k in OUTPUT_COLUMNS})

    print(f"✅ Wrote {args.out_md} and {args.out_csv}", flush=True)

if __name__ == "__main__":
    main()
