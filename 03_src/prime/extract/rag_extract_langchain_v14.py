#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_extract_langchain_v14.py

PRISMA-style metadata extractor for VLM papers.

What’s new vs v13:
- Retrieval: MMR with `doc_id` filter (better on-paper coverage), then similarity fallback.
- Context: Always appends full snippet after retrieval (recall booster), capped by CTX_CHAR_LIMIT.
- Backfills: Restores v12-style backfill for datasets & modality (and pragmatic enc/dec backfill).
- Gating: `--strict-evidence` remains optional (off by default); gating logic tightened but pragmatic.
- JSON: Robust balanced-JSON extraction with retries; falls back to regex if LLM JSON fails.
- Outputs: CSV + Markdown; generous defaults (k=12, retries=2, max_new_tokens=512).

Quick usage:

    # (index only if snippets changed)
    # python rag_build_index.py

    # Strict pass (precision)
    unset OPENAI_API_KEY
    CUDA_VISIBLE_DEVICES=0 python rag_extract_langchain_v14.py \
      --snippets-dir snippets \
      --backend llm \
      --local-model meta-llama/Meta-Llama-3-8B-Instruct \
      --k 12 --max-new-tokens 512 --retries 2 \
      --strict-evidence \
      --out-md  filled_papers_vlm_v14_llm_strict.md \
      --out-csv filled_papers_vlm_v14_llm_strict.csv \
      --debug

    # Lenient pass (recall)
    unset OPENAI_API_KEY
    CUDA_VISIBLE_DEVICES=0 python rag_extract_langchain_v14.py \
      --snippets-dir snippets \
      --backend llm \
      --local-model meta-llama/Meta-Llama-3-8B-Instruct \
      --k 20 --max-new-tokens 768 --retries 3 \
      --out-md  filled_papers_vlm_v14_llm_lenient.md \
      --out-csv filled_papers_vlm_v14_llm_lenient.csv \
      --debug
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- LangChain / Chroma (legacy imports to match your environment) ---
# If you upgrade later, swap to langchain_community.* equivalents.
from langchain.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings  # type: ignore
from langchain.vectorstores import Chroma  # type: ignore
from langchain.schema import Document  # type: ignore

# --- Transformers for local LLM ---
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # type: ignore


# -----------------------
# Constants & Canonicals
# -----------------------

EMBED_MODEL = "all-mpnet-base-v2"
DB_DIR = "chroma_db"
CTX_CHAR_LIMIT = 18000  # cap total context length

COLUMNS = [
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

# Dataset (evidence) patterns
DATASET_EVIDENCE = {
    "MIMIC-CXR": r"\bMIMIC[- ]?CXR\b",
    "IU X-ray": r"\bIU[- ]?X[- ]?ray\b|\bIndiana University Chest X[- ]?rays?\b",
    "NIH ChestX-ray14": r"\bChestX[- ]?ray14\b|\bNIH\b",
    "Open-i": r"\bOpen[- ]?i\b|\bOpen[- ]?I\b",
    "REFLACX": r"\bREFLACX\b",
    "CXR-RePaiR": r"\bCXR[- ]?RePaiR\b",
    "CheXpert": r"\bCheXpert\b",
    "PadChest": r"\bPadChest\b",
}

# Encoders / decoders aliases used for backfill
VISION_ENCODER_ALIASES: Dict[str, str] = {
    r"\bViT\b|\bVision Transformer\b": "ViT",
    r"\bCLIP\b": "CLIP",
    r"\bDeiT\b": "DeiT",
    r"\bSwin\b": "Swin",
    r"\bResNet(?:-?50)?\b": "ResNet50",
    r"\bConvNeXt\b": "ConvNeXt",
    r"\bCNN\b|convolutional neural network": "CNN",
}
LANG_DECODER_ALIASES: Dict[str, str] = {
    r"\bLLaMA\b": "LLaMA",
    r"\bBERT\b": "BERT",
    r"\bGPT-?2\b": "GPT-2",
    r"\bT5\b": "T5",
    r"\bLSTM\b": "LSTM",
    r"\bGRU\b": "GRU",
    r"\bTransformer\b": "Transformer",
}

FAMILY_TOKENS = {"ALBEF", "BLIP", "BLIP-2", "LLaVA", "Flamingo", "Other", "Transformer", "CLIP"}

MODEL_CANON = {
    r"\btrmrg\b|Transformer Medical report generator": "TrMRG",
    r"\beggca[- ]?net\b": "EGGCA-Net",
    r"\bcxr[- ]?irgen\b": "CXR-IRGen",
    r"\balbef\b": "ALBEF",
}

EVIDENCE_PATTERNS = {
    # Vision encoders
    "ViT": r"\bViT\b|Vision Transformer",
    "CLIP": r"\bCLIP\b",
    "DeiT": r"\bDeiT\b",
    "Swin": r"\bSwin\b",
    "ResNet50": r"\bResNet(?:-?50)?\b",
    "CNN": r"\bCNN\b|convolutional neural network",

    # Language decoders / models
    "LLaMA": r"\bLLaMA\b",
    "BERT": r"\bBERT\b",
    "GPT-2": r"\bGPT-?2\b",
    "T5": r"\bT5\b",
    "LSTM": r"\bLSTM\b",
    "GRU": r"\bGRU\b",
    "Transformer": r"\bTransformer\b",

    # Fusion / class / task
    "cross-attention": r"\bcross[- ]?attention\b",
    "co-attention": r"\bco[- ]?attention\b",
    "concatenation": r"\bconcatenation|\bconcat\b",
    "two-stream": r"\b(two[- ]?stream|dual[- ]?encoder)\b",
    "single-stream": r"\bsingle[- ]?stream\b",
    "RAG": r"\bRAG\b|\bretrieval[- ]?augmented\b",
    "report-generation": r"\breport[- ]?generation\b",
    "image-text matching": r"\bimage[- ]?text matching\b",
    "image-report generation": r"\bimage[- ]?report\b",
}

METRIC_TOKENS = [
    "BLEU", "ROUGE", "CIDEr", "METEOR", "BERTScore", "SPICE", "F1", "Accuracy",
    "RadCliQ", "CheXbert", "RadGraph", "GLEU",
]

OBJECTIVE_TOKENS = [
    "ITC(contrastive)", "ITM", "captioning CE/NLL", "RL(CIDEr/SCST)", "alignment loss", "coverage",
]

BOOL_TRUE = {"true", "yes", "y", "1"}
BOOL_FALSE = {"false", "no", "n", "0"}

MODALITY_CANON = {
    "x-ray": "X-Ray", "xray": "X-Ray", "x ray": "X-Ray",
    "ct": "CT", "mri": "MRI", "ultrasound": "Ultrasound", "mixed": "Mixed",
    "radiology": "X-Ray", "not reported": "Not reported",
}


# -----------------------
# Utils
# -----------------------

def println(s: str) -> None:
    print(s)
    sys.stdout.flush()


def debugln(s: str, debug: bool) -> None:
    if debug:
        print(s)
        sys.stdout.flush()


def list_papers(snippets_dir: Path) -> List[str]:
    exts = (".txt", ".md")
    names = set()
    for p in snippets_dir.glob("*"):
        if p.suffix.lower() in exts:
            names.add(p.stem)
        elif p.name.endswith(".pdf.txt"):
            names.add(p.name[:-8])  # strip ".pdf.txt"
    return sorted(names)


def read_full_snippet(snippets_dir: Path, base: str) -> Optional[str]:
    candidates = [
        snippets_dir / f"{base}.txt",
        snippets_dir / f"{base}.md",
        snippets_dir / f"{base}.pdf.txt",
    ]
    for c in candidates:
        if c.exists():
            try:
                return c.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return None


def canon_bool(x: Any) -> str:
    if isinstance(x, bool):
        return "Yes" if x else "No"
    if x is None:
        return "Not reported"
    s = str(x).strip().lower()
    if s in BOOL_TRUE:
        return "Yes"
    if s in BOOL_FALSE:
        return "No"
    return "Not reported"


def canon_modality(x: Any) -> str:
    if x is None:
        return "Not reported"
    if isinstance(x, list):
        if not x:
            return "Not reported"
        s = str(x[0]).strip().lower()
    else:
        s = str(x).strip().lower()
    return MODALITY_CANON.get(s, MODALITY_CANON.get(s.replace("-", " "), "Not reported"))


def canon_list(x: Any) -> str:
    if x is None:
        return "Not reported"
    if isinstance(x, list):
        vals = [str(v).strip() for v in x if str(v).strip()]
        return ", ".join(vals) if vals else "Not reported"
    s = str(x).strip()
    return s if s else "Not reported"


def canon_family(s: Any) -> str:
    if s is None:
        return "Not reported"
    val = str(s).strip()
    if not val:
        return "Not reported"
    for tok in FAMILY_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", val, flags=re.I):
            return tok
    return val


def canon_model(raw: Any, context: str) -> str:
    if raw is None:
        return "Not reported"
    s = str(raw).strip()
    if not s:
        return "Not reported"
    for pat, canon in MODEL_CANON.items():
        if re.search(pat, s, flags=re.I) or re.search(pat, context, flags=re.I):
            return canon
    return s


def extract_tokens_from_text(text: str, token_list: List[str]) -> List[str]:
    found = []
    for t in token_list:
        if re.search(rf"\b{re.escape(t)}\b", text, flags=re.I):
            found.append(t)
    return sorted(set(found))


def _present(value: str, ctx: str, patterns_map: dict) -> bool:
    if value == "Not reported":
        return False
    pat = patterns_map.get(value)
    if not pat:
        return re.search(re.escape(value), ctx, flags=re.I) is not None
    return re.search(pat, ctx, flags=re.I) is not None


# -----------------------
# Retrieval / LLM
# -----------------------

def build_vectorstore():
    embed = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embed, collection_name="papers")


def retrieve_context_mmr(vectordb, base: str, snippets_dir: Path, k: int, debug: bool) -> str:
    """MMR + filter by doc_id; then append full snippet, capped to CTX_CHAR_LIMIT."""
    query = base.replace("_", " ")
    texts, total = [], 0

    # 1) Try MMR with metadata filter
    try:
        docs = vectordb.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=max(40, 4 * k),
            lambda_mult=0.5,
            filter={"doc_id": base},
        )
    except Exception:
        docs = []

    # 2) If none, fall back to plain similarity with filter
    if not docs:
        try:
            docs = vectordb.similarity_search(query=query, k=k, filter={"doc_id": base})
        except Exception:
            docs = []

    # 3) Collect retrieved chunks (respect char cap)
    for d in docs or []:
        t = (d.page_content or "").strip()
        if not t:
            continue
        if total + len(t) > CTX_CHAR_LIMIT:
            break
        texts.append(t)
        total += len(t)

    # 4) Always append the full snippet (recall booster)
    full_text = read_full_snippet(snippets_dir, base)
    if full_text:
        remain = CTX_CHAR_LIMIT - total
        if remain > 0:
            texts.append(full_text[:remain])
            total += min(remain, len(full_text))

    if debug:
        print(f"[DEBUG] paper={base} | retrieved={len(docs or [])} | context_chars={total}")
        if texts:
            preview = texts[0][:400].replace("\n", " ")
            print(f"[DEBUG] context preview:\n{preview}\n")
    return "\n\n".join(texts)


def load_local_llm(model_name: str, max_new_tokens: int = 512):
    println("Loading local model…")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    gen_pipe = pipeline(
        task="text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    dev = getattr(mdl, "hf_device_map", None)
    if dev:
        println(f"Device map: {dev}")
    return gen_pipe


LLM_SYSTEM_INSTRUCTIONS = (
    "You are an expert annotator extracting exact metadata from the provided context. "
    "Return ONLY a single JSON object with the exact keys and types requested. "
    "Do not include explanations or extra text."
)

LLM_JSON_SCHEMA = {
    "file": "string",
    "modality": "string or list (X-Ray, CT, MRI, Ultrasound, Mixed, Not reported)",
    "datasets_train": "string or list (e.g., MIMIC-CXR, IU X-ray, NIH ChestX-ray14, Open-i, REFLACX, CXR-RePaiR, CheXpert, PadChest)",
    "datasets_eval": "string or list",
    "paired": "boolean or 'Not reported'",
    "vlm": "boolean or 'Not reported'",
    "model": "string (short model name, e.g., TrMRG, EGGCA-Net, CXR-IRGen, ALBEF, Not reported)",
    "class": "string (e.g., encoder-decoder, single-stream, two-stream, Not reported)",
    "task": "string (e.g., report-generation, image-text matching, image-report generation, Not reported)",
    "vision_encoder": "string or list (e.g., ViT, CLIP, ResNet50, CNN, Not reported)",
    "language_decoder": "string or list (e.g., LLaMA, BERT, GPT-2, LSTM, Transformer, Not reported)",
    "fusion": "string or list (e.g., cross-attention, co-attention, concatenation, Not reported)",
    "objectives": "list (subset of: ITC(contrastive), ITM, captioning CE/NLL, RL(CIDEr/SCST), alignment loss, coverage) or 'Not reported'",
    "family": "string (e.g., ALBEF, BLIP, BLIP-2, LLaVA, Flamingo, Transformer, Other, Not reported)",
    "rag": "boolean or 'Not reported'",
    "metrics": "list (subset of: BLEU, ROUGE, CIDEr, METEOR, BERTScore, SPICE, F1, Accuracy, RadCliQ, CheXbert, RadGraph, GLEU) or 'Not reported'",
}

def make_llm_prompt(context: str, filename: str) -> str:
    schema_str = json.dumps(LLM_JSON_SCHEMA, indent=2)
    return (
        f"{LLM_SYSTEM_INSTRUCTIONS}\n\n"
        f"FILENAME: {filename}\n\n"
        "CONTEXT (verbatim, do not ignore):\n"
        "---------------------------------\n"
        f"{context}\n"
        "---------------------------------\n\n"
        "Extract ONLY what is explicitly supported by the context. If unknown, use exactly 'Not reported'.\n"
        "Respond with ONE JSON object (no prose, no markdown), with keys exactly:\n"
        f"{schema_str}\n\n"
        "Rules:\n"
        "- Return JSON only, no backticks, no prefix/suffix text.\n"
        "- If a field allows a list, you MAY return a JSON array; otherwise return a string/boolean.\n"
        "- 'paired' and 'rag' should be booleans when clear, else 'Not reported'.\n"
        "- Prefer short names for 'model' (e.g., TrMRG, EGGCA-Net, CXR-IRGen, ALBEF).\n"
        "- Do NOT invent values not supported by the context.\n"
    )


def extract_json_block(s: str) -> Optional[str]:
    """Extract the first balanced JSON object from arbitrary text."""
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.I | re.M)
    start_pos = s.find("{")
    if start_pos == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start_pos:], start=start_pos):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start_pos:i + 1]
    return None


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    block = extract_json_block(raw_text)
    if not block:
        raise ValueError("No JSON object found in LLM output.")
    try:
        return json.loads(block)
    except Exception as e:
        raise ValueError(f"Invalid JSON from LLM: {e}")


def llm_generate(gen_pipe, prompt: str) -> str:
    out = gen_pipe(prompt, max_new_tokens=None)
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return out[0]["generated_text"]
    if isinstance(out, list) and out and "text" in out[0]:
        return out[0]["text"]
    return str(out)


# -----------------------
# Record building / gating
# -----------------------

def _findall(text: str, patterns: Dict[str, str]) -> List[str]:
    found: List[str] = []
    for pat, canon in patterns.items():
        if re.search(pat, text, flags=re.I):
            found.append(canon)
    # keep unique order
    uniq: List[str] = []
    for f in found:
        if f not in uniq:
            uniq.append(f)
    return uniq


def ensure_record(
    base: str,
    data: Dict[str, Any],
    context: str,
    strict: bool = False
) -> Dict[str, str]:
    # Raw fields
    modality = data.get("modality")
    datasets_train = data.get("datasets_train")
    datasets_eval = data.get("datasets_eval")
    paired = data.get("paired")
    vlm = data.get("vlm")
    model_name = data.get("model")
    arch_class = data.get("class")
    task = data.get("task")
    vision_enc = data.get("vision_encoder")
    lang_dec = data.get("language_decoder")
    fusion = data.get("fusion")
    objectives = data.get("objectives")
    family = data.get("family")
    rag = data.get("rag")
    metrics = data.get("metrics")

    # Canonicals
    modality = canon_modality(modality)
    datasets_train = canon_list(datasets_train)
    datasets_eval = canon_list(datasets_eval)
    paired = canon_bool(paired)
    vlm = canon_bool(vlm)
    rag = canon_bool(rag)
    model_name = canon_model(model_name, context)
    arch_class = "Not reported" if arch_class is None else str(arch_class).strip() or "Not reported"
    task = "Not reported" if task is None else str(task).strip() or "Not reported"
    vision_enc = "Not reported" if vision_enc is None else (", ".join(v.strip() for v in vision_enc) if isinstance(vision_enc, list) else str(vision_enc).strip() or "Not reported")
    lang_dec = "Not reported" if lang_dec is None else (", ".join(v.strip() for v in lang_dec) if isinstance(lang_dec, list) else str(lang_dec).strip() or "Not reported")
    fusion = "Not reported" if fusion is None else (", ".join(v.strip() for v in fusion) if isinstance(fusion, list) else str(fusion).strip() or "Not reported")

    # Auto-mine metrics/objectives from context if missing
    if (not objectives) or (isinstance(objectives, str) and objectives.strip().lower() == "not reported"):
        objectives = extract_tokens_from_text(context, OBJECTIVE_TOKENS)
    if (not metrics) or (isinstance(metrics, str) and metrics.strip().lower() == "not reported"):
        metrics = extract_tokens_from_text(context, METRIC_TOKENS)
    objectives = canon_list(objectives)
    metrics = canon_list(metrics)
    family = canon_family(family)

    # ---- v12-style pragmatic backfills ----

    # datasets from context if missing
    def _datasets_from_context(ctx: str) -> str:
        hits = []
        for name, pat in DATASET_EVIDENCE.items():
            if re.search(pat, ctx, flags=re.I):
                hits.append(name)
        hits = sorted(set(hits))
        return ", ".join(hits) if hits else "Not reported"

    if datasets_train == "Not reported" or not datasets_train.strip():
        datasets_train = _datasets_from_context(context)
    if datasets_eval == "Not reported" or not datasets_eval.strip():
        datasets_eval = datasets_train

    # modality from context & datasets if missing
    if modality == "Not reported":
        cx = context.lower()
        if re.search(r"\b(chest )?x[- ]?ray|cxr\b", cx):
            modality = "X-Ray"
        elif re.search(r"\bmri\b", cx):
            modality = "MRI"
        elif re.search(r"\bct\b", cx):
            modality = "CT"
        elif re.search(r"\bultrasound\b", cx):
            modality = "Ultrasound"
        elif any(ds in datasets_train for ds in ["IU X-ray", "MIMIC-CXR", "NIH ChestX-ray14", "Open-i", "REFLACX", "CXR-RePaiR", "CheXpert", "PadChest"]):
            modality = "X-Ray"

    # enc/dec backfill from context tokens if missing
    def _canon_from_aliases(text: str, alias_map: Dict[str, str]) -> List[str]:
        hits = []
        for pat, canon in alias_map.items():
            if re.search(pat, text, flags=re.I):
                hits.append(canon)
        # unique ordered
        uniq = []
        for h in hits:
            if h not in uniq:
                uniq.append(h)
        return uniq

    if vision_enc == "Not reported":
        enc_from_ctx = _canon_from_aliases(context, VISION_ENCODER_ALIASES)
        if enc_from_ctx:
            vision_enc = ", ".join(enc_from_ctx)
    if lang_dec == "Not reported":
        dec_from_ctx = _canon_from_aliases(context, LANG_DECODER_ALIASES)
        if dec_from_ctx:
            lang_dec = ", ".join(dec_from_ctx)

    # paired/vlm/rag heuristic nudges if still unknown
    if paired == "Not reported":
        paired = "Yes" if re.search(r"\bimage[- ]?report pairs?\b|\bpaired\b", context, flags=re.I) else "Not reported"
    if vlm == "Not reported":
        if (vision_enc != "Not reported" and lang_dec != "Not reported") \
            or family in {"ALBEF", "BLIP", "BLIP-2", "LLaVA", "Flamingo", "CLIP"} \
            or re.search(r"\bvision[- ]?language\b|\bVLM\b", context, flags=re.I):
            vlm = "Yes"
    if rag == "Not reported":
        rag = "Yes" if re.search(r"\bretrieval[- ]?augmented\b|\bRAG\b|\bretriev", context, flags=re.I) else "Not reported"

    # ---- strict evidence gating (optional) ----
    if strict:
        if vision_enc != "Not reported":
            vs = [s.strip() for s in vision_enc.split(",")]
            vision_enc = ", ".join([v for v in vs if _present(v, context, EVIDENCE_PATTERNS)]) or "Not reported"
        if lang_dec != "Not reported":
            ls = [s.strip() for s in lang_dec.split(",")]
            lang_dec = ", ".join([v for v in ls if _present(v, context, EVIDENCE_PATTERNS)]) or "Not reported"

        if family != "Not reported" and not _present(family, context, EVIDENCE_PATTERNS):
            family = "Not reported"

        if model_name != "Not reported" and not re.search(re.escape(model_name), context, flags=re.I):
            keep = False
            for pat, canon in MODEL_CANON.items():
                if canon == model_name and re.search(pat, context, flags=re.I):
                    keep = True
                    break
            if not keep:
                model_name = "Not reported"

        if fusion != "Not reported":
            fs = [s.strip() for s in fusion.split(",")]
            fusion = ", ".join([v for v in fs if _present(v, context, EVIDENCE_PATTERNS)]) or "Not reported"
        if arch_class != "Not reported" and not _present(arch_class, context, EVIDENCE_PATTERNS):
            arch_class = "Not reported"
        if task != "Not reported" and not _present(task, context, EVIDENCE_PATTERNS):
            task = "Not reported"

        if rag == "Yes" and not _present("RAG", context, EVIDENCE_PATTERNS):
            rag = "Not reported"

        def _gate_datasets(ds: str) -> str:
            if ds == "Not reported":
                return ds
            kept = []
            for token in [t.strip() for t in ds.split(",")]:
                if not token:
                    continue
                pat = DATASET_EVIDENCE.get(token, re.escape(token))
                if re.search(pat, context, flags=re.I):
                    kept.append(token)
            return ", ".join(kept) if kept else "Not reported"

        datasets_train = _gate_datasets(datasets_train)
        datasets_eval = _gate_datasets(datasets_eval)

    rec = {
        "File": base,
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
        "Family": family,
        "RAG": rag,
        "Metrics(primary)": metrics,
    }
    return rec


# -----------------------
# Backends
# -----------------------

def run_regex_backend(base: str, context: str, strict: bool, debug: bool) -> Dict[str, str]:
    # Minimal regex-based extraction with token mining
    modality = "X-Ray" if re.search(r"\b(chest\s*)?x[- ]?ray\b", context, flags=re.I) else "Not reported"

    # datasets
    ds_hits = []
    for name, pat in DATASET_EVIDENCE.items():
        if re.search(pat, context, flags=re.I):
            ds_hits.append(name)
    ds_hits = sorted(set(ds_hits))
    datasets = ", ".join(ds_hits) if ds_hits else "Not reported"

    objectives = extract_tokens_from_text(context, OBJECTIVE_TOKENS)
    metrics = extract_tokens_from_text(context, METRIC_TOKENS)

    data = {
        "file": base,
        "modality": modality,
        "datasets_train": datasets,
        "datasets_eval": datasets,
        "paired": "Not reported",
        "vlm": "Not reported",
        "model": "Not reported",
        "class": "Not reported",
        "task": "report-generation" if re.search(r"report[- ]?generation", context, flags=re.I) else "Not reported",
        "vision_encoder": "ViT" if re.search(r"\bViT\b|Vision Transformer", context, flags=re.I) else "Not reported",
        "language_decoder": "LLaMA" if re.search(r"\bLLaMA\b", context, flags=re.I) else "Not reported",
        "fusion": "cross-attention" if re.search(r"cross[- ]?attention", context, flags=re.I) else "Not reported",
        "objectives": objectives or "Not reported",
        "family": "ALBEF" if re.search(r"\bALBEF\b", context, flags=re.I) else "Not reported",
        "rag": "Yes" if re.search(r"\bRAG\b|retrieval[- ]?augmented", context, flags=re.I) else "Not reported",
        "metrics": metrics or "Not reported",
    }
    return ensure_record(base, data, context, strict)


def run_llm_backend(
    base: str,
    context: str,
    gen_pipe,
    retries: int,
    debug: bool,
    strict: bool,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    prompt = make_llm_prompt(context, base)
    attempt = 0
    last_err = None
    while attempt <= retries:
        attempt += 1
        try:
            raw = llm_generate(gen_pipe, prompt)
            debugln(f"[DEBUG] raw LLM text for {base} (attempt {attempt}):\n{raw}\n", debug)
            data = parse_llm_json(raw)
            rec = ensure_record(base, data, context, strict)
            return rec, None
        except Exception as e:
            last_err = str(e)
            time.sleep(0.2)
    return None, last_err


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Extract PRISMA/VLM metadata from paper snippets (v14).")
    ap.add_argument("--snippets-dir", type=str, required=True, help="Directory with text snippets")
    ap.add_argument("--backend", choices=["llm", "regex"], default="llm")
    ap.add_argument("--local-model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF model name for LLM backend")
    ap.add_argument("--k", type=int, default=12, help="Top-k docs to retrieve from Chroma (MMR)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=2, help="LLM JSON parse retries")
    ap.add_argument("--out-md", type=str, required=True)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument("--strict-evidence", action="store_true", help="Keep only values that appear in the context text")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    snippets_dir = Path(args.snippets_dir)
    if not snippets_dir.exists():
        println(f"ERROR: snippets dir not found: {snippets_dir}")
        sys.exit(1)

    println("▶ Extracting papers…")

    # Open vectorstore
    vectordb = None
    try:
        vectordb = build_vectorstore()
    except Exception as e:
        debugln(f"[DEBUG] Failed to open vectorstore, will use full snippets only: {e}", args.debug)

    # Load LLM if needed
    gen_pipe = None
    if args.backend == "llm":
        gen_pipe = load_local_llm(args.local_model, max_new_tokens=args.max_new_tokens)

    # Papers
    bases = list_papers(snippets_dir)
    if not bases:
        println("No snippet files found. Make sure your snippets/*.txt|*.md exist.")
        sys.exit(1)

    records: List[Dict[str, str]] = []
    for base in bases:
        # Retrieval + always-append full snippet (capped)
        if vectordb:
            context = retrieve_context_mmr(vectordb, base, snippets_dir, args.k, args.debug)
        else:
            context = read_full_snippet(snippets_dir, base) or ""
            if args.debug and context:
                debugln(f"[DEBUG] {base}: fallback full snippet (chars={len(context)})", args.debug)

        # Backend selection
        if args.backend == "regex" or not gen_pipe:
            rec = run_regex_backend(base, context or "", args.strict_evidence, args.debug)
            records.append(rec)
            println(f"✓ {base}")
            continue

        # LLM backend with regex fallback on JSON failure
        rec, err = run_llm_backend(base, context or "", gen_pipe, retries=args.retries, debug=args.debug, strict=args.strict_evidence)
        if rec:
            records.append(rec)
            println(f"✓ {base}")
        else:
            debugln(f"[DEBUG] LLM JSON failed for {base}: {err}. Falling back to regex.", args.debug)
            rec = run_regex_backend(base, context or "", args.strict_evidence, args.debug)
            records.append(rec)
            println(f"✓ {base} (regex fallback)")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Write Markdown
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "---|" * len(COLUMNS) + "\n")
        for r in records:
            row = [r.get(c, "Not reported") for c in COLUMNS]
            f.write("| " + " | ".join(row) + " |\n")

    println(f"✅ Wrote {args.out_md} and {args.out_csv}")


if __name__ == "__main__":
    main()
