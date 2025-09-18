#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v11 — LLM extractor with:
  • MMR retrieval (diversity-aware) and per-paper filtering via doc_id
  • Fallback to full snippet file when context is too short
  • Deterministic generation params (temperature=0, do_sample=False)
  • Robust JSON parsing and normalization
  • Debug prints to inspect retrieval + context

Usage example:
  unset OPENAI_API_KEY
  CUDA_VISIBLE_DEVICES=0 python rag_extract_langchain_v11.py \
    --snippets-dir snippets \
    --backend llm \
    --local-model meta-llama/Meta-Llama-3-8B-Instruct \
    --k 12 \
    --max-new-tokens 512 \
    --out-md  filled_papers_vlm_v11_llm.md \
    --out-csv filled_papers_vlm_v11_llm.csv \
    --debug
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keep these imports to match your current environment (deprecation warnings OK)
from langchain.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings  # noqa: E402
from langchain.vectorstores import Chroma  # noqa: E402

import pandas as pd  # make sure pandas is installed in your venv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# -----------------------------
# Constants / Defaults
# -----------------------------

DB_DIR = "chroma_db"
EMBED_MODEL = "all-mpnet-base-v2"

# Columns we will output (sheet + markdown)
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

# Single prompt (one-shot per paper) to extract all fields.
# Keep it strict about JSON and "Not reported" only when truly missing in the given CONTEXT.
EXTRACTION_PROMPT = """You are extracting a structured summary for a systematic literature review of **vision–language models for radiology report generation**.

Given the CONTEXT, fill **all** of the following fields. If a field is **truly absent** in the context, use exactly "Not reported". Otherwise extract the closest phrase you find (even if partial or approximate). Be conservative and do not guess beyond the given context.

Required JSON fields (use these exact keys):
- "modality": one of {X-Ray, CT, MRI, Ultrasound, Mixed, Not reported}
- "datasets_train": list of dataset names used for training (e.g., ["MIMIC-CXR", "IU X-ray"]) or "Not reported"
- "datasets_eval": list of dataset names used for eval (same style) or "Not reported"
- "paired": true/false if image–report pairs are explicitly used; or "Not reported"
- "vlm": "Yes" if vision-language or multimodal learning is present; "No" otherwise; or "Not reported"
- "model_name": model name or "Not reported"
- "arch_class": one of {encoder-decoder, single-stream, two-tower, Not reported}
- "task": short phrase, e.g., "report-generation" or "image-to-text generation" or "Not reported"
- "vision_encoder": backbone name(s) (e.g., "ViT-B/16", "CLIP", "ResNet50") or "Not reported"
- "language_decoder": decoder/backbone (e.g., "LLaMA", "BERT", "GPT-2", "Transformer", "LSTM") or "Not reported"
- "fusion": how modalities are fused (e.g., "cross-attention", "concatenation", "ITM/ITC pretrain", "cross-modal info fusion"), or "Not reported"
- "objectives": list of training objectives (from this set when applicable:
   ITC(contrastive), ITM, captioning CE/NLL, RL(CIDEr/SCST), alignment loss, coverage, other) or "Not reported"
- "family": if clearly based on a known family (e.g., ALBEF, BLIP, CLIP, other) else "Not reported"
- "rag": "Yes" if retrieval-augmented generation is used; "No" otherwise; or "Not reported"
- "metrics": primary metrics reported (e.g., "BLEU", "ROUGE", "CIDEr", "F1") or "Not reported"

Rules:
- Prefer concise values.
- Only output **valid JSON** object with those exact keys. No extra commentary.
- If unsure but there is a closest phrase in context, extract it rather than "Not reported".

CONTEXT:
========
{context}
========

Return only the JSON object.
"""


# -----------------------------
# Helpers
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snippets-dir", type=str, required=True, help="Directory of *.txt snippet files (one per paper).")
    p.add_argument("--backend", choices=["llm"], default="llm", help="Extraction backend. (v11 focuses on llm)")
    p.add_argument("--local-model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF model id.")
    p.add_argument("--k", type=int, default=12, help="# of chunks to keep per paper after MMR.")
    p.add_argument("--fetch-k", type=int, default=None, help="Candidate pool for MMR (default: 3*k).")
    p.add_argument("--lambda-mult", type=float, default=0.5, help="MMR diversity/relevance trade-off.")
    p.add_argument("--max-new-tokens", type=int, default=512, help="LLM generation cap.")
    p.add_argument("--min-context-chars", type=int, default=1200, help="Fallback threshold to full snippet.")
    p.add_argument("--out-md", type=str, default="filled_papers_vlm_v11_llm.md", help="Markdown output path.")
    p.add_argument("--out-csv", type=str, default="filled_papers_vlm_v11_llm.csv", help="CSV output path.")
    p.add_argument("--debug", action="store_true", help="Print retrieval/context debug info.")
    return p.parse_args()


def load_vectorstore() -> Chroma:
    embed = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embed,
        collection_name="papers",
    )
    return vectordb


def list_papers(snippets_dir: Path) -> List[str]:
    stems = sorted([p.stem for p in snippets_dir.glob("*.txt")])
    return stems


def load_snippet_text(snippets_dir: Path, paper_id: str) -> str:
    path = snippets_dir / f"{paper_id}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def load_local_llm(model_name: str, max_new_tokens: int):
    """Load a local HF causal LLM with accelerate device_map; deterministic decoding."""
    print("Loading local model…", flush=True)
    # dtype heuristic
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",           # let accelerate place weights on the visible GPU(s)
    )

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",           # DO NOT pass a `device` int here (collides with accelerate)
        torch_dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        return_full_text=False,
    )
    # A tiny sanity print for clarity
    try:
        print(f"Device map: {model.hf_device_map}", flush=True)
    except Exception:
        pass
    return gen_pipe


def json_from_text(txt: str) -> Dict[str, Any]:
    """Extract first JSON object from a text block robustly."""
    # Strip code fences if any
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.MULTILINE)
    # Find first {...} block
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    blob = txt[start:end+1]
    return json.loads(blob)


def to_list_or_not_report(v: Any) -> Any:
    if v is None:
        return "Not reported"
    if isinstance(v, str):
        # some users might write a CSV in a string; split lightly if commas exist
        s = v.strip()
        if not s:
            return "Not reported"
        if "," in s:
            parts = [x.strip() for x in s.split(",") if x.strip()]
            return parts if parts else "Not reported"
        return s
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, bool):
        return v
    return str(v)


def norm_str_list(x: Any) -> str:
    """Turn a list or scalar into the table-friendly string."""
    if x is None:
        return "Not reported"
    if isinstance(x, list):
        if not x:
            return "Not reported"
        return ", ".join(str(e) for e in x)
    if isinstance(x, bool):
        # Only Paired should be bool -> we'll convert later
        return "true" if x else "false"
    s = str(x).strip()
    return s if s else "Not reported"


def canon_modality(s: Any) -> str:
    s = norm_str_list(s).lower()
    if s == "not reported":
        return "Not reported"
    if "x-ray" in s or "cxr" in s or "chest x" in s:
        return "X-Ray"
    if "ct" in s:
        return "CT"
    if "mri" in s:
        return "MRI"
    if "ultrasound" in s or "sono" in s:
        return "Ultrasound"
    if "mixed" in s:
        return "Mixed"
    return "Not reported"


def yes_no_or_not(s: Any) -> str:
    s = norm_str_list(s).lower()
    if s in ["yes", "y", "true"]:
        return "Yes"
    if s in ["no", "n", "false"]:
        return "No"
    return "Not reported"


def paired_bool_or_not(x: Any) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    s = norm_str_list(x).lower()
    if s in ["true", "yes"]:
        return "true"
    if s in ["false", "no"]:
        return "false"
    return "Not reported"


def make_retriever(vectordb: Chroma, paper_id: str, k: int, fetch_k: Optional[int], lambda_mult: float):
    """MMR retriever with per-paper filter on doc_id."""
    if fetch_k is None:
        fetch_k = k * 3
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "filter": {"doc_id": paper_id},
        },
    )
    return retriever


def extract_for_paper_llm(
    paper_id: str,
    vectordb: Chroma,
    gen_pipe,
    snippets_dir: Path,
    k: int,
    fetch_k: Optional[int],
    lambda_mult: float,
    min_context_chars: int,
    debug: bool = False,
) -> Dict[str, str]:
    """Run retrieval + LLM extraction for one paper, return dict with final normalized fields."""
    query = f"Extract metadata for paper '{paper_id}' related to radiology report generation VLMs."
    retriever = make_retriever(vectordb, paper_id, k, fetch_k, lambda_mult)

    # Deprecation: get_relevant_documents -> invoke; keep old for compatibility
    try:
        docs = retriever.get_relevant_documents(query)  # type: ignore[attr-defined]
    except Exception:
        docs = retriever.invoke(query)  # newer LC

    context = "\n\n".join(d.page_content for d in docs)
    if debug:
        print(f"[DEBUG] paper={paper_id} | retrieved={len(docs)} | context_chars={len(context)}", flush=True)

    if len(context) < min_context_chars:
        # Fallback to the entire snippet file if retrieval was too thin
        full_txt = load_snippet_text(snippets_dir, paper_id)
        if full_txt:
            if debug:
                print(f"[DEBUG] paper={paper_id} | fallback to full snippet (chars={len(full_txt)})", flush=True)
            context = full_txt

    if debug:
        prev = (context[:400] + " …") if len(context) > 400 else context
        print("[DEBUG] context preview:\n" + prev.replace("\n", " ") + "\n", flush=True)

    prompt = EXTRACTION_PROMPT.format(context=context)
    out = gen_pipe(prompt)[0]["generated_text"]

    try:
        data = json_from_text(out)
    except Exception:
        # Very defensive fallback: no JSON extracted
        data = {}

    # Normalize fields into final strings for table
    modality = canon_modality(data.get("modality", "Not reported"))
    ds_train = norm_str_list(to_list_or_not_report(data.get("datasets_train", "Not reported")))
    ds_eval  = norm_str_list(to_list_or_not_report(data.get("datasets_eval", "Not reported")))
    paired   = paired_bool_or_not(data.get("paired", "Not reported"))
    vlm      = yes_no_or_not(data.get("vlm", "Not reported"))
    model    = norm_str_list(data.get("model_name", "Not reported"))
    aclass   = norm_str_list(data.get("arch_class", "Not reported"))
    task     = norm_str_list(data.get("task", "Not reported"))
    venc     = norm_str_list(data.get("vision_encoder", "Not reported"))
    ldec     = norm_str_list(data.get("language_decoder", "Not reported"))
    fusion   = norm_str_list(data.get("fusion", "Not reported"))
    objs     = norm_str_list(to_list_or_not_report(data.get("objectives", "Not reported")))
    family   = norm_str_list(data.get("family", "Not reported"))
    rag      = yes_no_or_not(data.get("rag", "Not reported"))
    metrics  = norm_str_list(to_list_or_not_report(data.get("metrics", "Not reported")))

    return {
        "File": paper_id,
        "Modality": modality,
        "Datasets (train)": ds_train,
        "Datasets (eval)": ds_eval,
        "Paired": paired,
        "VLM?": vlm,
        "Model": model,
        "Class": aclass,
        "Task": task,
        "Vision Enc": venc,
        "Lang Dec": ldec,
        "Fusion": fusion,
        "Objectives": objs,
        "Family": family,
        "RAG": rag,
        "Metrics(primary)": metrics,
    }


def write_outputs(rows: List[Dict[str, str]], out_md: Path, out_csv: Path) -> None:
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # Write CSV
    df.to_csv(out_csv, index=False)

    # Write Markdown table
    md_lines = []
    header = "| " + " | ".join(OUTPUT_COLUMNS) + " |"
    sep = "|---" * len(OUTPUT_COLUMNS) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for _, r in df.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in OUTPUT_COLUMNS) + " |")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ Wrote {out_md.name} and {out_csv.name}")


def main():
    args = parse_args()

    snippets_dir = Path(args.snippets_dir)
    assert snippets_dir.exists(), f"snippets dir not found: {snippets_dir}"

    papers = list_papers(snippets_dir)
    if not papers:
        print(f"No *.txt snippets found in {snippets_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"▶ Extracting {len(papers)} papers…", flush=True)

    vectordb = load_vectorstore()

    if args.backend != "llm":
        raise ValueError("v11 only supports --backend llm")

    gen_pipe = load_local_llm(args.local_model, max_new_tokens=args.max_new_tokens)

    rows: List[Dict[str, str]] = []
    for pid in papers:
        try:
            rec = extract_for_paper_llm(
                paper_id=pid,
                vectordb=vectordb,
                gen_pipe=gen_pipe,
                snippets_dir=snippets_dir,
                k=args.k,
                fetch_k=args.fetch_k,
                lambda_mult=args.lambda_mult,
                min_context_chars=args.min_context_chars,
                debug=args.debug,
            )
            rows.append(rec)
            print(f"✓ {pid}", flush=True)
        except Exception as e:
            print(f"✗ {pid}  (error: {e})", file=sys.stderr)

    write_outputs(rows, Path(args.out_md), Path(args.out_csv))


if __name__ == "__main__":
    main()
