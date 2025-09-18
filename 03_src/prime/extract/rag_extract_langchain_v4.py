#!/usr/bin/env python3
"""
rag_extract_langchain_v4.py  (fixed)

- Per-paper retrieval (filter by doc_id) to prevent cross-paper leakage
- Lexical sweep of the *entire* paper text before accepting "Not reported"
- Clamp model answers to things that actually appear in the retrieved context
- Writes both Markdown and CSV (defaults: filled_papers_llama_v4.*)
- Works with local LLaMA (HF) or GPT-4o via OPENAI_API_KEY
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

DB_DIR = "chroma_db"

COLUMNS = [
    "File", "Title", "Authors", "Modality", "Datasets",
    "Model Name", "Vision Encoder", "Language Decoder", "Fusion Strategy"
]

FIELD_QUERIES: Dict[str, str] = {
    "Title": "From the context, give the exact paper title as published.",
    "Authors": "From the context, list the author names in order as one short phrase (e.g., Doe J., Smith A.).",
    "Modality": "Which imaging modality does this study use (e.g., X-Ray / CXR, CT, MRI)? Return one short phrase.",
    "Datasets": "Which dataset names are used (e.g., MIMIC-CXR, IU X-ray, Open-i)? Return names only.",
    "Model Name": "What is the model name(s) used? Return a short name if present.",
    "Vision Encoder": "Which vision encoder architecture is used (e.g., ViT, ViT-B/16, CLIP, ResNet, CNN)?",
    "Language Decoder": "Which language/text decoder is used (e.g., LLaMA, BERT, GPT-2, LSTM, Transformer, T5)?",
    "Fusion Strategy": "How are image and text fused (e.g., Cross-Attention, Co-Attention, Multimodal Encoder, Early/Late Fusion, Concatenation)?",
}

ALLOWED = {
    "Modality": ["X-Ray", "CXR", "CT", "MRI", "Ultrasound", "X-ray"],
    "Vision Encoder": ["ViT", "ViT-B/16", "ResNet", "CLIP", "CNN", "DINO", "Swin", "EfficientNet"],
    "Language Decoder": ["LLaMA", "BERT", "GPT-2", "LSTM", "Transformer", "T5", "DistilGPT-2"],
    "Fusion Strategy": ["Cross-Attention", "Co-Attention", "Multimodal Encoder", "Early Fusion", "Late Fusion",
                        "Concatenation", "Gated Fusion", "Attention Fusion"],
}

DATASET_LIST = [
    "MIMIC-CXR", "MIMIC CXR", "IU X-ray", "Indiana University Chest X-rays", "Open-i", "Open-I",
    "CheXpert", "PadChest", "VinDr-CXR", "NIH ChestX-ray14", "COVIDx", "ChestX-ray14"
]
LEXICAL_TERMS = {
    "Vision Encoder": ["ViT-B/16", "ViT", "CLIP", "ResNet", "CNN", "DINO", "Swin", "EfficientNet"],
    "Language Decoder": ["LLaMA", "BERT", "GPT-2", "DistilGPT", "LSTM", "Transformer", "T5"],
    "Fusion Strategy": ["Cross-Attention", "Co-Attention", "Multimodal Encoder", "Early Fusion", "Late Fusion",
                        "Concatenation", "Gated Fusion", "Attention Fusion"]
}

CLAMP_FIELDS = ("Modality", "Vision Encoder", "Language Decoder", "Fusion Strategy")

INSTRUCTIONS = (
    "You are filling a PRISMA metadata table.\n"
    "Answer with a SINGLE SHORT PHRASE for the requested field using ONLY the provided context.\n"
    "If the field is not present in the context, answer exactly: Not reported.\n"
    "Do not include any code, examples, or explanations.\n"
)

# ---------------------- small utils ----------------------
def clean_one_phrase(text: str) -> str:
    if not text:
        return "Not reported"
    t = text.strip()

    # If the model ever echoed the prompt and we still have it, cut to content after 'Answer:'
    if "Answer:" in t:
        t = t.split("Answer:")[-1].strip()

    # Strip code fences and extra lines
    t = t.replace("```", " ").strip()
    # Keep the first non-empty line after trimming
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = lines[0] if lines else ""

    # Remove obvious prefixes
    t = re.sub(r"^(Answer|A|Title|Authors|Modality|Datasets|Model Name|Vision Encoder|Language Decoder|Fusion Strategy)\s*:\s*", "", t, flags=re.I)

    # Guard against instruction echoes / junk
    if not t or t.lower().startswith("you are ") or t.lower().startswith("from the context"):
        return "Not reported"

    # Collapse whitespace and strip quotes
    t = re.sub(r"\s+", " ", t).strip(" '\"\u200b\u200c\u200d")
    return t if t else "Not reported"

def clamp_from_context(field_name: str, context_text: str, raw_answer: str) -> str:
    ctx_lc = (context_text or "").lower()
    ans = (raw_answer or "").strip()
    if ans and ans.lower() in ctx_lc:
        return ans
    for cand in ALLOWED.get(field_name, []):
        if cand.lower() in ctx_lc:
            return cand
    return "Not reported"

def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def lexical_sweep(full_text: str, field: str) -> List[str]:
    if not full_text:
        return []
    hits = []
    terms = DATASET_LIST if field == "Datasets" else LEXICAL_TERMS.get(field, [])
    ft_lc = full_text.lower()
    for term in terms:
        if term.lower() in ft_lc:
            hits.append(term)
    return uniq_preserve(hits)

def window_around_hits(full_text: str, terms: List[str], radius: int = 400) -> str:
    if not full_text or not terms:
        return ""
    fl = full_text; fl_lc = fl.lower()
    spans = []
    for term in terms:
        start = 0; tl = term.lower()
        while True:
            idx = fl_lc.find(tl, start)
            if idx == -1: break
            s = max(0, idx - radius); e = min(len(fl), idx + len(term) + radius)
            spans.append(fl[s:e]); start = idx + len(term)
    boosted = "\n...\n".join(spans)
    return boosted[:6000] if len(boosted) > 6000 else boosted

def sanitize_cell(s: Optional[str]) -> str:
    if not s: return "Not reported"
    s = str(s).replace("|", "/").replace("`","")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "Not reported"

def row_to_markdown(cells: List[str]) -> str:
    cells = [sanitize_cell(x) for x in cells]
    if len(cells) < 9:
        cells += ["Not reported"] * (9 - len(cells))
    return "| " + " | ".join(cells) + " |"

# ---------------------- LLM loaders ----------------------
def load_openai_llm(model_name: str = "gpt-4o-mini"):
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    return ChatOpenAI(model=model_name, temperature=0)

def load_local_llm(local_model: str) -> Tuple[TextGenerationPipeline, AutoTokenizer]:
    print("Loading local model…")
    tok = AutoTokenizer.from_pretrained(local_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        offload_state_dict=True,
        offload_folder="offload",
    )
    pipe = TextGenerationPipeline(model=model, tokenizer=tok)
    return pipe, tok

def call_llm(llm, prompt: str, tok: Optional[AutoTokenizer]) -> str:
    """ChatOpenAI or HF pipeline -> one-line answer."""
    if hasattr(llm, "invoke"):  # OpenAI path
        out = llm.invoke(prompt).content
        return clean_one_phrase(out)

    # HF pipeline path: return only the completion (NOT the prompt)
    gen = llm(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        return_full_text=False,   # <<< key fix
    )
    text = gen[0]["generated_text"]
    # As extra safety, if 'Answer:' is present, slice after it
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    return clean_one_phrase(text)

# ---------------------- Retrieval helpers ----------------------
def get_retriever_for(vectordb: Chroma, pid: str, k: int):
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": {"doc_id": pid}},
    )

def build_context_for_field(vectordb: Chroma, pid: str, query: str, k: int = 8) -> Tuple[str, List[str]]:
    retriever = get_retriever_for(vectordb, pid, k=k)
    docs = retriever.get_relevant_documents(query)  # deprecation OK
    chunks = [d.page_content for d in docs if getattr(d, "page_content", None)]
    uniq, seen = [], set()
    for ch in chunks:
        key = ch[:200]
        if key not in seen:
            uniq.append(ch); seen.add(key)
    ctx = "\n\n---\n\n".join(uniq)
    return (ctx[:9000] if len(ctx) > 9000 else ctx), uniq

def read_full_text(snippets_dir: str, pid: str) -> str:
    p = Path(snippets_dir) / f"{pid}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")

# ---------------------- Core extraction ----------------------
def ask_field(llm, tok, field: str, pid: str, context_text: str) -> str:
    q = FIELD_QUERIES[field]
    prompt = (
        f"{INSTRUCTIONS}\n"
        f"Field: {field}\n\n"
        f"Context for paper {pid}.pdf:\n"
        f"\"\"\"\n{context_text}\n\"\"\"\n\n"
        "Answer:"
    )
    return call_llm(llm, prompt, tok)

def extract_row_for_paper(llm, tok, vectordb: Chroma, snippets_dir: str, pid: str) -> List[str]:
    field_contexts: Dict[str, str] = {}
    for field in FIELD_QUERIES.keys():
        ctx, _ = build_context_for_field(vectordb, pid, query=FIELD_QUERIES[field], k=8)
        field_contexts[field] = ctx

    full_text = read_full_text(snippets_dir, pid)

    answers: Dict[str, str] = {}
    for field in FIELD_QUERIES.keys():
        ans = ask_field(llm, tok, field, pid, field_contexts[field])

        if field in CLAMP_FIELDS:
            ans = clamp_from_context(field, field_contexts[field], ans)

        if ans == "Not reported":
            hits = lexical_sweep(full_text, field if field != "Modality" else "Modality")
            if field == "Modality" and not hits and full_text:
                if re.search(r"\bx[- ]?ray\b|\bcxr\b", full_text, flags=re.I): hits = ["X-Ray"]
                elif re.search(r"\bmri\b", full_text, flags=re.I): hits = ["MRI"]
                elif re.search(r"\bct\b", full_text, flags=re.I): hits = ["CT"]
            if hits:
                boosted = window_around_hits(full_text, hits, radius=450)
                retry = ask_field(llm, tok, field, pid, boosted)
                if field in CLAMP_FIELDS:
                    retry = clamp_from_context(field, boosted, retry)
                ans = retry

        if field == "Datasets":
            ds_hits = lexical_sweep(full_text, "Datasets")
            if ds_hits:
                ans = ", ".join(ds_hits)
            else:
                ctx_lc = field_contexts[field].lower()
                ctx_ds = [ds for ds in DATASET_LIST if ds.lower() in ctx_lc]
                ans = ", ".join(uniq_preserve(ctx_ds)) if ctx_ds else "Not reported"

        answers[field] = sanitize_cell(ans)

    cells = [
        f"{pid}.pdf",
        answers.get("Title", "Not reported"),
        answers.get("Authors", "Not reported"),
        answers.get("Modality", "Not reported"),
        answers.get("Datasets", "Not reported"),
        answers.get("Model Name", "Not reported"),
        answers.get("Vision Encoder", "Not reported"),
        answers.get("Language Decoder", "Not reported"),
        answers.get("Fusion Strategy", "Not reported"),
    ]
    return cells

# ---------------------- Main ----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True, help="Directory containing <paper>.txt snippet files")
    ap.add_argument("--local-model", default=os.environ.get("LOCAL_LLM", "meta-llama/Meta-Llama-3-8B-Instruct"),
                    help="HF model id for local inference (used if OPENAI_API_KEY is not set)")
    ap.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model if OPENAI_API_KEY is set")
    ap.add_argument("--out-md", default="filled_papers_llama_v4.md")
    ap.add_argument("--out-csv", default="filled_papers_llama_v4.csv")
    ap.add_argument("--k", type=int, default=8, help="Top-k chunks per field retrieval (per paper)")
    return ap.parse_args()

def main():
    args = parse_args()

    embed = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embed, collection_name="papers")

    llm = load_openai_llm(model_name=args.openai_model)
    tok = None
    if llm is None:
        llm, tok = load_local_llm(args.local_model)

    papers = sorted([p.stem for p in Path(args.snippets_dir).glob("*.txt")])
    print(f"▶ Extracting {len(papers)} papers…")

    with open(args.out_md, "w", encoding="utf-8") as fmd, \
         open(args.out_csv, "w", encoding="utf-8", newline="") as fcsv:

        fmd.write("| " + " | ".join(COLUMNS) + " |\n")
        fmd.write("|" + "|".join(["---"] * len(COLUMNS)) + "|\n")
        cw = csv.writer(fcsv); cw.writerow(COLUMNS)

        for pid in papers:
            try:
                cells = extract_row_for_paper(llm, tok, vectordb, args.snippets_dir, pid)
                fmd.write(row_to_markdown(cells) + "\n")
                cw.writerow(cells)
                print(f"✓ {pid}")
            except Exception as e:
                cells = [f"{pid}.pdf"] + ["Not reported"] * (len(COLUMNS) - 1)
                fmd.write(row_to_markdown(cells) + "\n")
                cw.writerow(cells)
                print(f"[WARN] {pid}: {e}")

    print(f"✅ Wrote {args.out_md} and {args.out_csv}")

if __name__ == "__main__":
    main()
