#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# LangChain (using legacy imports; deprecation warnings are ok in your env)
from langchain.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings  # type: ignore
from langchain.vectorstores import Chroma  # type: ignore
from langchain.docstore.document import Document

# Optional BM25
try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25Retriever = None  # type: ignore

# Local LLM (Transformers)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Canonical vocab & helpers
# =========================

DATASET_ALIASES = {
    "mimic-cxr": "MIMIC-CXR",
    "mimic cxr": "MIMIC-CXR",
    "mimic": "MIMIC-CXR",
    "iu x-ray": "IU X-ray",
    "iu xray": "IU X-ray",
    "open-i": "Open-i",
    "open i": "Open-i",
    "chestx-ray14": "NIH ChestX-ray14",
    "chestxray14": "NIH ChestX-ray14",
    "chexpert": "CheXpert",
    "padchest": "PadChest",
    "rsna": "RSNA Pneumonia",
    "rsna pneumonia": "RSNA Pneumonia",
    "reflacx": "REFLACX",
    "cxr-repair": "CXR-RePaiR",
    "cxr repair": "CXR-RePaiR",
}

METRIC_ALIASES = {
    "bleu": "BLEU", "bleu4": "BLEU",
    "cider": "CIDEr",
    "meteor": "METEOR",
    "rouge": "ROUGE", "rouge-l": "ROUGE",
    "radgraph": "RadGraph",
    "chexbert": "CheXbert",
    "radcliq": "RadCliQ",
    "gleu": "GLEU",
    "accuracy": "Accuracy",
    "f1": "F1",
}

# Light aliasing for common model tokens in evidence checking
MODEL_CANON = {
    "llama": "LLaMA",
    "llama-2": "LLaMA",
    "gpt2": "GPT-2",
    "gpt-2": "GPT-2",
    "gpt 2": "GPT-2",
    "gpt4v": "GPT-4V",
    "gpt-4v": "GPT-4V",
    "vit": "ViT",
    "clip": "CLIP",
    "resnet50": "ResNet50",
    "resnet-50": "ResNet50",
}

ALLOWED_VLM = {"yes": "Yes", "no": "No"}
ALLOWED_PAIRED = {"yes": "Yes", "no": "No", "not reported": "Not reported"}

DATASET_PAT = re.compile(
    r"\b(MIMIC[- ]?CXR|IU[- ]?X[- ]?ray|CheXpert|Open[- ]?i|PadChest|RSNA(?: Pneumonia)?|ChestX[- ]?ray14|CXR[- ]?RePaiR|REFLACX)\b",
    re.I,
)
METRIC_PAT = re.compile(
    r"\b(BLEU(?:-?\d)?|CIDEr|METEOR|ROUGE(?:-?L)?|RadGraph|CheXbert|RadCliQ|GLEU|Accuracy|F1)\b",
    re.I,
)


def _norm_token(s: str) -> str:
    return (s or "").strip()


def _norm_yes_no(s: str, allowed: dict) -> str:
    k = (s or "").strip().lower()
    return allowed.get(k, "Not reported")


def _canon_dataset(s: str) -> str:
    t = _norm_token(s)
    key = t.lower()
    return DATASET_ALIASES.get(key, t)


def _canon_metric(s: str) -> str:
    t = _norm_token(s)
    key = t.lower()
    return METRIC_ALIASES.get(key, t)


def _canon_modelish(s: str) -> str:
    t = _norm_token(s)
    key = t.lower()
    return MODEL_CANON.get(key, t)


def _soft_norm(s: str) -> str:
    """Lowercase and normalize punctuation/hyphens/spaces a bit for fuzzy contains."""
    s = (s or "").lower()
    s = re.sub(r"[\s_-]+", " ", s)
    return s.strip()


def value_in_context(value: str, ctx: str) -> bool:
    """Strict but robust: check exact or alias presence with mild normalization."""
    v = (value or "").strip()
    if not v:
        return False

    # direct contains (case-insensitive)
    if v.lower() in ctx.lower():
        return True

    # dataset/metric canonicalization
    for fn in (_canon_dataset, _canon_metric, _canon_modelish):
        canon = fn(v)
        if canon and canon.lower() in ctx.lower():
            return True

    # mild hyphen/space normalization
    if _soft_norm(v) in _soft_norm(ctx):
        return True

    return False


def enforce_evidence(row: dict, ctx_text: str) -> dict:
    """For each cell (except File), keep value only if evidenced in ctx_text."""
    verified = {}
    for k, v in row.items():
        if k == "File":
            verified[k] = v
            continue
        if isinstance(v, list):
            keep = [x for x in v if value_in_context(str(x), ctx_text)]
            verified[k] = keep if keep else "Not reported"
        else:
            # Only pass through non-empty, non-Not reported values with evidence
            val = str(v)
            if val and val != "Not reported" and value_in_context(val, ctx_text):
                verified[k] = val
            else:
                verified[k] = "Not reported"
    return verified


def _uniq(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_row(row: dict) -> dict:
    r = dict(row)
    r["VLM?"] = _norm_yes_no(r.get("VLM?"), ALLOWED_VLM)
    r["Paired"] = _norm_yes_no(r.get("Paired"), ALLOWED_PAIRED)

    def _norm_list(v):
        if isinstance(v, list):
            vals = [_canon_dataset(x) for x in v if str(x).strip()]
            return _uniq(vals) if vals else "Not reported"
        if not v or v == "Not reported":
            return "Not reported"
        return [_canon_dataset(v)]

    r["Datasets (train)"] = _norm_list(r.get("Datasets (train)"))
    r["Datasets (eval)"] = _norm_list(r.get("Datasets (eval)"))

    def _norm_metrics(v):
        if isinstance(v, list):
            vals = [_canon_metric(x) for x in v if str(x).strip()]
            return _uniq(vals) if vals else "Not reported"
        if not v or v == "Not reported":
            return "Not reported"
        return [_canon_metric(v)]

    r["Metrics(primary)"] = _norm_metrics(r.get("Metrics(primary)"))

    # Light model token normalization for evidence matching; keep original fields
    for key in ("Model", "Vision Enc", "Lang Dec", "Family"):
        val = r.get(key)
        if isinstance(val, list):
            r[key] = [_canon_modelish(x) for x in val]
        elif isinstance(val, str):
            r[key] = _canon_modelish(val)

    return r


def regex_candidates(text: str) -> Tuple[List[str], List[str]]:
    ds = {_canon_dataset(m.group(0)) for m in DATASET_PAT.finditer(text)}
    ms = {_canon_metric(m.group(0)) for m in METRIC_PAT.finditer(text)}
    return sorted(ds), sorted(ms)


# =========================
# I/O helpers
# =========================

def read_snippet_files(snippets_dir: str) -> List[Tuple[str, str]]:
    """Return list of (filename_stem, text). Accept .txt, .md, .json(l) (as text)."""
    items: List[Tuple[str, str]] = []
    paths = []
    paths += glob.glob(os.path.join(snippets_dir, "*"))
    for p in sorted(paths):
        if os.path.isdir(p):
            for q in sorted(glob.glob(os.path.join(p, "*"))):
                if os.path.isfile(q):
                    items.extend(read_snippet_files(q))
            continue
        name = os.path.basename(p)
        stem, _ext = os.path.splitext(name)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                items.append((stem, f.read()))
        except Exception:
            continue
    return items


# =========================
# Vector DB / Retrieval
# =========================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_db(snippets: List[Tuple[str, str]]) -> Tuple[Chroma, List[Document]]:
    """Create an in-memory Chroma from all snippets (no persistence)."""
    embed = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    texts, metadatas = [], []
    docs_all: List[Document] = []
    for fname, text in snippets:
        if not text.strip():
            continue
        texts.append(text)
        metadatas.append({"source": fname})
        docs_all.append(Document(page_content=text, metadata={"source": fname}))
    db = Chroma(collection_name="papers", embedding_function=embed)
    if texts:
        db.add_texts(texts=texts, metadatas=metadatas)
    return db, docs_all


def mmr_search(db: Chroma, query: str, k: int, filename: str) -> List[Document]:
    """MMR with per-file filter."""
    flt = {"source": filename}
    try:
        return db.max_marginal_relevance_search(query, k=k, fetch_k=max(k * 3, 12), filter=flt)
    except Exception:
        return db.similarity_search(query, k=k, filter=flt)


def bm25_for_file(corpus_docs: List[Document], filename: str, query: str, k: int) -> List[Document]:
    if not BM25Retriever:
        return []
    subset = [d for d in corpus_docs if d.metadata.get("source") == filename]
    if not subset:
        return []
    try:
        bm25 = BM25Retriever.from_texts([d.page_content for d in subset])
        bm25.k = min(6, max(3, k // 4))
        hits = bm25.get_relevant_documents(query)
        # Rewrap to keep metadata
        wrapped = []
        pages = {d.page_content: d for d in subset}
        for h in hits:
            base = pages.get(h.page_content)
            if base:
                wrapped.append(base)
        return wrapped
    except Exception:
        return []


def blend_mmr_bm25(query: str, k: int, db: Chroma, corpus_docs: List[Document], filename: str) -> List[Document]:
    mmr_docs = mmr_search(db, query, k, filename)
    out = {id(d): d for d in mmr_docs}
    for d in bm25_for_file(corpus_docs, filename, query, k):
        out.setdefault(id(d), d)
    docs = list(out.values())
    # Guarantee at least k items if possible by padding with more from same file
    if len(docs) < k:
        extras = [d for d in corpus_docs if d.metadata.get("source") == filename and d not in docs]
        docs += extras[: (k - len(docs))]
    return docs[: max(k, len(docs))]


# =========================
# Local LLM wrapper
# =========================

@dataclass
class LocalLLM:
    model_name: str
    max_new_tokens: int

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    @torch.inference_mode()
    def generate_json(self, prompt: str, temperature: float = 0.2) -> str:
        tok = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **tok,
            max_new_tokens=self.max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "```json" in text:
            seg = text.split("```json")[-1]
            if "```" in seg:
                seg = seg.split("```")[0]
            return seg.strip()
        m = re.findall(r"\{.*\}", text, flags=re.S)
        if m:
            return m[-1]
        return text.strip()


# =========================
# Extraction
# =========================

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

SYSTEM_INSTRUCTIONS = """You are extracting FACTS strictly from the provided context.
Return ONLY a compact JSON object with these keys:
["Modality","Datasets (train)","Datasets (eval)","Paired","VLM?","Model","Class","Task","Vision Enc","Lang Dec","Fusion","Objectives","Family","RAG","Metrics(primary)"].
Rules:
- Only output values that are explicitly present in the context (verbatim or obvious alias). If a field is not present, write "Not reported".
- For boolean-like fields use "Yes" or "No".
- For lists, return a JSON array of strings (no duplicates); otherwise a string.
- Do NOT invent models, datasets, or metrics. If unknown in context, use "Not reported".
- No extra commentary. Only the JSON object.
"""


def build_prompt(filename: str, context: str, ds_hint: List[str], m_hint: List[str]) -> str:
    hint = {
        "dataset_candidates_seen": ds_hint[:10],
        "metric_candidates_seen": m_hint[:10],
        "note": "Only use candidates if they literally appear in context."
    }
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Paper: {filename}\n"
        f"Context:\n{context}\n\n"
        f"Hints (from regex): {json.dumps(hint)}\n"
        f"Return JSON now:"
    )


def clip_context(docs: List[Document], budget_chars: int) -> str:
    buf = []
    total = 0
    for d in docs:
        t = d.page_content
        if not t:
            continue
        if total + len(t) > budget_chars:
            t = t[: max(0, budget_chars - total)]
        buf.append(t)
        total += len(t)
        if total >= budget_chars:
            break
    return "\n\n".join(buf)


def parse_row(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return {}
        return obj
    except Exception:
        return {}


# =========================
# Runner
# =========================

def extract_for_one_file(
    filename: str,
    db: Chroma,
    corpus_docs: List[Document],
    llm: LocalLLM,
    args,
) -> Dict[str, Any]:
    # Retrieve ONLY this file's text
    query = f"{filename} radiology vision-language report generation datasets metrics model"
    retrieved = blend_mmr_bm25(query, args.k, db, corpus_docs, filename)

    # Clip context
    approx_chars = int(args.ctx_budget_toks * 5)  # ~chars per token
    ctx_full = clip_context(retrieved, approx_chars)

    # Regex hints
    ds_hint, m_hint = regex_candidates(ctx_full)

    # Prompt LLM
    prompt = build_prompt(filename, ctx_full, ds_hint, m_hint)
    raw_json = llm.generate_json(prompt, temperature=0.2)
    fields = parse_row(raw_json)

    # Build row with forced attribution
    row = {k: fields.get(k, "Not reported") for k in COLUMNS if k != "File"}
    row["File"] = filename

    # Normalize
    row = normalize_row(row)

    # Strict evidence (cell-level)
    if args.strict_evidence:
        row = enforce_evidence(row, ctx_full)

    return row


def write_outputs(rows: List[Dict[str, Any]], out_csv: str, out_md: str):
    # Keep best row per file (more evidenced cells wins)
    best_by_file: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, int] = {}

    def score_row(r: Dict[str, Any]) -> int:
        sc = 0
        for k, v in r.items():
            if k == "File":
                continue
            if isinstance(v, list) and v and v != "Not reported":
                sc += len(v)
            elif isinstance(v, str) and v != "Not reported":
                sc += 1
        return sc

    for r in rows:
        f = r.get("File", "")
        sc = score_row(r)
        if f not in best_by_file or sc > scores.get(f, -1):
            best_by_file[f] = r
            scores[f] = sc

    final_rows = [best_by_file[k] for k in sorted(best_by_file.keys())]

    # CSV
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in final_rows:
            row_vals = []
            for c in COLUMNS:
                v = r.get(c, "Not reported")
                if isinstance(v, list):
                    v = ", ".join(v)
                row_vals.append(v)
            w.writerow(row_vals)

    # Markdown
    def _md_escape(x: str) -> str:
        return x.replace("|", r"\|")

    lines = []
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "---|" * len(COLUMNS)
    lines.append(header)
    lines.append(sep)
    for r in final_rows:
        row_vals = []
        for c in COLUMNS:
            v = r.get(c, "Not reported")
            if isinstance(v, list):
                v = ", ".join(v)
            row_vals.append(_md_escape(str(v)))
        lines.append("| " + " | ".join(row_vals) + " |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(args):
    pairs = read_snippet_files(args.snippets_dir)
    if not pairs:
        print("No snippets found.")
        write_outputs([], args.out_csv, args.out_md)
        return

    db, corpus_docs = build_db(pairs)
    llm = LocalLLM(model_name=args.local_model, max_new_tokens=args.max_new_tokens)

    filenames = sorted({fname for fname, _ in pairs})

    rows: List[Dict[str, Any]] = []
    print("▶ Extracting papers…")
    for fname in filenames:
        try:
            r = extract_for_one_file(fname, db, corpus_docs, llm, args)
            rows.append(r)
            print(f"✓ {fname}")
        except Exception as e:
            print(f"⚠️  {fname}: {e}")

    write_outputs(rows, args.out_csv, args.out_md)
    print(f"✅ Wrote {args.out_md} and {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True)
    ap.add_argument("--backend", choices=["llm", "regex"], default="llm")
    ap.add_argument("--retrieval", choices=["mmr", "similarity"], default="mmr")
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--ctx-budget-toks", type=int, default=8000)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--local-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--strict-evidence", action="store_true")
    args = ap.parse_args()
    main(args)
