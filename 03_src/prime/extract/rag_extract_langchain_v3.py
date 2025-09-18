#!/usr/bin/env python3
"""
RAG-based PRISMA metadata extraction (v3)
- Chroma + sentence-transformer embeddings for per-field retrieval (MMR)
- Strict JSON-only answers from the LLM ("{\"value\":\"...\"}")
- "Not reported" policy:
    1) Try RAG context for the field.
    2) Lexical sweep across *all* chunks (snippets/<paper>.txt) for field-specific keywords.
       If hits exist that weren't in RAG, re-ask using those windows.
    3) If zero hits anywhere, confidently return "Not reported".
- Local LLaMA (HF Transformers) by default; optional OpenAI if OPENAI_API_KEY is set.

Example:
  # Ensure Chroma index is built (run rag_build_index.py when snippets change)
  # Then run with local LLaMA:
  unset OPENAI_API_KEY
  CUDA_VISIBLE_DEVICES=0 python rag_extract_langchain_v3.py \
    --snippets-dir snippets \
    --local-model meta-llama/Meta-Llama-3-8B-Instruct \
    --out-md filled_papers_llama_v3.md \
    --out-csv filled_papers_llama_v3.csv
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# LangChain / Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Local LLM (HF)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    GenerationConfig,
)

# Optional OpenAI (only used if OPENAI_API_KEY is present)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ---------------------------- Schema / Columns ----------------------------
COLUMNS = [
    "File", "Title", "Authors", "Modality", "Datasets",
    "Model Name", "Vision Encoder", "Language Decoder", "Fusion Strategy"
]

# Retrieval query per field
FIELD_QUERIES: Dict[str, str] = {
    "Title": "paper title as published; exact title line; 'title:'; 'Abstract' header context",
    "Authors": "authors list; author affiliations; 'et al.'; correspondence",
    "Modality": "imaging modality description (X-ray/CXR/CT/MRI/Ultrasound); chest X-ray; modality section",
    "Datasets": "datasets used (MIMIC-CXR, IU X-ray, Open-i, CheXpert, ChestX-ray14, PadChest, VinDr-CXR, COVIDx); data section",
    "Model Name": "model name or approach (ALBEF, BLIP, CXR-IRGen, R2Gen, KERP, PPKED, M2KT, HRGR, TieNet)",
    "Vision Encoder": "vision encoder backbone (ViT, ResNet, Swin, CLIP, CNN, DINO, MAE, DeiT); feature extractor",
    "Language Decoder": "language model/decoder (LLaMA, BERT, GPT-2, DistilGPT-2, T5, Transformer, LSTM, BART, MiniLM)",
    "Fusion Strategy": "image-text fusion mechanism (cross-attention, co-attention, multimodal encoder, early fusion, late fusion, concatenation, gated fusion)",
}

# Keywords for lexical sweep (case-insensitive)
LEX_KEYWORDS: Dict[str, List[str]] = {
    "Title": ["title", "abstract"],
    "Authors": ["author", "authors", "et al", "corresponding", "affiliation"],
    "Modality": ["x-ray", "xray", "cxr", "ct", "mri", "ultrasound", "radiograph", "modality"],
    "Datasets": [
        "mimic-cxr", "iu x-ray", "iu-xray", "open-i", "open i", "chexpert",
        "padchest", "vindr-cxr", "chestx-ray14", "chestxray14", "covidx"
    ],
    "Model Name": [
        "albef", "blip", "cxr-irgen", "cxrirgen", "r2gen", "kerp", "ppked",
        "m2kt", "hrgr", "tienet", "roentgen", "ldm", "ealbef", "rag"
    ],
    "Vision Encoder": ["vit", "resnet", "swin", "clip", "cnn", "dino", "mae", "deit"],
    "Language Decoder": ["llama", "bert", "gpt-2", "distilgpt", "t5", "transformer", "lstm", "bart", "minilm", "gpt2"],
    "Fusion Strategy": [
        "cross-attention", "co-attention", "multimodal encoder",
        "early fusion", "late fusion", "concatenation", "gated fusion", "fusion"
    ],
}

DB_DIR = "chroma_db"
DEBUG_DIR = Path("rag_debug")
DEBUG_DIR.mkdir(exist_ok=True)


# ---------------------------- Utilities ----------------------------
def sanitize_cell(s: Optional[str]) -> str:
    if s is None:
        return "Not reported"
    s = str(s)
    s = s.replace("|", "/").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "Not reported"


def markdown_row(cells: List[str]) -> str:
    cells = [sanitize_cell(x) for x in cells]
    if len(cells) < 9:
        cells += ["Not reported"] * (9 - len(cells))
    if len(cells) > 9:
        cells = cells[:9]
    return "| " + " | ".join(cells) + " |"


def save_outputs(rows: List[List[str]], md_path: str, csv_path: str) -> None:
    # Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(COLUMNS)) + "|\n")
        for row in rows:
            f.write(markdown_row(row) + "\n")
    # CSV
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([sanitize_cell(x) for x in row])


def load_vectordb_and_embedder() -> Tuple[Chroma, SentenceTransformerEmbeddings]:
    embed = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embed,
        collection_name="papers"
    )
    return vectordb, embed


def load_snippet_text(snippets_dir: str, paper: str) -> str:
    path = Path(snippets_dir) / f"{paper}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def mmr_search_for_paper(
    vectordb: Chroma,
    query: str,
    paper: str,
    k: int = 6,
    fetch_k: int = 24,
    lambda_mult: float = 0.55
) -> List[Document]:
    """
    Use MMR search with a filter to restrict to this paper's chunks.
    We assume rag_build_index.py stored metadata 'paper' for each chunk.
    """
    try:
        docs = vectordb.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter={"paper": paper}
        )
        return docs
    except Exception:
        # Fallback: similarity search + manual distinct filter
        docs = vectordb.similarity_search(query=query, k=fetch_k, filter={"paper": paper})
        uniq = []
        seen = set()
        for d in docs:
            key = (d.page_content.strip(), d.metadata.get("chunk_id", d.metadata.get("source", "")))
            if key not in seen:
                uniq.append(d)
                seen.add(key)
            if len(uniq) >= k:
                break
        return uniq


def extract_windows(text: str, keywords: List[str], win: int = 320) -> List[str]:
    """
    Return distinct text windows around each keyword hit.
    """
    text_lc = text.lower()
    spans: List[Tuple[int, int]] = []
    for kw in keywords:
        pat = re.escape(kw.lower())
        for m in re.finditer(pat, text_lc):
            s = max(0, m.start() - win)
            e = min(len(text), m.end() + win)
            spans.append((s, e))
    # Merge overlapping spans
    if not spans:
        return []
    spans.sort()
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e + 50:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    # Extract
    windows = [text[s:e] for s, e in merged]
    # Deduplicate on content signature
    out, seen = [], set()
    for w in windows:
        sig = re.sub(r"\s+", " ", w.strip())
        if sig and sig not in seen:
            out.append(w.strip())
            seen.add(sig)
    return out


# ---------------------------- LLMs ----------------------------
def load_local_llm(model_name: str) -> TextGenerationPipeline:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        offload_state_dict=True,
        offload_folder="offload",
        trust_remote_code=True,
    )

    # IMPORTANT: set generation config on the *model* (do not pass into pipeline init)
    model.generation_config = GenerationConfig(
        max_new_tokens=96,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )

    pipe = TextGenerationPipeline(model=model, tokenizer=tok)
    return pipe


def extract_value_from_jsonish(s: str) -> Optional[str]:
    """
    Try to parse {"value": "..."}; fall back to naive first-line cleaning.
    """
    # Strip code fences if present
    s = s.strip().replace("```json", "```").replace("```JSON", "```").replace("```", "").strip()

    # Try JSON parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "value" in obj:
            v = obj["value"]
            if isinstance(v, str):
                return v.strip()
    except Exception:
        pass

    # Try to find a JSON object inside the string
    m = re.search(r'\{\s*"value"\s*:\s*"(.*?)"\s*\}', s, flags=re.S)
    if m:
        return m.group(1).strip()

    # Fallback: take first line, strip quotes/backticks
    line = s.splitlines()[0].strip()
    line = line.strip('"').strip("'").strip()
    return line if line else None


def ask_llm_local(pipe: TextGenerationPipeline, instruction: str, context: str) -> str:
    """
    Ask local LLaMA with strict JSON-only format.
    """
    prompt = (
        "You are extracting one field for a PRISMA metadata table from a scientific paper.\n"
        "Use ONLY the provided context.\n"
        "If the field is not present, answer exactly: Not reported\n\n"
        "Rules:\n"
        "1) Output EXACTLY one JSON object of the form: {\"value\":\"...\"}\n"
        "2) No extra text before or after the JSON. No markdown. No code.\n"
        "3) Keep it to a short phrase, not a sentence.\n\n"
        f"Instruction: {instruction}\n\n"
        "Context:\n"
        "-----\n"
        f"{context}\n"
        "-----\n"
        "Now output only: {\"value\":\"...\"}"
    )
    raw = pipe(prompt, return_full_text=False)[0]["generated_text"]
    val = extract_value_from_jsonish(raw)
    return val or "Not reported"


def ask_llm_openai(instruction: str, context: str) -> str:
    """
    Ask OpenAI (if OPENAI_API_KEY set); returns the extracted value string.
    """
    if not (OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        raise RuntimeError("OpenAI path requested but not available.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sysmsg = (
        "You are extracting one field for a PRISMA metadata table from a scientific paper. "
        "Use ONLY the provided context. If the field is not present, answer exactly: Not reported. "
        "Output EXACTLY a single JSON object: {\"value\":\"...\"}. No extra text."
    )
    user = (
        f"Instruction: {instruction}\n\n"
        "Context:\n-----\n"
        f"{context}\n-----\n"
        "Now output only: {\"value\":\"...\"}"
    )
    msg = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
    out = llm.invoke(msg).content
    val = extract_value_from_jsonish(out)
    return val or "Not reported"


# ---------------------------- Field extraction ----------------------------
def build_instruction(field: str) -> str:
    if field == "Title":
        return "Extract the exact published title of the paper."
    if field == "Authors":
        return "List authors in-order as a short comma-separated phrase (e.g., Doe J., Smith A.)."
    if field == "Modality":
        return "Which imaging modality (e.g., X-Ray/CXR, CT, MRI, Ultrasound)? Answer with the modality only."
    if field == "Datasets":
        return "List dataset names used (e.g., MIMIC-CXR, IU X-ray, Open-i); short comma-separated."
    if field == "Model Name":
        return "Provide the model/approach name(s) used; short phrase."
    if field == "Vision Encoder":
        return "Provide the vision encoder backbone (e.g., ViT-B/16, ResNet-50, CLIP)."
    if field == "Language Decoder":
        return "Provide the language model/decoder (e.g., LLaMA, BERT, GPT-2, Transformer, LSTM)."
    if field == "Fusion Strategy":
        return "Provide the image-text fusion mechanism (e.g., Cross-Attention, Co-Attention, Multimodal Encoder, Early/Late Fusion, Concatenation)."
    return f"Extract the field: {field}."


def dedupe_docs_by_content(docs: List[Document]) -> List[Document]:
    seen = set()
    uniq = []
    for d in docs:
        sig = re.sub(r"\s+", " ", d.page_content.strip())
        if sig and sig not in seen:
            uniq.append(d)
            seen.add(sig)
    return uniq


def compile_context(primary_docs: List[Document], extra_windows: List[str], max_chars: int = 6000) -> str:
    parts: List[str] = []
    if primary_docs:
        parts.append("RAG Retrieved:\n")
        for d in primary_docs:
            parts.append(d.page_content.strip())
            parts.append("\n---\n")
    if extra_windows:
        parts.append("\nLexical Windows:\n")
        for w in extra_windows:
            parts.append(w.strip())
            parts.append("\n---\n")
    txt = "".join(parts).strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars]
    return txt


def extract_field_for_paper(
    field: str,
    paper: str,
    snippets_dir: str,
    vectordb: Chroma,
    local_pipe: Optional[TextGenerationPipeline],
    use_openai: bool,
    k: int = 6,
    fetch_k: int = 24,
    lambda_mult: float = 0.55,
    debug_prompts: bool = False
) -> str:
    instr = build_instruction(field)

    # 1) Per-field retrieval via MMR
    q = FIELD_QUERIES.get(field, field)
    docs = mmr_search_for_paper(vectordb, q, paper, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    docs = dedupe_docs_by_content(docs)
    rag_text = "\n".join([d.page_content for d in docs])

    # 2) Lexical sweep across *full* snippet text
    full_text = load_snippet_text(snippets_dir, paper)
    hits_windows: List[str] = []
    if full_text:
        keywords = LEX_KEYWORDS.get(field, [])
        hits_windows = extract_windows(full_text, keywords, win=320)

    # If no retrieved docs and no keyword hits at all -> confidently Not reported
    if (not rag_text.strip()) and (not hits_windows):
        return "Not reported"

    # Ask using RAG context first
    ctx1 = compile_context(docs, [], max_chars=6000)
    if debug_prompts:
        Path(DEBUG_DIR, f"{paper}.{field}.ctx1.txt").write_text(ctx1, encoding="utf-8")
        Path(DEBUG_DIR, f"{paper}.{field}.instr.txt").write_text(instr, encoding="utf-8")

    if use_openai:
        val1 = ask_llm_openai(instr, ctx1)
    else:
        assert local_pipe is not None, "Local pipe must be provided when not using OpenAI."
        val1 = ask_llm_local(local_pipe, instr, ctx1)

    # If we got a non-"Not reported" answer, return it
    if val1 and val1.strip().lower() != "not reported":
        return val1.strip()

    # Otherwise, if we DO have lexical hits not yet used, re-ask with those windows
    if hits_windows:
        # Make a reduced (distinct) set of windows that are not already included in ctx1
        reduced = []
        for w in hits_windows:
            sig = re.sub(r"\s+", " ", w.strip())
            if sig and sig not in ctx1:
                reduced.append(w)
        if reduced:
            ctx2 = compile_context([], reduced, max_chars=6000)
            if debug_prompts:
                Path(DEBUG_DIR, f"{paper}.{field}.ctx2.txt").write_text(ctx2, encoding="utf-8")
            if use_openai:
                val2 = ask_llm_openai(instr, ctx2)
            else:
                val2 = ask_llm_local(local_pipe, instr, ctx2)
            if val2 and val2.strip().lower() != "not reported":
                return val2.strip()

    # Fall-through: still Not reported
    return "Not reported"


# ---------------------------- Main ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True, help="Folder with <paper>.txt extracted text")
    ap.add_argument("--out-md", required=True, help="Output markdown table path")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--local-model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF model id for local LLaMA")
    ap.add_argument("--k", type=int, default=6, help="MMR: final distinct chunks")
    ap.add_argument("--fetch-k", type=int, default=24, help="MMR: candidates pool")
    ap.add_argument("--lambda-mult", type=float, default=0.55, help="MMR balance relevance/diversity")
    ap.add_argument("--debug-prompts", action="store_true", help="Save per-field contexts/instructions to rag_debug/")
    return ap.parse_args()


def main():
    args = parse_args()

    # Use OpenAI only if key present; otherwise local LLaMA
    use_openai = bool(os.environ.get("OPENAI_API_KEY"))
    vectordb, _ = load_vectordb_and_embedder()

    local_pipe: Optional[TextGenerationPipeline] = None
    if not use_openai:
        print("Loading local model…")
        local_pipe = load_local_llm(args.local_model)

    # Papers = all snippet files present
    papers = sorted([p.stem for p in Path(args.snippets_dir).glob("*.txt")])  # noqa (dash in name)
    # The above is invalid Python due to hyphen in attribute; fix:
    # (Workaround since argparse stores as args.snippets_dir)
    papers = sorted([p.stem for p in Path(args.snippets_dir).glob("*.txt")])

    print(f"▶ Extracting {len(papers)} papers…")

    rows: List[List[str]] = []
    for paper in papers:
        file_pdf = f"{paper}.pdf"
        row = [file_pdf]
        for field in COLUMNS[1:]:
            try:
                val = extract_field_for_paper(
                    field=field,
                    paper=paper,
                    snippets_dir=args.snippets_dir,
                    vectordb=vectordb,
                    local_pipe=local_pipe,
                    use_openai=use_openai,
                    k=args.k,
                    fetch_k=args.fetch_k,
                    lambda_mult=args.lambda_mult,
                    debug_prompts=args.debug_prompts,
                )
            except Exception as e:
                val = "Not reported"
                if args.debug_prompts:
                    Path(DEBUG_DIR, f"{paper}.{field}.error.txt").write_text(str(e), encoding="utf-8")
            row.append(val)
        rows.append(row)
        print(f"✓ {paper}")

    save_outputs(rows, args.out_md, args.out_csv)
    print(f"✅ Wrote {args.out_md} and {args.out_csv}")


if __name__ == "__main__":
    main()
