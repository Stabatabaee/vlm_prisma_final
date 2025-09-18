#!/usr/bin/env python3
"""
RAG extractor (section-aware + strict single-item answers + lexical sweep)

Local LLaMA:
  CUDA_VISIBLE_DEVICES=0 python rag_extract_langchain_v2.py \
    --snippets-dir snippets \
    --local-model meta-llama/Meta-Llama-3-8B-Instruct \
    --out-md filled_papers_llama.md \
    --out-csv filled_papers_llama.csv \
    --debug-prompts

OpenAI (if OPENAI_API_KEY is set):
  python rag_extract_langchain_v2.py \
    --snippets-dir snippets \
    --openai-model gpt-4o-mini \
    --out-md filled_papers_openai.md \
    --out-csv filled_papers_openai.csv
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

OPENAI_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

COLUMNS = [
    "File", "Title", "Authors", "Modality", "Datasets",
    "Model Name", "Vision Encoder", "Language Decoder", "Fusion Strategy",
]

FIELD_QUERIES = {
    "Title": "Extract the exact published paper title.",
    "Authors": "List authors as 'Surname et al.' if long; otherwise first author surname + 'et al.'.",
    "Modality": "Which imaging modality is used (e.g., X-Ray, MRI, CT, Ultrasound)? Return one term.",
    "Datasets": "Which dataset name(s) are used (e.g., MIMIC-CXR, IU X-ray, CheXpert)? Return one dataset name most central to experiments.",
    "Model Name": "What is the name of the model/approach used by the authors (e.g., ALBEF, BLIP-2, CXR-IRGen, R2Gen)? Return one name, not baselines or related work.",
    "Vision Encoder": "Which vision encoder architecture/variant is used (e.g., ViT-B/16, ResNet-50, CLIP)? Return one.",
    "Language Decoder": "Which language model backbone is used (e.g., LLaMA, BERT, GPT-2, T5, LSTM)? Return one.",
    "Fusion Strategy": "How are image/text fused: cross-attention, co-attention, multimodal encoder, early fusion, late fusion, concatenation, gated fusion? Return one.",
}

# Canonical maps / known labels
VISION_MAP = [
    ("vit-b/16", "ViT-B/16"), ("vision transformer", "ViT"), ("vit", "ViT"),
    ("resnet-50", "ResNet-50"), ("resnet", "ResNet"),
    ("clip", "CLIP"), ("dino", "DINO"), ("cnn", "CNN"), ("u-net", "U-Net"),
]
LANG_MAP = [
    ("llama", "LLaMA"), ("bert-base", "BERT-base"), ("bertbase", "BERT-base"),
    ("bert", "BERT"), ("gpt-2", "GPT-2"), ("t5", "T5"), ("lstm", "LSTM"),
    ("transformer", "Transformer"),
]
FUSION_MAP = [
    ("cross-attention", "Cross-Attention"), ("co-attention", "Co-Attention"),
    ("multimodal encoder", "Multimodal Encoder"), ("early fusion", "Early Fusion"),
    ("late fusion", "Late Fusion"), ("concatenation", "Concatenation"),
    ("gated fusion", "Gated Fusion"), ("attention fusion", "Attention Fusion"),
]
KNOWN_MODELS = [
    # extendable:
    "ALBEF", "BLIP", "BLIP-2", "CXR-IRGen", "R2Gen", "PPKED", "M2Trans",
    "TieNet", "EGGCA-Net", "RoentGen", "TrMRG"
]

SWEEP = {
    "Datasets": [r"\bMIMIC[- ]?CXR\b", r"\bIU[- ]?X[- ]?ray\b", r"\bOpen-?I\b",
                 r"\bCheXpert\b", r"\bPadChest\b", r"\bVinDr[- ]?CXR\b", r"\bNIH ChestX-?ray14\b"],
    "Vision Encoder": [r"\bViT\b", r"\bVision Transformer\b", r"\bResNet\b", r"\bCLIP\b", r"\bDINO\b", r"\bCNN\b", r"\bU[- ]?Net\b"],
    "Language Decoder": [r"\bLLaMA\b", r"\bBERT\b", r"\bGPT[- ]?2\b", r"\bT5\b", r"\bLSTM\b", r"\bTransformer\b"],
    "Fusion Strategy": [r"\bcross[- ]?attention\b", r"\bco[- ]?attention\b", r"\bmultimodal encoder\b",
                        r"\bearly fusion\b", r"\blate fusion\b", r"\bconcatenation\b", r"\bgated fusion\b"],
    "Modality": [r"\bX[- ]?Ray\b", r"\bCXR\b", r"\bMRI\b", r"\bCT\b", r"\bUltrasound\b"],
    "Model Name": [r"\bALBEF\b", r"\bBLIP-?2?\b", r"\bR2Gen\b", r"\bPPKED\b", r"\bM2Trans\b", r"\bTieNet\b", r"\bCXR[- ]?IRGen\b", r"\bTrMRG\b", r"\bRoentGen\b"],
    "Title": [r"^\s*[A-Z][^\n]{10,120}$"],
    "Authors": [r"\bet al\.\b", r"\b[A-Z][a-z]+ [A-Z][a-z]+"],
}

def chunk_text(text: str, window: int = 1600, stride: int = 800) -> List[str]:
    out, n = [], len(text)
    for i in range(0, n, stride):
        ch = text[i:i+window]
        if not ch: break
        out.append(ch)
    return out

def label_section(t: str) -> str:
    tl = t.lower()
    if "references" in tl: return "references"
    if "related work" in tl or "literature review" in tl: return "related"
    if "acknowledg" in tl: return "ack"
    if any(k in tl for k in ["method", "approach", "architecture", "proposed model"]): return "methods"
    if any(k in tl for k in ["experiment", "evaluation", "results"]): return "experiments"
    if "dataset" in tl or "data set" in tl: return "dataset"
    if "abstract" in tl: return "abstract"
    return "other"

def build_faiss_for_paper(paper_id: str, txt_path: Path) -> Tuple[FAISS, List[str]]:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, 1600, 800)
    embed = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    docs = []
    for i, c in enumerate(chunks):
        docs.append(Document(page_content=c, metadata={"paper": paper_id, "chunk_id": i, "section": label_section(c)}))
    vs = FAISS.from_documents(docs, embed)
    return vs, chunks

class LocalLlamaChat:
    def __init__(self, model_name: str, max_new: int = 64):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto",
            offload_state_dict=True, offload_folder="offload", trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tok)
        self.max_new = max_new

    def chat(self, system_msg: str, user_msg: str, max_new_tokens: int | None = None) -> str:
        prompt = self.tok.apply_chat_template(
            [{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            tokenize=False, add_generation_prompt=True
        )
        out = self.pipe(
            prompt, max_new_tokens=(max_new_tokens or self.max_new),
            do_sample=False, pad_token_id=self.tok.eos_token_id, eos_token_id=self.tok.eos_token_id,
            return_full_text=False,
        )
        return out[0]["generated_text"]

class OpenAIChat:
    def __init__(self, model_name: str):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("langchain_openai not installed")
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def chat(self, system_msg: str, user_msg: str, max_new_tokens: int | None = None) -> str:
        resp = self.llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
        return resp.content

SYSTEM = (
    "You are extracting a single metadata field for a PRISMA table. "
    "Use ONLY the provided context. "
    "If the field is not present, answer exactly 'Not reported'. "
    "Return a ONE-LINE JSON object: {'<FIELD>': '<VALUE>'}. "
    "Output exactly ONE short noun phrase (no commas, no 'and'); otherwise return 'Not reported'. "
    "Do NOT list related work or baselines; only what THIS paper uses."
)

def field_prompt(field_name: str, instruction: str, context_chunks: List[str]) -> str:
    ctx = "\n\n---\n\n".join(context_chunks)
    return (
        f"Field: {field_name}\n"
        f"Instruction: {instruction}\n\n"
        f"Context:\n```\n{ctx}\n```\n\n"
        f"Return JSON exactly like: {{\"{field_name}\": \"<short value or Not reported>\"}}"
    )

def parse_json_value(text: str, field_name: str) -> str:
    t = text.strip().replace("```json","").replace("```","").strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m: t = m.group(0)
    try:
        obj = json.loads(t)
        val = str(obj.get(field_name, "")).strip()
        return val if val else "Not reported"
    except Exception:
        if re.search(r"\bNot reported\b", text, flags=re.I): return "Not reported"
        line = text.splitlines()[0].strip()
        return line if 0 < len(line) <= 120 else "Not reported"

def sanitize_cell(s: str | None) -> str:
    if not s: return "Not reported"
    s = str(s).replace("|","/").replace("`","").strip()
    s = re.sub(r"\s+"," ", s)
    return s if s else "Not reported"

def markdown_row(cells: List[str]) -> str:
    cells = [sanitize_cell(x) for x in cells]
    if len(cells) < 9: cells += ["Not reported"]*(9-len(cells))
    if len(cells) > 9: cells = cells[:9]
    return "| " + " | ".join(cells) + " |"

def lexical_hits(full_text: str, patterns: List[str], max_snips: int = 6, window: int = 240) -> List[str]:
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, full_text, flags=re.I | re.M):
            start = max(0, m.start() - window)
            end   = min(len(full_text), m.end() + window)
            hits.append(full_text[start:end])
            if len(hits) >= max_snips: break
        if len(hits) >= max_snips: break
    return hits

def first_map(blob_lc: str, mapping: List[Tuple[str,str]]) -> str | None:
    for k, v in mapping:
        if k in blob_lc: return v
    return None

def pick_single_token(val: str) -> str:
    # cut at comma/semicolon/ ' and '
    v = re.split(r",|;|\band\b", val, maxsplit=1, flags=re.I)[0].strip()
    # too long? probably a sentence → reject
    if len(v.split()) > 6: return "Not reported"
    return v

def canonicalize(field: str, val: str, context_blob: str) -> str:
    if val == "Not reported": return val
    v0 = pick_single_token(val)
    v = v0.lower()

    if field == "Vision Encoder":
        can = first_map(v, VISION_MAP)
        return can if can else ("Not reported" if any(x in v for x,_ in VISION_MAP) is False else v0)

    if field == "Language Decoder":
        can = first_map(v, LANG_MAP)
        return can if can else ("Not reported" if any(x in v for x,_ in LANG_MAP) is False else v0.capitalize())

    if field == "Fusion Strategy":
        can = first_map(v, FUSION_MAP)
        return can if can else "Not reported"

    if field == "Model Name":
        # keep only if matches a known model (exact or substring)
        for nm in KNOWN_MODELS:
            if nm.lower().replace("-", "") in v.replace("-", ""):
                return nm
        return "Not reported"

    if field == "Modality":
        # normalize a few common variants
        if re.search(r"\bx[- ]?ray\b|\bcxr\b", v, flags=re.I): return "X-Ray"
        if re.search(r"\bmri\b", v, flags=re.I): return "MRI"
        if re.search(r"\bct\b", v, flags=re.I): return "CT"
        if re.search(r"\bultrasound\b", v, flags=re.I): return "Ultrasound"

    if field == "Datasets":
        # Allow common names only (single item)
        ds = pick_single_token(v0)
        # sanity: must look like a known dataset word or appear in context
        if not re.search(r"(mimic|iu|open-?i|chexpert|padchest|vindr|nih)", ds, flags=re.I):
            if ds.lower() not in context_blob.lower():
                return "Not reported"
        return ds

    if field in ("Title","Authors"):
        return v0

    return v0

def extract_paper(paper_id: str, txt_path: Path, llm, debug_prompts: bool=False) -> List[str]:
    vs, chunks = build_faiss_for_paper(paper_id, txt_path)
    full_text = txt_path.read_text(encoding="utf-8", errors="ignore")
    context_blob = full_text  # for canonicalization sanity checks

    def retrieve(query: str, k: int = 10) -> List[Document]:
        docs = vs.similarity_search(query, k=k)
        # prefer non-related sections
        good, bad = [], []
        for d in docs:
            sec = d.metadata.get("section","other")
            if sec in ("related","references","ack"):
                bad.append(d)
            else:
                good.append(d)
        return good[:6] if good else docs[:6]

    out = {
        "File": f"{paper_id}.pdf",
        "Title": "Not reported",
        "Authors": "Not reported",
        "Modality": "Not reported",
        "Datasets": "Not reported",
        "Model Name": "Not reported",
        "Vision Encoder": "Not reported",
        "Language Decoder": "Not reported",
        "Fusion Strategy": "Not reported",
    }

    for field in COLUMNS[1:]:
        q = FIELD_QUERIES[field]
        docs = retrieve(q, k=12)
        ctx = [d.page_content for d in docs]

        prompt = field_prompt(field, q,
                  ctx + [ "Answer with ONE item only (no commas, no 'and'); else 'Not reported'." ])
        if debug_prompts:
            Path("debug_prompts").mkdir(exist_ok=True)
            Path("debug_prompts", f"{paper_id}.{field}.prompt.txt").write_text(prompt, encoding="utf-8")

        raw = llm.chat(SYSTEM, prompt)
        val = parse_json_value(raw, field)

        # Lexical sweep fallback
        if val == "Not reported" and field in SWEEP:
            hits = lexical_hits(full_text, SWEEP[field], max_snips=6)
            if hits:
                prompt2 = field_prompt(field, q, hits + ["ONE item only."])
                if debug_prompts:
                    Path("debug_prompts", f"{paper_id}.{field}.lexical_prompt.txt").write_text(prompt2, encoding="utf-8")
                raw2 = llm.chat(SYSTEM, prompt2)
                val2 = parse_json_value(raw2, field)
                if val2 and val2 != "Not reported":
                    val = val2

        # Canonicalize / validate / de-list
        val = canonicalize(field, val, context_blob)
        out[field] = sanitize_cell(val)

    return [out[c] for c in COLUMNS]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets-dir", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--local-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    ap.add_argument("--debug-prompts", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    papers = sorted(Path(args.snippets-dir if hasattr(args,'snippets-dir') else args.snippets_dir).glob("*.txt"))
    if not papers:
        raise SystemExit(f"No snippets found in {args.snippets_dir}")

    use_openai = bool(os.environ.get("OPENAI_API_KEY")) and OPENAI_AVAILABLE and args.openai_model
    if use_openai:
        llm = OpenAIChat(args.openai_model)
        print(f"▶ Using OpenAI model: {args.openai_model}")
    else:
        print("Loading local model…")
        llm = LocalLlamaChat(args.local_model, max_new=64)

    md_lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join(["---"]*len(COLUMNS)) + "|"]
    csv_lines = [",".join(COLUMNS)]

    print(f"▶ Extracting {len(papers)} papers…")
    for p in papers:
        pid = p.stem
        try:
            row = extract_paper(pid, p, llm, debug_prompts=args.debug_prompts)
            md_lines.append(markdown_row(row))
            def q(x: str) -> str:
                x = x.replace('"','""')
                return f"\"{x}\"" if ("," in x or '"' in x) else x
            csv_lines.append(",".join(q(x) for x in row))
            print(f"✓ {pid}")
        except Exception as e:
            print(f"[WARN] {pid}: {e}")
            row = [f"{pid}.pdf"] + ["Not reported"]*8
            md_lines.append(markdown_row(row))
            csv_lines.append(",".join(row))

    Path(args.out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    Path(args.out_csv).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote {args.out_md} and {args.out_csv}")

if __name__ == "__main__":
    main()
