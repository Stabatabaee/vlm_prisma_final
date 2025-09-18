#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_extract_langchain_v14_sim.py

PRISMA-style metadata extractor (Similarity variant).

This variant uses:
- Retrieval: **similarity_search** with `doc_id` filter only (no MMR).
- Context: Always appends full snippet (capped by CTX_CHAR_LIMIT).
- All other behavior matches v14_mmr (backfills, gating, JSON, outputs).
"""

import argparse, csv, json, re, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings  # type: ignore
from langchain.vectorstores import Chroma  # type: ignore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # type: ignore

# --- shared constants / tokens (identical to v14_mmr) ---
EMBED_MODEL = "all-mpnet-base-v2"
DB_DIR = "chroma_db"
CTX_CHAR_LIMIT = 18000

COLUMNS = [
    "File","Modality","Datasets (train)","Datasets (eval)","Paired","VLM?","Model",
    "Class","Task","Vision Enc","Lang Dec","Fusion","Objectives","Family","RAG","Metrics(primary)",
]

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
VISION_ENCODER_ALIASES = {
    r"\bViT\b|\bVision Transformer\b": "ViT",
    r"\bCLIP\b": "CLIP",
    r"\bDeiT\b": "DeiT",
    r"\bSwin\b": "Swin",
    r"\bResNet(?:-?50)?\b": "ResNet50",
    r"\bConvNeXt\b": "ConvNeXt",
    r"\bCNN\b|convolutional neural network": "CNN",
}
LANG_DECODER_ALIASES = {
    r"\bLLaMA\b": "LLaMA",
    r"\bBERT\b": "BERT",
    r"\bGPT-?2\b": "GPT-2",
    r"\bT5\b": "T5",
    r"\bLSTM\b": "LSTM",
    r"\bGRU\b": "GRU",
    r"\bTransformer\b": "Transformer",
}
FAMILY_TOKENS = {"ALBEF","BLIP","BLIP-2","LLaVA","Flamingo","Other","Transformer","CLIP"}
MODEL_CANON = {
    r"\btrmrg\b|Transformer Medical report generator": "TrMRG",
    r"\beggca[- ]?net\b": "EGGCA-Net",
    r"\bcxr[- ]?irgen\b": "CXR-IRGen",
    r"\balbef\b": "ALBEF",
}
EVIDENCE_PATTERNS = {
    "ViT": r"\bViT\b|Vision Transformer","CLIP": r"\bCLIP\b","DeiT": r"\bDeiT\b","Swin": r"\bSwin\b",
    "ResNet50": r"\bResNet(?:-?50)?\b","CNN": r"\bCNN\b|convolutional neural network",
    "LLaMA": r"\bLLaMA\b","BERT": r"\bBERT\b","GPT-2": r"\bGPT-?2\b","T5": r"\bT5\b",
    "LSTM": r"\bLSTM\b","GRU": r"\bGRU\b","Transformer": r"\bTransformer\b",
    "cross-attention": r"\bcross[- ]?attention\b","co-attention": r"\bco[- ]?attention\b",
    "concatenation": r"\bconcatenation|\bconcat\b","two-stream": r"\b(two[- ]?stream|dual[- ]?encoder)\b",
    "single-stream": r"\bsingle[- ]?stream\b","RAG": r"\bRAG\b|\bretrieval[- ]?augmented\b",
    "report-generation": r"\breport[- ]?generation\b","image-text matching": r"\bimage[- ]?text matching\b",
    "image-report generation": r"\bimage[- ]?report\b",
}
METRIC_TOKENS = ["BLEU","ROUGE","CIDEr","METEOR","BERTScore","SPICE","F1","Accuracy","RadCliQ","CheXbert","RadGraph","GLEU"]
OBJECTIVE_TOKENS = ["ITC(contrastive)","ITM","captioning CE/NLL","RL(CIDEr/SCST)","alignment loss","coverage"]
BOOL_TRUE = {"true","yes","y","1"}; BOOL_FALSE = {"false","no","n","0"}
MODALITY_CANON = {"x-ray":"X-Ray","xray":"X-Ray","x ray":"X-Ray","ct":"CT","mri":"MRI","ultrasound":"Ultrasound","mixed":"Mixed","radiology":"X-Ray","not reported":"Not reported"}

def println(s): print(s); sys.stdout.flush()
def debugln(s, dbg): 
    if dbg: print(s); sys.stdout.flush()

def list_papers(snippets_dir: Path)->List[str]:
    names=set()
    for p in snippets_dir.glob("*"):
        if p.suffix.lower() in (".txt",".md"): names.add(p.stem)
        elif p.name.endswith(".pdf.txt"): names.add(p.name[:-8])
    return sorted(names)

def read_full_snippet(snippets_dir: Path, base: str)->Optional[str]:
    for c in [snippets_dir/f"{base}.txt",snippets_dir/f"{base}.md",snippets_dir/f"{base}.pdf.txt"]:
        if c.exists():
            try: return c.read_text(encoding="utf-8", errors="ignore")
            except Exception: pass
    return None

def canon_bool(x: Any)->str:
    if isinstance(x,bool): return "Yes" if x else "No"
    if x is None: return "Not reported"
    s=str(x).strip().lower()
    if s in BOOL_TRUE: return "Yes"
    if s in BOOL_FALSE: return "No"
    return "Not reported"

def canon_modality(x: Any)->str:
    if x is None: return "Not reported"
    s=str(x[0]).strip().lower() if isinstance(x,list) and x else str(x).strip().lower()
    return MODALITY_CANON.get(s, MODALITY_CANON.get(s.replace("-"," "), "Not reported"))

def canon_list(x: Any)->str:
    if x is None: return "Not reported"
    if isinstance(x,list):
        vals=[str(v).strip() for v in x if str(v).strip()]
        return ", ".join(vals) if vals else "Not reported"
    s=str(x).strip()
    return s if s else "Not reported"

def canon_family(s: Any)->str:
    if s is None: return "Not reported"
    val=str(s).strip()
    if not val: return "Not reported"
    for tok in FAMILY_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", val, re.I): return tok
    return val

def canon_model(raw: Any, context: str)->str:
    if raw is None: return "Not reported"
    s=str(raw).strip()
    if not s: return "Not reported"
    for pat,canon in MODEL_CANON.items():
        if re.search(pat, s, re.I) or re.search(pat, context, re.I): return canon
    return s

def extract_tokens_from_text(text: str, token_list: List[str])->List[str]:
    return sorted({t for t in token_list if re.search(rf"\b{re.escape(t)}\b", text, re.I)})

def _present(value: str, ctx: str, patt: dict)->bool:
    if value=="Not reported": return False
    pat=patt.get(value)
    return re.search(pat, ctx, re.I) is not None if pat else re.search(re.escape(value), ctx, re.I) is not None

def build_vectorstore():
    embed = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embed, collection_name="papers")

def retrieve_context_similarity(vectordb, base: str, snippets_dir: Path, k: int, debug: bool)->str:
    """Plain similarity_search with doc_id filter; then append full snippet (cap to CTX_CHAR_LIMIT)."""
    query = base.replace("_"," ")
    texts, total = [], 0
    try:
        docs = vectordb.similarity_search(query=query, k=k, filter={"doc_id": base})
    except Exception:
        docs = []
    for d in docs or []:
        t=(d.page_content or "").strip()
        if not t: continue
        if total+len(t)>CTX_CHAR_LIMIT: break
        texts.append(t); total+=len(t)
    full_text = read_full_snippet(snippets_dir, base)
    if full_text:
        remain = CTX_CHAR_LIMIT - total
        if remain>0:
            texts.append(full_text[:remain]); total+=min(remain, len(full_text))
    if debug:
        debugln(f"[DEBUG] paper={base} | retrieved={len(docs or [])} | context_chars={total}", debug)
        if texts: debugln(f"[DEBUG] context preview:\n{texts[0][:400].replace('\n',' ')}\n", debug)
    return "\n\n".join(texts)

def load_local_llm(model_name: str, max_new_tokens: int=512):
    println("Loading local model…")
    tok=AutoTokenizer.from_pretrained(model_name)
    mdl=AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    gen=pipeline(task="text-generation", model=mdl, tokenizer=tok, device_map="auto",
                 return_full_text=False, max_new_tokens=max_new_tokens, do_sample=False,
                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    if getattr(mdl,"hf_device_map",None): println(f"Device map: {mdl.hf_device_map}")
    return gen

LLM_SYSTEM_INSTRUCTIONS = (
    "You are an expert annotator extracting exact metadata from the provided context. "
    "Return ONLY a single JSON object with the exact keys and types requested. "
    "Do not include explanations or extra text."
)
LLM_JSON_SCHEMA = {
    "file":"string","modality":"string or list (X-Ray, CT, MRI, Ultrasound, Mixed, Not reported)",
    "datasets_train":"string or list (e.g., MIMIC-CXR, IU X-ray, NIH ChestX-ray14, Open-i, REFLACX, CXR-RePaiR, CheXpert, PadChest)",
    "datasets_eval":"string or list","paired":"boolean or 'Not reported'","vlm":"boolean or 'Not reported'",
    "model":"string","class":"string","task":"string","vision_encoder":"string or list","language_decoder":"string or list",
    "fusion":"string or list","objectives":"list or 'Not reported'","family":"string","rag":"boolean or 'Not reported'",
    "metrics":"list or 'Not reported'",
}
def make_llm_prompt(context: str, filename: str)->str:
    schema_str=json.dumps(LLM_JSON_SCHEMA, indent=2)
    return (
        f"{LLM_SYSTEM_INSTRUCTIONS}\n\nFILENAME: {filename}\n\n"
        "CONTEXT (verbatim, do not ignore):\n---------------------------------\n"
        f"{context}\n---------------------------------\n\n"
        "Extract ONLY what is explicitly supported by the context. If unknown, use exactly 'Not reported'.\n"
        "Respond with ONE JSON object (no prose, no markdown), with keys exactly:\n"
        f"{schema_str}\n\nRules:\n- Return JSON only, no backticks.\n- Prefer short model names.\n- Do NOT invent values.\n"
    )
def extract_json_block(s: str)->Optional[str]:
    s=re.sub(r"^```(?:json)?|```$","",s.strip(), flags=re.I|re.M)
    i=s.find("{"); 
    if i==-1: return None
    depth=0
    for j,ch in enumerate(s[i:], start=i):
        if ch=="{": depth+=1
        elif ch=="}":
            depth-=1
            if depth==0: return s[i:j+1]
    return None
def parse_llm_json(raw: str)->Dict[str,Any]:
    block=extract_json_block(raw)
    if not block: raise ValueError("No JSON object found in LLM output.")
    return json.loads(block)
def llm_generate(gen, prompt:str)->str:
    out=gen(prompt, max_new_tokens=None)
    if isinstance(out,list) and out and "generated_text" in out[0]: return out[0]["generated_text"]
    if isinstance(out,list) and out and "text" in out[0]: return out[0]["text"]
    return str(out)

# ---- ensure_record / backfills / gating (identical to v14_mmr) ----
# To keep this message concise, the implementation below is copied 1:1 from v14_mmr.
# (If you need me to inline it again verbatim, say the word.)
def _present(value: str, ctx: str, patt: dict)->bool:
    if value=="Not reported": return False
    pat=patt.get(value)
    return re.search(pat, ctx, re.I) is not None if pat else re.search(re.escape(value), ctx, re.I) is not None

def ensure_record(base: str, data: Dict[str,Any], context: str, strict: bool=False)->Dict[str,str]:
    # The body here is identical to v14_mmr.ensure_record(...)
    # --- START COPY FROM v14_mmr ---
    modality = canon_modality(data.get("modality"))
    datasets_train = canon_list(data.get("datasets_train"))
    datasets_eval  = canon_list(data.get("datasets_eval"))
    paired = canon_bool(data.get("paired"))
    vlm    = canon_bool(data.get("vlm"))
    rag    = canon_bool(data.get("rag"))
    model_name = canon_model(data.get("model"), context)
    arch_class = "Not reported" if data.get("class") is None else str(data.get("class")).strip() or "Not reported"
    task       = "Not reported" if data.get("task")  is None else str(data.get("task")).strip()  or "Not reported"
    vision_enc = "Not reported" if data.get("vision_encoder") is None else (", ".join(v.strip() for v in data.get("vision_encoder")) if isinstance(data.get("vision_encoder"), list) else str(data.get("vision_encoder")).strip() or "Not reported")
    lang_dec   = "Not reported" if data.get("language_decoder") is None else (", ".join(v.strip() for v in data.get("language_decoder")) if isinstance(data.get("language_decoder"), list) else str(data.get("language_decoder")).strip() or "Not reported")
    fusion     = "Not reported" if data.get("fusion") is None else (", ".join(v.strip() for v in data.get("fusion")) if isinstance(data.get("fusion"), list) else str(data.get("fusion")).strip() or "Not reported")
    objectives = data.get("objectives"); metrics = data.get("metrics")
    if (not objectives) or (isinstance(objectives,str) and objectives.strip().lower()=="not reported"):
        objectives = extract_tokens_from_text(context, OBJECTIVE_TOKENS)
    if (not metrics) or (isinstance(metrics,str) and metrics.strip().lower()=="not reported"):
        metrics = extract_tokens_from_text(context, METRIC_TOKENS)
    objectives = canon_list(objectives); metrics = canon_list(metrics)
    family = canon_family(data.get("family"))
    def _datasets_from_ctx(ctx:str)->str:
        hits=[name for name,pat in DATASET_EVIDENCE.items() if re.search(pat, ctx, re.I)]
        hits=sorted(set(hits)); return ", ".join(hits) if hits else "Not reported"
    if datasets_train=="Not reported" or not datasets_train.strip():
        datasets_train=_datasets_from_ctx(context)
    if datasets_eval=="Not reported" or not datasets_eval.strip():
        datasets_eval=datasets_train
    if modality=="Not reported":
        cx=context.lower()
        if re.search(r"\b(chest )?x[- ]?ray|cxr\b", cx): modality="X-Ray"
        elif re.search(r"\bmri\b", cx): modality="MRI"
        elif re.search(r"\bct\b", cx): modality="CT"
        elif re.search(r"\bultrasound\b", cx): modality="Ultrasound"
        elif any(ds in datasets_train for ds in ["IU X-ray","MIMIC-CXR","NIH ChestX-ray14","Open-i","REFLACX","CXR-RePaiR","CheXpert","PadChest"]): modality="X-Ray"
    def _canon_from_aliases(text:str, alias_map:Dict[str,str])->List[str]:
        uniq=[]
        for pat,canon in alias_map.items():
            if re.search(pat, text, re.I) and canon not in uniq: uniq.append(canon)
        return uniq
    if vision_enc=="Not reported":
        enc=_canon_from_aliases(context, VISION_ENCODER_ALIASES)
        if enc: vision_enc=", ".join(enc)
    if lang_dec=="Not reported":
        dec=_canon_from_aliases(context, LANG_DECODER_ALIASES)
        if dec: lang_dec=", ".join(dec)
    if paired=="Not reported":
        paired="Yes" if re.search(r"\bimage[- ]?report pairs?\b|\bpaired\b", context, re.I) else "Not reported"
    if vlm=="Not reported":
        if (vision_enc!="Not reported" and lang_dec!="Not reported") or family in {"ALBEF","BLIP","BLIP-2","LLaVA","Flamingo","CLIP"} or re.search(r"\bvision[- ]?language\b|\bVLM\b", context, re.I):
            vlm="Yes"
    if rag=="Not reported":
        rag="Yes" if re.search(r"\bretrieval[- ]?augmented\b|\bRAG\b|\bretriev", context, re.I) else "Not reported"
    if strict:
        if vision_enc!="Not reported":
            vs=[v.strip() for v in vision_enc.split(",")]
            vision_enc=", ".join([v for v in vs if _present(v, context, EVIDENCE_PATTERNS)]) or "Not reported"
        if lang_dec!="Not reported":
            ls=[v.strip() for v in lang_dec.split(",")]
            lang_dec=", ".join([v for v in ls if _present(v, context, EVIDENCE_PATTERNS)]) or "Not reported"
        if family!="Not reported" and not _present(family, context, EVIDENCE_PATTERNS): family="Not reported"
        if model_name!="Not reported" and not re.search(re.escape(model_name), context, re.I):
            keep=False
            for pat,canon in MODEL_CANON.items():
                if canon==model_name and re.search(pat, context, re.I): keep=True; break
            if not keep: model_name="Not reported"
        if fusion!="Not reported":
            fs=[v.strip() for v in fusion.split(",")]
            fusion=", ".join([v for v in fs if _present(v, context, EVIDENCE_PATTERNS)]) or "Not reported"
        if arch_class!="Not reported" and not _present(arch_class, context, EVIDENCE_PATTERNS): arch_class="Not reported"
        if task!="Not reported" and not _present(task, context, EVIDENCE_PATTERNS): task="Not reported"
        if rag=="Yes" and not _present("RAG", context, EVIDENCE_PATTERNS): rag="Not reported"
        def _gate(ds:str)->str:
            if ds=="Not reported": return ds
            kept=[]
            for token in [t.strip() for t in ds.split(",") if t.strip()]:
                pat=DATASET_EVIDENCE.get(token, re.escape(token))
                if re.search(pat, context, re.I): kept.append(token)
            return ", ".join(kept) if kept else "Not reported"
        datasets_train=_gate(datasets_train); datasets_eval=_gate(datasets_eval)
    return {
        "File": base,"Modality": modality,"Datasets (train)": datasets_train,"Datasets (eval)": datasets_eval,
        "Paired": paired,"VLM?": vlm,"Model": model_name,"Class": arch_class,"Task": task,"Vision Enc": vision_enc,
        "Lang Dec": lang_dec,"Fusion": fusion,"Objectives": objectives,"Family": family,"RAG": rag,"Metrics(primary)": metrics,
    }
# --- END COPY ---

def run_regex_backend(base:str, context:str, strict:bool, debug:bool)->Dict[str,str]:
    # Same as v14_mmr
    modality="X-Ray" if re.search(r"\b(chest\s*)?x[- ]?ray\b", context, re.I) else "Not reported"
    ds=sorted({name for name,pat in DATASET_EVIDENCE.items() if re.search(pat, context, re.I)})
    datasets=", ".join(ds) if ds else "Not reported"
    objectives=extract_tokens_from_text(context, OBJECTIVE_TOKENS)
    metrics=extract_tokens_from_text(context, METRIC_TOKENS)
    data={"file":base,"modality":modality,"datasets_train":datasets,"datasets_eval":datasets,"paired":"Not reported","vlm":"Not reported",
          "model":"Not reported","class":"Not reported","task":"report-generation" if re.search(r"report[- ]?generation", context, re.I) else "Not reported",
          "vision_encoder":"ViT" if re.search(r"\bViT\b|Vision Transformer", context, re.I) else "Not reported",
          "language_decoder":"LLaMA" if re.search(r"\bLLaMA\b", context, re.I) else "Not reported",
          "fusion":"cross-attention" if re.search(r"cross[- ]?attention", context, re.I) else "Not reported",
          "objectives":objectives or "Not reported","family":"ALBEF" if re.search(r"\bALBEF\b", context, re.I) else "Not reported",
          "rag":"Yes" if re.search(r"\bRAG\b|retrieval[- ]?augmented", context, re.I) else "Not reported","metrics":metrics or "Not reported"}
    return ensure_record(base, data, context, strict)

def run_llm_backend(base:str, context:str, gen, retries:int, debug:bool, strict:bool)->Tuple[Optional[Dict[str,str]], Optional[str]]:
    prompt=make_llm_prompt(context, base); attempt=0; last_err=None
    while attempt<=retries:
        attempt+=1
        try:
            raw=llm_generate(gen, prompt); debugln(f"[DEBUG] raw LLM text for {base} (attempt {attempt}):\n{raw}\n", debug)
            data=parse_llm_json(raw); rec=ensure_record(base, data, context, strict); return rec, None
        except Exception as e:
            last_err=str(e); time.sleep(0.2)
    return None, last_err

def main():
    ap=argparse.ArgumentParser(description="Extract PRISMA/VLM metadata (v14 Similarity).")
    ap.add_argument("--snippets-dir", required=True)
    ap.add_argument("--backend", choices=["llm","regex"], default="llm")
    ap.add_argument("--local-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--strict-evidence", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args=ap.parse_args()

    snippets_dir=Path(args.snippets_dir)
    if not snippets_dir.exists(): println(f"ERROR: snippets dir not found: {snippets_dir}"); sys.exit(1)
    println("▶ Extracting papers…")

    try:
        vectordb=build_vectorstore()
    except Exception as e:
        vectordb=None; debugln(f"[DEBUG] Failed to open vectorstore, will use full snippets only: {e}", args.debug)

    gen=None
    if args.backend=="llm":
        gen=load_local_llm(args.local_model, max_new_tokens=args.max_new_tokens)

    bases=list_papers(snippets_dir)
    if not bases: println("No snippet files found."); sys.exit(1)

    records=[]
    for base in bases:
        if vectordb:
            context=retrieve_context_similarity(vectordb, base, snippets_dir, args.k, args.debug)
        else:
            context=read_full_snippet(snippets_dir, base) or ""
            if args.debug and context: debugln(f"[DEBUG] {base}: fallback full snippet (chars={len(context)})", args.debug)

        if args.backend=="regex" or not gen:
            rec=run_regex_backend(base, context or "", args.strict_evidence, args.debug); records.append(rec); println(f"✓ {base}"); continue
        rec, err=run_llm_backend(base, context or "", gen, args.retries, args.debug, args.strict_evidence)
        if rec: records.append(rec); println(f"✓ {base}")
        else:
            debugln(f"[DEBUG] LLM JSON failed for {base}: {err}. Falling back to regex.", args.debug)
            rec=run_regex_backend(base, context or "", args.strict_evidence, args.debug); records.append(rec); println(f"✓ {base} (regex fallback)")

    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); [w.writerow(r) for r in records]
    with open(args.out_md,"w",encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n"); f.write("|" + "---|"*len(COLUMNS) + "\n")
        for r in records: f.write("| " + " | ".join([r.get(c,"Not reported") for c in COLUMNS]) + " |\n")
    println(f"✅ Wrote {args.out_md} and {args.out_csv}")

if __name__=="__main__": main()
