from dotenv import load_dotenv
load_dotenv()



import os, json
from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DB_DIR = "chroma_db"
OUT_MD = "filled_papers_TEST.md"
OUT_CSV = "filled_papers_TEST.csv"

COLUMNS = ["File","Title","Authors","Modality","Datasets",
           "Model Name","Vision Encoder","Language Decoder","Fusion Strategy"]

FIELD_QUERIES = {
    "Title": "From the provided snippets only, extract the exact paper title.",
    "Authors": "From the snippets, list the authors as 'Surname et al.' or 'Surname, Surname, ...' if present.",
    "Modality": "Which imaging modality is used (e.g., X-Ray, CT, MRI)?",
    "Datasets": "List dataset names mentioned (e.g., MIMIC-CXR, IU X-ray, Open-i).",
    "Model Name": "What model name(s) are claimed (e.g., ALBEF, CXR-IRGen, R2Gen, PPKED)?",
    "Vision Encoder": "Which vision encoder is used (ViT, ResNet-50, CLIP, CNN)?",
    "Language Decoder": "Which language model/decoder is used (LLaMA, BERT, GPT-2, LSTM, Transformer, T5)?",
    "Fusion Strategy": "How are image & text fused (cross-attention, co-attention, multimodal encoder, early/late fusion, concatenation)?"
}

def load_db():
    embed = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=embed, collection_name="papers")

def get_llm():
    if os.getenv("OPENAI_API_KEY"): 
        # GPT-4o path
        return ChatOpenAI(model="gpt-4o", temperature=0)
    # local LLaMA 8B fallback
    name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mod = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", torch_dtype="auto", trust_remote_code=True
    )
    pipe = pipeline("text-generation", model=mod, tokenizer=tok, max_new_tokens=200, do_sample=False)
    return HuggingFacePipeline(pipeline=pipe)

SYSTEM_RULES = (
    "You are extracting fields for a PRISMA metadata table. "
    "Use ONLY the provided context. If a field is not present in the context, answer exactly 'Not reported'. "
    "Answer with a single short phrase, not a sentence."
)

def ask_field(llm, context: str, question: str) -> str:
    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    out = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    text = out.content if hasattr(out, "content") else str(out)
    ans = text.strip().splitlines()[0].strip()
    # trim any trailing punctuation noise
    return ans.strip(" .|`")

def collect_paper_ids(db) -> List[str]:
    # read distinct paper_ids from metadatas (Chroma API)
    # simplest approach: fetch all, then unique
    all_ids = set()
    for d in db.get(include=["metadatas"])["metadatas"]:
        if d and "paper_id" in d:
            all_ids.add(d["paper_id"])
    return sorted(all_ids)

def md_header():
    head = "| " + " | ".join(COLUMNS) + " |\n"
    sep  = "|" + "|".join(["---"]*len(COLUMNS)) + "|\n"
    return head + sep

def md_row(cells: List[str]) -> str:
    norm = [ (c if c and c.strip() else "Not reported") for c in cells ]
    # protect pipes
    norm = [c.replace("|","/").replace("`","").strip() for c in norm]
    return "| " + " | ".join(norm) + " |"

def main():
    db = load_db()
    llm = get_llm()
    paper_ids = collect_paper_ids(db)

    # write MD + CSV
    Path(OUT_MD).write_text(md_header(), encoding="utf-8")
    with open(OUT_CSV, "w", encoding="utf-8") as fcsv:
        fcsv.write(",".join(COLUMNS) + "\n")

    for pid in paper_ids:
        # per-paper retriever (MMR helps diversity)
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 36, "lambda_mult": 0.5, "filter": {"paper_id": pid}}
        )
        docs = retriever.get_relevant_documents("architecture, datasets, encoder/decoder, fusion")
        ctx = "\n\n---\n\n".join(d.page_content for d in docs)

        cells = [f"{pid}.pdf"]
        for col in COLUMNS[1:]:
            ans = ask_field(llm, ctx, FIELD_QUERIES[col])
            cells.append(ans if ans else "Not reported")

        # write MD
        with open(OUT_MD, "a", encoding="utf-8") as fmd:
            fmd.write(md_row(cells) + "\n")

        # write CSV
        with open(OUT_CSV, "a", encoding="utf-8") as fcsv:
            fcsv.write(",".join('"' + c.replace('"','""') + '"' for c in cells) + "\n")

        print(f"✓ {pid}")

    print(f"✅ Wrote {OUT_MD} and {OUT_CSV}")

if __name__ == "__main__":
    main()
