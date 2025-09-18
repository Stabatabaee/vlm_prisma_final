#!/usr/bin/env python3
"""
extract_metadata.py

Run final metadata extraction over all eligible PDFs using GPT-4 Turbo and
Llama-3.3-70B-Instruct, with your refined field prompts.
"""

import os
import glob
import json
import argparse
import openai
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_papers(folder):
    docs = {}
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt: pages.append(txt)
        docs[doc_id] = "\n".join(pages)
    return docs

def init_openai(key):
    openai.api_key = key

def extract_openai(model, prompt, text):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Extract structured metadata from this scientific paper."},
            {"role": "user",   "content": prompt + "\n\n" + text}
        ],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

def init_llama(name):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    m   = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
    return pipeline("text-generation", model=m, tokenizer=tok,
                    max_new_tokens=256, temperature=0.0, do_sample=False)

def extract_llama(gen, prompt, text):
    out = gen(prompt + "\n\n" + text)[0]["generated_text"]
    return out.split("\n",1)[-1].strip()

def reconcile(a, b):
    # simple: prefer GPT-4 if nonempty, else Llama
    return a if a else b

def main(args):
    # load
    docs    = load_papers(args.docs_folder)
    prompts = json.load(open(args.prompts, 'r'))
    init_openai(args.openai_key)
    llama_gen = init_llama(args.local_model)

    results = {}
    for doc_id, text in docs.items():
        record = {}
        for field, prompt in prompts.items():
            a = extract_openai(args.openai_model, prompt, text)
            l = extract_llama(llama_gen, prompt, text)
            record[field] = reconcile(a, l)
        results[doc_id] = record
        print(f"Extracted {len(prompts)} fields for {doc_id}")

    # save JSON
    out_path = args.output or "metadata_extracted.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("All done! Results written to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs-folder", required=True, help="Folder of PDFs")
    p.add_argument("--prompts",     required=True, help="Refined prompts JSON")
    p.add_argument("--openai-key",  required=True, help="OpenAI API key")
    p.add_argument("--openai-model",default="gpt-4-turbo", help="OpenAI model")
    p.add_argument("--local-model", required=True, help="HF name for Llama-3.3-70B-Instruct")
    p.add_argument("--output",      help="Output JSON path")
    args = p.parse_args()
    main(args)
