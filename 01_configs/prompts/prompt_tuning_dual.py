#!/usr/bin/env python3
"""
prompt_tuning_dual.py

Iteratively tune extraction prompts on a small set of PDF papers
against ground-truth annotations, using LLaMA-3.3-70B-Instruct.
This version:
  - Runs in stable FP16 on GPU.
  - Truncates each page to 256 tokens before calling the model.
  - Scans page-by-page, skipping any GPU errors to avoid crashes.
  - Attaches a GenerationConfig to the model rather than passing it to pipeline.
  - Normalizes expected values (joins lists into strings) before comparison.
"""
import os
import glob
import json
import argparse
import pdfplumber
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)

def load_papers_from_pdf(folder):
    """Return dict: {stem: [page1_text, page2_text, …]}."""
    docs = {}
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        stem = os.path.splitext(os.path.basename(path))[0]
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        docs[stem] = pages
    return docs


def init_local_llama(model_name):
    """
    Initialize text-generation pipeline in stable FP16 mode.
    Attach GenerationConfig to the model; let pipeline pick it up.
    Returns (pipeline, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, token=os.getenv("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    # Attach GenerationConfig for max_new_tokens, etc.
    model.generation_config = GenerationConfig(
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return gen, tokenizer


def extract_from_pages(gen, tokenizer, prompt, pages, max_tokens=128):
    """
    Scan each page; return the first non-empty extraction.
    Truncate to max_tokens tokens (default=128) and skip pages that error out.
    """
    """
    Scan each page; return the first non-empty extraction.
    Truncate to max_tokens and skip pages that error out.
    """
    for i, page_text in enumerate(pages, start=1):
        toks = tokenizer(
            page_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )["input_ids"][0]
        chunk = tokenizer.decode(toks, skip_special_tokens=True)
        print(f"  [Page {i}] → {len(toks)} tokens…", end="", flush=True)
        try:
            out = gen(prompt + "\n\n" + chunk)[0].get("generated_text", "")
        except Exception as e:
            print(f"  🚨 Error: {e.__class__.__name__}, skipping")
            continue
        answer = out.split("\n", 1)[-1].strip()
        if answer:
            print(f"  ✔️  “{answer}”")
            return answer
        print("  (no result)")
    return ""


def evaluate_prompts(docs, expected, prompts, gen, tokenizer):
    acc = {}
    total = len(expected)
    for field, prompt in prompts.items():
        correct = 0
        for doc_id, pages in docs.items():
            if doc_id not in expected or field not in expected[doc_id]:
                continue
            raw_val = expected[doc_id][field]
            # Normalize truth: join lists into string
            if isinstance(raw_val, list):
                truth = ", ".join(raw_val)
            else:
                truth = str(raw_val)
            truth = truth.strip()
            print(f"\n→ Extracting '{field}' from '{doc_id}':")
            pred = extract_from_pages(gen, tokenizer, prompt, pages)
            print(f"    → pred = {pred!r}, truth = {truth!r}")
            if pred == truth:
                correct += 1
        acc[field] = correct / total if total else 0.0
    return acc


def interactive_tune(args):
    docs = load_papers_from_pdf(args.docs_folder)
    prompts = json.load(open(args.prompts,'r',encoding='utf-8'))
    raw = json.load(open(args.expected,'r',encoding='utf-8'))
    expected = {
        os.path.splitext(e['file'])[0]:
        {k: v for k, v in e.items() if k != 'file'}
        for e in raw
    }

    print("→ PDFs found:   ", sorted(docs.keys()))
    print("→ Expected keys:", sorted(expected.keys()))

    gen, tokenizer = init_local_llama(args.local_model)

    while True:
        accs = evaluate_prompts(docs, expected, prompts, gen, tokenizer)
        to_refine = [f for f in prompts if accs.get(f,0) < args.threshold]
        if not to_refine:
            print(f"\n✅ All prompts ≥ {args.threshold:.0%}.")
            break

        print(f"\nFields below {args.threshold:.0%}:")
        for f in to_refine:
            print(f"  - {f}: {accs[f]:.2%}")

        for f in to_refine:
            print(f"\nCurrent prompt for '{f}':\n{prompts[f]}")
            new = input("Enter new prompt (ENTER to keep):\n> ").strip()
            if new:
                prompts[f] = new

        with open(args.prompts,'w',encoding='utf-8') as fp:
            json.dump(prompts, fp, indent=2)
        print("\nPrompts updated — re-evaluating.\n")

    out = os.path.splitext(args.prompts)[0] + '_refined.json'
    with open(out,'w',encoding='utf-8') as fp:
        json.dump(prompts, fp, indent=2)
    print("Saved refined prompts to", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs-folder", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--local-model", required=True)
    p.add_argument("--threshold", type=float, default=0.9)
    args = p.parse_args()
    interactive_tune(args)
