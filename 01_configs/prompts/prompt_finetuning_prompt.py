#!/usr/bin/env python3
import argparse
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

def parse_prompt_file(path):
    """
    Splits your fill_blanks_with_snippets.prompt into:
      - header: everything up to the first "### *.pdf"
      - sections: { "Foo.pdf": "<200 lines of text>", ... }
    """
    header_lines = []
    sections = {}
    current_pdf = None
    in_snippet = False
    snippet = []

    pdf_re = re.compile(r"^### (.+\.pdf)\s*$")
    fence_re = re.compile(r"^```")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # start of a section
            m = pdf_re.match(line)
            if m:
                current_pdf = m.group(1)
                in_snippet = False
                continue

            # header accumulation
            if current_pdf is None:
                header_lines.append(line.rstrip("\n"))
                continue

            # detect snippet opening
            if line.startswith("```text"):
                in_snippet = True
                snippet = []
                continue

            # collect snippet
            if in_snippet:
                if fence_re.match(line):
                    # end of snippet
                    sections[current_pdf] = "\n".join(snippet)
                    current_pdf = None
                    in_snippet = False
                else:
                    snippet.append(line.rstrip("\n"))

    return "\n".join(header_lines), sections

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-file",   required=True)
    p.add_argument("--model",         required=True)
    p.add_argument("--output-file",   required=True)
    args = p.parse_args()

    header, sections = parse_prompt_file(args.prompt_file)
    if not sections:
        print("❌ No PDFs found in prompt file.")
        return

    print(f"ℹ️  Loading model {args.model} …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    )
    model.eval()

    with open(args.output_file, "w", encoding="utf-8") as fout:
        # write header once
        fout.write(header + "\n\n")

        for pdf, snippet in sections.items():
            print(f"▶ Processing {pdf} …")
            prompt = (
                header + "\n\n"
                f"### {pdf}\n"
                "```text\n"
                f"{snippet}\n"
                "```\n\n"
                "Please fill in the missing “Vision Encoder”, “Language Decoder” and “Fusion Strategy” columns for this paper.\n"
            )

            # tokenize + move to GPU
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(device)

            # greedy generation
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            # decode only the newly generated portion
            gen_text = tokenizer.decode(
                out_ids[0, inputs["input_ids"].size(-1):],
                skip_special_tokens=True,
            ).strip()

            fout.write(gen_text + "\n\n")

    print(f"✅ Done!  Output written to {args.output_file}")

if __name__ == "__main__":
    main()
