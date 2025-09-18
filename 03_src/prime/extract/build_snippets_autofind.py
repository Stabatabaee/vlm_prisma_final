#!/usr/bin/env python3
"""
Auto-find PDFs and build snippet .txt files (one per PDF) for the v18–v22 pipeline.

- If --pdf-dir is valid and has PDFs, use it (recursively).
- Otherwise, scan from CWD for *.pdf/*.PDF, skipping junk dirs.
- Output: <snippets_dir>/<pdf_stem>.txt

Usage:
  python build_snippets_autofind.py \
      --snippets-dir gold_snippets \
      --min-chars 500 \
      [--pdf-dir VLM_PRISMA/pdfs]   # optional

Install extractors (if needed):
  pip install pymupdf pdfminer.six
"""
import argparse, os, re, sys, pathlib, unicodedata

SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".mypy_cache", ".pytest_cache"}

def try_extract_pymupdf(pdf_path):
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)
    except Exception:
        return None

def try_extract_pdfminer(pdf_path):
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return None
    try:
        return extract_text(pdf_path)
    except Exception:
        return None

def clean_text(t: str) -> str:
    if not t: return ""
    t = unicodedata.normalize("NFC", t)
    t = "".join(ch for ch in t if (ch >= " " or ch in "\n\t"))
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)       # de-hyphenate at line breaks
    t = re.sub(r"\n{3,}", "\n\n", t)             # collapse blank lines
    t = re.sub(r"[ \t]{2,}", " ", t)             # squeeze spaces
    return t.strip()

def list_pdfs_under(root: pathlib.Path):
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"])

def scan_repo_for_pdfs(cwd: pathlib.Path):
    pdfs = []
    for base, dirs, files in os.walk(cwd):
        # skip junk dirs
        parts = set(pathlib.Path(base).parts)
        if parts & SKIP_DIRS:
            dirs[:] = []  # don't descend further
            continue
        # small speed-up: prefer dirs likely to have PDFs
        # but still scan all (since user’s layout is unknown)
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(pathlib.Path(base) / fn)
    return sorted(pdfs)

def maybe_guess_pdf_root(cwd: pathlib.Path):
    # Try common subpaths if they exist
    candidates = [
        cwd / "VLM_PRISMA" / "pdfs",
        cwd / "VLM_PRISMA" / "PDFs",
        cwd / "VLM_PRISMA",
        cwd / "pdfs",
        cwd / "PDFs",
        cwd / "papers",
        cwd / "Papers",
        cwd / "data" / "pdfs",
        cwd / "datasets" / "pdfs",
    ]
    for c in candidates:
        if c.exists():
            hits = list_pdfs_under(c)
            if hits:
                return hits
    # Fallback: global scan (skips junk dirs)
    return scan_repo_for_pdfs(cwd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", help="Optional: directory to search for PDFs (recursive). If missing/empty, auto-scan from CWD.")
    ap.add_argument("--snippets-dir", required=True, help="Where to write one .txt per PDF.")
    ap.add_argument("--min-chars", type=int, default=500, help="Skip outputs shorter than this many characters.")
    args = ap.parse_args()

    cwd = pathlib.Path.cwd().resolve()
    out_dir = pathlib.Path(args.snippets_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose discovery strategy
    if args.pdf_dir:
        pdf_root = pathlib.Path(args.pdf_dir).resolve()
        if pdf_root.exists():
            pdfs = list_pdfs_under(pdf_root)
        else:
            print(f"⚠️  Provided --pdf-dir '{pdf_root}' does not exist. Auto-scanning from {cwd}…")
            pdfs = maybe_guess_pdf_root(cwd)
    else:
        pdfs = maybe_guess_pdf_root(cwd)

    if not pdfs:
        print(f"❌ No PDFs found under {args.pdf_dir or cwd}. "
              f"Try a different --pdf-dir or verify the repo path.")
        # Helpful hint
        print("Hint: run\n  find . -type f -iname '*.pdf' | head -n 10\nfrom your repo root to confirm locations.")
        sys.exit(1)

    # Show a brief preview
    print("Found PDFs (showing up to 10):")
    for p in pdfs[:10]:
        print("  -", p)
    if len(pdfs) > 10:
        print(f"  … and {len(pdfs) - 10} more")

    made = 0
    skipped = 0
    errors = 0
    for pdf in pdfs:
        stem = pdf.stem
        out_txt = out_dir / f"{stem}.txt"
        if out_txt.exists():
            skipped += 1
            continue

        text = try_extract_pymupdf(str(pdf)) or try_extract_pdfminer(str(pdf))
        if not text:
            print(f"❌ Failed to extract: {pdf}")
            errors += 1
            continue

        text = clean_text(text)
        if len(text) < args.min_chars:
            print(f"⚠️  Too short after cleaning ({len(text)} chars), skipping: {pdf}")
            skipped += 1
            continue

        try:
            out_txt.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"❌ Write error for {out_txt}: {e}")
            errors += 1
            continue

        made += 1

    # Also count existing .txt files (pre-existing coverage)
    existing_txts = len(list(out_dir.glob("*.txt")))

    print("\n=== Snippet build summary ===")
    print(f"Scan root         : {args.pdf_dir or cwd}")
    print(f"Snippets dir      : {out_dir}")
    print(f"PDFs discovered   : {len(pdfs)}")
    print(f"TXT created now   : {made}")
    print(f"Skipped (exists/short): {skipped}")
    print(f"Errors            : {errors}")
    print(f"Total .txt present: {existing_txts}")
    if existing_txts >= 84:
        print("✓ You have snippet coverage for at least 84 papers.")
    else:
        print(f"ℹ️ Currently {existing_txts} text files present; target is 84.")

if __name__ == "__main__":
    main()
