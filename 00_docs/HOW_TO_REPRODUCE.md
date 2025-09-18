# How to Reproduce Core Results

## Environment
- Python 3.10+ with requirements from the main repo (not bundled here).
- Graphviz installed for `.gv` → PNG/SVG figure builds.

## Typical flow
1) Ingest snippets from PDFs (toy example):
   - See `03_src/prime/ingest/*.py`
2) Extract & fill fields (LLM/RAG calls):
   - See `03_src/prime/extract/*.py`
3) Normalize & reconcile versions:
   - `03_src/prime/normalize/clean_extractions*.py`
4) Evaluate:
   - `03_src/prime/evaluate/evaluate_fields*.py`, `compare_v18_v22.py`, `fuse_results_v22.py`
5) Visualize figures (paper Figures 3–5):
   - `03_src/prime/viz/fig3_grouped_bar.py`
   - `03_src/prime/viz/fig4_error_taxonomy.py`
   - `03_src/prime/viz/fig5_wilson_cis.py`

Final artifacts produced in this kit are under `04_artifacts/`.
