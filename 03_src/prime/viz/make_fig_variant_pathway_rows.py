from graphviz import Digraph
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
STEM = "fig_variant_pathway_rows"
SVG = OUTDIR / f"{STEM}.svg"
PDF = OUTDIR / f"{STEM}.pdf"
PNG = OUTDIR / f"{STEM}.png"

# ---------- Graph ----------
g = Digraph("variant_pathway_rows", format="png")
g.attr(rankdir="TB", splines="spline", fontname="Helvetica", fontsize="18", labelloc="t")

# ---------- helper styles ----------
def nstyle_fixed(boxcolor="#ffffff"):
    return dict(shape="box", style="rounded,filled", fontname="Helvetica",
                fontsize="14", width="3.0", height="0.85", fixedsize="true",
                margin="0.10,0.08", fillcolor=boxcolor)

def nstyle_flow(boxcolor="#ffffff"):
    return dict(shape="box", style="rounded,filled", fontname="Helvetica",
                fontsize="14", margin="0.10,0.08", fillcolor=boxcolor)

arrow_lbl = dict(fontname="Helvetica", fontsize="12")

# ---------- ROW 1 : v18 ----------
with g.subgraph(name="cluster_v18") as v18:
    v18.attr(label="v18: High-Recall RAG Extractor", color="#5577ff", style="dashed", fontsize="16")
    s = nstyle_fixed("#f2f2f2")
    v18.node("v18_corpus", "Corpus (PDF snippets)", **s)
    s = nstyle_fixed("#cce5ff")
    v18.node("v18_ret", "Retriever\n(embeddings + BM25)", **s)
    s = nstyle_fixed("#d9ead3")
    v18.node("v18_llm", "LLM Extractor\n(structured prompting)", **s)
    s = nstyle_fixed("#e2f0d9")
    v18.node("v18_json", "JSON Validator\n(schema, braces)", **s)
    s = nstyle_fixed("#ffffff")
    v18.node("v18_out", "v18 Output\n(strict evidence)", **s)
    v18.edges([("v18_corpus","v18_ret"), ("v18_ret","v18_llm"),
               ("v18_llm","v18_json"), ("v18_json","v18_out")])

# ---------- ROW 2 : v20 Deterministic Filler ----------
with g.subgraph(name="cluster_v20") as v20:
    v20.attr(label="v20: Deterministic Filler Module", color="#ffb84d", style="dashed", fontsize="16")
    s = nstyle_fixed("#f2f2f2")
    v20.node("v20_in", "Input JSON", **s)
    s = nstyle_fixed("#ffe5d0")
    v20.node("v20_alias", "Alias Canonicalizer\n(regex and lexicons)", **s)
    s = nstyle_fixed("#ffe6aa")
    v20.node("v20_backfill", "Heuristic Backfill\n(missing fields)", **s)
    s = nstyle_fixed("#e2f0d9")
    v20.node("v20_canon", "Canonical JSON\n(v20 hints and features)", **s)
    v20.edges([("v20_in","v20_alias"), ("v20_alias","v20_backfill"), ("v20_backfill","v20_canon")])

# Align row 2 directly below row 1
with g.subgraph() as r12:
    r12.attr(rank="same")
    r12.node("v18_anchor", "", shape="point", width="0.01", style="invis")
    r12.node("v20_anchor", "", shape="point", width="0.01", style="invis")

# ---------- ROW 3 : v21 Cascade Hybrid ----------
with g.subgraph(name="cluster_v21") as v21:
    v21.attr(label="v21: Cascade Hybrid (LLM → Deterministic)", color="#a07dff", style="dashed", fontsize="16")
    s = nstyle_fixed("#d9ead3")
    v21.node("v21_strict", "Retriever + LLM Extractor\n(strict mode)", **s)
    s = nstyle_fixed("#e2f0d9")
    v21.node("v21_val", "Validator", **s)
    s = nstyle_fixed("#ffe5d0")
    v21.node("v21_fill", "Deterministic Filler\n(v20 module)", **s)
    s = nstyle_fixed("#ffe6aa")
    v21.node("v21_gate", "Confidence Gate\n(fieldwise overwrite rules)", **s)
    s = nstyle_fixed("#e2f0d9")
    v21.node("v21_out", "v21 Output\n(balanced coverage)", **s)
    v21.edges([("v21_strict","v21_val"), ("v21_val","v21_fill"),
               ("v21_fill","v21_gate"), ("v21_gate","v21_out")])

# ---------- ROW 4 : v22 Ensemble Fusion ----------
with g.subgraph(name="cluster_v22") as v22:
    v22.attr(label="v22: Ensemble Fusion (exact architecture)", color="#99dd99", style="dashed", fontsize="16")
    s = nstyle_flow("#f2f2f2")
    v22.node("v22_norm", "Normalization and Alignment\n(alias pass, schema alignment)", **s)
    s = nstyle_flow("#fff2cc")
    v22.node("v22_conf", "Confidence Scoring\n(snippet overlap, rule signals)", **s)
    s = nstyle_flow("#ffe599")
    v22.node("v22_rules", "Consensus and Conflict Rules\n(exact match, prefer strict, fallback deterministic)", **s)
    s = nstyle_flow("#ddeedd")
    v22.node("v22_qc", "Quality Gate\n(schema and cross-field checks)", **s)
    s = nstyle_flow("#d4edda")
    v22.node("v22_final", "Final Table (v22)", **s)
    v22.edges([("v22_norm","v22_conf"), ("v22_conf","v22_rules"), ("v22_rules","v22_qc"), ("v22_qc","v22_final")])

# ---------- Connections between rows ----------
# From v18 output → v20 input, and → v21 validator
g.edge("v18_out", "v20_in", label="raw input", **arrow_lbl)
g.edge("v18_out", "v21_val", label="strict evidence", **arrow_lbl)

# Row 4 inputs
# Small “input” labels on the left
g.node("inpA", "Input A: v18 Output", shape="box", style="rounded", fontsize="12")
g.node("inpB", "Input B: v21 Output", shape="box", style="rounded", fontsize="12")
g.node("inpC", "Input C: v20 Hints", shape="box", style="rounded", fontsize="12")

g.edge("inpA", "v22_norm")
g.edge("inpB", "v22_norm")
g.edge("inpC", "v22_norm")

# Anchor those labels to the left of v22
with g.subgraph() as left_rank:
    left_rank.attr(rank="same")
    left_rank.node("inpA")
    left_rank.node("inpB")
    left_rank.node("inpC")
    left_rank.node("v22_norm")

# Connect sources to the inputs
g.edge("v18_out", "inpA", arrowhead="none")
g.edge("v21_out", "inpB", arrowhead="none")
g.edge("v20_canon", "inpC", arrowhead="none")

# ---------- Render ----------
g.render(str(SVG.with_suffix("")), format="svg", cleanup=True)
g.render(str(PDF.with_suffix("")), format="pdf", cleanup=True)
g.render(str(PNG.with_suffix("")), format="png", cleanup=True)
print(f"Saved: {SVG}, {PDF}, {PNG}")
