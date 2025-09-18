from graphviz import Digraph
from pathlib import Path

# ------------------------------------------------------------------
# Output locations
# ------------------------------------------------------------------
OUTDIR = Path(__file__).resolve().parent
BASENAME = "fig_method_overview_aligned"

# ------------------------------------------------------------------
# Graph setup (vertical stack, aligned)
# ------------------------------------------------------------------
g = Digraph("prime_overview_aligned", format="pdf")
g.attr(
    rankdir="TB",          # Top -> Bottom stack
    bgcolor="white",
    splines="ortho",
    concentrate="true",
    nodesep="0.25",
    ranksep="0.55",
    dpi="300"
)

# Common style
node_style = dict(
    shape="box",
    style="rounded,filled",
    penwidth="1.8",
    fontname="Helvetica-Bold",
    fontsize="20",
    margin="0.30,0.22",
    color="#2F3A3D"
)
edge_style = dict(color="#2F3A3D", penwidth="1.6", arrowsize="0.9")
g.attr("node", **node_style)
g.attr("edge", **edge_style)

# Helper to add a dashed cluster with a vertical chain of nodes
def vertical_cluster(name, label, color, fill, nodes):
    c = Digraph(name)
    c.attr(label=label, fontsize="22", color=color, penwidth="2", style="dashed,rounded")
    # make them align nicely by keeping similar widths (short labels wrapped)
    prev = None
    for nid, text in nodes:
        c.node(nid, text, fillcolor=fill)
        if prev is not None:
            c.edge(prev, nid)
        prev = nid
    return c

# ------------------------------------------------------------------
# 1) Corpus & Data Prep
# ------------------------------------------------------------------
corpus = vertical_cluster(
    "cluster_corpus",
    "Corpus & Data Prep",
    "#5B8FF9",
    "#EFF5FF",
    [
        ("corpus",  "PRISMA selection\n(84 studies)"),
        ("pdf2txt", "PDF → text\n(snippets)"),
        ("embed",   "Embedding index\n(SentenceTransformer)"),
    ]
)
g.subgraph(corpus)

# ------------------------------------------------------------------
# 2) ROOT
# ------------------------------------------------------------------
root = vertical_cluster(
    "cluster_root",
    "ROOT (v1–v7): Baseline feasibility",
    "#52C41A",
    "#F0FFF0",
    [
        ("root_regex", "Regex + early prompts"),
        ("root_retr",  "Snippet retrieval (v3)\n(top-k evidence)"),
        ("root_json",  "JSON stability & retries\n(v4–v7)"),
    ]
)
g.subgraph(root)

# ------------------------------------------------------------------
# 3) CORE
# ------------------------------------------------------------------
core = vertical_cluster(
    "cluster_core",
    "CORE (v8–v17): Schema stabilization",
    "#FA8C16",
    "#FFF7E6",
    [
        ("core_vlm", "VLM-specific fields\n(Family, RAG)"),
        ("core_json","Strict JSON validation"),
        ("core_full","Retrieval fallback to full-text\nwhen snippet evidence is sparse"),
    ]
)
g.subgraph(core)

# ------------------------------------------------------------------
# 4) PRIME
# ------------------------------------------------------------------
prime = vertical_cluster(
    "cluster_prime",
    "PRIME (v18–v22): Hybrid & ensemble",
    "#9254DE",
    "#F9F0FF",
    [
        ("v18","PRIME-v18:\nLLM + RAG (high recall)"),
        ("v21","PRIME-v21:\nCascade (LLM → deterministic aliasing/filler)"),
        ("v22","PRIME-v22:\nEnsemble fusion (rule-based resolution)"),
    ]
)
g.subgraph(prime)

# ------------------------------------------------------------------
# 5) Evaluation & Reporting
# ------------------------------------------------------------------
evalc = vertical_cluster(
    "cluster_eval",
    "Evaluation & Reporting",
    "#40A9FF",
    "#E6F7FF",
    [
        ("avail",  "Availability masks\n(field present?)"),
        ("metrics","Two-stage metrics:\nAvailability-adjusted coverage, Yield"),
        ("diag",   "Diagnostics:\nMismatches, true missing"),
    ]
)
g.subgraph(evalc)

# ------------------------------------------------------------------
# Cross-cluster sequence arrows (left→right logical flow, stacked top→bottom)
# ------------------------------------------------------------------
g.edge("embed", "root_regex")
g.edge("root_json", "core_vlm")
g.edge("core_full", "v18")
g.edge("v22", "avail")

# ------------------------------------------------------------------
# Export SVG/PDF/PNG
# ------------------------------------------------------------------
for fmt in ("svg", "pdf", "png"):
    g.render(str(OUTDIR / f"{BASENAME}.{fmt}"), format=fmt, cleanup=True)

print("Saved:")
for x in ("svg","pdf","png"):
    print("-", OUTDIR / f"{BASENAME}.{x}")
