# figures/01_methods_overview/make_fig_variant_pathway.py
from graphviz import Digraph
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_SVG = HERE / "fig_variant_pathway.svg"
OUT_PDF = HERE / "fig_variant_pathway.pdf"
OUT_PNG = HERE / "fig_variant_pathway.png"

# --- palette (soft, print-friendly) ---
C_TEXT = "#2E3138"
C_EDGE = "#42464D"

C_V18_FILL = "#E9F2FF"   # soft blue
C_V19_FILL = "#FFEAE8"   # soft coral
C_V20_FILL = "#EFE8FF"   # soft violet
C_TUNE_FILL = "#EAF7ED"  # soft green
C_RES_FILL  = "#E6F7FB"  # soft cyan

C_V18_LINE = "#6DA3FF"
C_V19_LINE = "#FF9A8F"
C_V20_LINE = "#9C88FF"
C_TUNE_LINE = "#6FC07A"
C_RES_LINE  = "#4EC3D8"

# --- base graph ---
g = Digraph("variant_pathway", format="svg")
g.attr(rankdir="LR", splines="spline", dpi="192", fontsize="18", fontname="Helvetica")
g.attr("node",
       shape="box",
       style="rounded,filled,setlinewidth(1.8)",
       penwidth="1.8",
       fontname="Helvetica",
       fontsize="18",
       color=C_TEXT,
       fillcolor="white",
       margin="0.20,0.12")
g.attr("edge",
       color=C_EDGE,
       penwidth="1.6",
       arrowsize="0.9")

# ---------- clusters to guide the eye ----------
# v18 backbone
with g.subgraph(name="cluster_v18") as c:
    c.attr(label="Backbone (v18)", labelloc="t", labeljust="c",
           fontsize="20", fontname="Helvetica-Bold",
           color=C_V18_LINE, style="rounded,dashed")
    c.node("v18",
           "v18\nRAG + LLM\n(strict JSON schema)",
           fillcolor=C_V18_FILL, color=C_V18_LINE)

# v19 cascade + its knob
with g.subgraph(name="cluster_v19") as c:
    c.attr(label="Cascade (v19)", labelloc="t", labeljust="c",
           fontsize="20", fontname="Helvetica-Bold",
           color=C_V19_LINE, style="rounded,dashed")
    c.node("v19",
           "v19\nCascade (LLM → deterministic filler)",
           fillcolor=C_V19_FILL, color=C_V19_LINE)
    c.node("filler",
           "Filler strength\n(alias coverage, NA rules)",
           fillcolor=C_TUNE_FILL, color=C_TUNE_LINE)

# v20 ensemble + policy
with g.subgraph(name="cluster_v20") as c:
    c.attr(label="Ensemble (v20) → Resolution", labelloc="t", labeljust="c",
           fontsize="20", fontname="Helvetica-Bold",
           color=C_V20_LINE, style="rounded,dashed")
    c.node("v20",
           "v20\nEnsemble fusion (rule-based)",
           fillcolor=C_V20_FILL, color=C_V20_LINE)
    c.node("policy",
           "Resolution policy\n(field-wise precedence)",
           fillcolor=C_RES_FILL, color=C_RES_LINE)

# tuning knob for v18
g.node("retrieval",
       "Retrieval depth\n(top-k)",
       fillcolor=C_TUNE_FILL, color=C_TUNE_LINE)

# ---------- edges (primary flow bold, auxiliaries dashed) ----------
# main pipeline
g.edge("v18", "v19")
g.edge("v19", "v20")
g.edge("v20", "policy", color=C_RES_LINE)

# auxiliary controls
g.edge("v18", "retrieval", style="dashed", color=C_TUNE_LINE)
g.edge("v19", "filler", style="dashed", color=C_TUNE_LINE)

# ---------- render ----------
g.render(filename=OUT_SVG.stem, directory=str(HERE), format="svg", cleanup=True)
print(f"Saved: {OUT_SVG}")
g.render(filename=OUT_PDF.stem, directory=str(HERE), format="pdf", cleanup=True)
print(f"Saved: {OUT_PDF}")
g.render(filename=OUT_PNG.stem, directory=str(HERE), format="png", cleanup=True)
print(f"Saved: {OUT_PNG}")
