#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from typing import Dict, List, Tuple, Iterable, Optional

# =========================
# Runtime config (set in CLI)
# =========================
PRIMARY_MODEL_ONLY: bool = False
FAMILY_ONTOLOGY: str = "named"  # 'named' or 'coarse'

# ========== small utils ==========

def lower_clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def norm_spaces(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*\+\s*", " + ", s)
    return s

def split_multi(s: str) -> List[str]:
    """
    Split multi-valued cells.
    Separators: comma, semicolon, plus, slash, ampersand, ' and ' (loose), vertical bar.
    """
    if not s:
        return []
    tmp = re.sub(r"[\uFF0C；;|/]", ",", s)          # alt commas/semicolons/slashes/pipes
    tmp = re.sub(r"\s*\+\s*", ",", tmp)             # plus
    tmp = re.sub(r"\s*&\s*", ",", tmp)              # ampersand
    tmp = re.sub(r"\s+and\s+", ",", tmp, flags=re.I)
    parts = [p.strip() for p in tmp.split(",")]
    return [p for p in parts if p]

def uniq_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

_MISSING_TOKENS = {
    "", "n/a", "na", "none", "null", "unk", "unknown",
    "not available", "not applicable", "not reported", "nr", "n/r"
}

def is_missing(s: str) -> bool:
    return lower_clean(s) in _MISSING_TOKENS

def yes_no_notreported(s: str) -> str:
    ls = lower_clean(s)
    if ls in _MISSING_TOKENS:
        return "Not reported"
    if ls in ("yes", "y", "true", "1", "present"):
        return "Yes"
    if ls in ("no", "n", "false", "0", "absent"):
        return "No"
    return norm_spaces(s)

def _norm(s: str) -> str:
    return (s or "").lower()

# ========== alias helpers ==========

def compile_aliases(pairs: List[Tuple[str, str]]) -> List[Tuple[re.Pattern, str]]:
    out = []
    for pat, canon in pairs:
        out.append((re.compile(pat, re.I), canon))
    return out

def apply_alias_once(token: str, alias: List[Tuple[re.Pattern, str]]) -> str:
    t = token.strip()
    if not t:
        return ""
    for rx, canon in alias:
        if rx.fullmatch(t) or rx.search(t):
            return canon
    return t

def canonicalize_list(cell: str, alias: List[Tuple[re.Pattern, str]]) -> str:
    if not cell:
        return ""
    toks = split_multi(cell)
    mapped = [apply_alias_once(t, alias) for t in toks]
    mapped = [m for m in mapped if m and lower_clean(m) != "not reported"]
    mapped = uniq_keep_order(mapped)
    return ", ".join(mapped)

# ========== Canonical dictionaries ==========

MODEL_ALIASES = compile_aliases([
    # GPT family
    (r"\bgpt[-\s]?2\b|distilgpt2|generative pre[-\s]?trained transformer\s*2", "GPT-2"),
    (r"\bgpt[-\s]?3\.5\b", "GPT-3.5"),
    (r"\bgpt[-\s]?3\b", "GPT-3"),
    (r"\bgpt[-\s]?4v?\b|gpt[-\s]?4[-\s]?vision|gpt-4o|gpt4o|gpt-4v", "GPT-4V"),
    (r"\bgpt[-\s]?4\b", "GPT-4"),

    # LLaMA / Vicuna / Mistral
    (r"\bllama[-\s]?3\b|\bmeta[-\s]?llama[-\s]?3\b", "LLaMA-3"),
    (r"\bllama[-\s]?2\b|\bmeta[-\s]?llama[-\s]?2\b", "LLaMA-2"),
    (r"\bllama\b|\bmeta[-\s]?llama\b", "LLaMA"),
    (r"\bvicuna\b", "Vicuna"),
    (r"\bmistral\b", "Mistral"),
    (r"\bmixtral\b", "Mixtral"),

    # PaLM / Qwen
    (r"\bmed[-\s]?palm(?:[-\s]?2)?\b", "Med-PaLM"),
    (r"\bpalm[-\s]?2\b|\bpalm\b", "PaLM"),
    (r"\bqwen[-\s]?2[-\s]?vl\b", "Qwen2-VL"),
    (r"\bqwen[-\s]?vl\b", "Qwen-VL"),
    (r"\bqwen\b", "Qwen"),

    # VLM families often cited as "Model"
    (r"\binstruct[-\s]?blip\b", "InstructBLIP"),
    (r"\bblip[-\s]?2\b|blip v?2\b", "BLIP-2"),
    (r"\bblip\b", "BLIP"),
    (r"\balbef\b", "ALBEF"),
    (r"\bflamingo\b", "Flamingo"),
    (r"\bllava[-\s]?med\b", "LLaVA-Med"),
    (r"\bllava\b", "LLaVA"),
    (r"\b(minigpt[-\s]?4)\b", "MiniGPT-4"),
    (r"\botter\b", "Otter"),
    (r"\bx[-\s]?ray[-\s]?gpt\b|xraygpt", "XrayGPT"),
    (r"\bpali[-\s]?x\b", "PaLI-X"),
    (r"\bpali\b", "PaLI"),
    (r"\bkosmos[-\s]?1\b", "Kosmos-1"),
    (r"\bsimvlm\b", "SimVLM"),
    (r"\bofa\b", "OFA"),

    # Classic NLP/seq models
    (r"\bb[eE][rR][tT]\b", "BERT"),
    (r"\broberta\b", "RoBERTa"),
    (r"\bt5\b", "T5"),
    (r"\blstm\b", "LSTM"),
    (r"\bgru\b", "GRU"),
    (r"\btransformer\b", "Transformer"),

    # Domain-specific
    (r"\bchexnet\b", "CheXNet"),
    (r"\beggca[-\s]?net\b", "EGGCA-Net"),
    (r"\btrmrg\b", "TrMRG"),
    (r"\bcxr[-\s]?irgen\b", "CXR-IRGen"),
])

VISION_ALIASES = compile_aliases([
    (r"\bvision transformer\b|\bvit\b", "ViT"),
    (r"\bvitb/?16\b|\bvit[-\s]?b/16\b", "ViT"),
    (r"\bvanilla image transformer\b", "ViT"),
    (r"\bswin\b|\bswin[-\s]?transformer\b", "Swin"),
    (r"\bclip\b", "CLIP"),
    (r"\bresnet[-\s]?50\b|resnet 50\b", "ResNet50"),
    (r"\bresnet[-\s]?101\b|resnet 101\b", "ResNet101"),
    (r"\bconvnext\b", "ConvNeXt"),
    (r"\befficientnet\b", "EfficientNet"),
    (r"\bdensenet\b", "DenseNet"),
    (r"\binception\b", "Inception"),
    (r"\bc?nn\b|\bcnn\b", "CNN"),
])

LANG_ALIASES = compile_aliases([
    (r"\bllama[-\s]?3\b", "LLaMA-3"),
    (r"\bllama[-\s]?2\b", "LLaMA-2"),
    (r"\bllama\b", "LLaMA"),
    (r"\bvicuna\b", "Vicuna"),
    (r"\bgpt[-\s]?2\b|distilgpt2", "GPT-2"),
    (r"\bgpt[-\s]?3\.5\b", "GPT-3.5"),
    (r"\bgpt[-\s]?3\b", "GPT-3"),
    (r"\bgpt[-\s]?4v?\b|gpt[-\s]?4o\b", "GPT-4V"),
    (r"\bt5\b", "T5"),
    (r"\bb[eE][rR][tT]\b", "BERT"),
    (r"\broberta\b", "RoBERTa"),
    (r"\btransformer\b", "Transformer"),
    (r"\blstm\b", "LSTM"),
    (r"\bgru\b", "GRU"),
])

# Note: we keep "conditioned" as a recognizable token, but will drop it in normalize_fusion
FUSION_ALIASES = compile_aliases([
    (r"\bcross[-\s]?attention\b", "cross-attention"),
    (r"\bco[-\s]?attention\b", "co-attention"),
    (r"\bself[-\s]?attention\b", "self-attention"),
    (r"\bcondition(ed|ing)?\b", "conditioned"),
    (r"\bconcat(enation)?\b", "concatenation"),
    (r"\blate[-\s]?fusion\b", "late fusion"),
    (r"\bearly[-\s]?fusion\b", "early fusion"),
    (r"\bgated\b|\bgating\b", "gated"),
    (r"\bfilm\b", "FiLM"),
])

OBJECTIVE_ALIASES = compile_aliases([
    (r"\bitm\b|image[-\s]?text matching", "ITM"),
    (r"\bitc\b|image[-\s]?text contrast(ive)?\b|contrastive", "contrastive"),
    (r"\bmlm\b|masked language model(ing)?\b", "MLM"),
    (r"\bmim\b|masked image model(ing)?\b", "MIM"),
    (r"\bmrm\b|masked region model(ing)?\b", "MRM"),
    (r"\bcaption(ing)?\b", "captioning"),
    (r"\bcoverage\b", "coverage"),
    # New patterns
    (r"\bcross[-\s]?entropy\b", "cross-entropy"),
    (r"\bnext sentence prediction\b|\bnsp\b", "NSP"),
    (r"\breinforcement learning\b|\brl\b", "RL"),
])

FAMILY_ALIASES = compile_aliases([
    (r"\balbef\b", "ALBEF"),
    (r"\bblip[-\s]?2\b|blip v?2\b", "BLIP-2"),
    (r"\bblip\b", "BLIP"),
    (r"\bflamingo\b", "Flamingo"),
    (r"\bllava[-\s]?med\b", "LLaVA-Med"),
    (r"\bllava\b", "LLaVA"),
    (r"\bminigpt[-\s]?4\b", "MiniGPT-4"),
    (r"\binstruct[-\s]?blip\b", "InstructBLIP"),
    (r"\botter\b", "Otter"),
    (r"\bx[-\s]?ray[-\s]?gpt\b|xraygpt", "XrayGPT"),
    (r"\bpali[-\s]?x\b", "PaLI-X"),
    (r"\bpali\b", "PaLI"),
    (r"\bkosmos[-\s]?1\b", "Kosmos-1"),
    (r"\bsimvlm\b", "SimVLM"),
    (r"\bofa\b", "OFA"),
    # We also treat CLIP as a "family" candidate when needed for Model preference
    (r"\bclip\b", "CLIP"),
])

# Family preference order for "primary model"
PRIMARY_FAMILY_ORDER = [
    "LLaVA", "BLIP-2", "BLIP", "ALBEF", "Flamingo", "MiniGPT-4",
    "InstructBLIP", "Otter", "XrayGPT", "PaLI-X", "PaLI",
    "Kosmos-1", "SimVLM", "OFA", "CLIP"
]

# Narrow RAG aliases (cell text)
RAG_ALIASES = compile_aliases([
    (r"\brag\b", "Yes"),
    (r"\bretrieval[-\s]?augmented( generation)?\b", "Yes"),
    (r"\bretrieve[-\s]?augmented( generation)?\b", "Yes"),
    (r"\bdoc(ument)?[-\s]?retrieval\b", "Yes"),
    (r"\bdense[-\s]?retrieval\b", "Yes"),
    (r"\bretriever\b", "Yes"),
])

# Wide RAG cues (row-aware)
RAG_STRONG_CUES = [
    r"\bretrieval[- ]?augmented(?: generation)?\b",
    r"\bretrieve[- ]?augmented(?: generation)?\b",
    r"\brag\b",
    r"\bretriever(s)?\b",
    r"\bevidence[- ]?retrieval\b",
    r"\bknowledge[- ]?(base|bank|retrieval)\b",
    r"\bmemory\b.*\b(index|retrieve|retrieval)\b",
    r"\bnearest[- ]?neighbor(s)?\b",
    r"\bk[- ]?nn\b",
    r"\btemplate[- ]?retrieval\b",
    r"\breport[- ]?retrieval\b",
    r"\bcase[- ]?retrieval\b",
    r"\brerank(er)?\b|\bre[- ]?rank(er)?\b",
    r"\bbm25\b",
    r"\bfaiss\b",
    r"\bvector[- ]?(db|store|index)\b",
    r"\bdense[- ]?retrieval\b",
    r"\bdoc(ument)?[- ]?retrieval\b",
    r"\bretrieve[- ]?then[- ]?read\b",
]
RAG_WIDE_ALIASES = compile_aliases([(pat, "Yes") for pat in RAG_STRONG_CUES])

RAG_NEG_CUES = compile_aliases([
    (r"\bno\b.*\bretriev", "NO"),
    (r"\bwithout\b.*\bretriev", "NO"),
    (r"\bnon[- ]?retrieval\b", "NO"),
])

# ========== field normalizers ==========

def normalize_vlm_cell(s: str) -> str:
    ls = lower_clean(s)
    if "vlm" in ls and "(" in ls:
        return norm_spaces(s)
    return yes_no_notreported(s)

def normalize_model(cell: str) -> str:
    """
    Canonicalize Model tokens; we will later optionally collapse to a primary model.
    """
    return canonicalize_list(cell, MODEL_ALIASES)

def normalize_vision(cell: str) -> str:
    return canonicalize_list(cell, VISION_ALIASES)

def normalize_lang(cell: str) -> str:
    return canonicalize_list(cell, LANG_ALIASES)

def normalize_fusion(cell: str) -> str:
    """
    Canonicalize and drop harmless modifiers like 'conditioned'.
    """
    toks = split_multi(cell)
    mapped = [apply_alias_once(t, FUSION_ALIASES) for t in toks]
    mapped = [m for m in mapped if m and lower_clean(m) != "not reported"]
    # drop bare modifiers
    mapped = [m for m in mapped if lower_clean(m) not in {"conditioned"}]
    mapped = uniq_keep_order(mapped)
    return ", ".join(mapped)

def normalize_objectives(cell: str) -> str:
    return canonicalize_list(cell, OBJECTIVE_ALIASES)

def normalize_family(cell: str) -> str:
    fam = canonicalize_list(cell, FAMILY_ALIASES)
    if FAMILY_ONTOLOGY == "coarse":
        # Very light conversion: map named VLMs to 'Transformer' family if present
        if fam:
            return "Transformer"
    return fam

def normalize_metrics(cell: str) -> str:
    toks = split_multi(cell)
    toks = [norm_spaces(t) for t in toks if t]
    return ", ".join(uniq_keep_order(toks))

def normalize_simple(cell: str) -> str:
    s = (cell or "").strip()
    return s

# ========== derivations (RAG + Model/Class) ==========

def scan_text(text: str, alias: List[Tuple[re.Pattern, str]]) -> str:
    t = text or ""
    for rx, canon in alias:
        if rx.search(t):
            return canon
    return ""

def normalize_rag_cell(cell: str) -> str:
    ls = lower_clean(cell)
    if ls in _MISSING_TOKENS:
        return ""
    if ls in ("yes", "y", "true", "1", "present", "rag"):
        return "Yes"
    if re.search(r"\bno\b", ls) and re.search(r"rag|retriev", ls):
        return ""
    if scan_text(cell, RAG_ALIASES):
        return "Yes"
    return ""

def derive_rag_rowaware(row: Dict[str, str]) -> str:
    direct = normalize_rag_cell(row.get("RAG", ""))
    if direct:
        return direct

    blob_neg = " | ".join([row.get(k, "") or "" for k in row.keys()])
    if scan_text(blob_neg, RAG_NEG_CUES):
        return ""

    parts = []
    for k in [
        "File", "Title", "Abstract",
        "Model", "Family", "Objectives", "Fusion", "Task", "Notes",
        "Method", "Approach", "System", "Lang Dec", "Vision Enc",
    ]:
        v = row.get(k, "")
        if v:
            parts.append(str(v))
    blob = " | ".join(parts)

    if scan_text(blob, RAG_ALIASES) or scan_text(blob, RAG_WIDE_ALIASES):
        return "Yes"

    return ""

def derive_model_if_empty(model: str, family: str, vision: str, lang: str) -> str:
    if not is_missing(model):
        return model
    fam = normalize_family(family)
    if fam:
        return fam
    lang_norm = normalize_lang(lang)
    vision_norm = normalize_vision(vision)
    parts = [p for p in [lang_norm, vision_norm] if p]
    return ", ".join(uniq_keep_order(parts))

# ====== Model refinement: prefer named VLM family; avoid generic "Transformer" unless no better ======

GENERIC_MODEL_TOKENS = {"transformer", "cnn", "rnn", "cnn-rnn", "bert", "roberta", "t5", "lstm", "gru"}

def tokenize_cell(cell: str) -> List[str]:
    return [t.strip() for t in split_multi(cell) if t.strip()]

def pick_primary_family(candidates: List[str]) -> Optional[str]:
    # candidates already canonicalized; choose by PRIMARY_FAMILY_ORDER priority or appearance
    for fam in PRIMARY_FAMILY_ORDER:
        if fam in candidates:
            return fam
    return candidates[0] if candidates else None

def refine_model_prefer_family(model: str, family: str, vision: str) -> str:
    """
    If model contains only generic tokens (e.g., 'Transformer') but the row includes a
    named VLM family (Family) or CLIP (Vision Enc), prefer that.
    If PRIMARY_MODEL_ONLY is set, collapse to the single best candidate.
    """
    model_tokens = tokenize_cell(model)
    # Build family candidates from Model itself (if any named families present)
    from_model_named = [apply_alias_once(t, FAMILY_ALIASES) for t in model_tokens]
    from_model_named = [t for t in from_model_named if t and t not in {"Transformer"} and t not in GENERIC_MODEL_TOKENS]

    # From Family column
    fam = normalize_family(family)
    fam_tokens = tokenize_cell(fam)
    fam_named = [t for t in fam_tokens if t and t not in {"Transformer"}]

    # From Vision: add CLIP as a candidate if present
    vis = normalize_vision(vision)
    vis_tokens = tokenize_cell(vis)
    vis_candidates = ["CLIP"] if any(t == "CLIP" for t in vis_tokens) else []

    candidates = uniq_keep_order(from_model_named + fam_named + vis_candidates)

    if PRIMARY_MODEL_ONLY:
        primary = pick_primary_family(candidates)
        if primary:
            return primary
        # If nothing better and Model had something, but it's generic 'Transformer' etc., keep the first generic token
        if model_tokens:
            # Prefer NOT to return 'Transformer' if Family had something coarse-mapped to Transformer due to ontology
            # but there is truly nothing else
            return model_tokens[0]
        return ""

    # Not collapsing: ensure that if Model is only generic and we have a named family candidate, prepend it
    if candidates:
        # If model already contains a named fam, keep as-is
        if any(t in candidates for t in model_tokens):
            return ", ".join(uniq_keep_order(model_tokens))
        # else, put the primary in front, keep rest of model_tokens (but drop duplicates)
        primary = pick_primary_family(candidates)
        out = [primary] if primary else []
        out.extend([t for t in model_tokens if t not in out])
        return ", ".join(uniq_keep_order(out))

    # No candidates → return existing model (generic or otherwise)
    return ", ".join(uniq_keep_order(model_tokens))

# ========== NEW / EXTENDED defaults & backfills (automatic) ==========

OBJ_MAP = {
    # core VLM families
    "albef": "contrastive, ITM",
    "blip-2": "contrastive, ITM",
    "blip": "contrastive, ITM",
    "flamingo": "ITM",
    "llava": "contrastive",
    "llava-med": "contrastive",
    "minigpt-4": "contrastive",
    "otter": "contrastive",
    "xraygpt": "contrastive",
    # radiology-specific
    "cxr-irgen": "contrastive, ITM",
    "token-mixer": "contrastive, ITM",
    "pairaug": "contrastive, ITM",
    # extras
    "instructblip": "contrastive, ITM",
    "pali-x": "contrastive, ITM",
    "pali": "contrastive, ITM",
    "simvlm": "ITM",
    "ofa": "ITM",
}

VLM_FAMILY_KEYS = [
    "albef", "blip-2", "blip", "flamingo", "llava-med", "llava",
    "minigpt-4", "instructblip", "otter", "xraygpt", "pali-x", "pali",
    "kosmos-1", "simvlm", "ofa",
]

def looks_like_vlm(row: Dict[str, str]) -> bool:
    fam_mod = " ".join([_norm(row.get("Model","")), _norm(row.get("Family",""))])
    has_vis = bool((row.get("Vision Enc") or "").strip())
    has_lang = bool((row.get("Lang Dec") or "").strip())
    vlm_flag = _norm(row.get("VLM?", "")) in {"yes", "true", "1", "y"}
    vlm_marker = any(k in fam_mod for k in VLM_FAMILY_KEYS)
    return vlm_flag or (has_vis and has_lang) or vlm_marker

def backfill_objectives_and_fusion(row: Dict[str, str]) -> Tuple[bool, bool]:
    """Return (obj_filled, fusion_filled). More aggressive defaults for VLMs."""
    obj_filled = False
    fus_filled = False

    # Fusion
    if is_missing(row.get("Fusion", "")):
        if looks_like_vlm(row):
            row["Fusion"] = "cross-attention"
            fus_filled = True

    # Objectives
    if is_missing(row.get("Objectives", "")):
        model_fam_blob = " ".join([_norm(row.get("Model", "")), _norm(row.get("Family", ""))]).replace(" ", "")
        for k, v in OBJ_MAP.items():
            if k in model_fam_blob:
                row["Objectives"] = v
                obj_filled = True
                break
        # If still missing but looks like a VLM, force default
        if not obj_filled and looks_like_vlm(row):
            row["Objectives"] = "contrastive, ITM"
            obj_filled = True

    return obj_filled, fus_filled


def derive_family_if_empty_from_model(row: Dict[str, str]) -> bool:
    """If Family is missing but Model contains a known family token, copy it into Family."""
    if not is_missing(row.get("Family", "")):
        return False
    model = _norm(row.get("Model", "")).replace(" ", "")
    for k in VLM_FAMILY_KEYS:
        if k in model:
            fam = apply_alias_once(k, FAMILY_ALIASES)
            row["Family"] = fam
            return True
    return False

def derive_class_if_missing(row: Dict[str, str]) -> bool:
    """
    Derive Class:
      - VLM (multimodal): VLM marker OR (vision & language)
      - Vision-only: vision present, language missing
      - Text-only: language present, vision missing
    """
    if not is_missing(row.get("Class", "")):
        if lower_clean(row.get("Class","")) != "not reported":
            return False

    has_vis = bool((row.get("Vision Enc") or "").strip())
    has_lang = bool((row.get("Lang Dec") or "").strip())

    if looks_like_vlm(row):
        row["Class"] = "VLM (multimodal)"
        return True
    if has_vis and not has_lang:
        row["Class"] = "Vision-only"
        return True
    if has_lang and not has_vis:
        row["Class"] = "Text-only"
        return True
    return False

def mirror_datasets_if_one_missing(row: Dict[str, str]) -> bool:
    """If exactly one of train/eval is present, mirror it to the other."""
    t = row.get("Datasets (train)", "")
    e = row.get("Datasets (eval)", "")
    t_missing = is_missing(t)
    e_missing = is_missing(e)
    if not t_missing and e_missing:
        row["Datasets (eval)"] = norm_spaces(t)
        return True
    if t_missing and not e_missing:
        row["Datasets (train)"] = norm_spaces(e)
        return True
    return False

def apply_text_only_na(row: Dict[str, str]) -> Tuple[bool, bool]:
    """
    For *non-VLM* rows without a vision encoder, if Fusion/Objectives are missing,
    mark them 'Not applicable'. Returns (fusion_set, objectives_set).
    """
    fusion_set = objectives_set = False
    if not looks_like_vlm(row):
        has_vis = bool((row.get("Vision Enc") or "").strip())
        if not has_vis:
            if is_missing(row.get("Fusion", "")):
                row["Fusion"] = "Not applicable"
                fusion_set = True
            if is_missing(row.get("Objectives", "")):
                row["Objectives"] = "Not applicable"
                objectives_set = True
    return fusion_set, objectives_set

# Map of column-specific cleaners (RAG is handled row-aware later)
FIELD_NORMALIZERS = {
    "Modality": normalize_simple,
    "Datasets (train)": normalize_simple,
    "Datasets (eval)": normalize_simple,
    "Paired": yes_no_notreported,
    "VLM?": normalize_vlm_cell,
    "Model": normalize_model,
    "Class": normalize_simple,
    "Task": normalize_simple,
    "Vision Enc": normalize_vision,
    "Lang Dec": normalize_lang,
    "Fusion": normalize_fusion,
    "Objectives": normalize_objectives,
    "Family": normalize_family,
    # "RAG": handled in derive_rag_rowaware
    "Metrics(primary)": normalize_metrics,
}

# ========== row cleaning ==========

def clean_row(row: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)

    # 1) field-wise normalization (except RAG)
    for col, fn in FIELD_NORMALIZERS.items():
        if col in out:
            if col == "RAG":
                continue
            out[col] = fn(out[col])

    # 2) derive/normalize RAG from full context
    out["RAG"] = derive_rag_rowaware({**row, **out})

    # 3) derive Model if still missing
    if "Model" in out:
        out["Model"] = derive_model_if_empty(
            out.get("Model", ""),
            out.get("Family", ""),
            out.get("Vision Enc", ""),
            out.get("Lang Dec", ""),
        )

    # 3b) refine Model to prefer named VLM family (ignore generic 'Transformer' if better exists)
    out["Model"] = refine_model_prefer_family(
        out.get("Model", ""),
        out.get("Family", ""),
        out.get("Vision Enc", ""),
    )

    # 4) automatic backfills (family, class, objectives, fusion, datasets)
    derive_family_if_empty_from_model(out)
    derive_class_if_missing(out)
    backfill_objectives_and_fusion(out)
    mirror_datasets_if_one_missing(out)

    # 5) N/A for text-only rows (non-VLM, no vision)
    apply_text_only_na(out)

    # 6) final whitespace tidy
    for k, v in list(out.items()):
        out[k] = norm_spaces(v)

    return out

# ========== IO helpers ==========

def read_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows

def write_rows(path: str, header: List[str], rows: List[Dict[str, str]]):
    if "RAG" not in header:
        header = list(header) + ["RAG"]
    if "Metrics(primary)" not in header:
        header = list(header) + ["Metrics(primary)"]

    with open(path, "w", newline="", encoding="utf-8") as g:
        writer = csv.DictWriter(g, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in header})

# ========== Mask backfill (RAG) ==========

def load_mask(mask_csv: str):
    with open(mask_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        hdr = r.fieldnames or []
        key_col = next((c for c in ["File", "file", "filename", "paper", "Paper", ""] if c in hdr), hdr[0] if hdr else "")
        A = {}
        for row in r:
            A[row.get(key_col, "")] = row
    return hdr, A, key_col

def truthy(v: str) -> bool:
    return (v or "").strip() in {"1", "1.0", "true", "True", "yes", "Yes", "Y", "y"}

def backfill_rag_from_mask(rows: List[Dict[str, str]], mask_csv: str) -> int:
    hdr, A, _ = load_mask(mask_csv)
    rag_col = next((c for c in hdr if c.lower().strip() == "rag"), None)
    updated = 0
    for r in rows:
        key = r.get("File", "")
        if not key or (r.get("RAG", "") or "").strip():
            continue
        m = A.get(key)
        if not m:
            continue
        val = m.get(rag_col, "") if rag_col else ""
        if not rag_col:
            for c in hdr:
                if c and c.lower().strip() == "rag":
                    val = m.get(c, "")
                    break
        if truthy(val):
            r["RAG"] = "Yes"
            updated += 1
    return updated

# ========== Best-per-file reducer ==========

VALUE_COLS = [
    "Modality","Datasets (train)","Datasets (eval)","Paired","VLM?","Model",
    "Class","Task","Vision Enc","Lang Dec","Fusion","Objectives","Family","RAG","Metrics(primary)"
]

def norm_key(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'\.(txt|pdf|md)$','', s, flags=re.I)
    s = re.sub(r'\s+',' ', s)
    return s.lower()

def choose_best_per_file(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Choose the row with most filled value cols; when tied, prefer one whose Model
    names a VLM family (and not just 'Transformer').
    """
    def has_named_family(model: str) -> bool:
        toks = tokenize_cell(model)
        named = [apply_alias_once(t, FAMILY_ALIASES) for t in toks]
        named = [t for t in named if t and t not in {"Transformer"} and t not in GENERIC_MODEL_TOKENS]
        return len(named) > 0

    bins: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        bins.setdefault(norm_key(r.get("File","")), []).append(r)

    chosen: List[Dict[str, str]] = []
    for _, group in bins.items():
        group.sort(
            key=lambda r: (
                sum(1 for c in VALUE_COLS if not is_missing(r.get(c,""))),
                1 if has_named_family(r.get("Model","")) else 0
            ),
            reverse=True
        )
        chosen.append(group[0])
    return chosen

# ========== CLI ==========

def main():
    global PRIMARY_MODEL_ONLY, FAMILY_ONTOLOGY

    ap = argparse.ArgumentParser(
        description="Clean & canonicalize extraction CSV with row-aware RAG + automatic model-aware backfills and optional mask + best-per-file."
    )
    ap.add_argument("input_csv", help="Input CSV path")
    ap.add_argument("output_csv", help="Output CSV path")
    ap.add_argument("--mask", type=str, default=None, help="Availability mask CSV to backfill RAG (by File key)")
    ap.add_argument("--best", action="store_true", help="Emit best-per-file CSV")
    ap.add_argument("--best-out", type=str, default=None, help="Output path for best-per-file CSV")

    # New switches
    ap.add_argument("--primary-model-only", action="store_true",
                    help="Collapse Model to a single primary named VLM family when available; avoid generic 'Transformer' if specific family exists.")
    ap.add_argument("--family-ontology", choices=["named","coarse"], default="named",
                    help="How to canonicalize Family column: 'named' (ALBEF/BLIP-2/...) or 'coarse' (Transformer).")

    args = ap.parse_args()
    PRIMARY_MODEL_ONLY = args.primary_model_only
    FAMILY_ONTOLOGY = args.family_ontology

    # 1) read & clean (automatic backfills included)
    header, rows = read_rows(args.input_csv)
    cleaned = [clean_row(r) for r in rows]
    write_rows(args.output_csv, header, cleaned)
    print(f"✓ Cleaned -> {args.output_csv}")

    # 2) mask-based RAG backfill (optional)
    if args.mask:
        updated = backfill_rag_from_mask(cleaned, args.mask)
        # Re-apply NA rule & tidy
        for r in cleaned:
            apply_text_only_na(r)
            derive_class_if_missing(r)
            for k in list(r.keys()):
                r[k] = norm_spaces(r[k])
        write_rows(args.output_csv, header, cleaned)
        print(f"✓ RAG backfill from mask: {updated} rows updated")

    # 3) best-per-file
    if args.best:
        best_rows = choose_best_per_file(cleaned)
        best_out = args.best_out or args.output_csv.replace(".csv", "_best.csv")
        write_rows(best_out, header, best_rows)
        print(f"✓ Best-per-file -> {best_out}: kept {len(best_rows)} rows from {len(cleaned)}")

if __name__ == "__main__":
    main()
