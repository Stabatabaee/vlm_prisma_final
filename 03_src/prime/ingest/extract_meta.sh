#!/usr/bin/env bash

# print header
printf "| File | Title | Authors | Modality | Datasets | Model Name | Vision Encoder | Language Decoder | Fusion Strategy |\n"
printf "|---|---|---|---|---|---|---|---|---|\n"

for f in tuning_samples/*.pdf; do
  file=$(basename "$f")

  # Title
  title=$(pdfinfo "$f" | awk -F': ' '/^Title:/ {print $2}')

  # Authors (lines 2–8, stop at Abstract)
  authors=$(pdftotext "$f" - \
    | sed -n '2,8p' \
    | sed '/Abstract/,$d' \
    | tr '\n' ' ' \
    | sed 's/  */ /g; s/^ //; s/ $//')

  # Modality
  modality=$(pdftotext "$f" - | grep -m1 -Eo 'Chest X[- ]?[rR]ay' || echo 'N/A')

  # Datasets
  datasets=$(pdftotext "$f" - \
    | grep -Eo 'MIMIC[- ]?CXR|IU[- ]?XRay|NIH Chest X[- ]?ray|Open[- ]?i|CXR-PRO|PhysioNet' \
    | sort -u | paste -sd', ' -)

  # Model Name
  model=$(pdftotext "$f" - | grep -m1 -Eo 'TrMRG|CXR-IRGen|EGGCA-Net|MM-CXRG|RAG \+ EALBEF \+ LLaMA' || echo 'N/A')

  # Vision Encoder
  vision_encoder=$(pdftotext "$f" - | grep -m1 -Eo 'ViT|ResNet-[0-9]+' || echo 'N/A')

  # Language Decoder
  language_decoder=$(pdftotext "$f" - | grep -m1 -Eo 'BART|MiniLM|LLaMA|Transformer-based autoregressive decoder' || echo 'N/A')

  # Fusion Strategy
  fusion_strategy=$(pdftotext "$f" - | grep -m1 -Eo 'cross-attention|channel attention|concatenation|averaging|Retrieval-Augmented Generation|Dual Fine-Grained Branch' || echo 'N/A')

  # emit row
  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "$file" "$title" "$authors" "$modality" "$datasets" \
    "$model" "$vision_encoder" "$language_decoder" "$fusion_strategy"
done
