# Multimodal Fusion for Multi-Label Movie Genre Classification

A comparative study of text–image fusion strategies for predicting multiple genres per film, built on a self-collected TMDb dataset of 6,446 movies across 19 genres.

> **Master's thesis project** · Sogang University, Graduate School of AI・SW
> Focus: a systematic, controlled comparison of fusion strategies under a single experimental setup.

---

## 概要（日本語）

本リポジトリは、**マルチラベル映画ジャンル分類におけるマルチモーダル融合戦略の比較研究**（修士論文）をまとめたものです。TMDbから独自に収集した6,446本・19ジャンルのデータセットを用い、テキスト／画像エンコーダと融合手法の組み合わせを体系的に比較しました。最良構成（Late Fusion・重み付き）で **Micro-F1 = 0.6463** を達成し、保存済みの予測確率としきい値から **GPU なしで結果を再現**できます。

---

## TL;DR — Key Results

- **Best configuration:** Late Fusion (weighted) — **DistilBERT (text) + ConvNeXt-Tiny (image)**
- **Best score:** **Micro-F1 = 0.6463** (image weight α = 0.35, per-label thresholding)
- **Key finding:** Gated Fusion *underperformed* even single-modality baselines on this dataset — interpreted as a **data-scale dependency** effect (see [Key Findings](#key-findings)).
- **Reproducible without a GPU:** saved prediction probabilities + per-label thresholds let anyone recompute the headline metrics in seconds.

<!-- TODO: optional — add a small results badge/summary table image here once figures are finalized -->

---

## Motivation

Movies rarely belong to a single genre — a film can be *Action*, *Adventure*, and *Sci-Fi* at once. This makes genre prediction a **multi-label** problem. It is also naturally **multimodal**: the plot overview carries semantic cues (text), while the poster carries stylistic and visual cues (image).

This project asks a focused question: **which fusion strategy combines text and image signals most effectively for this task, and under what conditions?** Rather than proposing one new model, it compares encoders and fusion methods under a single, controlled setup so the differences are attributable to the design choices themselves.

---

## Dataset

- **Source:** [The Movie Database (TMDb)](https://www.themoviedb.org/), collected via the TMDb API
- **Size:** 6,446 films
- **Labels:** 19 genres (multi-label)
- **Modalities:** plot overview (text) + poster (image)
- **Splitting:** iterative stratification for multi-label data via `MultilabelStratifiedKFold` ([iterative-stratification](https://github.com/trent-b/iterative-stratification)), preserving label distribution across train / validation / test splits

> **On posters & licensing:** poster images are **not redistributed** in this repository. Instead, the repo provides the list of TMDb movie IDs and a collection script, so the dataset can be reconstructed from the original source under TMDb's terms. See [`data/README.md`](data/README.md). <!-- TODO: create this file -->

---

## Methods

The study compares combinations along three axes:

| Axis | Options |
|------|---------|
| **Text encoder** | TF-IDF · BERT-mini · DistilBERT |
| **Image encoder** | ResNet50 · ConvNeXt-Tiny |
| **Fusion strategy** | Late Fusion · Early Fusion · Gated Fusion |

**Fusion strategies (brief):**

- **Late Fusion** — each modality is scored independently, then combined at the probability level. The weighted variant uses `p = (1 − α)·p_text + α·p_img`, where **α is the image weight** (best: α = 0.35, i.e. text 0.65 / image 0.35).
- **Early Fusion** — modality features are concatenated and jointly classified.
- **Gated Fusion** — a learned gating mechanism modulates each modality's contribution.

Encoder representations are projected to a common 512-dimensional space before fusion (a power-of-two design convention).

<!-- TODO: add the architecture diagram (draw.io PNG) here, e.g.: -->
<!-- ![Fusion architectures](figures/fusion_architectures.png) -->

---

## Results

Evaluation uses **per-label thresholding** (each genre gets its own decision threshold tuned on the validation set) and reports Micro-F1 as the primary metric.

| Configuration | Text | Image | Micro-F1 |
|---|---|---|---|
| **Late Fusion (weighted)** | **DistilBERT** | **ConvNeXt-Tiny** | **0.6463** |
| Late Fusion (average) | DistilBERT | ConvNeXt-Tiny | 0.6189 |
| Early Fusion | DistilBERT | ConvNeXt-Tiny | 0.6087 |
| Gated Fusion | DistilBERT | ConvNeXt-Tiny | 0.5453 |
| Text only | DistilBERT | — | 0.5976 |
| Image only | — | ConvNeXt-Tiny | 0.5326 |

---

## Key Findings

- **Late Fusion (weighted) was the strongest configuration**, and a modest tilt toward text (α = 0.35 on the image side) worked best — consistent with the plot overview carrying more discriminative genre signal than the poster alone.
- **Gated Fusion underperformed even single-modality baselines** on this dataset. Since gated-style fusion has shown its advantage on substantially larger datasets (e.g. GMU on MM-IMDb, ~26k films, in Arevalo et al. 2017), this result is interpreted as a **data-scale dependency** effect rather than a flaw in the mechanism itself: the gating parameters appear under-supported at the ~6.4k-film scale used here.

> These are interpretations of observed behavior, not claims about internal mechanisms beyond what the experiments support.

---

## Reproducibility

Reproduction is organized in layers, from "runs anywhere" to "reference only," so it's clear what you can expect to reproduce:

| Layer | What's included | Can you reproduce it? |
|---|---|---|
| **Evaluation** | Saved prediction probabilities + per-label thresholds + scoring script | ✅ **Yes — no GPU needed.** Recompute Micro-F1 and thresholds directly. |
| **Training** | Model training code | ⚠️ Reference only — requires GPU and the reconstructed dataset. |
| **Data collection** | TMDb collection script + movie ID list | ⚠️ Requires a TMDb API key; rebuilds the dataset from source. |

**Quick start (evaluation layer):**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Reproduce the headline metrics from saved probabilities + thresholds
python src/evaluate/reproduce_results.py  
```

Environment (pinned in `requirements.txt`):

- Python, PyTorch 2.10.0, Transformers 5.0.0
- iterative-stratification (trent-b), NumPy, scikit-learn

<!-- TODO: finalize requirements.txt and the evaluate script entry point -->

---

## Repository Structure

```
multimodal-genre-fusion/
├── README.md
├── requirements.txt
│
├── src/
│   └── evaluate/
│       └── reproduce_results.py    # reproduce headline Micro-F1 (no GPU)
│
└── results/                        # ★ reproduction artifacts
    ├── probs/                      # saved prediction probabilities (.npy)
    │   ├── text_distilbert_test_prob.npy
    │   ├── image_convnext_test_prob.npy
    │   └── lf_weighted_test_prob.npy
    ├── thresholds/
    │   └── lf_weighted_perlabel_thresholds.npy
    ├── y_test.npy
    └── label_order.npy
```

---

## Limitations & Future Work

- **Dataset scale.** At ~6.4k films, the dataset is smaller than benchmarks like MM-IMDb; some fusion strategies (notably gated fusion) may benefit from more data.
- **Single data source.** TMDb-only collection may introduce distribution bias relative to other movie databases.
- **Future directions.** Larger / multi-source data, stronger pre-trained encoders, and threshold-calibration studies are natural next steps.

---

## References

- Arevalo et al. (2017), *Gated Multimodal Units for Information Fusion* — GMU on MM-IMDb (~25,959 films).
- Mangolin et al. (2022) — multimodal movie genre classification on TMDb (~10,594-film experimental set; ~152,622-film source pool); Late Fusion effective.
- Shaukat et al. (2025) — BERT-family pre-trained model (DeBERTa-based) for the task.
- Unal et al. (2023) — ConvNeXt-based image study for genre classification.

<!-- TODO: complete reference list with full citations and links, matching the thesis bibliography -->

---

## Author

**Kim Doyun (金祹昀)** — M.S., Graduate School of AI・SW, Sogang University
<!-- TODO: add contact / LinkedIn / portfolio links as desired -->
