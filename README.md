# MicroSurg-AI-Wavelet-Enhanced-Deep-Vision-for-Sub-Millimeter-Tool-Perception
# 

### Expanding the Zooming-without-Zooming (ZwZ) Framework: Spatial-Information Incorporation, Frequency, and Geometric Extensions

## Abstract

Fine-grained perception in robotic surgery demands resolving sub-millimeter structures such as suture threads, needle tips, and micro-scissor edges within global $1024 \times 1024$ endoscopic frames. While Multimodal Large Language Models (MLLMs) excel at broad scene understanding, they struggle when decisive evidence is small and overwhelmed by global context. Traditional physical optical zoom incurs mechanical latency and operator distraction, while agentic "Thinking-with-Images" approaches incur heavy computational delays through iterative visual re-encoding.

We present **Wave-R2I**, an architectural and procedural expansion of the Zooming-without-Zooming (ZwZ) framework. By treating "zooming" as a training-time primitive rather than an inference-time tool, we equip Qwen2-VL with four technical extensions:
1. **Saliency-Guided SAM Extraction Pipeline**: Automatically curates high-utility micro-crops using Segment Anything (SAM ViT-H) and Grad-CAM saliency scoring across 53,526 surgical frames.
2. **Discrete Wavelet Transform (DWT) Branch**: Extracts high-frequency ($\text{HH}$) diagonal edge details from native high-resolution inputs to mathematically restore microscopic boundaries smoothed by standard ViT resizing.
3. **Transformer-Based Mask Decoder**: Translates structural `[SEG]` token activations into pixel-precise $1024 \times 1024$ binary segmentation masks via a 4-layer Transformer decoder and transposed convolutions.
4. **Cross-View Homography Regularization**: Leverages SIFT feature matching and RANSAC homography estimation to enforce geometric consistency under dynamic endoscopic camera movement and extreme perspective shifts.

The project achieves sub-millimeter surgical precision from global frames in a single forward pass (~7.5s, 11GB VRAM on an NVIDIA RTX 3090), eliminating mechanical zoom while halving GPU memory requirements over naive multi-crop querying.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │    Input Operative Frame (1024 x 1024)        │
                      └───────┬───────────────────────────────┬───────┘
                              │                               │
                      [ViT Preprocessing]               [2-Level Haar DWT]
                              │                               │
                              ▼                               ▼
                      ┌───────────────┐              ┌────────────────┐
                      │ Low-Res View  │              │ High-High (HH) │
                      │   (448x448)   │              │ Sub-Band Edges │
                      └───────┬───────┘              └────────┬───────┘
                              │                               │
                              ▼                               ▼
                      ┌───────────────┐              ┌────────────────┐
                      │ Qwen2-VL ViT  │              │ 64-dim Linear  │
                      │ Visual Tokens │              │   Projection   │
                      └───────┬───────┘              └────────┬───────┘
                              │                               │
                              └───────────────┬───────────────┘
                                              ▼ (Weighted Fusion α = 0.3)
                                     ┌─────────────────┐
                                     │ Fused Embedding │
                                     └────────┬────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │  Qwen2-VL Causal LM Backbone (4-bit NF4 QLoRA)│
                      └───────┬───────────────────────────────┬───────┘
                              │                               │
                              ▼                               ▼
                      ┌───────────────┐              ┌────────────────┐
                      │  Text Output  │              │  [SEG] Trigger │
                      │ (VQA + BBox)  │              │  Hidden State  │
                      └───────────────┘              └────────┬───────┘
                                                              │
                                                              ▼
                                                     ┌────────────────┐
                                                     │  Transformer   │
                                                     │  Mask Decoder  │
                                                     │  (64x Upscale) │
                                                     └────────┬───────┘
                                                              │
                                                              ▼
                                                     ┌────────────────┐
                                                     │ 1024x1024 Mask │
                                                     └────────────────┘
```

---

## Key Modules & Contributions

### 1. Saliency-Guided SAM Extraction Pipeline
To ensure knowledge distillation focuses on high-utility surgical regions rather than uninformative background tissue:
* **Candidate Proposals:** Class-agnostic masks proposed via SAM (ViT-H).
* **Saliency Filtering:** Grad-CAM activation maps generated from a ResNet-50 backbone. Only regions occupying $<10\%$ of total frame area with mean saliency $\bar{S} > 0.25$ are selected as $256 \times 256$ micro-crops.
* **Fast Phonebook Indexer:** Built to uniquely map 53,526 candidate frames across SurgMLLMBench splits without redundant filesystem traversals.

### 2. Wavelet Vision Branch (Frequency-Domain Edge Recovery)
Vision Transformers inherently exhibit low-pass filtering behavior, suppressing sharp high-frequency edges:
* Input images are decomposed using a 2-level Haar Discrete Wavelet Transform into sub-bands $\{LL, LH, HL, HH\}$.
* The High-High ($\text{HH}$) sub-band isolates diagonal edge derivatives (capturing metallic tool sheen, needle tips, and suture threads).
* Extracted $\text{HH}$ representations are projected into a 64-dimensional token space and fused with primary visual tokens using weight $\alpha = 0.3$.

### 3. Transformer Mask Decoder & Vocabulary Triggering
To overcome the text-only output constraint of traditional ZwZ:
* Expanded tokenizer vocabulary: `[SEG]`, `[MASK]`, `[TOOL]`, `[TISSUE]`.
* When the causal LM generates `[SEG]`, its 4096-dimensional hidden state is routed to a 4-layer Transformer decoder (hidden dimension 256, 8 attention heads).
* Three successive transposed convolution layers provide a $64\times$ spatial upscale, reconstructing a full-resolution $1024 \times 1024$ binary segmentation mask.

### 4. Cross-View Geometric Homography Consistency
Moving laparoscopic cameras induce perspective distortions and scale shifts:
* Consecutive surgical video frames are matched using SIFT descriptors and a FLANN matcher.
* Projective homography matrices $\mathcal{H}$ are estimated using RANSAC, yielding ~500 verified cross-view pairs.
* Geometric consistency regularization forces the student's latent representations under perspective distortion to match canonical unwarped teacher representations.

---

## Multi-Loss Objective

The network is trained end-to-end with a composite multi-task loss:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{text} + \lambda_2 \mathcal{L}_{coord} + \lambda_3 \mathcal{L}_{dice} + \lambda_4 \mathcal{L}_{geo}$$

* **$\mathcal{L}_{text}$ (Causal Language Modeling):** Cross-entropy loss computed strictly over target assistant responses with prompt tokens masked to $-100$.
* **$\mathcal{L}_{coord}$ (Tool Bounding Box Regression):** Smooth-$L_1$ loss over normalized coordinates $[y_1, x_1, y_2, x_2] \in [0.0, 1.0]$.
* **$\mathcal{L}_{dice}$ (Soft Dice Segmentation Loss):** Penalizes mask overlap discrepancies to address severe foreground/background imbalance:
  $$\mathcal{L}_{dice} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$
* **$\mathcal{L}_{geo}$ (Cross-View Regularization):** Euclidean distance between canonical teacher features and homography-warped student feature representations.

---

## Distillation Hierarchy

A three-priority knowledge loading strategy curates balanced, high-fidelity surgical instruction data:

| Priority | Source | Samples | Role |
| :--- | :--- | :--- | :--- |
| **1 (Gold)** | `dsyn_dataset.jsonl` | ~1,200 | Pre-labeled surgical ground-truth VQA |
| **2 (Silver)** | Gemini 2.5 Flash-Lite | ~1,000 | Automated API-driven micro-reasoning & tool-tip bboxes |
| **3 (Bronze)** | Qwen2-VL (Student) | ~6,000 | Self-consistency gap-filling on unannotated crops |

---

## Experimental Results

All experiments were conducted on an **NVIDIA RTX 3090 (24GB VRAM)** using 4-bit NormalFloat (NF4) quantization and LoRA ($r=64, \alpha=128$, learning rate $2 \times 10^{-5}$).

### 1. Inference Latency & Resource Consumption

| Method | Avg. Latency | VRAM Usage | Zoom Required? |
| :--- | :--- | :--- | :--- |
| **Physical Optical Zoom (Clinical)** | ~30s | N/A | Yes (Mechanical) |
| **Naive Digital Crop-and-Query** | ~8.0s | ~20 GB | Yes (Iterative ViT pass) |
| **Wave-R2I (Ours)** | **~7.5s** | **~11 GB** | **No (Single glance)** |

*Wave-R2I achieves a 4× speedup over mechanical adjustment while cutting VRAM utilization nearly in half compared to multi-crop querying.*

### 2. Qualitative Findings & Grad-CAM Analysis
* **Instrument Identification & Grounding:** Accurately localizes graspers, scissors, and hooks (e.g., bounding box `[0.38, 0.49, 0.56, 0.70]` during cholecystectomy procedures).
* **Vignette Suppression:** Grad-CAM Class Activation Maps show attention concentrated tightly on the tool-tissue interface, effectively ignoring the dark circular vignette characteristic of laparoscopic optics.
* **Low-Exposure Robustness:** In underexposed frames, uniform low activation correctly signals insufficient evidence, avoiding hallucinations.

---

## Repository Structure

```text
MicroSurg-AI/
├── configs/
│   └── api_key.env.example                  # Environment configuration template
├── data_pipeline/                           # End-to-end data curation pipeline
│   ├── setup_and_subset.py                  # Step 01: Multi-dataset balancing & subsetting
│   ├── micro_region_extractor.py            # Step 02: SAM ViT-H + Grad-CAM saliency micro-crops
│   ├── teacher_oracle.py                    # Step 03: Gemini teacher oracle distillation
│   ├── multi_tier_oracle.py                 # Step 03.1: 3-tier priority knowledge merger
│   ├── filter_crops_oracle.py               # Step 03.2: Top-4000 crop area selector
│   ├── cleanup_vqa.py                       # Step 03.5: VQA cleaner & tool nomenclature normalizer
│   ├── build_metadata.py                    # Step 04: Local crop to global pixel coordinate mapper
│   ├── grounding_transform.py               # Visual bounding box prompt anchor
│   ├── homography_consistency.py            # Step 05: SIFT + RANSAC inter-frame homography
│   ├── dsyn_assembly.py                     # Step 06: Multimodal dataset assembly
│   ├── dataloader.py                        # PyTorch Dataset with dynamic path resolution
│   ├── dataalign.py                         # Pickle metadata alignment
│   ├── datacheck.py                         # Image path verification utility
│   ├── extract_endovis.py                   # Parquet frame extractor
│   └── create_mock_dataset.py               # Synthetic test frame generator
├── models/                                  # Neural network modules
│   ├── wavelet_branch.py                    # WaveletVisionBranch (2-level Haar DWT HH band)
│   ├── mask_decoder.py                      # TransformerMaskDecoder (4-layer, 64x upscale)
│   ├── losses.py                            # DiceLoss, GeometricConsistencyLoss, ZoomingMultiLoss
│   └── metrics.py                           # Word-F1, BBox IoU, mIoU, and mAP
├── training/                                # Training scripts
│   ├── train_zooming.py                     # Qwen2-VL 4-bit QLoRA multi-task training loop
│   ├── tokenizer_expansion.py               # Adds [SEG], [MASK], [TOOL], [TISSUE] tokens
│   └── run_phase2_master.py                 # Training sprint runner
├── evaluation/                              # Evaluation & explainability
│   ├── evaluate_surgical.py                 # VQA, BBox IoU, mAP, and Grad-CAM attention maps
│   ├── evaluate_scibert_segmentation.py     # SciBERT Medical BERTScore & Otsu/Watershed overlays
│   ├── benchmark_evaluation.py              # EndoVis2018 semantic similarity benchmark
│   └── colon_benchevaluation.py             # ColonBench streaming MCQ benchmark
├── scripts/                                 # Execution and SLURM runner scripts
│   ├── run_data_pipeline.sh                 # SLURM script for data pipeline
│   ├── run_steps_02_03_04_gpu.sh            # SLURM script for GPU steps 02-04
│   ├── run_teacher_oracle.sh                # SLURM script for teacher distillation
│   ├── download_qwen.sh                     # Base Qwen2-VL model download script
│   └── download_dataset.py                  # Surgical benchmark downloader
├── tests/                                   # Automated unit tests
│   └── test_modules.py                      # Sanity checks for Wavelet, Decoder, Loss, Metrics
├── legacy/                                  # Preserved historical scripts
├── train.py                                 # Root training entrypoint
├── evaluate.py                              # Root evaluation entrypoint
├── run_pipeline.py                          # Root pipeline orchestrator
├── requirements.txt                         # Pinned dependencies
├── pyproject.toml                           # Package configuration
└── .gitignore                               # Git exclusions for weights, data, and cache
```

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Eshani-R-Sawant/MicroSurg-AI-Wavelet-Enhanced-Deep-Vision-for-Sub-Millimeter-Tool-Perception.git
cd MicroSurg-AI-Wavelet-Enhanced-Deep-Vision-for-Sub-Millimeter-Tool-Perception

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

```bash
cp configs/api_key.env.example configs/api_key.env
# Edit configs/api_key.env:
export GENAI_API_KEY="your_gemini_api_key"
export HF_TOKEN="your_huggingface_token"
```

### Verification Tests

Verify core architectural blocks (Wavelet branch, mask decoder, loss functions, metrics):
```bash
python3 tests/test_modules.py
```

### End-to-End Workflow

```bash
# 1. Run Data Preparation Pipeline (Subsetting, Saliency Extraction, Distillation, Assembly)
python3 run_pipeline.py --stage all

# 2. Train Qwen2-VL Student with Wavelet Branch and Mask Decoder
python3 train.py

# 3. Evaluate Grounding, VQA, and Grad-CAM Attention Maps
python3 evaluate.py
```

---


