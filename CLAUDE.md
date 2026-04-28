# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

INP-Former is a PyTorch deep learning framework for **Universal Anomaly Detection** (CVPR 2025). It introduces Intrinsic Normal Prototypes (INPs) — learned tokens that extract normal representations directly from test images rather than relying on external training references. Supports MVTec-AD, VisA, and Real-IAD datasets.

## Environment Setup

```bash
conda create -n INP python=3.8.12
conda activate INP
pip install -r requirements.txt
# Optional: pip install gradio (for Zero_Shot_App.py)
# Optional: pip install onnx==1.15.0 onnxruntime-gpu==1.15.0 onnxsim (for ONNX export)
```

Requires PyTorch 2.0.0+cu118, timm 0.9.12, kornia 0.7.3, adeval 1.1.0.

## Running Training and Testing

Each anomaly detection paradigm has its own entry script. All use `--phase train` or `--phase test`.

```bash
# Multi-class (one model for all categories in a dataset)
python INP_Former_Multi_Class.py --dataset MVTec-AD --data_path ../mvtec_anomaly_detection --phase train

# Single-class (one model per category)
python INP_Former_Single_Class.py --dataset MVTec-AD --data_path ../mvtec_anomaly_detection --phase train

# Few-shot (1/2/4-shot)
python INP_Former_Few_Shot.py --dataset VisA --data_path ../VisA_pytorch/1cls --shot 4 --phase train

# Super multi-class (train across all 3 datasets simultaneously)
python INP_Former_Super_Multi_Class.py --mvtec_data_path ../mvtec_anomaly_detection --visa_data_path ../VisA_pytorch/1cls --real_iad_data_path ../Real-IAD --phase train

# Zero-shot (no training — loads a pre-trained model and evaluates on a target dataset)
python INP_Former_Zero_Shot.py --source_dataset Real-IAD --dataset MVTec-AD --data_path ../mvtec_anomaly_detection

# Interactive Gradio demo
python Zero_Shot_App.py
```

Common flags: `--encoder dinov2reg_vit_base_14`, `--input_size 448`, `--crop_size 392`, `--INP_num 6`, `--total_epochs 200`, `--batch_size 16`, `--save_dir ./saved_results`.

There is no test suite, linter, or Makefile. Validation happens via evaluation metrics (AUROC, AP, F1, AUPRO) computed during the test phase.

## Architecture

**Data flow:**
1. Input image (448×448) → frozen DINOv2 encoder → multi-layer token features (layers 2–9)
2. Encoder features → bottleneck MLP → Aggregation Block extracts INP prototypes (6 learned tokens)
3. INP prototypes + encoder features → 8 Prototype Block decoder layers → reconstructed features
4. Anomaly map = 1 − cosine_similarity(encoder features, decoder features), Gaussian-smoothed

**Key modules in `models/`:**
- `uad.py` — `INP_Former` class: wires encoder, bottleneck, aggregation, and decoder together
- `vision_transformer.py` — `Aggregation_Block` (extracts INPs) and `Prototype_Block` (INP-guided decoding) attention mechanisms
- `vit_encoder.py` — loads frozen backbone (DINOv2/v1, BEiT, etc.) from timm or local `dinov2/`

**Training loss:** `global_cosine_hm_adaptive(en, de, y=3) + 0.2 * gather_loss` — soft-mining adaptive cosine loss plus INP coherence constraint.

**Evaluation (`utils.py`):** `evaluation_batch()` runs forward pass, computes anomaly maps via `cal_anomaly_maps()`, then uses `adeval.EvalAccumulatorCuda` for GPU-accelerated AUROC/AP/F1/AUPRO metrics.

**Dataset loading (`dataset.py`):** `MVTecDataset` handles MVTec-AD and VisA; `RealIADDataset` handles Real-IAD. Both return (image, ground_truth_mask, label, path).

**Feature fusion:** Encoder and decoder outputs are grouped into 2 feature groups before anomaly map computation (inspired by Dinomaly).

## Pre-trained Checkpoints

Stored in `saved_results/` with naming pattern:
`INP-Former-{Setting}_dataset={Dataset}_Encoder={encoder}_Resize={size}_Crop={crop}_INP_num={n}/model.pth`

## ONNX Export

```bash
python convert_onnx.py --pretrained_model_path <path_to_model.pth>
python inference_onnx.py  # Run inference with exported model
```
