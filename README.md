# KAN-MoA: Coarse Experts Generalize, Fine Experts Memorize

<p align="center">
  <strong>Spectral Decomposition of Adapters for Parameter-Efficient Domain Generalization</strong><br>
  DG-EBF Workshop @ CVPR 2026
</p>

<p align="center">
  <a href="#results"><img src="https://img.shields.io/badge/Avg_Accuracy-77.9%25-blue?style=flat-square" /></a>
  <a href="#results"><img src="https://img.shields.io/badge/Trainable_Params-0.2M-green?style=flat-square" /></a>
  <a href="#results"><img src="https://img.shields.io/badge/Training-2x_Faster-orange?style=flat-square" /></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?style=flat-square" /></a>
</p>


> **TL;DR** — We replace linear LoRA adapters with a mixture of KAN experts at different B-spline grid resolutions (G∈{3,5,10,20}). Low-G experts learn smooth, generalizable functions. High-G experts absorb complex domain-specific variations. A spectral router based on the FFT power spectrum of token embeddings directs tokens to experts. Result: **77.9% on DomainBed with 0.2M params**, matching 261M-param ensemble methods.

---

## Overview

**Problem.** Linear adapters (LoRA, KAdaptation) can't model non-linear domain shifts. Mixture-of-Adapters (MoA) varies capacity through rank, but experts remain linear.

**Idea.** Use *functional resolution* — the B-spline grid size G of Kolmogorov-Arnold Networks — as a complexity control knob. Different experts occupy different points on the complexity spectrum:

```
Expert 0 (G=3)  → 6 spline coefficients  → smooth, low-complexity    → generalizable
Expert 1 (G=5)  → 8 spline coefficients  → gentle complexity         → mostly invariant  
Expert 2 (G=10) → 13 spline coefficients → moderate complexity       → transitional
Expert 3 (G=20) → 23 spline coefficients → sharp, high-complexity    → domain-specific
```

**Architecture.** Each KAN-MoA layer:

```
y = W_frozen · x + Σ_i  g_i(x) · E_i(x)

where E_i(x) = W_up · KAN_Gi(LN(W_down · x))    [bottleneck expert]
      g(x)   = Softmax(W_s · LN(|FFT(x)|))       [spectral router]
```

Injected into 24 attention layers of a frozen CLIP ViT-B/16. Only 0.2M trainable parameters (0.23% of backbone).

---

## Results

### Main Comparison (DomainBed, training-domain validation)

| Method | Type | PACS | VLCS | OfficeHome | TerraInc. | DomainNet | Avg. | Train. Params |
|:-------|:-----|:----:|:----:|:----------:|:---------:|:---------:|:----:|:-------------:|
| LoRA (r=2) | PEFT | 96.4 | 82.6 | 86.7 | 46.1 | 61.5 | 74.7 | 0.11M |
| KAdaptation | PEFT | 97.5 | 83.0 | 90.3 | 51.9 | 62.7 | 77.1 | 0.14M |
| KA-MoA | MoA | 97.4 | 83.1 | 90.6 | 52.8 | 62.7 | 77.3 | 1.5M |
| GNN-MoE | PEFT | 97.8 | 83.6 | 90.7 | 53.8 | 62.3 | 77.7 | 1.8M |
| KA-MoA-Ens. (3 models) | Ensemble | 97.6 | 83.4 | 90.9 | 54.3 | 63.1 | 77.9 | 261.3M |
| **KAN-MoA (Ours)** | **PEFT** | **97.9** | **83.3** | **90.7** | **54.0** | **63.6** | **77.9** | **0.2M** |

### Heterogeneous vs. Uniform Grids

| Grid Config | PACS | OH | VLCS | TerraInc. | DN | Avg. |
|:------------|:----:|:--:|:----:|:---------:|:--:|:----:|
| [3,3,3,3] | 97.7 | 89.5 | 82.6 | 49.2 | 62.5 | 76.3 |
| [10,10,10,10] | 97.6 | 89.3 | 82.8 | 49.7 | 62.3 | 76.3 |
| [20,20,20,20] | 97.5 | 89.3 | 82.0 | 49.8 | 62.5 | 76.2 |
| **[3,5,10,20] (Ours)** | **97.9** | **90.7** | **83.3** | **54.0** | **63.6** | **77.9** |

> All uniform configurations perform identically (~76.3%). The +1.6% gain comes entirely from heterogeneity.

### Compute Efficiency (single A100)

| Method | Step Time | Eval Time |
|:-------|:---------:|:---------:|
| ERM-MoE | 0.963s | 31.17s |
| **KAN-MoA** | **0.473s** | **16.93s** |

---

## Installation

```bash
git clone https://github.com/XXXX/KAN-MoA.git
cd KAN-MoA
conda create -n kan_moa python=3.10 -y
conda activate kan_moa
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [DomainBed](https://github.com/facebookresearch/DomainBed)

---

## Quick Start

### Training

```bash
# PACS (leave-one-domain-out, spectral router, default config)
python train.py \
    --dataset PACS \
    --backbone openclip_vitb16 \
    --adapter kan_moa \
    --grid_sizes 3 5 10 20 \
    --rank 8 \
    --router spectral \
    --lr 5e-5 \
    --steps 5000 \
    --batch_size 32 \
    --lambda_swd 0.001 \
    --lambda_orth 0.01 \
    --lambda_lb 0.001 \
    --seed 0
```

### Evaluation

```bash
python eval.py \
    --dataset PACS \
    --checkpoint checkpoints/pacs_best.pth \
    --test_domain sketch
```

### Run All Benchmarks

```bash
# Reproduce main results (Table 1) across all 5 datasets
bash scripts/run_all.sh
```

---

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `--grid_sizes` | `3 5 10 20` | B-spline grid sizes for the 4 experts |
| `--rank` | `8` | Bottleneck rank for all experts |
| `--router` | `spectral` | Router type: `spectral`, `cosine`, or `kan` |
| `--lambda_swd` | `0.001` | Spline weight decay coefficient |
| `--lambda_orth` | `0.01` | Expert orthogonality coefficient |
| `--lambda_lb` | `0.001` | Load balancing coefficient |
| `--grad_scale_start` | `0.1` | Initial KAN gradient scaling factor |
| `--grad_scale_steps` | `1000` | Steps to linearly ramp gradient scale to 1.0 |

### Per-Dataset Adjustments

DomainNet benefits from stronger orthogonality (`--lambda_orth 0.10`). All other datasets use defaults.

---

## Project Structure

```
KAN-MoA/
├── models/
│   ├── kan_moa.py          # KAN-MoA layer and expert architecture
│   ├── kan_linear.py       # B-spline KAN layer implementation
│   ├── routers.py          # Cosine, KAN, and Spectral routers
│   └── backbone.py         # Frozen CLIP ViT-B/16 with adapter injection
├── losses/
│   ├── spline_weight_decay.py   # L2 regularization on spline coefficients
│   ├── orthogonality.py         # Expert output orthogonality loss
│   └── load_balance.py          # Soft load balancing penalty
├── data/
│   └── domainbed.py        # DomainBed dataset loaders
├── analysis/
│   ├── spline_complexity.py     # Roughness, curvature, magnitude metrics
│   ├── expert_knockout.py       # Test-time expert ablation
│   ├── routing_jsd.py           # JSD between domain routing distributions
│   └── gradcam.py               # Per-expert Grad-CAM visualization
├── scripts/
│   ├── run_all.sh               # Reproduce all main results
│   └── run_ablations.sh         # Reproduce ablation studies
├── train.py
├── eval.py
└── requirements.txt
```

---

## Analysis & Visualization

### Spline Complexity

```bash
# Compute roughness, curvature, and weight magnitude per expert
python analysis/spline_complexity.py --checkpoint checkpoints/pacs_best.pth
```

### Expert Knockout

```bash
# Measure accuracy drop when disabling each expert at test time
python analysis/expert_knockout.py --checkpoint checkpoints/pacs_best.pth --dataset PACS
```

### Grad-CAM Attention Maps

```bash
# Generate per-expert Grad-CAM visualizations
python analysis/gradcam.py --checkpoint checkpoints/pacs_best.pth --domain cartoon --num_images 10
```

### Routing Analysis

```bash
# Compute JSD between domain-pair routing distributions
python analysis/routing_jsd.py --checkpoint checkpoints/pacs_best.pth --dataset PACS
```

---

## Key Findings

**1. Complexity gradient is real.** Spline roughness increases 3.4× and weight magnitude 4.6× from G=3 to G=20.

**2. Heterogeneity is the key.** All uniform grid configs (G=3/10/20) perform identically at ~76.3%. The heterogeneous [3,5,10,20] reaches 77.9%.

**3. Experts are complementary.** Fine experts (G=20) are the most critical single experts (−2.42% on Sketch when removed). But removing ALL fine experts hurts more than removing ALL coarse → both groups are essential.

**4. Router learns domain-aware selection.** Sketch routing diverges from Photo (JSD=0.149), while Art Painting and Photo route nearly identically (JSD=0.006).

---

## Citation

```bibtex
@inproceedings{kanmoa2026,
  title={Coarse Experts Generalize, Fine Experts Memorize: Spectral Decomposition of Adapters for Parameter-Efficient Domain Generalization},
  author={Mahmoud Soliman, Mohamed S. Shehata},
  booktitle={DG-EBF Workshop at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## Acknowledgments

This work builds on [DomainBed](https://github.com/facebookresearch/DomainBed), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [efficient-kan](https://github.com/Blealtan/efficient-kan). We thank the authors of [MoA (Lee et al., WACV 2025)](https://arxiv.org/abs/XXXX.XXXXX) for the baseline framework.

## License

This project is released under the [MIT License](LICENSE).
