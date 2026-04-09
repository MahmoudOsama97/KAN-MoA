# KAN-MoA: Kolmogorov-Arnold Network Mixture of Adapters for Domain Generalization

## Detailed Implementation Analysis

---

## Table of Contents

1. [Core Idea](#1-core-idea)
2. [Background: What is a KAN?](#2-background-what-is-a-kan)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Adapter Variants](#4-adapter-variants)
5. [Routing Mechanisms](#5-routing-mechanisms)
6. [Injection Pipeline](#6-injection-pipeline)
7. [Training Algorithm: KAN_MOA](#7-training-algorithm-kan_moa)
8. [Regularization Losses](#8-regularization-losses)
9. [Configuration & Hyperparameters](#9-configuration--hyperparameters)
10. [End-to-End Data Flow](#10-end-to-end-data-flow)
11. [Design Decisions & Stability Mechanisms](#11-design-decisions--stability-mechanisms)

---

## 1. Core Idea

### The Problem

Standard Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA model weight updates as **low-rank linear** transformations:

$$\Delta W = B \times A, \quad W_{\text{new}} = W_{\text{frozen}} + \Delta W$$

This assumes the adaptation needed for new domains is a **linear** function of the latent representation. However, real-world domain shifts (e.g., Photo → Sketch, clear weather → fog) involve **non-linear** transformations — texture distortion, frequency shifts, and style changes that linear adapters cannot disentangle.

### The KAN-MoA Solution

KAN-MoA replaces linear adapters with **Kolmogorov-Arnold Networks (KANs)** — networks that learn **functions on edges** (B-splines) rather than scalar weights on nodes. Each adapter becomes a learnable non-linear function capable of modeling complex domain shifts.

The key insight is that **domain invariance is linked to signal frequency**:

| Feature Type | Frequency Content | KAN Grid Size | Role |
|:---|:---|:---|:---|
| **Invariant (shape/geometry)** | Low-frequency, smooth | Small ($G = 3$) | Captures domain-invariant structure |
| **Domain-specific (texture/style)** | High-frequency, detailed | Large ($G = 20$) | Absorbs domain-specific noise |

By constructing a **Mixture-of-Experts (MoE)** where each expert has a **different B-spline grid resolution**, we create a spectral decomposition of the adaptation: coarse experts learn smooth invariant features, fine experts absorb textured noise. During test-time on unseen domains, the invariant features from coarse experts dominate.

### Theoretical Grounding

From the KAN generalization bounds (Zhang & Zhou, 2025):

$$\text{Generalization Error} \propto \frac{\tilde{\alpha}^3 \log(2\tilde{d}\tilde{p})}{n}$$

where $\tilde{p}$ is the number of B-spline grid intervals ($G$). **Reducing $G$ directly reduces the complexity class**, making an expert with $G=3$ *mathematically incapable* of overfitting to high-frequency noise. This provides a stronger inductive bias for generalization than simply reducing LoRA's rank $r$.

---

## 2. Background: What is a KAN?

### The Kolmogorov-Arnold Representation Theorem

The classical theorem states that any continuous multivariate function can be decomposed as a sum of compositions of univariate functions and addition:

$$f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

KANs operationalize this by placing **learnable activation functions on edges** (rather than fixed activations on nodes as in MLPs).

### KANLinear — The Atomic Layer

Implemented in `domainbed/models/kan.py`, a single `KANLinear(in_features, out_features, grid_size)` layer computes:

$$y = \underbrace{W_{\text{base}} \cdot \sigma(x)}_{\text{Linear base}} + \underbrace{\sum_{i} c_i \cdot B_i(x)}_{\text{B-spline component}}$$

where:
- $W_{\text{base}} \in \mathbb{R}^{\text{out} \times \text{in}}$ — a standard linear weight matrix.
- $\sigma(\cdot)$ — base activation function (SiLU by default).
- $B_i(x)$ — B-spline basis functions of order 3 defined on a fixed grid.
- $c_i$ — learnable spline coefficients (the `spline_weight` parameter).

**Parameters of `KANLinear`:**

| Parameter | Shape | Description |
|:---|:---|:---|
| `base_weight` | `(out, in)` | Linear component weight |
| `spline_weight` | `(out, in, G + k)` | B-spline coefficients where $G$ = `grid_size`, $k$ = `spline_order` |
| `spline_scaler` | `(out, in)` | Per-edge scale factor for spline output |
| `grid` (buffer) | `(in, G + 2k + 1)` | Non-learnable grid knot positions |

**The grid** is the crucial hyperparameter:
- Default range: $[-1, 1]$
- Grid size $G$: number of intervals. More intervals → more B-spline basis functions → higher-frequency functions can be represented.
- Grid with $G = 3$ → only 6 ($G + k = 3 + 3$) spline coefficients per edge → smooth, low-frequency functions.
- Grid with $G = 20$ → 23 coefficients per edge → can represent rapid oscillations.

**Forward pass** (simplified):

```
1. Compute B-spline bases: bases[b, in, G+k] from x[b, in]
2. base_output = Linear(SiLU(x), W_base)
3. spline_output = Linear(bases.flatten(), scaled_spline_weight.flatten())
4. output = base_output + spline_output
```

**Memory optimization**: `torch.utils.checkpoint` is used during training to avoid storing the full B-spline expansion tensor `(batch, in, G+k)` in memory.

**Diagnostic probe**: The forward pass includes a runtime check (sampled at 0.1% of batches) that warns if >50% of input values fall outside the $[-1, 1]$ grid range — indicating the spline bases won't activate properly.

---

## 3. Architecture Deep Dive

### The Injected Layer Structure

Each dense layer (`nn.Linear`) targeted for injection gets **replaced** by one of three adapter variants, all following this pattern:

```
Input x ──► Frozen Linear W_frozen ──► base_output
     │
     └──► Router(x) ──► gating weights [g_1, ..., g_N]
     │
     └──► Expert_1(x) ──► weighted by g_1 ──┐
     └──► Expert_2(x) ──► weighted by g_2 ──┤
     └──► Expert_3(x) ──► weighted by g_3 ──┤──► Σ (weighted sum) ──► adapter_output
     └──► Expert_N(x) ──► weighted by g_N ──┘

Final output = base_output + adapter_output
```

This is a **residual connection**: the frozen pretrained weights always provide the base computation, and the MoE adapters add a learned correction.

### Expert Internal Architecture (The "Sandwich")

Each expert $E_k$ uses a bottleneck design:

$$E_k(x) = W_{\text{up}}^{(k)} \circ \Phi_k \circ \text{Norm} \circ W_{\text{down}}^{(k)}(x)$$

| Step | Operation | Shape Transform |
|:---|:---|:---|
| 1. Down-project | $W_{\text{down}}^{(k)}: \mathbb{R}^{d} \to \mathbb{R}^{r_k}$ | `(B, S, d)` → `(B, S, r_k)` |
| 2. Normalize | LayerNorm on the $r_k$ dimension | `(B*S, r_k)` → `(B*S, r_k)` |
| 3. KAN transform | $\Phi_k: \mathbb{R}^{r_k} \to \mathbb{R}^{r_k}$ with grid $G_k$ | `(B*S, r_k)` → `(B*S, r_k)` |
| 4. Up-project | $W_{\text{up}}^{(k)}: \mathbb{R}^{r_k} \to \mathbb{R}^{d'}$ | `(B, S, r_k)` → `(B, S, d')` |

The KAN is applied in the **bottleneck dimension** ($r_k$, typically 2-32), not in the full feature dimension ($d = 768$). This makes KAN computation feasible inside a Transformer.

---

## 4. Adapter Variants

Implemented in `domainbed/models/kan_adapters.py`, three adapter types are available:

### 4.1. `kan_kronecker` — MOEKANKroneckerInjectedLinear

**Best performing variant.** Uses Kronecker product decomposition for the down/up projections.

**Down-projection**: `KroneckerLinear(d_in, r, n=r)`
- Constructs $W = A \otimes B$ where $A \in \mathbb{R}^{n \times n}$ and $B \in \mathbb{R}^{(r/n) \times (d/n)}$.
- The full weight matrix $W \in \mathbb{R}^{r \times d}$ is never stored; it's computed on-the-fly via `torch.kron()`.
- Parameters: $n^2 + \frac{r \cdot d}{n^2}$ instead of $r \cdot d$ — significant savings for large $d$.
- Falls back to dense `nn.Linear` if dimensions aren't divisible by $n$.

**Middle**: `KANLinear(r, r, grid_size=G_k)` with `LayerNorm(r)` pre-normalization.

**Up-projection**: `KroneckerLinear(r, d_out, n=r)` — same Kronecker structure.

**Additional features**:
- `InstanceNorm1d(r)` on bottleneck outputs.
- KAN weights initialized near zero (`std = kan_init_scale * 0.01`) for stable training start.
- Output scaled by `kan_init_scale` (default 0.1) before adding to base output.

**Forward pass pseudocode**:
```python
base_out = frozen_linear(x)                    # (B, S, d_out)
x_flat = flatten(x)                            # (B*S, d_in)
routing = softmax(router(x_flat))              # (B*S, N_experts)

output = zeros(B*S, d_out)
for i in range(N_experts):
    h = kronecker_down[i](x)                   # (B, S, r_i)
    h = layer_norm[i](flatten(h))              # (B*S, r_i)
    h = kan[i](h)                              # (B*S, r_i)
    h = kronecker_up[i](reshape(h))            # (B, S, d_out)
    output += routing[:, i:i+1] * flatten(h) * kan_init_scale

return base_out + reshape(output)
```

### 4.2. `kan_lora` — MOEKANLoraInjectedLinear

Uses standard linear layers for down/up projections (like traditional LoRA).

**Down-projection**: `nn.Linear(d_in, r, bias=False)` — initialized with Kaiming uniform.
**Middle**: `KANLinear(r, r, grid_size=G_k)` with `LayerNorm(r)` pre-normalization.
**Up-projection**: `nn.Linear(r, d_out, bias=False)` — initialized to **zeros** (standard LoRA practice).

**Training noise**: Adds Gaussian noise to router logits during training: `logits += N(0, 1/N_experts)` to encourage exploration.

**Dropout**: Applied after the KAN transformation (rate 0.1).

### 4.3. `pure_kan` — MOEPureKANInjectedLinear

Uses KAN layers for **both** the down-projection and up-projection (fully non-linear adapter).

**Down-projection**: `KANLinear(d_in, r, grid_size=G_k)` — full KAN layer.
**Up-projection**: `KANLinear(r, d_out, grid_size=G_k)` — initialized to **zero** (all weights zeroed).

**Critical stability fix**: Up-projection KAN is zero-initialized:
```python
kan_up.base_weight.zero_()
kan_up.spline_weight.zero_()
kan_up.spline_scaler.zero_()
```
This ensures the adapter starts as an identity-plus-zero residual, preventing large initial perturbations.

**Parameter comparison** (for a 768→768 layer with 4 experts, r=8, G=5):

| Variant | Down Params | KAN Params | Up Params | Total |
|:---|:---|:---|:---|:---|
| `kan_lora` | $768 \times 8 = 6144$ | $8 \times 8 \times 8 = 512$ | $8 \times 768 = 6144$ | ~51K (×4) |
| `kan_kronecker` | $8^2 + 96^2 ≈ 9280$ | $512$ | $9280$ | ~76K (×4) |
| `pure_kan` | $768 \times 8 \times 8 ≈ 49K$ | — | $8 \times 768 \times 8 ≈ 49K$ | ~392K (×4) |

---

## 5. Routing Mechanisms

Three router types determine how input tokens are distributed across experts:

### 5.1. `cosine` — CosineTopKGate (Default)

```python
class CosineTopKGate(nn.Module):
    expert_embeddings: Parameter(num_experts, d_in)  # orthogonally initialized
```

- Normalizes both input and expert embeddings to unit vectors.
- Routing logits = cosine similarity: `logits = x_norm @ expert_norm.T`
- Softmax produces soft gating weights.
- Expert embeddings are initialized with `nn.init.orthogonal_()` to encourage diverse initial routing.

**Trainable parameters**: `num_experts × d_in` (e.g., $4 \times 768 = 3072$).

### 5.2. `kan` — KANRouter

```python
class KANRouter(nn.Module):
    layer: KANLinear(d_in, num_experts, grid_size=3)
```

- Uses a KAN layer as the routing function itself.
- Non-linear routing boundaries: each expert's gate is a **learned B-spline function** of the input.
- Grid size fixed at 3 (simple routing, fast computation).

**Trainable parameters**: `d_in × num_experts × (3 + 3) + d_in × num_experts ≈ d_in × num_experts × 7` (e.g., $768 \times 4 \times 7 ≈ 21K$).

### 5.3. `spectral` — SpectralRouter

```python
class SpectralRouter(nn.Module):
    spec_norm: LayerNorm(d_in // 2 + 1)
    gate: Linear(d_in // 2 + 1, num_experts, bias=False)
```

- Applies **Real FFT** to input tokens along the embedding dimension.
- Computes the **power spectrum** (magnitude of FFT coefficients).
- Normalizes the spectrum with LayerNorm.
- Routes based on frequency content: high-frequency-dominated tokens → fine experts, low-frequency → coarse experts.
- Directly implements the "frequency-aware routing" hypothesis.

**Trainable parameters**: `(d_in//2 + 1) × num_experts + (d_in//2 + 1)` (LayerNorm) ≈ ~1.5K for typical sizes.

---

## 6. Injection Pipeline

The `inject_kan_moe()` function in `kan_adapters.py` handles the model surgery:

### Step 1: Module Discovery

```python
def _find_modules(model, target_list):
    for name, module in model.named_modules():
        if any(t in name for t in target_list) and isinstance(module, nn.Linear):
            # Find parent module and child attribute name
            matches.append((parent, child_name, module))
```

**Injection targets** are specified as comma-separated substrings (e.g., `"qkv,proj,fc1,fc2"`). Any `nn.Linear` whose full module name contains one of these substrings gets replaced.

For ViT-Base with `qkv,proj`:
- `blocks.*.attn.qkv` — the fused Query/Key/Value projection (768→2304)
- `blocks.*.attn.proj` — the attention output projection (768→768)
- This means **24 layers** (12 blocks × 2 layers/block) are adapted.

With `qkv,proj,fc1,fc2`:
- Additionally `blocks.*.mlp.fc1` (768→3072) and `blocks.*.mlp.fc2` (3072→768)
- **48 layers** total.

### Step 2: Layer Replacement

For each matched `nn.Linear`:

1. Create new adapter layer (one of the three variants).
2. Copy the frozen weight: `new_layer.linear.weight = child.weight`
3. Replace in parent: `setattr(parent, name, new_layer)`

### Step 3: Freeze/Unfreeze

```python
for n, p in model.named_parameters():
    if "expert" in n or "router" in n or "phm" in n or "base_weight" in n or "spline" in n:
        p.requires_grad = True
    else:
        p.requires_grad = False
```

Only adapter-related parameters are trainable. The entire pretrained backbone is frozen.

### Backbone Integration

In `backbones.py`, the model `"nf_vitbase_moekan"` triggers KAN injection:

```python
elif name == "nf_vitbase_moekan":
    network = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
    n_outputs = network.num_features   # 768
    network.head = Identity()          # Remove classification head
    network = inject_kan_moe(network, args)  # Inject adapters
```

The backbone is **CLIP ViT-B/16** pretrained on LAION-2B (via timm).

---

## 7. Training Algorithm: KAN_MOA

Defined in `algorithms.py`, `KAN_MOA` extends `ERM` (Empirical Risk Minimization) with enhanced KAN-specific training.

### Initialization

```python
class KAN_MOA(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super().__init__(...)  # Creates featurizer + classifier, optimizer
```

Inherits from `ERM`:
- `self.featurizer` = `ResNet` wrapper → `get_backbone("nf_vitbase_moekan", ...)` → CLIP ViT-B/16 + KAN injection
- `self.classifier` = `nn.Linear(768, num_classes)`
- `self.network` = `Sequential(featurizer, classifier)`
- `self.optimizer` = Adam (default), applied to all parameters (but only adapter params have `requires_grad=True`)

### Update Step

Each `update(x, y)` call receives a list of mini-batches from different source domains:

```python
def update(self, x, y, **kwargs):
    # x = [x_domain_0, x_domain_1, x_domain_2]  (list of tensors)
    # y = [y_domain_0, y_domain_1, y_domain_2]

    all_x = torch.cat(x)   # Concatenate all domains
    all_y = torch.cat(y)

    logits = self.predict(all_x)          # Forward through network
    cls_loss = F.cross_entropy(logits, all_y)

    # Collect all MOE adapter modules
    adapters = [m for m in self.network.modules() if isinstance(m, MOEBase)]

    # Compute auxiliary losses from each adapter
    for m in adapters:
        lip_loss += m.compute_lipschitz_loss()
        ortho_loss += m.compute_orthogonality_loss()
        load_balance_loss += m.compute_load_balance_loss()
        grid_entropy_loss += m.compute_grid_entropy_loss()

    # Algorithm-level losses
    comp_loss = self.compute_complexity_invariance_loss(adapters)
    domain_routing_loss = self.compute_domain_routing_loss(adapters, x)

    # Weighted total
    loss = (cls_loss
          + lip_loss * λ_lip
          + ortho_loss * λ_ortho
          + comp_loss * λ_comp
          + load_balance_loss * λ_lb
          + grid_entropy_loss * λ_ge
          + domain_routing_loss * λ_dr)

    loss.backward()

    # KAN gradient scaling for stability
    for m in adapters:
        m.scale_kan_gradients()

    optimizer.step()
```

### Error Recovery

A `try/except` block wraps the entire update. If any loss computation or gradient operation fails, the algorithm **falls back to pure ERM** (only cross-entropy loss), ensuring training never crashes.

---

## 8. Regularization Losses

### 8.1. Classification Loss

$$\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(\hat{y}, y)$$

Standard cross-entropy on concatenated batches from all source domains.

### 8.2. Lipschitz Regularization ($\lambda = 0.001$)

$$\mathcal{L}_{\text{lip}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \left( \|W_{\text{spline}}^{(k)}\|_F^2 + 0.1 \cdot \|W_{\text{base}}^{(k)}\|_F^2 \right)$$

Computed per-adapter in `MOEBase.compute_lipschitz_loss()`. Penalizes large spline weights to encourage **smooth B-spline functions**. A smoother function has a smaller Lipschitz constant, meaning small input perturbations (domain shifts) produce small output changes.

### 8.3. Pairwise Orthogonality Loss ($\lambda = 0.01$)

$$\mathcal{L}_{\text{ortho}} = \frac{1}{N} \sum_{b=1}^{N} \text{mean}\left(\text{ReLU}\left(|S_{ij}^{(b)}| - 0.3\right)\right) \quad \text{for } i \neq j$$

where $S^{(b)} = \hat{E}^{(b)} {\hat{E}^{(b)}}^{\top}$ is the cosine similarity matrix of normalized expert outputs $\hat{E}^{(b)} \in \mathbb{R}^{K \times r}$ for sample $b$.

- Computed from `last_expert_outputs` stored during forward pass.
- **Threshold of 0.3**: Only penalizes expert pairs with similarity > 30%.
- Allows some beneficial correlation while preventing expert collapse (all experts learning the same thing).
- Enforces the spectral decomposition: coarse experts learn different representations from fine experts.

### 8.4. Load Balancing Loss ($\lambda = 0.001$)

$$\mathcal{L}_{\text{lb}} = \sum_{k} \text{ReLU}\left(\frac{0.5}{K} - \bar{g}_k\right) + \sum_{k} \text{ReLU}\left(\bar{g}_k - \frac{2.0}{K}\right)$$

where $\bar{g}_k = \frac{1}{N}\sum_{b} g_k(x_b)$ is the average routing weight for expert $k$.

- **Soft penalty with thresholds**: Only activates if an expert receives < 50% or > 200% of its fair share.
- This is softer than standard Switch Transformer load balancing — it allows some imbalance for specialization.
- Prevents degenerate routing where all tokens go to one expert.

### 8.5. Grid Entropy Loss ($\lambda = 0.001$)

$$\mathcal{L}_{\text{ge}} = \left| H(g) - H_{\text{target}} \right|$$

where $H(g) = -\sum_k \bar{g}_k \log \bar{g}_k$ and $H_{\text{target}} = \min(0.4, 0.5 \cdot \log K)$.

- Target entropy is **below maximum** ($\log K$): allows focused routing while maintaining diversity.
- For $K=4$: $H_{\text{target}} = 0.4$ vs. $H_{\text{max}} = 1.386$. This means the router should be somewhat focused (not uniform).
- Balances between expert utilization (high entropy → uniform) and specialization (low entropy → sparse).

### 8.6. Complexity Invariance Loss ($\lambda = 0.01$)

$$\mathcal{L}_{\text{comp}} = \text{Var}\left(\left[ C_1, C_2, \dots, C_K \right]\right), \quad C_k = \frac{1}{|\theta_k|}\sum_{p \in \theta_k} |p|$$

- $C_k$ is the mean absolute value of all trainable parameters in expert $k$.
- Penalizes **variance** of complexity measures across experts.
- Ensures no single expert dominates by having disproportionately large parameters.

### 8.7. Domain Routing Loss ($\lambda = 0.01$, disabled by default)

$$\mathcal{L}_{\text{dr}} = \text{clamp}\left(-\text{KL}\left(\mathcal{R}_d \| \mathcal{U}\right), -2, 2\right)$$

where $\mathcal{R}_d = \mathbb{E}_{x \sim d}[g(x)]$ is the mean routing distribution for domain $d$ and $\mathcal{U}$ is the uniform distribution.

- Minimizes negative KL divergence from uniform → **encourages non-uniform routing per domain**.
- Each domain should develop a specialized routing pattern (e.g., sketches → coarse experts, photos → balanced).
- Clamped to $[-2, 2]$ to prevent loss explosion.
- **Disabled by default** (`enable_domain_routing = False`), as early experiments showed mixed results.

### Loss Weight Summary

| Loss | Weight | Default | Controls |
|:---|:---|:---|:---|
| Cross-entropy | 1.0 | Always on | Task accuracy |
| Lipschitz | `lipschitz_weight` | 0.001 | Spline smoothness |
| Orthogonality | `ortho_weight` | 0.01 | Expert diversity |
| Complexity inv. | `complexity_weight` | 0.01 | Expert balance |
| Load balance | `load_balance_weight` | 0.001 | Routing fairness |
| Grid entropy | `grid_entropy_weight` | 0.001 | Routing focus |
| Domain routing | `domain_routing_weight` | 0.01 | Domain specialization |

---

## 9. Configuration & Hyperparameters

### Command-Line Arguments

```bash
python train_all.py <experiment_name> \
    --data_dir data \
    --algorithm KAN_MOA \
    --dataset OfficeHome \
    --model nf_vitbase_moekan \
    --adapter_type kan_kronecker \        # {kan_lora, kan_kronecker, pure_kan}
    --num_experts 4 \                      # Number of experts per layer
    --expert_ranks "2,4,8,16" \           # Bottleneck dim per expert (comma-separated)
    --kan_grids "3,3,20,20" \             # Grid size per expert (comma-separated)
    --injection_targets "qkv,proj" \       # Which layers to adapt
    --router_type spectral \               # {cosine, kan, spectral}
    --seed 1 \
    --steps 10000
```

### Key Architectural Parameters

| Parameter | CLI Flag | Default | Description |
|:---|:---|:---|:---|
| Adapter type | `--adapter_type` | `kan_lora` | Expert architecture variant |
| Num. experts | `--num_experts` | `4` | Experts per injected layer |
| Expert ranks | `--expert_ranks` | `"8"` | Bottleneck dim per expert (cycles if shorter than num_experts) |
| KAN grids | `--kan_grids` | `"5"` | B-spline grid size per expert (cycles if shorter) |
| Router | `--router_type` | `cosine` | Routing mechanism |
| Targets | `--injection_targets` | `qkv,proj,fc1,fc2` | Layer name substrings to inject into |
| Rank | `--r` | `4` | Global rank (used by some other adapter types) |

### How `expert_ranks` and `kan_grids` Work

These are parsed by `_parse_list()`:
- `"8"` → `[8, 8, 8, 8]` (broadcast to all experts)
- `"2,4,8,16"` → `[2, 4, 8, 16]` (one per expert)
- `"3,5"` with 4 experts → `[3, 5, 3, 5]` (cycled)

**Example multi-resolution configuration:**
```bash
--num_experts 4 \
--expert_ranks "8,8,8,8" \
--kan_grids "3,3,20,20"
```
- Experts 0-1: Grid=3 (coarse, shape experts)
- Experts 2-3: Grid=20 (fine, texture experts)

### hparams_registry.py Defaults

Additional hyperparameters set in the registry (configurable via config files):

| Parameter | Default | Description |
|:---|:---|:---|
| `lr` | `5e-5` | Learning rate |
| `weight_decay` | `0.0` | Weight decay |
| `batch_size` | `32` | Per-domain batch size |
| `optimizer` | `adam` | Optimizer type |
| `kan_init_scale` | `0.1` | Initial scale for KAN outputs |
| `kan_grad_warmup_steps` | `1000` | Steps over which KAN gradient scale ramps up |
| `router_noise` | `0.1` | Noise added to router logits |
| `router_temperature` | `1.0` | Temperature for softmax routing |

---

## 10. End-to-End Data Flow

### Training (one step)

```
Input: x_d for d in {domain_0, domain_1, domain_2} (3 source domains)

1. Concatenate: all_x = cat([x_0, x_1, x_2])    # (3*B, C, H, W)

2. Featurizer:
   a. CLIP ViT-B/16 patch embedding              # (3*B, 197, 768)
   b. For each of 12 transformer blocks:
      i.   LayerNorm
      ii.  ATTENTION:
           - qkv = MOEKANAdapter(x)              # (3*B, 197, 2304) via MoE
             → Router computes gating per token
             → Each expert: Kronecker↓ → Norm → KAN → Kronecker↑
             → Weighted sum + frozen qkv output
           - attn = softmax(Q @ K^T / sqrt(d)) @ V
           - proj = MOEKANAdapter(attn_out)       # (3*B, 197, 768) via MoE
      iii. Residual + LayerNorm
      iv.  MLP:
           - fc1 = MOEKANAdapter(x)               # (3*B, 197, 3072) [if targeted]
           - GELU
           - fc2 = MOEKANAdapter(x)               # (3*B, 197, 768)  [if targeted]
      v.   Residual
   c. CLS token extraction                        # (3*B, 768)
   d. Dropout

3. Classifier: logits = Linear(768, num_classes)(features)    # (3*B, C)

4. Loss computation:
   a. cls_loss = CrossEntropy(logits, all_y)
   b. Iterate over all MOEBase modules:
      - Lipschitz loss (from KAN spline weights)
      - Orthogonality loss (from stored expert outputs)
      - Load balance loss (from stored routing weights)
      - Grid entropy loss (from stored routing weights)
   c. Complexity invariance loss (cross-expert parameter variance)
   d. Domain routing loss (routing distribution diversity across domains)
   e. Total loss = weighted sum of all components

5. Backward + gradient scaling (KAN gradients scaled by 0.1, ramping to 1.0)
6. Optimizer step
```

### Inference

At test time, the forward pass is identical except:
- No router noise injection
- No gradient checkpointing
- Only `predict(x) = network(x)` is called
- All routing is deterministic (softmax of clean logits)

---

## 11. Design Decisions & Stability Mechanisms

### Why Normalize Before KAN?

KAN B-splines are defined on a fixed grid (default $[-1, 1]$). If input activations have a different range (e.g., $[0, 50]$), most B-spline bases evaluate to zero and the spline component is ineffective. Each expert uses `LayerNorm(r)` on its bottleneck output before feeding it to the KAN to ensure inputs are approximately in $[-1, 1]$.

### Why Zero-Initialize Up-Projections?

LoRA's key insight: initialize the adapter so its initial output is **zero**. This means:
$$y_{\text{initial}} = W_{\text{frozen}} x + 0 = W_{\text{frozen}} x$$

The model starts from its pretrained behavior and gradually learns corrections. Without this, random adapter outputs would destroy pretrained features on the first step.

- `kan_lora`: `nn.init.zeros_(up.weight)`
- `kan_kronecker`: `nn.init.normal_(spline_weight, std=0.001)` + output scaled by 0.1
- `pure_kan`: All KAN parameters in up-projection zeroed

### KAN Gradient Scaling

KAN spline weights can have larger gradients than standard linear layers due to the B-spline basis expansion. The `scale_kan_gradients()` method:

1. Scales all KAN parameter gradients by `kan_grad_scale` (starts at 0.1).
2. Gradually increases the scale by 0.0001 per step (warm-up to 1.0).
3. Prevents early training instability where large KAN gradients overwhelm the optimizer.

### Soft Expert Output Aggregation

All three variants use **soft routing** (all experts contribute, weighted by softmax probabilities) rather than top-k hard routing (only the top expert processes the token). This:
- Provides smoother gradients for all experts.
- Avoids the "dead expert" problem common in hard MoE.
- The entropy loss encourages some focusing without full sparsity.

### Expert Output Padding for Loss Computation

Since experts can have different bottleneck dimensions ($r_k$), their hidden outputs are **zero-padded to the maximum $r$** before stacking into a `(batch, num_experts, max_r)` tensor. This enables batched orthogonality loss computation.

### Error Recovery

Every forward pass and loss computation is wrapped in `try/except`. If any component fails:
- Adapter forward: returns `self.linear(x)` (frozen output only).
- KAN_MOA update: falls back to pure ERM (cross-entropy only).

This ensures robustness during experimentation with novel configurations.

---

## File Structure Summary

```
domainbed/
├── models/
│   ├── kan.py                    # KANLinear & KAN: B-spline network implementation
│   ├── kan_adapters.py           # MOE adapter variants + injection logic
│   └── lora/
│       └── lora_util.py          # Original LoRA/MoE-LoRA (comparison baselines)
├── networks/
│   ├── networks.py               # Featurizer factory (ResNet wrapper)
│   └── backbones.py              # Backbone registry (triggers injection for nf_vitbase_moekan)
├── algorithms/
│   └── algorithms.py             # KAN_MOA training algorithm + all DG baselines
├── hparams_registry.py           # Default hyperparameters for all algorithms
└── trainer.py                    # Training loop (dataset setup, checkpointing, evaluation)

train_all.py                      # Entry point: CLI parsing, experiment orchestration
config.yaml                       # Base configuration (model, SWAD, etc.)
```
