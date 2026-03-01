Here is a comprehensive description of the **KAN-MoA (Kolmogorov-Arnold Network Mixture of Adapters)** research proposal. This document covers the theoretical motivation, mathematical formulation, specific sub-ideas, and the hyperparameter configuration.

---

# Research Proposal: KAN-MoA
## Spectral Domain Generalization via Complexity-Aware Functional Adapters

### 1. Introduction & Motivation

**The Problem: Linearity vs. Reality**
Current Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA (Low-Rank Adaptation), assume that domain shifts can be modeled by linear transformations of the latent space ($W_{new} = W_{old} + \Delta W$, where $\Delta W = B \times A$). However, real-world domain shifts—such as the transition from a "Photo" to a "Sketch" or changes in weather conditions—are highly non-linear transformations involving texture distortion and frequency shifts. Standard linear adapters struggle to disentangle invariant geometric features (low-frequency content) from spurious style features (high-frequency noise), leading to overfitting on source domains.

**The Solution: Functional Learning**
We propose replacing linear adapters with **Kolmogorov-Arnold Networks (KANs)**. Unlike Multi-Layer Perceptrons (MLPs) which learn weights, KANs learn *functions* (B-splines) on edges. This allows us to mathematically decouple model complexity from parameter count.

**The Core Hypothesis**
We posit that **Domain Invariance is a function of Signal Frequency.**
*   **Invariant Features (Shape/Geometry):** Can be represented by smooth, low-complexity functions (Low Grid Size).
*   **Domain-Specific Features (Texture/Style):** Require high-frequency, "wiggly" functions (High Grid Size).

By constructing a Mixture-of-Experts (MoE) where experts are distinguished by their **Functional Resolution (Grid Size)** rather than just capacity, we can mechanically filter out domain-specific noise.

---

### 2. Theoretical Foundation

Our approach is grounded in the Generalization Bounds derived by *Zhang & Zhou (2025)* for KANs.

**Theorem (KAN Complexity):**
The generalization error of a KAN is bounded by its complexity measure $\tilde{\alpha}$:
$$ \text{Generalization Error} \propto \frac{\tilde{\alpha}^3 \log(2\tilde{d}\tilde{p})}{n} $$
Where $\tilde{p}$ is the number of intervals in the B-spline grid ($G$).

**Implication for Domain Generalization:**
Reducing the Grid Size ($G$) directly reduces the complexity class of the function. An expert with $G=3$ is mathematically incapable of overfitting to high-frequency noise (texture), guaranteeing that it only learns global trends (shape). This provides a stronger inductive bias for generalization than simply reducing the rank ($r$) in LoRA.

---

### 3. Architecture: The KAN-MoA Layer

We replace the standard Dense layers (or QKV projections) in a Vision Transformer (ViT) with the **Heterogeneous KAN-MoA Layer**.

#### **A. The Atomic Unit: KAN-LoRA (The Sandwich Architecture)**
To make KANs computationally feasible inside a Transformer, we use a bottleneck architecture that applies non-linearity only in a compressed latent space.

For an input $x \in \mathbb{R}^{d_{in}}$, the output of the $k$-th expert $E_k(x)$ is:
$$ E_k(x) = W_{up}^{(k)} \cdot \Phi_k \left( \text{Norm}(W_{down}^{(k)} x) \right) $$

*   **$W_{down}^{(k)} \in \mathbb{R}^{r \times d_{in}}$:** Linear Down-Projection to rank $r$.
*   **$\text{Norm}(\cdot)$:** Instance Normalization to ensure activations lie within the spline grid range (typically $[-3, 3]$).
*   **$\Phi_k: \mathbb{R}^r \to \mathbb{R}^r$:** The KAN Layer. It consists of learnable B-splines $\phi(z) = \sum_{i} c_i B_i(z)$ with grid size $G_k$.
*   **$W_{up}^{(k)} \in \mathbb{R}^{d_{out} \times r}$:** Linear Up-Projection.

#### **B. The Routing Mechanism**
We define a set of experts $\mathcal{E} = \{E_1, \dots, E_N\}$. The final output $y$ is a weighted sum:
$$ y = W_{frozen}x + \sum_{k=1}^{N} g_k(x) \cdot E_k(x) $$
Where $g(x)$ is a lightweight KAN-based router that outputs gating probabilities via Softmax.

---

### 4. Novel Sub-Ideas (The "Delta" over SOTA)

We introduce three specific mechanisms to enforce domain invariance.

#### **Idea 1: Multi-Resolution Spectral Experts (Foveated Vision)**
Instead of identical experts, we enforce a **Complexity Gradient**.
*   **Experts 1-2 (Shape Experts):** Initialized with **Coarse Grids ($G=3$)**. These act as "Low-Pass Filters," capturing only smooth, geometric transformations invariant across domains.
*   **Experts 3-4 (Texture Experts):** Initialized with **Fine Grids ($G=20$)**. These act as "High-Pass Filters," capturing high-frequency domain details.

#### **Idea 2: Orthogonality Loss (Diversity Enforcement)**
To prevent the "Texture Experts" from learning the "Shape" features (redundancy), we force the two groups to learn orthogonal representations.
$$ \mathcal{L}_{ortho} = \left\| \frac{\sum_{i \in \text{Coarse}} E_i(x)}{||\dots||} \cdot \frac{\sum_{j \in \text{Fine}} E_j(x)}{||\dots||} \right\|^2 $$
This penalizes the cosine similarity between the aggregate output of coarse experts and fine experts.

#### **Idea 3: Complexity-Invariant Routing**
We regularize the router to ensure routing decisions are based on semantic content, not domain style. We define the "Routing Distribution" for domain $d$ as $\mathcal{R}_d = \mathbb{E}_{x \sim d} [g(x)]$.
$$ \mathcal{L}_{inv} = \sum_{d_1, d_2 \in \mathcal{D}} || \text{Var}(\mathcal{R}_{d_1}) - \text{Var}(\mathcal{R}_{d_2}) ||^2 $$
This forces the router to utilize experts similarly across different domains, preventing it from creating a "Photo-only" expert.

---

### 5. Hyperparameters & Configuration

This setup is controlled by the following hyperparameters passed to `train_all.py`.

#### **Architectural Hyperparameters**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `adapter_type` | `kan_lora` | Using the Linear-KAN-Linear bottleneck structure. |
| `num_experts` | `4` | Total number of experts per layer. |
| `expert_ranks` | `[8, 8, 8, 8]` | The latent dimension ($r$) for each expert. Kept low for efficiency. |
| `kan_grids` | `[3, 3, 20, 20]` | **The Key Novelty.** Experts 0/1 are Coarse (Shape), Experts 2/3 are Fine (Texture). |
| `router_type` | `kan` | Using a KAN router ($G=3$) for non-linear decision boundaries. |
| `injection_targets` | `qkv, proj` | Applied to Attention Query/Key/Value and Output Projections. |

#### **Loss & Optimization Hyperparameters**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `enable_lipschitz` | `1` (True) | Enables L1 regularization on spline coefficients to enforce smoothness. |
| `lambda_lipschitz` | `0.01` | Weight of the smoothness penalty. |
| `enable_complexity` | `1` (True) | Enables the Router Invariance / Orthogonality loss. |
| `lambda_complexity` | `1.0` | Weight of the invariance penalty. |
| `batch_size` | `24` | Tuned to fit KAN memory requirements on 32GB VRAM. |

### 6. Expected Outcome
By training with this configuration, the model's **Coarse Experts** will learn a robust, domain-invariant representation of objects (shapes), while the **Fine Experts** absorb the domain-specific noise (texture). During inference on unseen domains, the robust features from the Coarse Experts will dominate, leading to higher accuracy compared to standard LoRA, which mixes these signals inextricably.