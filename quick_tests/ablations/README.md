# Ablation Study Guide

## Overview

This folder contains scripts for running comprehensive ablation studies across all datasets, covering both loss configurations and architectural hyperparameters.

## Ablation Categories

### 1. Loss Ablations (9 configs)

| Name | Load Balance | Grid Entropy | Pairwise Ortho | Ortho Weight | Load Weight |
|------|--------------|--------------|----------------|--------------|-------------|
| `baseline` | ❌ | ❌ | ❌ | 0.01 | 0.001 |
| `ortho_only` | ❌ | ❌ | ✅ | 0.01 | 0.001 |
| `ortho_strong` | ❌ | ❌ | ✅ | 0.1 | 0.001 |
| `load_balance` | ✅ | ❌ | ❌ | 0.01 | 0.001 |
| `load_balance_strong` | ✅ | ❌ | ❌ | 0.01 | 0.01 |
| `grid_entropy` | ❌ | ✅ | ❌ | 0.01 | 0.001 |
| `ortho_load` | ✅ | ❌ | ✅ | 0.01 | 0.001 |
| `all_features` | ✅ | ✅ | ✅ | 0.01 | 0.001 |
| `all_features_strong` | ✅ | ✅ | ✅ | 0.1 | 0.01 |

### 2. Number of Experts Sweep (4 configs)

| Config | Num Experts | Expert Ranks |
|--------|-------------|--------------|
| `experts2` | 2 | 2,4 |
| `experts4` | 4 | 2,4,8,16 |
| `experts6` | 6 | 2,4,8,16,32,64 |
| `experts8` | 8 | 1,2,4,8,16,32,64,128 |

### 3. KAN Grid Size Sweep (8 configs)

| Config | Grid Size | Description |
|--------|-----------|-------------|
| `grid2` | 2 | Minimal spline complexity |
| `grid3` | 3 | Default |
| `grid4` | 4 | Higher complexity |
| `grid5` | 5 | Moderate-high |
| `grid6` | 6 | High complexity |
| `grid8` | 8 | Very high complexity |
| `grid10` | 10 | Near maximum |
| `grid12` | 12 | Maximum complexity |

### 4. Injection Targets Sweep (4 configs)

| Config | Targets | Description |
|--------|---------|-------------|
| `inject_qkv` | qkv | Query, Key, Value only |
| `inject_proj` | proj | Projection only |
| `inject_qkv_proj` | qkv,proj | Both (default) |
| `inject_qkv_proj_mlp` | qkv,proj,mlp | All including MLP |

### 5. Router Type Sweep (4 configs)

| Config | Router | Description |
|--------|--------|-------------|
| `router_spectral` | spectral | Spectral-based routing (default) |
| `router_learned` | learned | Learned routing weights |
| `router_random` | random | Random routing (baseline) |
| `router_topk` | topk | Top-K expert selection |

### 6. Lipschitz Weight Sweep (5 configs)

| Config | Weight | Description |
|--------|--------|-------------|
| `lip_0p0` | 0.0 | No Lipschitz constraint |
| `lip_0p0001` | 0.0001 | Very weak |
| `lip_0p001` | 0.001 | Default |
| `lip_0p01` | 0.01 | Moderate |
| `lip_0p1` | 0.1 | Strong |

### 7. Complexity Weight Sweep (4 configs)

| Config | Weight | Description |
|--------|--------|-------------|
| `comp_0p0` | 0.0 | No complexity penalty |
| `comp_0p001` | 0.001 | Weak |
| `comp_0p01` | 0.01 | Default |
| `comp_0p1` | 0.1 | Strong |

### 8. Expert Ranks Sweep (5 configs)

| Config | Ranks | Num Experts |
|--------|-------|-------------|
| `ranks_2_4` | 2,4 | 2 |
| `ranks_2_4_8_16` | 2,4,8,16 | 4 |
| `ranks_2_4_8_16_32_64` | 2,4,8,16,32,64 | 6 |
| `ranks_1_2_4_8` | 1,2,4,8 | 4 (lower) |
| `ranks_4_8_16_32` | 4,8,16,32 | 4 (higher) |

### 9. Uniform Grid + Rank Configs (10 configs)

All experts have the **same rank** and **same grid size**.

| Config | Ranks | Grid | Description |
|--------|-------|------|-------------|
| `uniform_r2_g3` | 2,2,2,2 | 3 | Low rank, default grid |
| `uniform_r4_g3` | 4,4,4,4 | 3 | Medium rank, default grid |
| `uniform_r8_g3` | 8,8,8,8 | 3 | High rank, default grid |
| `uniform_r16_g3` | 16,16,16,16 | 3 | Very high rank, default grid |
| `uniform_r4_g2` | 4,4,4,4 | 2 | Medium rank, low grid |
| `uniform_r4_g4` | 4,4,4,4 | 4 | Medium rank, high grid |
| `uniform_r4_g5` | 4,4,4,4 | 5 | Medium rank, higher grid |
| `uniform_r4_g6` | 4,4,4,4 | 6 | Medium rank, very high grid |
| `uniform_r8_g4` | 8,8,8,8 | 4 | High rank, high grid |
| `uniform_r8_g5` | 8,8,8,8 | 5 | High rank, higher grid |

### 10. Mixed Grid Configs (4 configs)

Different **grid sizes per expert** to explore grid diversity.

| Config | Ranks | Grids | Description |
|--------|-------|-------|-------------|
| `mixgrid_2345_r4` | 4,4,4,4 | 2,3,4,5 | Uniform rank, ascending grids |
| `mixgrid_3456_r4` | 4,4,4,4 | 3,4,5,6 | Uniform rank, higher ascending grids |
| `mixgrid_2468_r4` | 4,4,4,4 | 2,4,6,8 | Uniform rank, even grids |
| `mixgrid_3333_r_mixed` | 2,4,8,16 | 3,3,3,3 | Mixed ranks, uniform grid |

### 11. Scaling Configs (4 configs)

**Grid size scales with rank** - exploring if higher-rank experts benefit from more grid points.

| Config | Ranks | Grids | Description |
|--------|-------|-------|-------------|
| `scale_low` | 2,4,8,16 | 2,3,4,5 | Low-rank experts get fewer grids |
| `scale_mid` | 4,8,16,32 | 3,4,5,6 | Mid-range scaling |
| `scale_high` | 8,16,32,64 | 4,5,6,8 | High-rank with high grids |
| `scale_inverse` | 16,8,4,2 | 5,4,3,2 | Inverse: high rank → low grid |

## Datasets

| Dataset | Steps | Time Limit | # Domains |
|---------|-------|------------|-----------|
| OfficeHome | 10000 | 24h | 4 |
| PACS | 10000 | 24h | 4 |
| VLCS | 10000 | 24h | 4 |
| TerraIncognita | 10000 | 24h | 4 |
| DomainNet | 15000 | 48h | 6 |

## Total Jobs

- 5 datasets × 67 configurations = **335 jobs**

Breakdown:
- Loss ablations: 5 × 9 = 45
- Num experts: 5 × 4 = 20
- KAN grids: 5 × 8 = 40
- Injection targets: 5 × 4 = 20
- Router types: 5 × 4 = 20
- Lipschitz weights: 5 × 5 = 25
- Complexity weights: 5 × 4 = 20
- Expert ranks: 5 × 5 = 25
- Uniform configs: 5 × 10 = 50
- Mixed grid configs: 5 × 4 = 20
- Scaling configs: 5 × 4 = 20

## Usage

### 1. Generate ablation scripts
```bash
cd /home/mosama97/scratch/MOKanA/quick_tests
bash generate_ablations.sh
```

### 2. Submit jobs

**Submit all ablations (all datasets, all categories):**
```bash
bash submit_ablations.sh all
```

**Submit for specific dataset:**
```bash
bash submit_ablations.sh PACS
bash submit_ablations.sh OfficeHome
bash submit_ablations.sh VLCS
bash submit_ablations.sh TerraIncognita
bash submit_ablations.sh DomainNet
```

**Submit specific category for all datasets:**
```bash
bash submit_ablations.sh all baseline   # Baseline only
bash submit_ablations.sh all experts    # Experts sweep
bash submit_ablations.sh all grid       # Grid sweep
bash submit_ablations.sh all router     # Router sweep
bash submit_ablations.sh all lip        # Lipschitz sweep
bash submit_ablations.sh all comp       # Complexity sweep
bash submit_ablations.sh all inject     # Injection targets
bash submit_ablations.sh all ranks      # Expert ranks
```

**Submit specific category for specific dataset:**
```bash
bash submit_ablations.sh PACS experts
bash submit_ablations.sh OfficeHome grid
bash submit_ablations.sh VLCS router
```

### 3. Monitor jobs
```bash
squeue -u $USER
```

### 4. Analyze results
```bash
bash analyze_ablations.sh
```

## Output Files

- Scripts: `ablations/test_<dataset>_<config>.sh`
- Logs: `ablations/output_<dataset>_<config>.out`
- Errors: `ablations/error_<dataset>_<config>.err`
- Results: `/home/mosama97/scratch/MOKanA/ablation_results.csv`

## Expected Insights

### Architectural Ablations
- **Num Experts**: Trade-off between capacity and efficiency
- **KAN Grids**: Spline complexity vs. overfitting
- **Injection Targets**: Where adaptation is most effective
- **Router Types**: Best routing strategy for domain generalization
- **Expert Ranks**: Optimal rank distribution for experts

### Loss Ablations
- **Lipschitz**: Effect of smoothness constraint
- **Complexity**: Effect of sparsity/simplicity penalty
- **Orthogonality**: Whether expert diversity helps
- **Load Balance**: Whether uniform expert usage helps
