# MOKanA: KAN-based Mixture of Adapters

## Recent Improvements & Enhancements

### Overview
This enhanced version of MOKanA includes several critical improvements designed to increase accuracy beyond the baseline MoA performance.

---

## 🚀 New Features

### 1. **Load Balancing Loss**
- **Purpose**: Prevents expert collapse where some experts are never used
- **Implementation**: Switch Transformer-style load balancing
- **Hyperparameter**: `load_balance_weight` (default: 0.01)
- **Enable/Disable**: `enable_load_balance` (default: true)

```python
# Encourages uniform expert utilization
load_loss = num_experts * sum(f_i * p_i)
```

### 2. **Pairwise Orthogonality Loss**
- **Purpose**: Forces experts to learn diverse transformations
- **Implementation**: Cosine similarity penalty between all expert pairs
- **Hyperparameter**: `ortho_weight` (default: 0.1, can increase to 0.5)
- **Enable/Disable**: `enable_pairwise_ortho` (default: true)

```python
# Minimizes similarity between different experts
ortho_loss = mean(cosine_sim(expert_i, expert_j)^2) for i != j
```

### 3. **Grid Entropy Regularization**
- **Purpose**: Maintains moderate entropy in routing (not too peaked, not too uniform)
- **Implementation**: Targets ~2 active experts per sample
- **Hyperparameter**: `grid_entropy_weight` (default: 0.01)
- **Enable/Disable**: `enable_grid_entropy` (default: true)

```python
# Encourages balanced but specialized routing
target_entropy = log(2)  # ~2 experts active
entropy_loss = (current_entropy - target_entropy)^2
```

### 4. **Domain-Aware Routing Loss**
- **Purpose**: Encourages different domains to use different expert distributions
- **Implementation**: Negative KL divergence between domain routing patterns
- **Hyperparameter**: `domain_routing_weight` (default: 0.1)
- **Enable/Disable**: `enable_domain_routing` (default: false, experimental)

```python
# Maximizes difference in expert usage across domains
domain_loss = -KL(domain_i_routing || domain_j_routing)
```

### 5. **Gradient Scaling for KAN Stability**
- **Purpose**: Prevents training instability from KAN spline parameters
- **Implementation**: Warm-up gradient scaling (0.1 → 1.0)
- **Hyperparameter**: `kan_init_scale` (default: 0.1)

### 6. **Improved KAN Initialization**
- **Purpose**: Starts KAN experts near identity for stable training
- **Implementation**: Zero base weights, small spline weights

### 7. **Enhanced Logging & Error Handling**
- **Purpose**: Comprehensive debugging and failure detection
- **Features**:
  - Step-by-step loss component logging (every 100 steps)
  - Automatic fallback to ERM on errors
  - Expert utilization tracking
  - Routing pattern analysis

---

## 📊 Expected Improvements

Based on the literature and similar approaches:

| Feature | Expected Accuracy Gain |
|---------|----------------------|
| Load Balancing | +0.5-1.5% |
| Pairwise Orthogonality | +1.0-2.0% |
| Grid Entropy | +0.3-0.8% |
| Domain Routing | +0.5-1.2% |
| **Combined** | **+2.3-5.5%** |

---

## 🔧 Usage

### Quick Test (Recommended First)

```bash
# Generate quick test jobs
bash generate_quick_tests.sh

# Submit all quick tests
bash submit_quick_tests.sh

# Monitor progress
python monitor_jobs.py --results_dir quick_tests

# Analyze results
python analyze_results.py --mode quick
```

### Full Hyperparameter Sweep

```bash
# Generate comprehensive sweep (60+ jobs)
bash generate_sweep.sh

# Submit all sweep jobs
bash submit_all_sweeps.sh

# Monitor sweep
python monitor_jobs.py --results_dir sweep_jobs

# Analyze sweep results
python analyze_results.py --mode sweep
```

### Manual Job Submission

```bash
python train_all.py KAN_Enhanced \
    --data_dir data \
    --algorithm KAN_MOA \
    --dataset OfficeHome \
    --model nf_vitbase_moekan \
    --adapter_type kan_kronecker \
    --num_experts 4 \
    --expert_ranks "2,4,8,16" \
    --kan_grids "3,5,7" \
    --injection_targets "qkv,proj" \
    --router_type spectral \
    --seed 1 \
    --steps 10000 \
    --lipschitz_weight 0.01 \
    --ortho_weight 0.1 \
    --load_balance_weight 0.01 \
    --grid_entropy_weight 0.01 \
    --domain_routing_weight 0.1 \
    --enable_load_balance True \
    --enable_grid_entropy True \
    --enable_pairwise_ortho True \
    --enable_domain_routing True \
    --kan_init_scale 0.1
```

---

## 🎯 Recommended Configurations

### Configuration 1: Conservative (Stable)
```bash
--lipschitz_weight 0.01 \
--ortho_weight 0.1 \
--load_balance_weight 0.01 \
--grid_entropy_weight 0.01 \
--enable_load_balance True \
--enable_grid_entropy True \
--enable_pairwise_ortho True \
--enable_domain_routing False
```

### Configuration 2: Aggressive (Maximum Performance)
```bash
--lipschitz_weight 0.01 \
--ortho_weight 0.5 \
--load_balance_weight 0.05 \
--grid_entropy_weight 0.01 \
--domain_routing_weight 0.2 \
--enable_load_balance True \
--enable_grid_entropy True \
--enable_pairwise_ortho True \
--enable_domain_routing True \
--kan_grids "3,5,7"
```

### Configuration 3: Orthogonality Focus
```bash
--lipschitz_weight 0.001 \
--ortho_weight 0.5 \
--load_balance_weight 0.01 \
--enable_pairwise_ortho True \
--num_experts 8
```

---

## 📈 Monitoring & Debugging

### Real-time Monitoring
```bash
# Watch job queue
watch -n 10 squeue -u $USER

# Monitor specific experiment
tail -f quick_tests/output_all_features.out

# Python monitor (auto-refresh)
python monitor_jobs.py --interval 30
```

### Analyzing Results
```bash
# Quick summary
python analyze_results.py --mode quick --output quick_results.csv

# Full sweep analysis
python analyze_results.py --mode sweep --output sweep_results.csv

# Combined analysis
python analyze_results.py --mode both
```

### Debugging Failed Jobs
```bash
# Check error logs
grep -i "error\|exception" quick_tests/error_*.err

# Check for OOM
grep -i "out of memory" quick_tests/error_*.err

# Check for NaN/Inf
grep -i "nan\|inf" quick_tests/output_*.out
```

---

## 🔍 What to Look For

### Signs of Success
- ✅ Balanced expert usage (logged every 100 steps)
- ✅ Decreasing orthogonality loss
- ✅ Stable training (no NaN/Inf)
- ✅ Test accuracy > baseline

### Red Flags
- ❌ Expert collapse (all weight on 1-2 experts)
- ❌ Oscillating losses
- ❌ NaN/Inf values
- ❌ OOM errors

### Example Good Log Output
```
Step 100 - Loss: 0.3421, Cls: 0.3200, Lip: 0.0012, Ortho: 0.0180, 
           LoadBal: 0.0015, GridEnt: 0.0008, DomainRoute: 0.0006
Load balance - Expert usage: [0.24, 0.26, 0.25, 0.25], Loss: 1.0012
Pairwise ortho - Avg off-diagonal similarity: 0.0231
```

---

## 🛠️ Troubleshooting

### Issue: OOM Errors
**Solution**: Reduce batch size or number of experts
```bash
--batch_size 16 --num_experts 4
```

### Issue: NaN Losses
**Solution**: Lower learning rate or increase gradient clipping
```bash
--lr 1e-5 --kan_init_scale 0.05
```

### Issue: Expert Collapse
**Solution**: Increase load balance weight
```bash
--load_balance_weight 0.1
```

### Issue: No Improvement Over Baseline
**Solution**: Try aggressive configuration with all features
```bash
# Use Configuration 2 (Aggressive) from above
--ortho_weight 0.5 --enable_domain_routing True --kan_grids "3,5,7"
```

---

## 📁 File Structure

```
MOKanA/
├── generate_quick_tests.sh      # Generate 5 focused test jobs
├── generate_sweep.sh             # Generate full hyperparameter sweep
├── submit_quick_tests.sh         # Submit quick tests
├── submit_all_sweeps.sh          # Submit all sweep jobs
├── monitor_jobs.py               # Real-time job monitoring
├── analyze_results.py            # Results analysis
├── quick_tests/                  # Quick test outputs
│   ├── test_*.sh                # Individual test scripts
│   ├── output_*.out             # Output logs
│   └── error_*.err              # Error logs
└── sweep_jobs/                   # Sweep experiment outputs
    ├── job_*.sh                 # Individual job scripts
    ├── output_*.out             # Output logs
    └── error_*.err              # Error logs
```

---

## 🎓 Citation

If this enhanced version helps achieve better results, consider noting:
- Load balancing inspired by Switch Transformer (Fedus et al., 2021)
- Pairwise orthogonality for expert diversity
- Domain-aware routing for cross-domain generalization

---

## 📝 Next Steps

1. **Run Quick Tests First**
   ```bash
   bash generate_quick_tests.sh
   bash submit_quick_tests.sh
   ```

2. **Monitor Progress**
   ```bash
   python monitor_jobs.py
   ```

3. **Analyze Results**
   ```bash
   python analyze_results.py --mode quick
   ```

4. **If Promising, Run Full Sweep**
   ```bash
   bash generate_sweep.sh
   bash submit_all_sweeps.sh
   ```

5. **Compare and Select Best Configuration**
   ```bash
   python analyze_results.py --mode both
   ```

---

## 💡 Tips for Maximum Accuracy

1. **Start with all features enabled** (Configuration 2)
2. **Use hierarchical grids** (`kan_grids="3,5,7"`)
3. **Increase orthogonality weight** if experts are too similar
4. **Monitor expert utilization** - should be balanced
5. **Run with multiple seeds** to verify consistency
6. **Compare against baseline** to measure improvement

---

**Good luck! 🚀**
