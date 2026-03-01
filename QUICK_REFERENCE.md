# MOKanA Quick Reference Card

## 🚀 Quick Start (Copy & Paste)

```bash
cd /home/mosama97/scratch/MOKanA
bash RUN_ME_FIRST.sh
```

## 📝 Job Submission Commands

### Submit Quick Tests (5 experiments, ~2-4 hours)
```bash
bash submit_quick_tests.sh
```

### Monitor Jobs
```bash
python monitor_jobs.py --results_dir quick_tests
# OR
watch -n 10 squeue -u $USER
```

### Analyze Results
```bash
python analyze_results.py --mode quick
```

### Generate & Submit Full Sweep (60+ experiments)
```bash
bash generate_sweep.sh
bash submit_all_sweeps.sh
```

## 🎯 Configuration Quick Reference

### All New Hyperparameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--lipschitz_weight` | 0.01 | 0.001-0.1 | Lipschitz regularization |
| `--ortho_weight` | 0.1 | 0.01-0.5 | Orthogonality penalty |
| `--load_balance_weight` | 0.01 | 0.001-0.1 | Load balancing |
| `--grid_entropy_weight` | 0.01 | 0.001-0.05 | Routing entropy |
| `--domain_routing_weight` | 0.1 | 0.05-0.2 | Domain routing |
| `--enable_load_balance` | True | True/False | Enable load balance |
| `--enable_grid_entropy` | True | True/False | Enable grid entropy |
| `--enable_pairwise_ortho` | True | True/False | Enable pairwise ortho |
| `--enable_domain_routing` | False | True/False | Enable domain routing |
| `--kan_init_scale` | 0.1 | 0.01-0.5 | KAN initialization |

### Recommended Presets

**Baseline (for comparison)**:
```bash
--enable_load_balance False --enable_grid_entropy False \
--enable_pairwise_ortho False --enable_domain_routing False
```

**Stable (recommended start)**:
```bash
--enable_load_balance True --enable_grid_entropy True \
--enable_pairwise_ortho True --enable_domain_routing False \
--ortho_weight 0.1 --load_balance_weight 0.01
```

**Aggressive (max performance)**:
```bash
--enable_load_balance True --enable_grid_entropy True \
--enable_pairwise_ortho True --enable_domain_routing True \
--ortho_weight 0.5 --load_balance_weight 0.05 \
--domain_routing_weight 0.2 --kan_grids "3,5,7"
```

## 📊 Monitoring Logs

### What to Look For (Good Signs)
```
✅ Load balance - Expert usage: [0.24, 0.26, 0.25, 0.25], Loss: 1.0012
✅ Pairwise ortho - Avg off-diagonal similarity: 0.0231
✅ Grid entropy - Current: 0.6931, Target: 0.6931
✅ Step 1000 - Loss: 0.3421 (decreasing)
```

### Red Flags
```
❌ Load balance - Expert usage: [0.98, 0.01, 0.01, 0.00]  # Collapse!
❌ Step 500 - Loss: nan  # Unstable training
❌ CUDA out of memory  # Need smaller batch
```

## 🔍 Debugging Commands

### Check for errors
```bash
grep -i "error\|exception" quick_tests/error_*.err
```

### Check for NaN/Inf
```bash
grep -i "nan\|inf" quick_tests/output_*.out
```

### View specific job output
```bash
tail -f quick_tests/output_all_features.out
```

### Check job status
```bash
squeue -u $USER
sacct -u $USER --format=JobID,JobName,State,Elapsed
```

## 📈 Expected Results Timeline

| Time | Milestone |
|------|-----------|
| 0 min | Setup complete, jobs submitted |
| 30 min | First metrics appear in logs |
| 2-4 hrs | Quick tests complete |
| 1 day | Initial sweep results |
| 3-4 days | Full sweep complete |

## 🎯 Target Accuracy Improvements

- Baseline (original): ~65-70% (your current)
- +Load Balance: +0.5-1.5%
- +Pairwise Ortho: +1.0-2.0%
- +Grid Entropy: +0.3-0.8%
- +Domain Routing: +0.5-1.2%
- **Total Target**: +2.3-5.5% over baseline

## 📁 Important Files

| File | Purpose |
|------|---------|
| `RUN_ME_FIRST.sh` | Master setup script |
| `submit_quick_tests.sh` | Submit validation tests |
| `monitor_jobs.py` | Real-time monitoring |
| `analyze_results.py` | Results analysis |
| `README_IMPROVEMENTS.md` | Full documentation |
| `ENHANCEMENT_SUMMARY.md` | Quick reference |
| `quick_tests/` | Quick test outputs |
| `sweep_jobs/` | Sweep outputs |

## 🆘 Quick Fixes

### Jobs not running?
```bash
squeue -u $USER  # Check queue
sbatch quick_tests/test_baseline.sh  # Retry
```

### Results not showing?
```bash
ls -lh quick_tests/output_*.out  # Check file sizes
tail quick_tests/output_baseline.out  # View latest
```

### Scripts not executable?
```bash
bash make_executable.sh
```

## 📞 Verification

Before running, verify setup:
```bash
python verify_setup.py
```

Should show all ✅ checks passed.

---

**Quick Help**: All enhancements are in place and ready to use!
