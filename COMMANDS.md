# MOKanA - Commands Reference Guide

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Running Experiments](#running-experiments)
3. [Job Submission (SLURM)](#job-submission-slurm)
4. [Monitoring Jobs](#monitoring-jobs)
5. [Analyzing Results](#analyzing-results)
6. [Debugging](#debugging)
7. [Quick Tests](#quick-tests)

---

## Environment Setup

### Activate Virtual Environment
```bash
source ~/myenv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Setup
```bash
python verify_setup.py
```

---

## Running Experiments

### Single Training Run
```bash
python train_all.py \
    --data_dir /home/mosama97/scratch/data \
    --algorithm KAN_MOA \
    --dataset PACS \
    --test_env 0 \
    --output_dir ./outputs \
    --hparams '{"lr": 5e-5, "batch_size": 32}'
```

### Training with Specific Features Enabled
```bash
# Baseline (no extra losses)
python train_all.py \
    --algorithm KAN_MOA \
    --dataset PACS \
    --test_env 0 \
    --hparams '{"enable_load_balance": false, "enable_grid_entropy": false, "enable_pairwise_ortho": false, "enable_domain_routing": false}'

# With Load Balancing
python train_all.py \
    --algorithm KAN_MOA \
    --dataset PACS \
    --test_env 0 \
    --hparams '{"enable_load_balance": true, "load_balance_weight": 0.01}'

# With Pairwise Orthogonality
python train_all.py \
    --algorithm KAN_MOA \
    --dataset PACS \
    --test_env 0 \
    --hparams '{"enable_pairwise_ortho": true, "ortho_weight": 0.1}'

# With Domain Routing
python train_all.py \
    --algorithm KAN_MOA \
    --dataset PACS \
    --test_env 0 \
    --hparams '{"enable_domain_routing": true, "domain_routing_weight": 0.1}'

# All Features Enabled
python train_all.py \
    --algorithm KAN_MOA \
    --dataset PACS \
    --test_env 0 \
    --hparams '{"enable_load_balance": true, "enable_grid_entropy": true, "enable_pairwise_ortho": true, "enable_domain_routing": true}'
```

### Available Datasets
- `PACS` - Photo, Art, Cartoon, Sketch
- `VLCS` - VOC2007, LabelMe, Caltech, Sun
- `OfficeHome` - Art, Clipart, Product, Real World
- `TerraIncognita` - Wildlife camera trap images
- `DomainNet` - Large-scale domain adaptation

### Available Algorithms
- `ERM` - Empirical Risk Minimization (baseline)
- `KAN_MOA` - KAN Mixture of Adapters (our method)
- `CORAL`, `MMD`, `DANN`, `IRM`, `VREx`, `Mixup`, etc.

---

## Job Submission (SLURM)

### Submit a Single Job
```bash
sbatch job.sh
```

### Submit Multiple Sweep Jobs
```bash
# Generate sweep configurations
bash generate_sweep.sh

# Submit all sweep jobs
bash submit_all_sweeps.sh
```

### Submit Quick Tests
```bash
bash submit_quick_tests.sh
```

### Example SLURM Job Script
```bash
#!/bin/bash
#SBATCH --job-name=kan_moa
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/myenv/bin/activate
cd ~/scratch/MOKanA

python train_all.py \
    --data_dir /home/mosama97/scratch/data \
    --algorithm KAN_MOA \
    --dataset PACS \
    --test_env 0
```

---

## Monitoring Jobs

### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# View specific job details
squeue -j <JOB_ID>

# View job in detail
scontrol show job <JOB_ID>
```

### Monitor Job Progress
```bash
# Watch job queue in real-time
watch -n 5 'squeue -u $USER'

# Use custom monitoring script
python monitor_jobs.py
```

### View Job Logs in Real-Time
```bash
# Follow output log
tail -f output_<JOB_ID>.out

# Follow error log
tail -f error_<JOB_ID>.err

# Follow both
tail -f output_<JOB_ID>.out error_<JOB_ID>.err
```

### Check GPU Usage (on compute node)
```bash
# If on interactive session or within job
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Cancel Jobs
```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel all pending jobs
scancel -u $USER --state=pending
```

### Check Account Status
```bash
bash check_account_status.sh

# Or manually
sacct -u $USER --starttime=2024-01-01 --format=JobID,JobName,State,Elapsed,MaxRSS
```

---

## Analyzing Results

### View Training Logs
```bash
# Search for accuracy in output files
grep -r "test_out" output_*.out

# Search for final accuracy
grep -r "Step 5000" output_*.out

# Find best accuracy across all runs
grep -r "test_out" output_*.out | sort -t':' -k2 -n | tail -10
```

### Analyze Results Script
```bash
python analyze_results.py --output_dir ./outputs
```

### Parse Specific Metrics
```bash
# Extract training loss over time
grep "Loss:" output_<JOB_ID>.out

# Extract test accuracy
grep "test_out" output_<JOB_ID>.out | tail -1

# Count errors
grep -c "Error" error_<JOB_ID>.err
```

### Compare Configurations
```bash
# Compare baseline vs load_balance
echo "=== Baseline ===" && grep "test_out" output_baseline.out | tail -1
echo "=== Load Balance ===" && grep "test_out" output_load_balance.out | tail -1
```

---

## Debugging

### Check for Common Errors
```bash
# Check for CUDA errors
grep -i "cuda" error_*.err

# Check for memory errors
grep -i "out of memory" error_*.err

# Check for tensor shape errors
grep -i "size mismatch\|shape\|dimension" error_*.err

# Check for our specific fix (float to tensor)
grep -i "'float' object has no attribute 'to'" error_*.err
```

### Run Interactive Debug Session
```bash
# Request interactive GPU session
salloc --gres=gpu:1 --mem=32G --time=2:00:00

# Then run your script
source ~/myenv/bin/activate
cd ~/scratch/MOKanA
python train_all.py --dataset PACS --test_env 0 --steps 100
```

### Test KAN Adapters
```bash
python -c "
from domainbed.models.kan_adapters import MOEKANLoraInjectedLinear
import torch
adapter = MOEKANLoraInjectedLinear(768, 768)
x = torch.randn(2, 197, 768)
out, aux = adapter(x)
print(f'Input: {x.shape}, Output: {out.shape}')
print('KAN adapter test passed!')
"
```

### Check Model Architecture
```bash
python -c "
from domainbed import networks
import torch
args = type('Args', (), {'model': 'vit_base_patch16_224', 'pretrained': True})()
hparams = {'resnet18': False}
featurizer = networks.Featurizer((3, 224, 224), hparams, args)
print(featurizer)
"
```

---

## Quick Tests

### Run All Quick Tests
```bash
cd quick_tests
for f in test_*.sh; do bash $f; done
```

### Run Specific Quick Test
```bash
# Baseline test
bash quick_tests/test_baseline.sh

# Load balance test
bash quick_tests/test_load_balance.sh

# Pairwise orthogonality test
bash quick_tests/test_pairwise_ortho.sh

# Domain routing test
bash quick_tests/test_domain_routing.sh

# All features test
bash quick_tests/test_all_features.sh
```

### View Quick Test Results
```bash
# Check outputs
cat quick_tests/output_*.out

# Check errors
cat quick_tests/error_*.err
```

---

## Useful One-Liners

### Find Latest Output File
```bash
ls -t output_*.out | head -1 | xargs tail -50
```

### Count Running/Pending Jobs
```bash
echo "Running: $(squeue -u $USER -t RUNNING | wc -l)"
echo "Pending: $(squeue -u $USER -t PENDING | wc -l)"
```

### Find All Completed Jobs with Errors
```bash
for f in error_*.err; do
    if [ -s "$f" ]; then
        echo "=== $f ===" && head -5 "$f"
    fi
done
```

### Extract Best Accuracy per Dataset
```bash
for dataset in PACS VLCS OfficeHome; do
    echo "=== $dataset ==="
    grep -l "$dataset" output_*.out | xargs grep "test_out" | sort -t':' -k2 -n | tail -1
done
```

### Clean Up Old Files
```bash
# Remove output files older than 7 days
find . -name "output_*.out" -mtime +7 -delete
find . -name "error_*.err" -mtime +7 -delete
```

### Check Disk Usage
```bash
du -sh ~/scratch/MOKanA/
du -sh ~/scratch/data/
```

---

## Configuration Reference

### Key Hyperparameters for KAN_MOA
```python
{
    # Learning
    "lr": 5e-5,
    "batch_size": 32,
    "weight_decay": 0.0,
    
    # Loss Weights
    "lipschitz_weight": 0.01,
    "ortho_weight": 0.1,
    "complexity_weight": 0.1,
    "load_balance_weight": 0.01,
    "grid_entropy_weight": 0.01,
    "domain_routing_weight": 0.1,
    
    # Feature Toggles
    "enable_load_balance": True,
    "enable_grid_entropy": True,
    "enable_pairwise_ortho": True,
    "enable_domain_routing": False,
    
    # KAN Specific
    "kan_init_scale": 0.1,
}
```

---

## Troubleshooting

### Error: "'float' object has no attribute 'to'"
This has been fixed. Ensure you have the latest version of `algorithms.py`.

### Error: "CUDA out of memory"
Reduce batch size:
```bash
--hparams '{"batch_size": 16}'
```

### Error: "No module named 'domainbed'"
```bash
cd ~/scratch/MOKanA
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Jobs Stuck in Pending
Check account limits:
```bash
sacctmgr show assoc user=$USER format=Account,User,MaxJobs,MaxSubmit
```

---

## Contact & Resources

- **Repository**: `/home/mosama97/scratch/MOKanA`
- **Data Directory**: `/home/mosama97/scratch/data`
- **Virtual Environment**: `~/myenv`
