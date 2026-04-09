#!/usr/bin/env python3
"""
Experiment 2: Cross-Domain Routing Consistency Analysis
========================================================
Loads a trained KAN-MoA checkpoint, runs forward passes per domain,
collects routing weights from all MOE adapters, and visualizes:
  - Heatmap: domain x expert routing distribution
  - Per-layer routing pattern comparison
  - Jensen-Shannon Divergence between domain pairs

Usage:
  python exp2_routing_consistency.py \
      --checkpoint path/to/TE0.pth \
      --data_dir data \
      --dataset PACS \
      --output_dir experiments/exp2_results
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_from_checkpoint(ckpt_path, data_dir, dataset_name):
    """Load model from checkpoint and reconstruct algorithm."""
    from domainbed import hparams_registry
    from domainbed.datasets import get_dataset
    from domainbed.lib.fast_data_loader import FastDataLoader
    from sconf import Config
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_dict = ckpt['args']
    hparams_dict = ckpt['model_hparams']
    test_envs = ckpt['test_envs']
    
    # Reconstruct args namespace
    args = argparse.Namespace(**args_dict)
    args.data_dir = Path(data_dir)
    
    # Reconstruct hparams (must be sconf.Config for attribute access)
    hparams = hparams_registry.default_hparams(args.algorithm, dataset_name)
    hparams.update(hparams_dict)
    hparams = Config(default=hparams)
    
    # Load dataset
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams)
    
    # Build algorithm
    from domainbed.algorithms import algorithms as algo_module
    algorithm_class = getattr(algo_module, args.algorithm)
    algorithm = algorithm_class(
        dataset.input_shape, dataset.num_classes, len(dataset) - len(test_envs), hparams, args
    )
    
    # Load weights
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.cuda()
    algorithm.eval()
    
    # Create data loaders per environment
    env_loaders = []
    for i in range(len(dataset)):
        # Use in_splits for all envs (training portion)
        loader = FastDataLoader(
            dataset=in_splits[i][0] if i < len(in_splits) else out_splits[i - len(in_splits)][0],
            batch_size=64,
            num_workers=4
        )
        env_loaders.append(loader)
    
    return algorithm, dataset, env_loaders, test_envs


def collect_routing_weights(algorithm, loader, max_batches=20):
    """
    Run forward pass and collect routing weights from all MOE adapters.
    Returns dict: {adapter_name: tensor of shape (N, num_experts)}
    """
    from domainbed.models.kan_adapters import MOEBase
    
    adapter_routing = defaultdict(list)
    
    algorithm.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = batch["x"].cuda()
            _ = algorithm.predict(x)
            
            # Collect routing weights from each adapter
            for name, module in algorithm.named_modules():
                if isinstance(module, MOEBase) and module.last_routing_weights is not None:
                    rw = module.last_routing_weights.detach().cpu()  # (N, num_experts)
                    adapter_routing[name].append(rw)
    
    # Concatenate across batches -> mean routing per adapter
    result = {}
    for name, weight_list in adapter_routing.items():
        all_weights = torch.cat(weight_list, dim=0)  # (total_tokens, num_experts)
        result[name] = all_weights.mean(dim=0).numpy()  # (num_experts,)
    
    return result


def compute_jsd(p, q):
    """Jensen-Shannon Divergence between two distributions."""
    m = 0.5 * (p + q)
    eps = 1e-10
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def main():
    parser = argparse.ArgumentParser(description="Exp2: Cross-Domain Routing Consistency")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--output_dir", type=str, default="experiments/exp2_results")
    parser.add_argument("--max_batches", type=int, default=20,
                        help="Max batches per domain to process")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    algorithm, dataset, env_loaders, test_envs = load_model_from_checkpoint(
        args.checkpoint, args.data_dir, args.dataset
    )
    
    env_names = dataset.environments
    num_envs = len(env_names)
    print(f"Dataset: {args.dataset}, Environments: {env_names}")
    print(f"Test envs: {test_envs}")
    
    # Collect routing per domain
    print("\nCollecting routing weights per domain...")
    domain_routings = {}  # {env_name: {adapter_name: mean_routing}}
    for i, (env_name, loader) in enumerate(zip(env_names, env_loaders)):
        print(f"  Processing domain: {env_name}")
        domain_routings[env_name] = collect_routing_weights(
            algorithm, loader, max_batches=args.max_batches
        )
    
    # Get adapter names (consistent across domains)
    adapter_names = sorted(list(next(iter(domain_routings.values())).keys()))
    num_adapters = len(adapter_names)
    num_experts = len(next(iter(next(iter(domain_routings.values())).values())))
    
    # Get grid sizes from model
    from domainbed.models.kan_adapters import MOEBase
    adapters = [(n, m) for n, m in algorithm.named_modules() if isinstance(m, MOEBase)]
    grid_sizes = adapters[0][1].grid_list if adapters and hasattr(adapters[0][1], 'grid_list') else list(range(num_experts))
    
    print(f"\nFound {num_adapters} adapters, {num_experts} experts each")
    print(f"Grid sizes: {grid_sizes}")
    
    # ======================================================================
    # Plot 1: Global Routing Heatmap (domains x experts, averaged over layers)
    # ======================================================================
    print("\nGenerating global routing heatmap...")
    global_routing = np.zeros((num_envs, num_experts))
    for i, env_name in enumerate(env_names):
        env_routing = domain_routings[env_name]
        for adapter_name in adapter_names:
            global_routing[i] += env_routing[adapter_name]
        global_routing[i] /= num_adapters  # Average over adapters
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        global_routing, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=[f"Expert {j}\n(G={grid_sizes[j]})" for j in range(num_experts)],
        yticklabels=env_names,
        ax=ax, vmin=0, vmax=1
    )
    ax.set_title(f"Cross-Domain Routing Distribution ({args.dataset})", fontsize=14)
    ax.set_xlabel("Expert Index", fontsize=12)
    ax.set_ylabel("Domain", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp2_{args.dataset}_routing_heatmap.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp2_{args.dataset}_routing_heatmap.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Plot 2: Per-Layer Routing Comparison
    # ======================================================================
    print("Generating per-layer routing comparison...")
    # Select a few representative layers
    selected_adapters = adapter_names[:6] if num_adapters > 6 else adapter_names
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, adapter_name in enumerate(selected_adapters):
        if idx >= 6:
            break
        ax = axes[idx]
        
        layer_data = np.zeros((num_envs, num_experts))
        for i, env_name in enumerate(env_names):
            layer_data[i] = domain_routings[env_name][adapter_name]
        
        # Shorten adapter name for display
        short_name = adapter_name.split('.')[-2] + '.' + adapter_name.split('.')[-1] \
            if '.' in adapter_name else adapter_name
        
        x = np.arange(num_experts)
        width = 0.8 / num_envs
        for i, env_name in enumerate(env_names):
            ax.bar(x + i * width, layer_data[i], width, label=env_name, alpha=0.8)
        
        ax.set_title(short_name, fontsize=10)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Routing Weight")
        ax.set_xticks(x + width * num_envs / 2)
        ax.set_xticklabels([f"E{j}" for j in range(num_experts)])
        if idx == 0:
            ax.legend(fontsize=7)
    
    # Hide unused subplots
    for idx in range(len(selected_adapters), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"Per-Layer Routing Patterns ({args.dataset})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp2_{args.dataset}_perlayer_routing.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp2_{args.dataset}_perlayer_routing.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Plot 3: Jensen-Shannon Divergence Matrix Between Domains
    # ======================================================================
    print("Computing Jensen-Shannon Divergence between domains...")
    jsd_matrix = np.zeros((num_envs, num_envs))
    for i in range(num_envs):
        for j in range(num_envs):
            jsds = []
            for adapter_name in adapter_names:
                p = domain_routings[env_names[i]][adapter_name]
                q = domain_routings[env_names[j]][adapter_name]
                jsds.append(compute_jsd(p, q))
            jsd_matrix[i, j] = np.mean(jsds)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        jsd_matrix, annot=True, fmt=".4f", cmap="Blues",
        xticklabels=env_names, yticklabels=env_names, ax=ax
    )
    ax.set_title(f"Jensen-Shannon Divergence of Routing ({args.dataset})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp2_{args.dataset}_jsd_matrix.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp2_{args.dataset}_jsd_matrix.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Save numerical results
    # ======================================================================
    results = {
        "dataset": args.dataset,
        "num_adapters": num_adapters,
        "num_experts": num_experts,
        "grid_sizes": grid_sizes,
        "environments": env_names,
        "test_envs": test_envs,
        "global_routing": global_routing.tolist(),
        "jsd_matrix": jsd_matrix.tolist(),
        "per_adapter_routing": {
            env_name: {
                adapter_name: domain_routings[env_name][adapter_name].tolist()
                for adapter_name in adapter_names
            }
            for env_name in env_names
        }
    }
    
    results_path = os.path.join(args.output_dir, f"exp2_{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"  - Routing heatmap: exp2_{args.dataset}_routing_heatmap.pdf")
    print(f"  - Per-layer routing: exp2_{args.dataset}_perlayer_routing.pdf")
    print(f"  - JSD matrix: exp2_{args.dataset}_jsd_matrix.pdf")
    print(f"  - JSON results: exp2_{args.dataset}_results.json")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"ROUTING SUMMARY ({args.dataset})")
    print(f"{'='*60}")
    print(f"{'Domain':<20}", end="")
    for j in range(num_experts):
        print(f"{'Expert '+str(j):<12}", end="")
    print()
    print("-" * (20 + 12 * num_experts))
    for i, env_name in enumerate(env_names):
        print(f"{env_name:<20}", end="")
        for j in range(num_experts):
            print(f"{global_routing[i, j]:<12.4f}", end="")
        print()
    
    print(f"\nJSD Matrix (lower = more similar routing):")
    for i, env_name in enumerate(env_names):
        print(f"  {env_name:<20}", end="")
        for j in range(num_envs):
            print(f"{jsd_matrix[i, j]:.4f}  ", end="")
        print()


if __name__ == "__main__":
    main()
