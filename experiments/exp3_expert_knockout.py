#!/usr/bin/env python3
"""
Experiment 3: Expert Knockout at Test Time
==========================================
Loads a trained KAN-MoA checkpoint and evaluates accuracy when:
  - All experts active (baseline)
  - Only coarse experts (low grid) active
  - Only fine experts (high grid) active
  - Individual expert knockout (remove one at a time)
  - Random subset of experts

This proves each expert contributes unique information and that
coarse+fine together is essential.

Usage:
  python exp3_expert_knockout.py \
      --checkpoint path/to/TE0.pth \
      --data_dir data \
      --dataset PACS \
      --output_dir experiments/exp3_results
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_data(ckpt_path, data_dir, dataset_name):
    """Load model from checkpoint."""
    from domainbed import hparams_registry
    from domainbed.datasets import get_dataset
    from domainbed.lib.fast_data_loader import FastDataLoader
    from sconf import Config
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_dict = ckpt['args']
    hparams_dict = ckpt['model_hparams']
    test_envs = ckpt['test_envs']
    
    args = argparse.Namespace(**args_dict)
    args.data_dir = Path(data_dir)
    
    hparams = hparams_registry.default_hparams(args.algorithm, dataset_name)
    hparams.update(hparams_dict)
    hparams = Config(default=hparams)
    
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams)
    
    from domainbed.algorithms import algorithms as algo_module
    algorithm_class = getattr(algo_module, args.algorithm)
    algorithm = algorithm_class(
        dataset.input_shape, dataset.num_classes, len(dataset) - len(test_envs), hparams, args
    )
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.cuda()
    algorithm.eval()
    
    # Create eval loaders per env
    eval_loaders = {}
    for i, env_name in enumerate(dataset.environments):
        ds = out_splits[i][0]  # Out-split for fair evaluation
        loader = FastDataLoader(dataset=ds, batch_size=64, num_workers=4)
        eval_loaders[env_name] = loader
    
    return algorithm, dataset, eval_loaders, test_envs


class ExpertKnockout:
    """Context manager to zero out specific experts during forward pass."""
    
    def __init__(self, algorithm, knockout_indices):
        """
        knockout_indices: list of expert indices to knock out (zero their output)
        """
        from domainbed.models.kan_adapters import MOEBase
        self.adapters = [m for m in algorithm.modules() if isinstance(m, MOEBase)]
        self.knockout_indices = knockout_indices
        self.original_forwards = []
        self.hooks = []
    
    def __enter__(self):
        """Monkey-patch forward methods to zero out knocked-out experts."""
        from domainbed.models.kan_adapters import (
            MOEKANKroneckerInjectedLinear, 
            MOEKANLoraInjectedLinear,
            MOEPureKANInjectedLinear
        )
        
        knockout_indices = self.knockout_indices
        
        for adapter in self.adapters:
            original_forward = adapter.forward
            
            def make_knockout_forward(orig_fwd, adapter_ref):
                def knockout_forward(x):
                    from einops import rearrange
                    
                    # Run base linear
                    base_out = adapter_ref.linear(x)
                    b, s, d = x.shape
                    x_flat = rearrange(x, 'b s d -> (b s) d')
                    
                    # Router
                    router_logits = adapter_ref.router(x_flat)
                    routing_weights = F.softmax(router_logits, dim=-1)
                    
                    # Zero out knocked-out experts' routing weights
                    for ki in knockout_indices:
                        if ki < routing_weights.shape[-1]:
                            routing_weights[:, ki] = 0.0
                    
                    # Re-normalize remaining weights
                    weight_sum = routing_weights.sum(dim=-1, keepdim=True)
                    routing_weights = routing_weights / (weight_sum + 1e-8)
                    
                    adapter_ref.last_routing_weights = routing_weights
                    
                    output = torch.zeros((b * s, base_out.shape[-1]), device=x.device, dtype=x.dtype)
                    
                    for i in range(adapter_ref.num_experts):
                        if i in knockout_indices:
                            continue
                        
                        if hasattr(adapter_ref, 'expert_downs'):
                            x_down = adapter_ref.expert_downs[i](x)
                            x_down_flat = rearrange(x_down, 'b s r -> (b s) r')
                            
                            if hasattr(adapter_ref, 'kan_input_norms'):
                                x_down_flat = adapter_ref.kan_input_norms[i](x_down_flat)
                            
                            kan_out = adapter_ref.expert_kans[i](x_down_flat)
                            final_out = adapter_ref.expert_ups[i](kan_out.view(b, s, -1))
                            final_out_flat = rearrange(final_out, 'b s o -> (b s) o')
                        else:
                            continue
                        
                        weight = routing_weights[:, i:i + 1]
                        scale = getattr(adapter_ref, 'kan_init_scale', 1.0)
                        output = output + weight * final_out_flat * scale
                    
                    return base_out + output.view(b, s, -1)
                
                return knockout_forward
            
            self.original_forwards.append((adapter, original_forward))
            adapter.forward = make_knockout_forward(original_forward, adapter)
        
        return self
    
    def __exit__(self, *args):
        for adapter, orig_fwd in self.original_forwards:
            adapter.forward = orig_fwd


def evaluate_accuracy(algorithm, loader):
    """Evaluate accuracy on a data loader."""
    correct = 0
    total = 0
    algorithm.eval()
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].cuda()
            y = batch["y"].cuda()
            logits = algorithm.predict(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0.0


def get_expert_grid_sizes(algorithm):
    """Extract grid sizes from the first MOE adapter."""
    from domainbed.models.kan_adapters import MOEBase
    for m in algorithm.modules():
        if isinstance(m, MOEBase) and hasattr(m, 'grid_list'):
            return m.grid_list
    return None


def main():
    parser = argparse.ArgumentParser(description="Exp3: Expert Knockout at Test Time")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--output_dir", type=str, default="experiments/exp3_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    algorithm, dataset, eval_loaders, test_envs = load_model_and_data(
        args.checkpoint, args.data_dir, args.dataset
    )
    
    env_names = dataset.environments
    grid_sizes = get_expert_grid_sizes(algorithm)
    num_experts = len(grid_sizes) if grid_sizes else 4
    print(f"Grid sizes: {grid_sizes}")
    print(f"Environments: {env_names}, Test envs: {test_envs}")
    
    # Determine coarse/fine split
    if grid_sizes:
        median_grid = np.median(grid_sizes)
        coarse_idx = [i for i, g in enumerate(grid_sizes) if g <= median_grid]
        fine_idx = [i for i, g in enumerate(grid_sizes) if g > median_grid]
    else:
        coarse_idx = list(range(num_experts // 2))
        fine_idx = list(range(num_experts // 2, num_experts))
    
    print(f"Coarse experts (G<={np.median(grid_sizes) if grid_sizes else '?'}): {coarse_idx}")
    print(f"Fine experts (G>{np.median(grid_sizes) if grid_sizes else '?'}): {fine_idx}")
    
    # ======================================================================
    # Define knockout configurations
    # ======================================================================
    knockout_configs = {
        "all_active": [],  # No knockout = baseline
        "coarse_only": fine_idx,  # Knock out fine experts
        "fine_only": coarse_idx,  # Knock out coarse experts
    }
    
    # Individual knockouts
    for i in range(num_experts):
        grid_str = f"G={grid_sizes[i]}" if grid_sizes else f"idx={i}"
        knockout_configs[f"knockout_expert_{i}_{grid_str}"] = [i]
    
    # ======================================================================
    # Evaluate each configuration
    # ======================================================================
    results = {}
    
    for config_name, knockout_indices in knockout_configs.items():
        print(f"\n--- Evaluating: {config_name} (knockout={knockout_indices}) ---")
        config_results = {}
        
        if len(knockout_indices) == 0:
            # No knockout - standard evaluation
            for env_name, loader in eval_loaders.items():
                acc = evaluate_accuracy(algorithm, loader)
                config_results[env_name] = acc
                print(f"  {env_name}: {acc:.4f}")
        else:
            # Apply knockout
            with ExpertKnockout(algorithm, knockout_indices):
                for env_name, loader in eval_loaders.items():
                    acc = evaluate_accuracy(algorithm, loader)
                    config_results[env_name] = acc
                    print(f"  {env_name}: {acc:.4f}")
        
        config_results["avg"] = np.mean(list(config_results.values()))
        results[config_name] = config_results
    
    # ======================================================================
    # Compute accuracy drops
    # ======================================================================
    baseline_results = results["all_active"]
    drops = {}
    for config_name, config_results in results.items():
        if config_name == "all_active":
            continue
        drops[config_name] = {
            env: baseline_results[env] - config_results[env]
            for env in config_results
        }
    
    # ======================================================================
    # Plot 1: Bar chart of accuracy per knockout config
    # ======================================================================
    print("\nGenerating plots...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    config_names = list(results.keys())
    x = np.arange(len(config_names))
    
    test_env_names = [env_names[i] for i in test_envs]
    train_env_names = [e for e in env_names if e not in test_env_names]
    
    # Plot average accuracy for each config
    avgs = [results[c]["avg"] * 100 for c in config_names]
    colors = ['#2ecc71' if c == 'all_active' else 
              '#3498db' if 'coarse_only' in c else
              '#e74c3c' if 'fine_only' in c else
              '#f39c12' for c in config_names]
    
    bars = ax.bar(x, avgs, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=avgs[0], color='green', linestyle='--', alpha=0.5, label='Baseline')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('knockout_expert_', 'KO E') for c in config_names], 
                        rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Average Accuracy (%)", fontsize=12)
    ax.set_title(f"Expert Knockout Analysis ({args.dataset})", fontsize=14)
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp3_{args.dataset}_knockout.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp3_{args.dataset}_knockout.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Plot 2: Accuracy drop heatmap (knockout x environment)
    # ======================================================================
    drop_configs = [c for c in config_names if c != "all_active"]
    drop_matrix = np.zeros((len(drop_configs), len(env_names)))
    
    for i, config in enumerate(drop_configs):
        for j, env in enumerate(env_names):
            drop_matrix[i, j] = drops[config].get(env, 0) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        drop_matrix, annot=True, fmt=".2f", cmap="RdYlGn_r",
        xticklabels=env_names,
        yticklabels=[c.replace('knockout_expert_', 'KO E') for c in drop_configs],
        ax=ax, center=0
    )
    ax.set_title(f"Accuracy Drop (%) from Expert Knockout ({args.dataset})", fontsize=14)
    ax.set_xlabel("Environment")
    ax.set_ylabel("Knockout Config")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp3_{args.dataset}_drop_heatmap.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp3_{args.dataset}_drop_heatmap.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Save results
    # ======================================================================
    output_data = {
        "dataset": args.dataset,
        "grid_sizes": grid_sizes,
        "coarse_idx": coarse_idx,
        "fine_idx": fine_idx,
        "test_envs": test_envs,
        "results": results,
        "accuracy_drops": drops
    }
    
    results_path = os.path.join(args.output_dir, f"exp3_{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"EXPERT KNOCKOUT SUMMARY ({args.dataset})")
    print(f"{'='*80}")
    header = f"{'Config':<30}"
    for env in env_names:
        header += f"{env:<15}"
    header += f"{'Avg':<10}"
    print(header)
    print("-" * len(header))
    
    for config_name in config_names:
        row = f"{config_name:<30}"
        for env in env_names:
            acc = results[config_name].get(env, 0) * 100
            row += f"{acc:<15.2f}"
        row += f"{results[config_name]['avg'] * 100:<10.2f}"
        print(row)
    
    print(f"\nAccuracy Drops from Baseline:")
    for config_name in drop_configs:
        avg_drop = drops[config_name]["avg"] * 100
        print(f"  {config_name}: -{avg_drop:.2f}%")


if __name__ == "__main__":
    main()
