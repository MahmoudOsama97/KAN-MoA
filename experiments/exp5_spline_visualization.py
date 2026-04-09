#!/usr/bin/env python3
"""
Experiment 5: Spline Function Visualization
============================================
Extracts and visualizes the learned B-spline functions from coarse (low G)
vs fine (high G) experts.

Hypothesis: Coarse experts learn smooth, broad functions (low-frequency)
            Fine experts learn sharp, localized functions (high-frequency)

This provides visual proof of the frequency decomposition hypothesis.

Usage:
  python exp5_spline_visualization.py \
      --checkpoint path/to/TE0.pth \
      --data_dir data \
      --dataset PACS \
      --output_dir experiments/exp5_results
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def load_model(ckpt_path, data_dir, dataset_name):
    """Load model from checkpoint."""
    from domainbed import hparams_registry
    from domainbed.datasets import get_dataset
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
    algorithm.cpu()
    algorithm.eval()
    
    return algorithm, dataset


def extract_spline_functions(kan_layer, input_range=(-2, 2), num_points=500):
    """
    Extract learned spline functions from a KANLinear layer.
    
    For each (in_dim, out_dim) pair, computes the learned univariate
    B-spline function by evaluating f(x) at multiple points.
    
    Returns:
        x_vals: (num_points,)
        function_values: (in_features, out_features, num_points)
    """
    in_features = kan_layer.in_features
    out_features = kan_layer.out_features
    
    x_vals = np.linspace(input_range[0], input_range[1], num_points)
    function_values = np.zeros((in_features, out_features, num_points))
    
    kan_layer.eval()
    with torch.no_grad():
        for idx, x_val in enumerate(x_vals):
            # Create input: each dimension set to x_val, others to 0
            for dim_idx in range(in_features):
                # Input: only dim_idx is active
                inp = torch.zeros(1, in_features)
                inp[0, dim_idx] = x_val
                
                out = kan_layer(inp)  # (1, out_features)
                function_values[dim_idx, :, idx] = out[0].numpy()
    
    return x_vals, function_values


def extract_spline_weights_analysis(kan_layer):
    """
    Analyze spline weight statistics without full function evaluation.
    Faster and works for any KAN layer size.
    """
    stats = {}
    
    if hasattr(kan_layer, 'spline_weight') and kan_layer.spline_weight is not None:
        w = kan_layer.spline_weight.detach().cpu()
        stats['spline_weight_mean'] = w.mean().item()
        stats['spline_weight_std'] = w.std().item()
        stats['spline_weight_max'] = w.abs().max().item()
        stats['spline_weight_l1'] = w.abs().mean().item()
        stats['spline_weight_shape'] = list(w.shape)
        
        # Compute "roughness" - higher order differences of spline weights
        # Measures how "spiky" the learned function is
        if w.dim() >= 2:
            # Compute 1st order differences along the grid dimension (last dim)
            diffs = torch.diff(w, dim=-1)
            stats['roughness_l1'] = diffs.abs().mean().item()
            stats['roughness_l2'] = (diffs ** 2).mean().sqrt().item()
            
            # 2nd order differences (curvature)
            diffs2 = torch.diff(diffs, dim=-1)
            stats['curvature_l1'] = diffs2.abs().mean().item()
            stats['curvature_l2'] = (diffs2 ** 2).mean().sqrt().item()
    
    if hasattr(kan_layer, 'base_weight') and kan_layer.base_weight is not None:
        bw = kan_layer.base_weight.detach().cpu()
        stats['base_weight_mean'] = bw.mean().item()
        stats['base_weight_std'] = bw.std().item()
    
    if hasattr(kan_layer, 'grid') and kan_layer.grid is not None:
        grid = kan_layer.grid.detach().cpu()
        stats['grid_size'] = grid.shape[-1]
        stats['grid_range'] = [grid.min().item(), grid.max().item()]
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Exp5: Spline Function Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--output_dir", type=str, default="experiments/exp5_results")
    parser.add_argument("--num_layers_to_plot", type=int, default=4,
                        help="Number of adapter layers to visualize")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    algorithm, dataset = load_model(args.checkpoint, args.data_dir, args.dataset)
    
    # Collect all KAN experts from MOE adapters
    from domainbed.models.kan_adapters import MOEBase
    from domainbed.models.kan import KANLinear
    
    adapters = [(name, m) for name, m in algorithm.named_modules() if isinstance(m, MOEBase)]
    print(f"Found {len(adapters)} MOE adapters")
    
    if not adapters:
        print("No MOE adapters found!")
        return
    
    grid_sizes = adapters[0][1].grid_list if hasattr(adapters[0][1], 'grid_list') else []
    num_experts = len(grid_sizes)
    print(f"Grid sizes: {grid_sizes}")
    
    # ======================================================================
    # Part 1: Spline Weight Statistics Analysis
    # ======================================================================
    print("\nAnalyzing spline weight statistics...")
    expert_stats_all = defaultdict(list)
    
    for adapter_name, adapter in adapters:
        if hasattr(adapter, 'expert_kans'):
            for i, kan in enumerate(adapter.expert_kans):
                stats = extract_spline_weights_analysis(kan)
                stats['adapter'] = adapter_name
                stats['expert_idx'] = i
                stats['grid_size'] = grid_sizes[i] if i < len(grid_sizes) else 0
                expert_stats_all[i].append(stats)
    
    # Aggregate statistics per expert
    aggregated = {}
    for expert_idx in range(num_experts):
        all_stats = expert_stats_all[expert_idx]
        if not all_stats:
            continue
        agg = {
            'grid_size': grid_sizes[expert_idx],
            'avg_roughness_l1': np.mean([s.get('roughness_l1', 0) for s in all_stats]),
            'avg_roughness_l2': np.mean([s.get('roughness_l2', 0) for s in all_stats]),
            'avg_curvature_l1': np.mean([s.get('curvature_l1', 0) for s in all_stats]),
            'avg_curvature_l2': np.mean([s.get('curvature_l2', 0) for s in all_stats]),
            'avg_spline_weight_l1': np.mean([s.get('spline_weight_l1', 0) for s in all_stats]),
            'avg_spline_weight_std': np.mean([s.get('spline_weight_std', 0) for s in all_stats]),
            'avg_spline_weight_max': np.mean([s.get('spline_weight_max', 0) for s in all_stats]),
        }
        aggregated[expert_idx] = agg
    
    # ======================================================================
    # Plot 1: Roughness & Curvature comparison (bar chart)
    # ======================================================================
    print("Generating roughness comparison plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    expert_labels = [f"E{i} (G={grid_sizes[i]})" for i in range(num_experts)]
    colors = ['#2ecc71' if grid_sizes[i] <= np.median(grid_sizes) else '#e74c3c' 
              for i in range(num_experts)]
    
    # Roughness (1st derivative)
    roughness = [aggregated.get(i, {}).get('avg_roughness_l1', 0) for i in range(num_experts)]
    axes[0].bar(range(num_experts), roughness, color=colors, edgecolor='black')
    axes[0].set_xticks(range(num_experts))
    axes[0].set_xticklabels(expert_labels)
    axes[0].set_ylabel("L1 Roughness")
    axes[0].set_title("Spline Roughness (1st-order)")
    
    # Curvature (2nd derivative)
    curvature = [aggregated.get(i, {}).get('avg_curvature_l1', 0) for i in range(num_experts)]
    axes[1].bar(range(num_experts), curvature, color=colors, edgecolor='black')
    axes[1].set_xticks(range(num_experts))
    axes[1].set_xticklabels(expert_labels)
    axes[1].set_ylabel("L1 Curvature")
    axes[1].set_title("Spline Curvature (2nd-order)")
    
    # Weight magnitude
    weight_l1 = [aggregated.get(i, {}).get('avg_spline_weight_l1', 0) for i in range(num_experts)]
    axes[2].bar(range(num_experts), weight_l1, color=colors, edgecolor='black')
    axes[2].set_xticks(range(num_experts))
    axes[2].set_xticklabels(expert_labels)
    axes[2].set_ylabel("Mean |spline weight|")
    axes[2].set_title("Spline Weight Magnitude")
    
    plt.suptitle(f"Spline Function Analysis ({args.dataset})\n"
                 f"Green=Coarse (smooth), Red=Fine (sharp)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp5_{args.dataset}_spline_stats.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp5_{args.dataset}_spline_stats.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Part 2: Visualize actual spline functions for first few layers
    # ======================================================================
    print("Visualizing learned spline functions...")
    
    layers_to_plot = min(args.num_layers_to_plot, len(adapters))
    
    fig, axes = plt.subplots(layers_to_plot, num_experts, figsize=(5 * num_experts, 4 * layers_to_plot))
    if layers_to_plot == 1:
        axes = axes.reshape(1, -1)
    if num_experts == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx in range(layers_to_plot):
        adapter_name, adapter = adapters[layer_idx]
        short_name = adapter_name.split('.')[-2] + '.' + adapter_name.split('.')[-1] \
            if '.' in adapter_name else adapter_name
        
        for expert_idx in range(num_experts):
            ax = axes[layer_idx, expert_idx]
            kan = adapter.expert_kans[expert_idx]
            
            # For small KAN layers, we can visualize individual spline functions
            in_dim = kan.in_features
            out_dim = kan.out_features
            
            # Sample a few (in, out) pairs to visualize
            num_samples = min(5, in_dim * out_dim)
            
            x_vals, func_vals = extract_spline_functions(kan, input_range=(-2, 2), num_points=200)
            
            # Plot a few randomly selected spline functions
            np.random.seed(42)
            pairs = [(i, j) for i in range(in_dim) for j in range(out_dim)]
            selected = np.random.choice(len(pairs), min(num_samples, len(pairs)), replace=False)
            
            for sel_idx in selected:
                in_idx, out_idx = pairs[sel_idx]
                y_vals = func_vals[in_idx, out_idx, :]
                ax.plot(x_vals, y_vals, alpha=0.6, linewidth=1.5)
            
            ax.set_title(f"E{expert_idx} (G={grid_sizes[expert_idx]})", fontsize=11)
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.grid(True, alpha=0.3)
            
            if layer_idx == 0:
                ax.set_title(f"Expert {expert_idx} (G={grid_sizes[expert_idx]})", fontsize=12, fontweight='bold')
            
            if expert_idx == 0:
                ax.set_ylabel(f"{short_name}\nOutput", fontsize=10)
    
    plt.suptitle(f"Learned Spline Functions ({args.dataset})\n"
                 f"Coarse (low G) → smooth curves | Fine (high G) → complex/sharp curves", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp5_{args.dataset}_spline_functions.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_dir, f"exp5_{args.dataset}_spline_functions.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ======================================================================
    # Plot 3: Frequency analysis of spline functions (FFT)
    # ======================================================================
    print("Computing frequency analysis of spline functions...")
    
    fig, axes = plt.subplots(1, num_experts, figsize=(5 * num_experts, 4))
    if num_experts == 1:
        axes = [axes]
    
    for expert_idx in range(num_experts):
        ax = axes[expert_idx]
        all_freq_powers = []
        
        for adapter_name, adapter in adapters[:5]:  # First 5 layers
            kan = adapter.expert_kans[expert_idx]
            if hasattr(kan, 'spline_weight') and kan.spline_weight is not None:
                w = kan.spline_weight.detach().cpu().numpy()
                # FFT along the grid dimension
                for row in w.reshape(-1, w.shape[-1]):
                    freq = np.abs(np.fft.rfft(row))
                    if len(freq) > 1:
                        freq = freq / (freq.sum() + 1e-10)
                        all_freq_powers.append(freq)
        
        if all_freq_powers:
            # Pad to same length
            max_len = max(len(f) for f in all_freq_powers)
            padded = np.array([np.pad(f, (0, max_len - len(f))) for f in all_freq_powers])
            mean_spectrum = padded.mean(axis=0)
            
            freqs = np.arange(len(mean_spectrum))
            ax.bar(freqs, mean_spectrum, color=colors[expert_idx], edgecolor='black', alpha=0.7)
            ax.set_xlabel("Frequency Bin")
            ax.set_ylabel("Normalized Power")
        
        ax.set_title(f"Expert {expert_idx} (G={grid_sizes[expert_idx]})", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Frequency Content of Spline Weights ({args.dataset})\n"
                 f"Coarse → concentrated in low freq | Fine → spread across high freq", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp5_{args.dataset}_freq_analysis.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp5_{args.dataset}_freq_analysis.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Save results
    # ======================================================================
    output_data = {
        "dataset": args.dataset,
        "grid_sizes": grid_sizes,
        "num_experts": num_experts,
        "num_adapters": len(adapters),
        "expert_statistics": {str(k): v for k, v in aggregated.items()},
    }
    
    results_path = os.path.join(args.output_dir, f"exp5_{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SPLINE ANALYSIS SUMMARY ({args.dataset})")
    print(f"{'='*60}")
    print(f"{'Expert':<15}{'Grid':<8}{'Roughness':<12}{'Curvature':<12}{'Weight L1':<12}")
    print("-" * 59)
    for i in range(num_experts):
        agg = aggregated.get(i, {})
        print(f"Expert {i:<8}{grid_sizes[i]:<8}"
              f"{agg.get('avg_roughness_l1', 0):<12.6f}"
              f"{agg.get('avg_curvature_l1', 0):<12.6f}"
              f"{agg.get('avg_spline_weight_l1', 0):<12.6f}")
    
    print(f"\nExpected: Fine experts (high G) → HIGHER roughness & curvature")
    print(f"          Coarse experts (low G) → LOWER  roughness & curvature")


if __name__ == "__main__":
    main()
