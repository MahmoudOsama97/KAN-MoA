#!/usr/bin/env python3
"""
Experiment 4: Expert Output Similarity Across Domains
=====================================================
For each class, collects expert bottleneck (KAN) outputs across domains
and computes cross-domain cosine similarity per expert.

Hypothesis: Fine-grid experts (high G) capture domain-specific textures
            → LOW cross-domain similarity
            Coarse-grid experts (low G) capture domain-invariant shapes
            → HIGH cross-domain similarity

Usage:
  python exp4_expert_similarity.py \
      --checkpoint path/to/TE0.pth \
      --data_dir data \
      --dataset PACS \
      --output_dir experiments/exp4_results
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
    
    eval_loaders = {}
    for i, env_name in enumerate(dataset.environments):
        ds = in_splits[i][0]
        loader = FastDataLoader(dataset=ds, batch_size=32, num_workers=4)
        eval_loaders[env_name] = loader
    
    return algorithm, dataset, eval_loaders, test_envs


def collect_expert_outputs_by_class(algorithm, loader, max_batches=15):
    """
    Collect expert bottleneck outputs grouped by class label.
    Returns: {class_label: {adapter_idx: {expert_idx: tensor(N, hidden_dim)}}}
    """
    from domainbed.models.kan_adapters import MOEBase
    
    adapters = [(name, m) for name, m in algorithm.named_modules() if isinstance(m, MOEBase)]
    
    # {class: {adapter_idx: {expert_idx: [tensors]}}}
    class_expert_outputs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    algorithm.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            
            x = batch["x"].cuda()
            y = batch["y"].cpu().numpy()
            _ = algorithm.predict(x)
            
            for adapter_idx, (name, adapter) in enumerate(adapters):
                if adapter.last_expert_outputs is not None:
                    # last_expert_outputs: (N, num_experts, hidden_dim)
                    expert_outs = adapter.last_expert_outputs.detach().cpu()
                    
                    # Average over sequence dimension if present
                    if expert_outs.dim() == 3:
                        # (batch*seq, num_experts, hidden_dim) - need to average per image
                        # Assume seq_len = expert_outs.shape[0] // len(y)
                        batch_size = len(y)
                        if expert_outs.shape[0] % batch_size == 0:
                            seq_len = expert_outs.shape[0] // batch_size
                            # Reshape: (batch, seq, experts, hidden) -> mean over seq
                            expert_outs = expert_outs.view(batch_size, seq_len, 
                                                            expert_outs.shape[1], 
                                                            expert_outs.shape[2])
                            expert_outs = expert_outs.mean(dim=1)  # (batch, experts, hidden)
                        else:
                            continue
                    
                    num_experts = expert_outs.shape[1]
                    for sample_idx, label in enumerate(y):
                        if sample_idx < expert_outs.shape[0]:
                            for expert_idx in range(num_experts):
                                vec = expert_outs[sample_idx, expert_idx]
                                class_expert_outputs[int(label)][adapter_idx][expert_idx].append(vec)
    
    # Average per class per expert
    class_means = {}
    for cls in class_expert_outputs:
        class_means[cls] = {}
        for adapter_idx in class_expert_outputs[cls]:
            class_means[cls][adapter_idx] = {}
            for expert_idx in class_expert_outputs[cls][adapter_idx]:
                vecs = torch.stack(class_expert_outputs[cls][adapter_idx][expert_idx])
                class_means[cls][adapter_idx][expert_idx] = vecs.mean(dim=0)
    
    return class_means


def compute_cross_domain_similarity(domain_class_means, env_names, num_experts, num_adapters):
    """
    For each class and expert, compute cross-domain cosine similarity.
    Returns: (num_experts, ) array of average cross-domain similarity
    """
    # Get common classes across all domains
    all_classes = None
    for env_name in env_names:
        if env_name in domain_class_means:
            env_classes = set(domain_class_means[env_name].keys())
            if all_classes is None:
                all_classes = env_classes
            else:
                all_classes = all_classes & env_classes
    
    if not all_classes:
        print("Warning: No common classes found across domains!")
        return np.zeros(num_experts)
    
    print(f"  Common classes: {sorted(all_classes)}")
    
    expert_similarities = defaultdict(list)
    
    for cls in all_classes:
        for adapter_idx in range(min(num_adapters, 5)):  # Sample adapters
            for expert_idx in range(num_experts):
                # Collect mean vectors for this class/expert across domains
                domain_vecs = []
                for env_name in env_names:
                    if (env_name in domain_class_means and 
                        cls in domain_class_means[env_name] and
                        adapter_idx in domain_class_means[env_name][cls] and
                        expert_idx in domain_class_means[env_name][cls][adapter_idx]):
                        domain_vecs.append(domain_class_means[env_name][cls][adapter_idx][expert_idx])
                
                if len(domain_vecs) < 2:
                    continue
                
                # Compute pairwise cosine similarities
                for i in range(len(domain_vecs)):
                    for j in range(i + 1, len(domain_vecs)):
                        sim = F.cosine_similarity(
                            domain_vecs[i].unsqueeze(0), 
                            domain_vecs[j].unsqueeze(0)
                        ).item()
                        expert_similarities[expert_idx].append(sim)
    
    # Average similarity per expert
    result = np.zeros(num_experts)
    for e in range(num_experts):
        if expert_similarities[e]:
            result[e] = np.mean(expert_similarities[e])
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Exp4: Expert Output Similarity Across Domains")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--output_dir", type=str, default="experiments/exp4_results")
    parser.add_argument("--max_batches", type=int, default=15)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    algorithm, dataset, eval_loaders, test_envs = load_model_and_data(
        args.checkpoint, args.data_dir, args.dataset
    )
    
    env_names = dataset.environments
    
    # Get grid sizes and expert info
    from domainbed.models.kan_adapters import MOEBase
    adapters = [(n, m) for n, m in algorithm.named_modules() if isinstance(m, MOEBase)]
    grid_sizes = adapters[0][1].grid_list if adapters and hasattr(adapters[0][1], 'grid_list') else [3, 5, 10, 20]
    num_experts = len(grid_sizes)
    num_adapters = len(adapters)
    
    print(f"Grid sizes: {grid_sizes}")
    print(f"Environments: {env_names}")
    print(f"Adapters: {num_adapters}, Experts: {num_experts}")
    
    # Collect expert outputs per domain, per class
    print("\nCollecting expert outputs per domain...")
    domain_class_means = {}
    for env_name, loader in eval_loaders.items():
        print(f"  Processing domain: {env_name}")
        domain_class_means[env_name] = collect_expert_outputs_by_class(
            algorithm, loader, max_batches=args.max_batches
        )
    
    # Compute cross-domain similarity per expert
    print("\nComputing cross-domain similarity...")
    overall_sim = compute_cross_domain_similarity(
        domain_class_means, env_names, num_experts, num_adapters
    )
    
    # Also compute per domain-pair similarity
    print("Computing per domain-pair similarity...")
    pair_sims = {}
    for i in range(len(env_names)):
        for j in range(i + 1, len(env_names)):
            pair_envs = [env_names[i], env_names[j]]
            sim = compute_cross_domain_similarity(
                domain_class_means, pair_envs, num_experts, num_adapters
            )
            pair_name = f"{env_names[i]}-{env_names[j]}"
            pair_sims[pair_name] = sim
    
    # ======================================================================
    # Plot 1: Cross-domain similarity per expert (bar chart)
    # ======================================================================
    print("\nGenerating plots...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_experts)
    
    colors = ['#2ecc71' if g <= np.median(grid_sizes) else '#e74c3c' for g in grid_sizes]
    bars = ax.bar(x, overall_sim, color=colors, edgecolor='black', linewidth=0.8)
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"Expert {i}\n(G={grid_sizes[i]})" for i in range(num_experts)])
    ax.set_ylabel("Cross-Domain Cosine Similarity", fontsize=12)
    ax.set_title(f"Expert Output Similarity Across Domains ({args.dataset})\n"
                 f"Green=Coarse (domain-invariant), Red=Fine (domain-specific)", fontsize=13)
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(overall_sim), color='gray', linestyle='--', alpha=0.5, 
               label=f'Mean={np.mean(overall_sim):.3f}')
    ax.legend()
    
    for bar, val in zip(bars, overall_sim):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp4_{args.dataset}_similarity.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp4_{args.dataset}_similarity.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Plot 2: Similarity heatmap (domain pairs x experts)
    # ======================================================================
    pair_names = sorted(pair_sims.keys())
    sim_matrix = np.array([pair_sims[p] for p in pair_names])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        sim_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=[f"E{i} (G={grid_sizes[i]})" for i in range(num_experts)],
        yticklabels=pair_names, ax=ax, vmin=0, vmax=1
    )
    ax.set_title(f"Cross-Domain Expert Similarity per Domain Pair ({args.dataset})", fontsize=13)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Domain Pair")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp4_{args.dataset}_pairwise_sim.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp4_{args.dataset}_pairwise_sim.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Plot 3: Scatter plot - Grid Size vs Cross-Domain Similarity
    # ======================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(grid_sizes, overall_sim, s=200, c=colors, edgecolors='black', zorder=5)
    
    for i in range(num_experts):
        ax.annotate(f'E{i}', (grid_sizes[i], overall_sim[i]), 
                    textcoords="offset points", xytext=(10, 5), fontsize=11)
    
    # Fit trend line
    if len(grid_sizes) > 2:
        z = np.polyfit(grid_sizes, overall_sim, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(grid_sizes), max(grid_sizes), 100)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
    
    ax.set_xlabel("Grid Size (G)", fontsize=12)
    ax.set_ylabel("Cross-Domain Cosine Similarity", fontsize=12)
    ax.set_title(f"Grid Size vs. Domain Invariance ({args.dataset})", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"exp4_{args.dataset}_grid_vs_sim.pdf"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, f"exp4_{args.dataset}_grid_vs_sim.png"), dpi=300)
    plt.close()
    
    # ======================================================================
    # Save results
    # ======================================================================
    output_data = {
        "dataset": args.dataset,
        "grid_sizes": grid_sizes,
        "num_experts": num_experts,
        "environments": env_names,
        "overall_similarity": overall_sim.tolist(),
        "pairwise_similarity": {k: v.tolist() for k, v in pair_sims.items()},
    }
    
    results_path = os.path.join(args.output_dir, f"exp4_{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXPERT SIMILARITY SUMMARY ({args.dataset})")
    print(f"{'='*60}")
    for i in range(num_experts):
        category = "COARSE" if grid_sizes[i] <= np.median(grid_sizes) else "FINE"
        print(f"  Expert {i} (G={grid_sizes[i]:>2}, {category}):  "
              f"Cross-domain similarity = {overall_sim[i]:.4f}")
    
    print(f"\nExpected: Coarse experts → HIGH similarity (>0.7)")
    print(f"          Fine experts   → LOW  similarity (<0.5)")
    print(f"Observation: {'CONFIRMED' if overall_sim[0] > overall_sim[-1] else 'NEED REVIEW'}")


if __name__ == "__main__":
    main()
