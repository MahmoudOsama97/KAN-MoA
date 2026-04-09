#!/usr/bin/env python3
"""
Experiment 1: Parse Grid Ablation Training Results (Table 6)
=============================================================
Parses exp1_*.out files to extract per-domain and average accuracy
for each grid configuration × seed. Produces:
  - JSON with all results
  - LaTeX table (Table 6)
  - Console summary

Usage:
  python experiments/exp1_parse_grid_ablation.py \
      --log_dir . \
      --dataset PACS \
      --output_dir experiments/exp1_results
"""

import argparse
import glob
import json
import os
import re
import numpy as np
from collections import defaultdict


def parse_exp1_out(filepath):
    """Parse a single exp1 .out file for the final accuracy table."""
    result = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the final table row: "| training-domain validation | 98.292% | 99.094% | ..."
    pattern = r'training-domain validation\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%'
    match = re.search(pattern, content)
    
    if match:
        result['domain_accs'] = [float(match.group(i)) for i in range(1, 5)]
        result['avg'] = float(match.group(5))
    
    # Extract domain names from the table header
    header_pattern = r'\|\s*Selection\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|'
    hmatch = re.search(header_pattern, content)
    if hmatch:
        result['domains'] = [hmatch.group(i) for i in range(1, 5)]
    
    # Extract grid config from the command
    grid_pattern = r'--kan_grids\s+([\d,]+)'
    gmatch = re.search(grid_pattern, content)
    if gmatch:
        result['kan_grids'] = gmatch.group(1)
    
    # Extract seed
    seed_pattern = r'--seed\s+(\d+)'
    smatch = re.search(seed_pattern, content)
    if smatch:
        result['seed'] = int(smatch.group(1))
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='.',
                        help='Directory containing exp1_*.out files')
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--output_dir', type=str, default='experiments/exp1_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all exp1 .out files for this dataset
    pattern = os.path.join(args.log_dir, f'exp1_{args.dataset}_*_s*_*.out')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No exp1 output files found matching: {pattern}")
        return
    
    print(f"Found {len(files)} exp1 output files for {args.dataset}")
    
    # Parse all files
    configs = defaultdict(list)  # config_name -> list of {seed, accs, avg}
    
    for filepath in files:
        fname = os.path.basename(filepath)
        # Extract config name from filename: exp1_PACS_uniform_3_s1_25692440.out
        # Config = everything between dataset and _s{seed}
        match = re.match(rf'exp1_{args.dataset}_(.+)_s(\d+)_\d+\.out', fname)
        if not match:
            continue
        
        config_name = match.group(1)
        seed = int(match.group(2))
        
        result = parse_exp1_out(filepath)
        if 'avg' not in result:
            print(f"  WARNING: Could not parse {fname}")
            continue
        
        result['seed'] = seed
        result['file'] = fname
        configs[config_name].append(result)
        print(f"  {config_name} s{seed}: {result['avg']:.3f}%")
    
    # Aggregate across seeds
    print(f"\n{'='*70}")
    print(f"GRID ABLATION RESULTS — {args.dataset} (Table 6)")
    print(f"{'='*70}")
    
    # Get domain names from any result
    domain_names = None
    for v in configs.values():
        if v and 'domains' in v[0]:
            domain_names = v[0]['domains']
            break
    domain_names = domain_names or ['D0', 'D1', 'D2', 'D3']
    
    # Map config names to display labels
    config_labels = {
        'uniform_3': 'G = (3,3,3,3)',
        'uniform_10': 'G = (10,10,10,10)',
        'uniform_20': 'G = (20,20,20,20)',
        'hetero_3_5_10_20': 'G = (3,5,10,20) [Ours]',
        'reverse_20_10_5_3': 'G = (20,10,5,3) [Reversed]',
    }
    
    # Sort: uniform_3, uniform_10, uniform_20, hetero, reverse
    desired_order = ['uniform_3', 'uniform_10', 'uniform_20', 'hetero_3_5_10_20', 'reverse_20_10_5_3']
    sorted_configs = [c for c in desired_order if c in configs]
    # Add any remaining configs not in the desired order
    sorted_configs += [c for c in sorted(configs.keys()) if c not in sorted_configs]
    
    results_table = {}
    
    header = f"{'Configuration':<30}"
    for d in domain_names:
        header += f" {d:>12}"
    header += f" {'Avg':>8} {'±Std':>8}"
    print(header)
    print("-" * len(header))
    
    for config_name in sorted_configs:
        runs = configs[config_name]
        label = config_labels.get(config_name, config_name)
        
        avgs = [r['avg'] for r in runs]
        mean_avg = np.mean(avgs)
        std_avg = np.std(avgs)
        
        # Per-domain means
        domain_means = []
        domain_stds = []
        for d_idx in range(len(domain_names)):
            d_accs = [r['domain_accs'][d_idx] for r in runs if 'domain_accs' in r]
            domain_means.append(np.mean(d_accs))
            domain_stds.append(np.std(d_accs))
        
        row = f"{label:<30}"
        for dm in domain_means:
            row += f" {dm:>11.2f}%"
        row += f" {mean_avg:>7.2f}% {std_avg:>6.2f}%"
        print(row)
        
        results_table[config_name] = {
            'label': label,
            'seeds': [r['seed'] for r in runs],
            'per_seed_avg': avgs,
            'mean_avg': mean_avg,
            'std_avg': std_avg,
            'domain_names': domain_names,
            'domain_means': domain_means,
            'domain_stds': domain_stds,
            'kan_grids': runs[0].get('kan_grids', ''),
        }
    
    # Best config
    best_config = max(results_table.keys(), key=lambda k: results_table[k]['mean_avg'])
    print(f"\nBest: {config_labels.get(best_config, best_config)} "
          f"({results_table[best_config]['mean_avg']:.2f}%)")
    
    # ======================================================================
    # Generate LaTeX Table
    # ======================================================================
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Grid ablation on " + args.dataset + r". All experts share the same rank allocation. "
        r"``Ours'' uses heterogeneous grids $G=(3,5,10,20)$.}",
        r"\label{tab:grid_ablation}",
        r"\begin{tabular}{l" + "c" * len(domain_names) + "c}",
        r"\toprule",
        r"Configuration & " + " & ".join(domain_names) + r" & Avg. \\",
        r"\midrule",
    ]
    
    for config_name in sorted_configs:
        r = results_table[config_name]
        label = r['label'].replace('_', r'\_')
        if 'Ours' in label:
            label = r"\textbf{" + label + "}"
        
        cells = [f"{dm:.1f}" for dm in r['domain_means']]
        avg_str = f"{r['mean_avg']:.1f}"
        if 'Ours' in r['label']:
            avg_str = r"\textbf{" + avg_str + "}"
        
        latex_lines.append(f"{label} & " + " & ".join(cells) + f" & {avg_str}" + r" \\")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex_table = "\n".join(latex_lines)
    
    # Save
    results_data = {
        'dataset': args.dataset,
        'num_configs': len(results_table),
        'total_runs': sum(len(configs[c]) for c in configs),
        'configs': results_table,
    }
    
    json_path = os.path.join(args.output_dir, f'exp1_{args.dataset}_grid_ablation.json')
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")
    
    latex_path = os.path.join(args.output_dir, f'exp1_{args.dataset}_table6.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX: {latex_path}")
    
    print(f"\n--- LaTeX Table 6 ---")
    print(latex_table)


if __name__ == '__main__':
    main()
