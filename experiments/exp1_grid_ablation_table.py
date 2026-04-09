#!/usr/bin/env python3
"""
Generate a consolidated Grid Ablation table (Table 6) across all datasets.

Reads from the already-generated log_analysis_results.csv (produced by
analyze_logs.py) and creates a single LaTeX table where:
  - Rows   = grid configurations (uniform_3, uniform_10, uniform_20, hetero, reverse)
  - Columns = datasets (PACS, OfficeHome, VLCS, TerraIncognita, DomainNet, Avg.)

Usage:
    python experiments/exp1_grid_ablation_table.py
    python experiments/exp1_grid_ablation_table.py --csv path/to/log_analysis_results.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
DATASETS = ["PACS", "OfficeHome", "VLCS", "TerraIncognita", "DomainNet"]
DATASET_SHORT = {
    "PACS": "PACS",
    "OfficeHome": "Office-Home",
    "VLCS": "VLCS",
    "TerraIncognita": "Terra Inc.",
    "DomainNet": "DomainNet",
}

# Ordered grid configs (rows of the table)
GRID_CONFIGS = [
    ("uniform_3",  "G=(3,3,3,3)"),
    ("uniform_10", "G=(10,10,10,10)"),
    ("uniform_20", "G=(20,20,20,20)"),
    ("hetero",     "G=(3,5,10,20) [Ours]"),
    ("reverse",    "G=(20,10,5,3) [Rev.]"),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "exp1_results")


def load_data(csv_path):
    """Load the CSV and filter to EXP1 grid ablation runs."""
    df = pd.read_csv(csv_path)

    # Keep only EXP1 rows
    mask = df["experiment_name"].str.startswith("EXP1", na=False)
    exp1 = df[mask].copy()

    if exp1.empty:
        print("ERROR: No EXP1 rows found in the CSV. Run analyze_logs.py first.")
        sys.exit(1)

    return exp1


def build_table(exp1):
    """
    Build a table dict:
        table[grid_config][dataset] = {"mean": ..., "std": ...}
    Also compute per-row average across datasets.
    """
    table = {}
    for gc_key, _ in GRID_CONFIGS:
        table[gc_key] = {}
        for ds in DATASETS:
            sub = exp1[(exp1["dataset"] == ds) & (exp1["grid_config"] == gc_key)]
            accs = sub["best_acc"].dropna()
            if len(accs) > 0:
                table[gc_key][ds] = {"mean": accs.mean(), "std": accs.std(), "n": len(accs)}
            else:
                table[gc_key][ds] = {"mean": None, "std": None, "n": 0}

        # Compute average across datasets
        means = [table[gc_key][ds]["mean"] for ds in DATASETS if table[gc_key][ds]["mean"] is not None]
        if means:
            table[gc_key]["Avg"] = {"mean": np.mean(means), "std": None, "n": len(means)}
        else:
            table[gc_key]["Avg"] = {"mean": None, "std": None, "n": 0}

    return table


def find_best_per_column(table):
    """Return the grid_config key with the highest mean for each dataset+Avg."""
    best = {}
    columns = DATASETS + ["Avg"]
    for col in columns:
        best_val = -1
        best_key = None
        for gc_key, _ in GRID_CONFIGS:
            m = table[gc_key][col]["mean"]
            if m is not None and m > best_val:
                best_val = m
                best_key = gc_key
        best[col] = best_key
    return best


def fmt_cell(entry, is_best=False):
    """Format a table cell as 'mean±std' or just 'mean'."""
    if entry["mean"] is None:
        return "—"
    m = entry["mean"]
    s = entry["std"]
    if s is not None and s > 0 and entry["n"] > 1:
        text = f"{m:.1f}{{\\scriptsize±{s:.1f}}}"
    else:
        text = f"{m:.1f}"
    if is_best:
        text = f"\\textbf{{{text}}}"
    return text


def generate_latex(table, best):
    """Generate LaTeX table string."""
    ncols = len(DATASETS) + 2  # config + datasets + avg
    col_spec = "l" + "c" * (len(DATASETS) + 1)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Grid ablation study (Table~6). Mean accuracy (\\%) over 3 seeds. "
                  "\\textbf{Bold} = best per dataset. "
                  "``Ours'' uses heterogeneous grids $G\\!=\\!(3,5,10,20)$.}")
    lines.append("\\label{tab:grid_ablation_all}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header_parts = ["Configuration"]
    for ds in DATASETS:
        header_parts.append(DATASET_SHORT[ds])
    header_parts.append("Avg.")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for gc_key, gc_label in GRID_CONFIGS:
        is_ours = gc_key == "hetero"
        if is_ours:
            row_label = f"\\textbf{{{gc_label}}}"
        else:
            row_label = gc_label

        cells = [row_label]
        for col in DATASETS + ["Avg"]:
            entry = table[gc_key][col]
            is_best = (best[col] == gc_key)
            cells.append(fmt_cell(entry, is_best=is_best))

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_console_table(table, best):
    """Print a readable ASCII table."""
    columns = DATASETS + ["Avg"]
    col_w = {ds: max(len(DATASET_SHORT.get(ds, ds)), 12) for ds in columns}
    col_w["Avg"] = 10

    # Header
    hdr = f"{'Configuration':<25s}"
    for col in columns:
        label = DATASET_SHORT.get(col, col)
        hdr += f" | {label:>{col_w[col]}s}"
    print(hdr)
    print("-" * len(hdr))

    for gc_key, gc_label in GRID_CONFIGS:
        row = f"{gc_label:<25s}"
        for col in columns:
            entry = table[gc_key][col]
            if entry["mean"] is None:
                cell = "—"
            elif entry["std"] is not None and entry["n"] > 1:
                cell = f"{entry['mean']:.1f}±{entry['std']:.1f}"
            else:
                cell = f"{entry['mean']:.1f}"
            marker = " *" if best[col] == gc_key else ""
            row += f" | {cell + marker:>{col_w[col]}s}"
        print(row)


def save_json(table, path):
    """Save table as JSON for programmatic use."""
    out = {}
    for gc_key, gc_label in GRID_CONFIGS:
        out[gc_key] = {"label": gc_label}
        for col in DATASETS + ["Avg"]:
            entry = table[gc_key][col]
            out[gc_key][col] = {
                "mean": round(entry["mean"], 2) if entry["mean"] is not None else None,
                "std": round(entry["std"], 2) if entry["std"] is not None else None,
                "n": entry["n"],
            }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated grid ablation table.")
    parser.add_argument("--csv", default="log_analysis_results.csv",
                        help="Path to log_analysis_results.csv (default: log_analysis_results.csv)")
    args = parser.parse_args()

    # Resolve CSV path
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        # Try relative to script location first, then cwd
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        candidate = os.path.join(repo_root, csv_path)
        if os.path.exists(candidate):
            csv_path = candidate
        # else: assume cwd-relative (argparse default)

    print(f"Loading data from: {csv_path}")
    exp1 = load_data(csv_path)
    print(f"Found {len(exp1)} EXP1 runs across {exp1['dataset'].nunique()} datasets\n")

    # Build table
    table = build_table(exp1)
    best = find_best_per_column(table)

    # Console output
    print("=" * 90)
    print("GRID ABLATION — Table 6 (All Datasets)")
    print("=" * 90)
    generate_console_table(table, best)
    print("=" * 90)
    print("(* = best per column)\n")

    # Generate outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # LaTeX
    latex = generate_latex(table, best)
    tex_path = os.path.join(RESULTS_DIR, "table6_grid_ablation_all.tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  LaTeX: {tex_path}")

    # JSON
    json_path = os.path.join(RESULTS_DIR, "table6_grid_ablation_all.json")
    save_json(table, json_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
