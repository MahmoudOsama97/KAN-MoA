#!/usr/bin/env python3
# filepath: /home/mosama97/scratch/MOKanA/analyze_logs.py
"""
Analyze all .out log files recursively across all subdirectories and generate
a comprehensive CSV report. Extracts hyperparameters (including G values / 
kan_grids, expert_ranks, router_type, regularization flags, etc.), training
metrics, and per-environment accuracy with oracle model selection.

Usage:
    python analyze_logs.py                    # recursive from current dir
    python analyze_logs.py /path/to/repo      # recursive from given root
    python analyze_logs.py . --no-recursive   # only top-level folder
"""

import re
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Define patterns for table parsing
DECIMAL_PATTERN = r"(-?\d+\.\d+)"
INTEGER_PATTERN = r"(\d+)"
WHITESPACE = r"\s+"

# Table columns from log format (basic 16-column format for PACS/OfficeHome/etc)
TABLE_COLUMNS = [
    "test_in",
    "test_out", 
    "train_in",
    "train_out",
    "tr_outloss",
    "data1",
    "data2",
    "data3",
    "data4",
    "data5",
    "step",
    "epoch",
    "loss",
    "loss1",
    "step_time",
    "eval_time",
]

# DomainNet has 18 columns (6 envs + extra columns)
TABLE_COLUMNS_DOMAINNET = [
    "test_in",
    "test_out", 
    "train_in",
    "train_out",
    "tr_outloss",
    "env0_out",
    "env1_out",
    "env2_out",
    "env3_out",
    "env4_out",
    "env5_out",
    "extra1",  # Sometimes env_in appears here
    "step",
    "epoch",
    "loss",
    "cls_loss",
    "step_time",
    "eval_time",
]

# Positions for decimal vs integer columns
DECIMAL_POSITIONS = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15]
INTEGER_POSITIONS = [10]
TOTAL_COLUMNS = 16

# Build regex pattern dynamically
def build_table_pattern():
    patterns = []
    for i in range(TOTAL_COLUMNS):
        if i in INTEGER_POSITIONS:
            patterns.append(INTEGER_PATTERN)
        else:
            patterns.append(DECIMAL_PATTERN)
    return WHITESPACE.join(patterns)

TABLE_PATTERN = build_table_pattern()


def extract_summary_table(log_content):
    """Extract final summary table from DomainNet-style logs."""
    # Look for training-domain validation row with percentages
    pattern = r"training-domain validation\s*=\s*([\d.]+)%"
    matches = re.findall(pattern, log_content)
    if matches:
        # Each match is a per-environment final accuracy
        per_env_accs = [float(m) for m in matches]
        if per_env_accs:
            avg_acc = np.mean(per_env_accs)
            return {
                'per_env_accs': per_env_accs,
                'avg_acc': avg_acc
            }
    return None


def extract_domainnet_table_data(log_content):
    """Extract table data specifically for DomainNet format (18 columns).
    
    DomainNet log format per test env block:
    - 18 numeric columns per data row
    - Columns: test_in, test_out, train_in, train_out, tr_outloss, 
               env0_X, env1_X, env2_X, env3_X, env4_X, env5_X, extra_env_col,
               step, epoch, loss, cls_loss, step_time, eval_time
    - The env columns shift depending on which env is being tested
    """
    # DomainNet has 18 columns: 12 decimals, 1 integer (step), 5 more decimals
    DECIMAL_POSITIONS_DN = list(range(12)) + list(range(13, 18))
    INTEGER_POSITIONS_DN = [12]
    TOTAL_COLUMNS_DN = 18
    
    patterns = []
    for i in range(TOTAL_COLUMNS_DN):
        if i in INTEGER_POSITIONS_DN:
            patterns.append(INTEGER_PATTERN)
        else:
            patterns.append(DECIMAL_PATTERN)
    
    dn_pattern = WHITESPACE.join(patterns)
    matches = re.findall(dn_pattern, log_content)
    
    if not matches:
        return None
    
    columns_dn = [
        "test_in", "test_out", "train_in", "train_out", "tr_outloss",
        "data0", "data1", "data2", "data3", "data4", "data5", "data6",
        "step", "epoch", "loss", "cls_loss", "step_time", "eval_time",
    ]
    
    df = pd.DataFrame(matches, columns=columns_dn)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def analyze_domainnet_log(log_content, config):
    """Analyze DomainNet log with correct 6-environment logic."""
    result = {}
    
    n_steps = config.get('n_steps', 15000)
    steps_per_row = 1000  # DomainNet uses 1000-step intervals
    test_envs = [0, 1, 2, 3, 4, 5]
    rows_per_env = n_steps // steps_per_row + 1  # 16 rows for 15000 steps (0, 1000, ..., 15000)
    
    # Extract table data using DomainNet-specific pattern
    df = extract_domainnet_table_data(log_content)
    
    if df is None or df.empty:
        # Try alternative extraction from summary lines
        return None
    
    # Detect rows_per_env by looking for step resets (step goes back to 0)
    step_col = df['step'].values
    detected_rows_per_env = None
    for i in range(1, len(step_col)):
        if step_col[i] < step_col[i-1]:  # Step decreased = new env
            detected_rows_per_env = i
            break
    
    if detected_rows_per_env:
        rows_per_env = detected_rows_per_env
    
    # Split data by test environment
    test_env_dfs = []
    cur_start_row = 0
    num_complete_envs = 0
    for test_env in test_envs:
        cur_end_row = cur_start_row + rows_per_env
        if cur_start_row < len(df):
            test_env_df = df.iloc[cur_start_row:min(cur_end_row, len(df))]
            if not test_env_df.empty:
                test_env_dfs.append(test_env_df.reset_index(drop=True))
                # Check if this env is complete (has all expected rows)
                if len(test_env_df) >= rows_per_env:
                    num_complete_envs += 1
        cur_start_row = cur_end_row
    
    if len(test_env_dfs) < 1:
        return None  # No data
    
    # Get final test_in values for each environment
    final_test_ins = []
    final_test_outs = []
    for i, env_df in enumerate(test_env_dfs):
        if not env_df.empty:
            final_val = env_df.iloc[-1]['test_in']
            final_test_ins.append(final_val)
            result[f'te{i}_final_test_in'] = final_val
            if 'test_out' in env_df.columns:
                final_out = env_df.iloc[-1]['test_out']
                final_test_outs.append(final_out)
                result[f'te{i}_final_test_out'] = final_out
    
    if final_test_ins:
        result['avg_final_test_in'] = np.mean(final_test_ins) * 100
        result['std_final_test_in'] = np.std(final_test_ins) * 100
    if final_test_outs:
        result['avg_final_test_out'] = np.mean(final_test_outs) * 100
    
    # Oracle model selection (same stopping step for all envs)
    best_avg_train_out = 0
    best_step_train_out = None
    best_avg_tr_out_loss = 0
    best_step_tr_out_loss = None
    
    # Get min step to start from (at least 2 checkpoints in)
    min_step = steps_per_row * 2
    
    for stopping_step in range(min_step, n_steps + 1, steps_per_row):
        chosen_vals_train_out = []
        chosen_vals_tr_out_loss = []
        
        for env_df in test_env_dfs:
            if env_df.empty:
                continue
            # Get rows up to stopping step
            env_df_tmp = env_df[env_df["step"] <= stopping_step]
            if env_df_tmp.empty:
                continue
            
            # Best based on max train_out
            best_idx = env_df_tmp["train_out"].idxmax()
            chosen_vals_train_out.append(env_df_tmp.loc[best_idx, "test_in"])
            
            # Best based on min tr_outloss
            best_idx = env_df_tmp["tr_outloss"].idxmin()
            chosen_vals_tr_out_loss.append(env_df_tmp.loc[best_idx, "test_in"])
        
        if chosen_vals_train_out:
            avg_train_out = np.mean(chosen_vals_train_out)
            if avg_train_out > best_avg_train_out:
                best_avg_train_out = avg_train_out
                best_step_train_out = stopping_step
        
        if chosen_vals_tr_out_loss:
            avg_tr_out_loss = np.mean(chosen_vals_tr_out_loss)
            if avg_tr_out_loss > best_avg_tr_out_loss:
                best_avg_tr_out_loss = avg_tr_out_loss
                best_step_tr_out_loss = stopping_step
    
    result['avg_best_trainout_testin'] = best_avg_train_out * 100
    result['best_step_trainout'] = best_step_train_out
    result['avg_best_troutloss_testin'] = best_avg_tr_out_loss * 100
    result['best_step_troutloss'] = best_step_tr_out_loss
    
    # Get training metrics from last complete environment dataframe
    if test_env_dfs:
        # Use the last complete env, or last available if none complete
        last_df = test_env_dfs[-1]
        if not last_df.empty:
            result['final_loss'] = last_df['loss'].iloc[-1] if 'loss' in last_df.columns else None
            result['min_loss'] = last_df['loss'].min() if 'loss' in last_df.columns else None
            result['total_steps'] = n_steps
    
    # Mark if incomplete
    result['num_envs_completed'] = num_complete_envs
    
    result['best_acc'] = max(
        result.get('avg_final_test_in', 0),
        result.get('avg_best_trainout_testin', 0),
        result.get('avg_best_troutloss_testin', 0)
    )
    
    return result


def extract_config_from_log(log_content):
    """Extract configuration/hyperparameters from log content.
    
    Parses three sources in the log:
      1. Command :: line  (the actual CLI invocation)
      2. Args: section    (tab-indented key: value pairs)
      3. HParams: section (tab-indented key: value pairs)
    """
    config = {}
    
    # ===== 0. Quick check: skip logs that have no training data =====
    if 'Args:' not in log_content and 'Command ::' not in log_content:
        return config  # Empty / non-training log
    
    # ===== 1. Parse the Command :: line for CLI arguments =====
    cmd_match = re.search(r"Command\s*::\s*(.+)", log_content)
    cmd_line = cmd_match.group(1).strip() if cmd_match else ""
    
    # Helper: extract --flag value from command line
    def _cmd_val(flag):
        m = re.search(rf"--{flag}\s+(\S+)", cmd_line)
        return m.group(1) if m else None
    
    # ===== 2. Parse Args: section (tab-indented key: value) =====
    args_section = {}
    args_match = re.search(r"Args:\s*\n((?:\s+\S.*\n)*)", log_content)
    if args_match:
        for line in args_match.group(1).split('\n'):
            line = line.strip()
            if ':' in line:
                k, v = line.split(':', 1)
                args_section[k.strip()] = v.strip()
    
    # ===== 3. Parse HParams: section =====
    hparams_section = {}
    hp_match = re.search(r"HParams:\s*\n((?:\s+\S.*\n)*)", log_content)
    if hp_match:
        for line in hp_match.group(1).split('\n'):
            line = line.strip()
            if ':' in line:
                k, v = line.split(':', 1)
                hparams_section[k.strip()] = v.strip()
    
    # ===== 4. Extract fields (prefer Args section, fallback to cmd line) =====
    
    # --- Core identifiers ---
    config['dataset'] = args_section.get('dataset') or _cmd_val('dataset') or ''
    config['algorithm'] = args_section.get('algorithm') or _cmd_val('algorithm') or ''
    
    # Experiment name (from Args name field)
    exp_name = args_section.get('name', '')
    if exp_name:
        config['experiment_name'] = exp_name
    
    # --- KAN grid sizes (THE critical G values) ---
    kan_grids_raw = args_section.get('kan_grids') or _cmd_val('kan_grids') or ''
    if kan_grids_raw:
        config['kan_grids'] = kan_grids_raw
        # Parse individual grid values
        try:
            g_values = [int(g) for g in kan_grids_raw.split(',')]
            config['n_grid_values'] = len(g_values)
            config['min_grid'] = min(g_values)
            config['max_grid'] = max(g_values)
            config['grid_values'] = g_values  # list for programmatic use
            
            # Derive grid_config type
            if len(set(g_values)) == 1:
                config['grid_config'] = f'uniform_{g_values[0]}'
            elif g_values == sorted(g_values):
                config['grid_config'] = 'hetero'  # ascending = heterogeneous (our method)
            elif g_values == sorted(g_values, reverse=True):
                config['grid_config'] = 'reverse'
            else:
                config['grid_config'] = 'custom'
            
            # Also try to confirm from experiment name
            if exp_name:
                name_lower = exp_name.lower()
                if 'hetero' in name_lower:
                    config['grid_config'] = 'hetero'
                elif 'reverse' in name_lower or 'rev' in name_lower:
                    config['grid_config'] = 'reverse'
                elif 'uniform' in name_lower:
                    config['grid_config'] = f'uniform_{g_values[0]}'
        except (ValueError, TypeError):
            config['grid_config'] = 'unknown'
    
    # --- Expert ranks ---
    expert_ranks_raw = args_section.get('expert_ranks') or _cmd_val('expert_ranks') or ''
    if expert_ranks_raw:
        config['expert_ranks'] = expert_ranks_raw
        try:
            r_values = [int(r) for r in expert_ranks_raw.split(',')]
            config['rank_values'] = r_values
            config['min_rank'] = min(r_values)
            config['max_rank'] = max(r_values)
        except (ValueError, TypeError):
            pass
    
    # --- Number of experts ---
    ne = args_section.get('num_experts') or _cmd_val('num_experts')
    if ne:
        try:
            config['num_experts'] = int(ne)
        except ValueError:
            pass
    
    # --- Adapter type ---
    config['adapter_type'] = args_section.get('adapter_type') or _cmd_val('adapter_type') or ''
    
    # --- Router type ---
    config['router_type'] = args_section.get('router_type') or _cmd_val('router_type') or ''
    
    # --- Injection targets ---
    config['injection_targets'] = args_section.get('injection_targets') or _cmd_val('injection_targets') or ''
    
    # --- Model ---
    config['model'] = args_section.get('model') or _cmd_val('model') or ''
    
    # --- Seed ---
    seed_val = args_section.get('seed') or _cmd_val('seed')
    if seed_val:
        try:
            config['seed'] = int(seed_val)
        except ValueError:
            pass
    
    # --- Steps ---
    steps_val = args_section.get('steps') or _cmd_val('steps')
    if steps_val:
        try:
            config['n_steps'] = int(steps_val)
        except ValueError:
            pass
    
    # --- Regularization flags (from Args) ---
    for flag in ['enable_orthogonality', 'enable_lipschitz', 'enable_complexity_invariance']:
        val = args_section.get(flag) or _cmd_val(flag)
        if val is not None:
            try:
                config[flag] = int(val)
            except ValueError:
                config[flag] = 1 if val.lower() in ('true', '1') else 0
    
    # --- Lambda values (from Args) ---
    for lam in ['lambda_orthogonality', 'lambda_lipschitz', 'lambda_complexity']:
        val = args_section.get(lam) or _cmd_val(lam)
        if val:
            try:
                config[lam] = float(val)
            except ValueError:
                pass
    
    # --- l_aux flag ---
    l_aux_val = args_section.get('l_aux')
    if l_aux_val is not None:
        config['l_aux'] = l_aux_val.strip() == 'True'
    
    # ===== 5. HParams section (training hyperparameters) =====
    # Learning rate, batch_size, weight_decay
    for hp_key, config_key in [('lr', 'lr'), ('batch_size', 'batch_size'), ('weight_decay', 'weight_decay')]:
        val = hparams_section.get(hp_key)
        if val:
            try:
                config[config_key] = float(val) if '.' in val or 'e' in val.lower() else int(val)
            except ValueError:
                pass
    
    # KAN-MOA loss weights
    for param in ['lipschitz_weight', 'ortho_weight', 'complexity_weight', 
                  'load_balance_weight', 'grid_entropy_weight', 'domain_routing_weight']:
        val = hparams_section.get(param)
        if val:
            try:
                config[param] = float(val)
            except ValueError:
                pass
    
    # Feature toggles
    for toggle in ['enable_load_balance', 'enable_grid_entropy', 'enable_pairwise_ortho', 'enable_domain_routing']:
        val = hparams_section.get(toggle)
        if val is not None:
            config[toggle] = val.strip().lower() in ('true', '1')
    
    # KAN-specific HParams
    for hp in ['kan_init_scale', 'kan_grad_warmup_steps', 'router_noise', 'router_temperature']:
        val = hparams_section.get(hp)
        if val:
            try:
                config[hp] = float(val)
            except ValueError:
                pass
    
    # ===== 6. Parameter counts =====
    match = re.search(r"# of params\s*=\s*(\d+)", log_content)
    if match:
        config['total_params'] = int(match.group(1))
    
    match = re.search(r"# of trainable params\s*=\s*(\d+)", log_content)
    if match:
        config['trainable_params'] = int(match.group(1))
    
    return config


def extract_table_data(log_content, dataset=None):
    """Extract tabular metrics from log content."""
    # For DomainNet, use the specialized extractor
    if dataset and dataset.lower() == 'domainnet':
        return extract_domainnet_table_data(log_content)
    
    # For non-DomainNet datasets, parse lines that have numeric data
    # Look for lines with the pattern: multiple float/int values separated by whitespace
    lines = log_content.split('\n')
    data_rows = []
    
    for line in lines:
        # Skip lines with INFO prefix - extract the data part
        if 'INFO' in line:
            # Find where the timestamp ends and data begins
            parts = line.split('|')
            if len(parts) >= 2:
                line = parts[-1].strip()
        
        # Skip header lines and non-data lines
        if 'test_in' in line.lower() or 'step' in line.lower() and 'epoch' in line.lower():
            continue
        
        # Try to parse as numeric data row
        values = line.split()
        if len(values) >= 10:  # Minimum expected columns
            try:
                # Check if first few values are numeric
                float(values[0])
                float(values[1])
                # This looks like a data row
                data_rows.append(values)
            except (ValueError, IndexError):
                continue
    
    if not data_rows:
        # Fallback to original pattern matching
        matches = re.findall(TABLE_PATTERN, log_content)
        if not matches:
            return None
        df = pd.DataFrame(matches, columns=TABLE_COLUMNS)
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    
    # Determine column count and create DataFrame
    # Use the first row to determine structure
    num_cols = len(data_rows[0])
    
    # Create generic column names, but ensure we have test_in, test_out, step
    if num_cols >= 16:
        col_names = TABLE_COLUMNS[:num_cols]
    else:
        col_names = [f'col{i}' for i in range(num_cols)]
        # Try to identify key columns
        if num_cols >= 2:
            col_names[0] = 'test_in'
            col_names[1] = 'test_out'
    
    # Ensure all rows have same length
    max_cols = max(len(row) for row in data_rows)
    padded_rows = []
    for row in data_rows:
        if len(row) < max_cols:
            row = row + [''] * (max_cols - len(row))
        padded_rows.append(row[:len(col_names)])
    
    df = pd.DataFrame(padded_rows, columns=col_names[:len(padded_rows[0])])
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Ensure we have a step column - look for integer column that increments
    if 'step' not in df.columns:
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                vals = df[col].dropna().values
                if len(vals) > 1 and all(v >= 0 and v == int(v) for v in vals[:5] if not np.isnan(v)):
                    if vals[1] > vals[0]:  # Incrementing
                        df = df.rename(columns={col: 'step'})
                        break
    
    return df


def analyze_single_log(log_filepath, root_folder=None):
    """Analyze a single log file and return metrics."""
    result = {
        'filepath': str(log_filepath),
        'filename': os.path.basename(log_filepath),
    }
    
    # Add subfolder relative to root
    if root_folder:
        try:
            rel = os.path.relpath(os.path.dirname(log_filepath), root_folder)
            result['subfolder'] = rel if rel != '.' else '(root)'
        except ValueError:
            result['subfolder'] = os.path.dirname(str(log_filepath))
    
    try:
        with open(log_filepath, 'r') as f:
            log_content = f.read()
    except Exception as e:
        result['error'] = str(e)
        return result
    
    # Skip empty or very short files (no training data)
    if len(log_content.strip()) < 100:
        result['error'] = 'Empty or too short'
        return result
    
    # Extract configuration
    config = extract_config_from_log(log_content)
    
    # Skip files that have no meaningful config (e.g. just CUDA info)
    if not config.get('dataset') and not config.get('algorithm'):
        result['error'] = 'No training config found (not a training log)'
        return result
    
    result.update(config)
    
    # For DomainNet, use the specialized analyzer
    dataset_name = config.get('dataset', '').lower()
    if dataset_name == 'domainnet':
        dn_result = analyze_domainnet_log(log_content, config)
        if dn_result:
            result.update(dn_result)
            return result
        # If specialized analyzer fails, try summary table extraction
        summary = extract_summary_table(log_content)
        if summary:
            result['avg_final_test_in'] = summary['avg_acc']
            result['avg_best_trainout_testin'] = summary['avg_acc']
            result['avg_best_troutloss_testin'] = summary['avg_acc']
            result['best_acc'] = summary['avg_acc']
            # Store per-env accuracies if available
            for i, acc in enumerate(summary.get('per_env_accs', [])):
                result[f'te{i}_final_test_in'] = acc / 100.0  # Store as decimal
            return result
        # Still return the result with config info even if no data extracted
        result['error'] = 'Could not extract DomainNet data'
        return result
    
    # For other datasets (PACS, OfficeHome, VLCS, TerraIncognita), use original logic
    # Extract table data
    dataset = config.get('dataset', '')
    df = extract_table_data(log_content, dataset=dataset)
    
    if df is None or df.empty:
        result['error'] = 'No table data found'
        return result
    
    # Get number of steps
    n_steps = config.get('n_steps', 5000)
    
    # For non-DomainNet datasets: PACS, OfficeHome, VLCS, TerraIncognita
    # They use 200-step intervals and have 4 test environments
    steps_per_row = 200  # Typical evaluation frequency for these datasets
    num_test_envs = 4
    
    try:
        # Detect rows_per_env from actual data by looking for step resets
        # Steps go 0, 200, 400, ... then reset to 0 for next env
        step_col = df['step'].values
        rows_per_env = 1
        for i in range(1, len(step_col)):
            if step_col[i] < step_col[i-1]:  # Step decreased = new env
                rows_per_env = i
                break
        else:
            # No reset found, assume all data is for one run or use calculation
            rows_per_env = n_steps // steps_per_row + 1
        
        # Calculate actual number of environments we can extract from data
        actual_num_envs = len(df) // rows_per_env if rows_per_env > 0 else 0
        # Use the minimum of expected and actual
        num_envs_to_use = min(num_test_envs, actual_num_envs) if actual_num_envs > 0 else num_test_envs
        
        if len(df) >= rows_per_env and rows_per_env > 1:
            # Split by test environment
            env_dfs = []
            for i in range(num_envs_to_use):
                start = i * rows_per_env
                end = start + rows_per_env
                if start < len(df):
                    env_df = df.iloc[start:min(end, len(df))].reset_index(drop=True)
                    env_dfs.append(env_df)
            
            # Final values per environment
            final_test_ins = []
            final_test_outs = []
            for i, env_df in enumerate(env_dfs):
                if env_df.empty:
                    continue
                final_row = env_df.iloc[-1]
                final_test_ins.append(final_row['test_in'])
                final_test_outs.append(final_row['test_out'])
                result[f'te{i}_final_test_in'] = final_row['test_in']
                result[f'te{i}_final_test_out'] = final_row['test_out']
            
            result['avg_final_test_in'] = np.mean(final_test_ins) * 100
            result['avg_final_test_out'] = np.mean(final_test_outs) * 100
            result['std_final_test_in'] = np.std(final_test_ins) * 100
            
            # ===== Oracle model selection (same stopping step for all envs) =====
            # Find optimal stopping step that maximizes average test_in across all envs
            best_avg_train_out = 0
            best_step_train_out = None
            best_avg_tr_outloss = 0
            best_step_tr_outloss = None
            
            # Iterate over possible stopping steps (starting from early steps)
            min_step = steps_per_row * 10  # Start from 10th checkpoint
            for stopping_step in range(min_step, n_steps + 1, steps_per_row):
                chosen_vals_train_out = []
                chosen_vals_tr_outloss = []
                
                for env_df in env_dfs:
                    if env_df.empty:
                        continue
                    # Get rows up to stopping step
                    env_df_tmp = env_df[env_df["step"] <= stopping_step]
                    if env_df_tmp.empty:
                        continue
                    
                    # Best based on max train_out up to this step
                    best_idx = env_df_tmp["train_out"].idxmax()
                    chosen_vals_train_out.append(env_df_tmp.loc[best_idx, "test_in"])
                    
                    # Best based on min tr_outloss up to this step
                    best_idx = env_df_tmp["tr_outloss"].idxmin()
                    chosen_vals_tr_outloss.append(env_df_tmp.loc[best_idx, "test_in"])
                if chosen_vals_train_out:
                    avg_train_out = np.mean(chosen_vals_train_out)
                    if avg_train_out > best_avg_train_out:
                        best_avg_train_out = avg_train_out
                        best_step_train_out = stopping_step
                
                if chosen_vals_tr_outloss:
                    avg_tr_outloss = np.mean(chosen_vals_tr_outloss)
                    if avg_tr_outloss > best_avg_tr_outloss:
                        best_avg_tr_outloss = avg_tr_outloss
                        best_step_tr_outloss = stopping_step
            
            # Store oracle selection results
            result['avg_best_trainout_testin'] = best_avg_train_out * 100
            result['best_step_trainout'] = best_step_train_out
            result['avg_best_troutloss_testin'] = best_avg_tr_outloss * 100
            result['best_step_troutloss'] = best_step_tr_outloss
            
            # Compute best accuracy across all selection methods
            result['best_acc'] = max(
                result['avg_final_test_in'],
                result['avg_best_trainout_testin'],
                result['avg_best_troutloss_testin']
            )
            
            # Per-environment values at best stopping step (based on train_out)
            if best_step_train_out:
                for i, env_df in enumerate(env_dfs):
                    if env_df.empty:
                        continue
                    env_df_tmp = env_df[env_df["step"] <= best_step_train_out]
                    if not env_df_tmp.empty:
                        best_idx = env_df_tmp["train_out"].idxmax()
                        result[f'te{i}_best_trainout_testin'] = env_df_tmp.loc[best_idx, "test_in"]
            
        else:
            # Single test environment or different format
            result['final_test_in'] = df['test_in'].iloc[-1] * 100 if 'test_in' in df.columns else None
            result['final_test_out'] = df['test_out'].iloc[-1] * 100 if 'test_out' in df.columns else None
            result['best_test_in'] = df['test_in'].max() * 100 if 'test_in' in df.columns else None
            result['max_train_out'] = df['train_out'].max() * 100 if 'train_out' in df.columns else None
            
    except Exception as e:
        result['parse_error'] = str(e)
        # Fallback: just get final row metrics
        if not df.empty:
            result['final_test_in'] = df['test_in'].iloc[-1] * 100 if 'test_in' in df.columns else None
            result['final_test_out'] = df['test_out'].iloc[-1] * 100 if 'test_out' in df.columns else None
    
    # Training metrics
    if df is not None and not df.empty:
        result['final_loss'] = df['loss'].iloc[-1] if 'loss' in df.columns else None
        result['min_loss'] = df['loss'].min() if 'loss' in df.columns else None
        result['total_steps'] = df['step'].max() if 'step' in df.columns else None
    
    return result


def analyze_folder(folder_path, pattern="*.out", recursive=True, skip_non_training=True):
    """Analyze all log files in a folder (recursively by default)."""
    folder = Path(folder_path).resolve()
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return pd.DataFrame()
    
    # Find all matching files, skip dcgm_*.out and analysis_*.out and error_*.out
    skip_prefixes = ('dcgm_', 'analysis_', 'error_')
    if recursive:
        files = [f for f in folder.rglob(pattern) if not any(f.name.startswith(p) for p in skip_prefixes)]
    else:
        files = [f for f in folder.glob(pattern) if not any(f.name.startswith(p) for p in skip_prefixes)]
    
    if not files:
        print(f"No files matching '{pattern}' found in {folder_path}")
        return pd.DataFrame()
    
    # Sort files for deterministic ordering
    files.sort(key=lambda f: str(f))
    
    print(f"Found {len(files)} log files to analyze across {len(set(f.parent for f in files))} directories...")
    
    results = []
    skipped = 0
    for i, filepath in enumerate(files):
        rel_path = filepath.relative_to(folder) if filepath.is_relative_to(folder) else filepath
        print(f"  [{i+1}/{len(files)}] {rel_path}")
        result = analyze_single_log(filepath, root_folder=str(folder))
        
        # Optionally skip non-training logs from the final results
        if skip_non_training and result.get('error') == 'No training config found (not a training log)':
            skipped += 1
            continue
        if skip_non_training and result.get('error') == 'Empty or too short':
            skipped += 1
            continue
        
        results.append(result)
    
    if skipped:
        print(f"  (Skipped {skipped} non-training/empty files)")
    
    if not results:
        print("No valid training logs found.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability — grid info is prominent
    priority_cols = [
        'filename', 'subfolder', 'dataset', 'algorithm', 'experiment_name',
        'grid_config', 'kan_grids', 'n_grid_values', 'min_grid', 'max_grid',
        'best_acc', 'avg_final_test_in', 'avg_best_trainout_testin', 'avg_best_troutloss_testin',
        'std_final_test_in',
        'adapter_type', 'num_experts', 'expert_ranks', 'min_rank', 'max_rank',
        'router_type', 'injection_targets',
        'trainable_params', 'total_params',
        'enable_orthogonality', 'enable_lipschitz', 'enable_complexity_invariance',
        'lambda_orthogonality', 'lambda_lipschitz', 'lambda_complexity',
        'enable_load_balance', 'enable_grid_entropy', 'enable_pairwise_ortho', 'enable_domain_routing',
        'lipschitz_weight', 'ortho_weight', 'complexity_weight',
        'load_balance_weight', 'grid_entropy_weight', 'domain_routing_weight',
        'kan_init_scale', 'kan_grad_warmup_steps', 'router_noise', 'router_temperature',
        'lr', 'batch_size', 'weight_decay', 'n_steps', 'seed', 'model',
    ]
    
    # Put priority columns first, then the rest
    existing_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[existing_priority + other_cols]
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze .out log files recursively and generate a CSV report with metrics and hyperparameters.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all .out files recursively from current directory (DEFAULT)
  python analyze_logs.py .
  
  # Analyze specific folder and save to custom CSV
  python analyze_logs.py /path/to/repo --output my_results.csv
  
  # Analyze only top-level directory (no recursion)
  python analyze_logs.py /path/to/repo --no-recursive
  
  # Analyze only files matching a pattern
  python analyze_logs.py /path/to/logs --pattern "exp1_*.out"
  
  # Include non-training logs (e.g. empty files, CUDA-only logs)
  python analyze_logs.py . --include-all
"""
    )
    
    parser.add_argument('folder', type=str, nargs='?', default='.',
                       help='Root folder to search for .out log files (default: current directory)')
    parser.add_argument('--output', '-o', type=str, default='log_analysis_results.csv',
                       help='Output CSV file path (default: log_analysis_results.csv)')
    parser.add_argument('--pattern', '-p', type=str, default='*.out',
                       help='File pattern to match (default: *.out)')
    parser.add_argument('--no-recursive', action='store_true',
                       help='Only search top-level folder (default: recursive)')
    parser.add_argument('--include-all', action='store_true',
                       help='Include non-training logs in output (empty, CUDA-only, etc.)')
    parser.add_argument('--sort-by', '-s', type=str, default='avg_best_trainout_testin',
                       help='Column to sort results by (default: avg_best_trainout_testin)')
    parser.add_argument('--descending', '-d', action='store_true', default=True,
                       help='Sort in descending order (default: True)')
    parser.add_argument('--show-top', '-t', type=int, default=10,
                       help='Show top N results in console (default: 10)')
    
    args = parser.parse_args()
    
    recursive = not args.no_recursive
    
    print(f"\n{'='*80}")
    print("LOG ANALYSIS TOOL")
    print(f"{'='*80}")
    print(f"Folder: {os.path.abspath(args.folder)}")
    print(f"Pattern: {args.pattern}")
    print(f"Recursive: {recursive}")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")
    
    # Analyze folder
    df = analyze_folder(args.folder, args.pattern, recursive=recursive,
                       skip_non_training=not args.include_all)
    
    if df.empty:
        print("No results to save.")
        return
    
    # Sort results
    if args.sort_by in df.columns:
        df = df.sort_values(by=args.sort_by, ascending=not args.descending, na_position='last')
    
    # Drop internal list columns before saving to CSV
    drop_cols = [c for c in ['grid_values', 'rank_values'] if c in df.columns]
    df_save = df.drop(columns=drop_cols, errors='ignore')
    
    # Save to CSV
    df_save.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total logs analyzed: {len(df)}")
    
    # Show accuracy columns if present
    acc_cols = [c for c in df.columns if 'test_in' in c.lower() or 'testin' in c.lower()]
    if acc_cols and not df[acc_cols[0]].isna().all():
        print(f"\nAccuracy Statistics ({acc_cols[0]}):")
        print(f"  Mean: {df[acc_cols[0]].mean():.2f}%")
        print(f"  Std:  {df[acc_cols[0]].std():.2f}%")
        print(f"  Min:  {df[acc_cols[0]].min():.2f}%")
        print(f"  Max:  {df[acc_cols[0]].max():.2f}%")
    
    # Show top results
    print(f"\n{'='*80}")
    print(f"TOP {args.show_top} RESULTS")
    print(f"{'='*80}")
    
    display_cols = ['filename', 'subfolder']
    for col in ['dataset', 'algorithm', 'grid_config', 'kan_grids',
                'best_acc', 'avg_best_trainout_testin', 'avg_final_test_in',
                'adapter_type', 'num_experts', 'expert_ranks',
                'router_type', 'injection_targets',
                'trainable_params', 'lr', 'seed']:
        if col in df.columns:
            display_cols.append(col)
    
    top_df = df.head(args.show_top)[display_cols].copy()
    
    # Format percentages
    for col in top_df.columns:
        if 'test' in col.lower() or col == 'best_acc':
            top_df[col] = top_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Format trainable_params as human-readable
    if 'trainable_params' in top_df.columns:
        def _fmt_params(x):
            if pd.isna(x) or not x:
                return "N/A"
            x = int(x)
            if x >= 1_000_000:
                return f"{x/1_000_000:.2f}M"
            elif x >= 1_000:
                return f"{x/1_000:.1f}K"
            return str(x)
        top_df['trainable_params'] = top_df['trainable_params'].apply(_fmt_params)
    
    print(top_df.to_string(index=False))
    print(f"\n{'='*80}")
    
    # --- Custom summary: Highest best_acc per dataset and their average ---
    datasets = ['PACS', 'OfficeHome', 'VLCS', 'TerraIncognita', 'DomainNet']
    best_accs = []
    best_rows = {}
    print(f"\n{'='*80}")
    print("HIGHEST best_acc PER DATASET (with method config & trainable params)")
    print(f"{'='*80}")
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        if not ds_df.empty and 'best_acc' in ds_df.columns:
            # For DomainNet, handle possible missing/NaN best_acc
            ds_df = ds_df[pd.notnull(ds_df['best_acc'])]
            if not ds_df.empty:
                best_idx = ds_df['best_acc'].idxmax()
                best_row = ds_df.loc[best_idx]
                max_acc = best_row['best_acc']
                best_accs.append(max_acc)
                best_rows[ds] = best_row
                
                # Format trainable params
                train_p = best_row.get('trainable_params', None)
                total_p = best_row.get('total_params', None)
                if pd.notna(train_p) and train_p:
                    train_p = int(train_p)
                    if train_p >= 1_000_000:
                        param_str = f"{train_p/1_000_000:.2f}M"
                    elif train_p >= 1_000:
                        param_str = f"{train_p/1_000:.1f}K"
                    else:
                        param_str = str(train_p)
                else:
                    param_str = "N/A"
                    
                if pd.notna(total_p) and total_p:
                    total_p = int(total_p)
                    if total_p >= 1_000_000:
                        total_str = f"{total_p/1_000_000:.2f}M"
                    else:
                        total_str = str(total_p)
                else:
                    total_str = "N/A"
                
                # Method config
                adapter = best_row.get('adapter_type', 'N/A') if pd.notna(best_row.get('adapter_type', None)) else 'N/A'
                n_exp = best_row.get('num_experts', 'N/A') if pd.notna(best_row.get('num_experts', None)) else 'N/A'
                ranks = best_row.get('expert_ranks', 'N/A') if pd.notna(best_row.get('expert_ranks', None)) else 'N/A'
                grids = best_row.get('kan_grids', 'N/A') if pd.notna(best_row.get('kan_grids', None)) else 'N/A'
                grid_cfg = best_row.get('grid_config', 'N/A') if pd.notna(best_row.get('grid_config', None)) else 'N/A'
                router = best_row.get('router_type', 'N/A') if pd.notna(best_row.get('router_type', None)) else 'N/A'
                targets = best_row.get('injection_targets', 'N/A') if pd.notna(best_row.get('injection_targets', None)) else 'N/A'
                subfld = best_row.get('subfolder', 'N/A') if pd.notna(best_row.get('subfolder', None)) else 'N/A'
                fname = best_row.get('filename', 'N/A')
                
                print(f"\n  {ds}:")
                print(f"    Accuracy:          {max_acc:.2f}%")
                print(f"    Grid Config:       {grid_cfg} (G={grids})")
                print(f"    Trainable Params:  {param_str} / {total_str} total")
                print(f"    Adapter Type:      {adapter}")
                print(f"    Num Experts:       {n_exp}")
                print(f"    Expert Ranks:      {ranks}")
                print(f"    Router Type:       {router}")
                print(f"    Injection Targets: {targets}")
                print(f"    Subfolder:         {subfld}")
                print(f"    Log File:          {fname}")
            else:
                print(f"\n  {ds}: N/A")
        else:
            print(f"\n  {ds}: N/A")
    
    if best_accs:
        avg_acc = np.mean(best_accs)
        print(f"\n{'-'*80}")
        print(f"Average of bests: {avg_acc:.2f}%")
        
        # Summary table of trainable params per dataset
        print(f"\n{'-'*80}")
        print(f"{'Dataset':15s} | {'Best Acc':>10s} | {'Grid Config':>14s} | {'Trainable Params':>18s} | {'Total Params':>14s} | {'Config':s}")
        print(f"{'-'*15}-+-{'-'*10}-+-{'-'*14}-+-{'-'*18}-+-{'-'*14}-+-{'-'*30}")
        for ds in datasets:
            if ds in best_rows:
                row = best_rows[ds]
                acc = row['best_acc']
                tp = row.get('trainable_params', None)
                total = row.get('total_params', None)
                
                if pd.notna(tp) and tp:
                    tp_int = int(tp)
                    tp_str = f"{tp_int:,}"
                else:
                    tp_str = "N/A"
                
                if pd.notna(total) and total:
                    total_str = f"{int(total):,}"
                else:
                    total_str = "N/A"
                
                adapter = row.get('adapter_type', '?') if pd.notna(row.get('adapter_type', None)) else '?'
                grids = row.get('kan_grids', '?') if pd.notna(row.get('kan_grids', None)) else '?'
                grid_cfg = row.get('grid_config', '?') if pd.notna(row.get('grid_config', None)) else '?'
                router = row.get('router_type', '?') if pd.notna(row.get('router_type', None)) else '?'
                config_str = f"{adapter} g={grids} ({grid_cfg}) r={router}"
                
                print(f"{ds:15s} | {acc:9.2f}% | {grid_cfg:>14s} | {tp_str:>18s} | {total_str:>14s} | {config_str}")
            else:
                print(f"{ds:15s} | {'N/A':>10s} | {'N/A':>14s} | {'N/A':>18s} | {'N/A':>14s} | N/A")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
