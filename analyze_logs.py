#!/usr/bin/env python3
# filepath: /home/mosama97/scratch/MOKanA/analyze_logs.py
"""
Analyze all .out log files in a folder and generate a comprehensive CSV report.
Extracts hyperparameters, training con        if len(df) >= rows_per_env and rows_per_env > 1 and num_envs_to_use > 0:
            # Split by test environment
            env_dfs = []
            for i in range(num_envs_to_use):
                start = i * rows_per_env
                end = start + rows_per_env
                if start < len(df):
                    env_df = df.iloc[start:min(end, len(df))].reset_index(drop=True)
                    if not env_df.empty:
                        env_dfs.append(env_df)
            
            if not env_dfs:
                # Fallback: treat entire dataframe as single environment
                env_dfs = [df]
            
            # Final values per environment and accuracy metrics.
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
    """Extract configuration/hyperparameters from log content."""
    config = {}
    
    # Dataset
    match = re.search(r"dataset\s*[=:]\s*(\w+)", log_content, re.IGNORECASE)
    if match:
        config['dataset'] = match.group(1)
    
    # Algorithm
    match = re.search(r"algorithm\s*[=:]\s*(\w+)", log_content, re.IGNORECASE)
    if match:
        config['algorithm'] = match.group(1)
    
    # Test environment
    match = re.search(r"test_env\w*\s*[=:]\s*\[?(\d+)", log_content, re.IGNORECASE)
    if match:
        config['test_env'] = int(match.group(1))
    
    # Number of steps
    match = re.search(r"n_steps\s*[=:]\s*(\d+)", log_content)
    if match:
        config['n_steps'] = int(match.group(1))
    
    # Learning rate
    match = re.search(r"['\"]?lr['\"]?\s*[=:]\s*([0-9.e\-]+)", log_content)
    if match:
        try:
            config['lr'] = float(match.group(1))
        except:
            pass
    
    # Batch size
    match = re.search(r"batch_size\s*[=:]\s*(\d+)", log_content)
    if match:
        config['batch_size'] = int(match.group(1))
    
    # Weight decay
    match = re.search(r"weight_decay\s*[=:]\s*([0-9.e\-]+)", log_content)
    if match:
        try:
            config['weight_decay'] = float(match.group(1))
        except:
            pass
    
    # KAN-MOA specific parameters
    for param in ['lipschitz_weight', 'ortho_weight', 'complexity_weight', 
                  'load_balance_weight', 'grid_entropy_weight', 'domain_routing_weight']:
        match = re.search(rf"['\"]?{param}['\"]?\s*[=:]\s*([0-9.e\-]+)", log_content)
        if match:
            try:
                config[param] = float(match.group(1))
            except:
                pass
    
    # Feature toggles
    for toggle in ['enable_load_balance', 'enable_grid_entropy', 'enable_pairwise_ortho', 'enable_domain_routing']:
        match = re.search(rf"['\"]?{toggle}['\"]?\s*[=:]\s*(True|False|true|false|1|0)", log_content)
        if match:
            val = match.group(1).lower()
            config[toggle] = val in ['true', '1']
    
    # Model
    match = re.search(r"model\s*[=:]\s*['\"]?(\S+)['\"]?", log_content)
    if match:
        config['model'] = match.group(1).strip("'\"")
    
    # Seed
    match = re.search(r"seed\s*[=:]\s*(\d+)", log_content)
    if match:
        config['seed'] = int(match.group(1))
    
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


def analyze_single_log(log_filepath):
    """Analyze a single log file and return metrics."""
    result = {
        'filepath': str(log_filepath),
        'filename': os.path.basename(log_filepath),
    }
    
    try:
        with open(log_filepath, 'r') as f:
            log_content = f.read()
    except Exception as e:
        result['error'] = str(e)
        return result
    
    # Extract configuration
    config = extract_config_from_log(log_content)
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


def analyze_folder(folder_path, pattern="*.out", recursive=False):
    """Analyze all log files in a folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return pd.DataFrame()
    
    # Find all matching files, skip dcgm_*.out
    if recursive:
        files = [f for f in folder.rglob(pattern) if not f.name.startswith('dcgm_')]
    else:
        files = [f for f in folder.glob(pattern) if not f.name.startswith('dcgm_')]
    
    if not files:
        print(f"No files matching '{pattern}' found in {folder_path}")
        return pd.DataFrame()
    
    print(f"Found {len(files)} log files to analyze...")
    
    results = []
    for i, filepath in enumerate(files):
        print(f"  [{i+1}/{len(files)}] Processing: {filepath.name}")
        result = analyze_single_log(filepath)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    priority_cols = [
        'filename', 'dataset', 'algorithm', 'test_env',
        'best_acc', 'avg_final_test_in', 'avg_best_trainout_testin', 'avg_best_troutloss_testin',
        'std_final_test_in',
        'lr', 'batch_size', 'n_steps', 'seed',
    ]
    
    # Put priority columns first, then the rest
    existing_priority = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[existing_priority + other_cols]
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze .out log files and generate a CSV report with metrics and hyperparameters.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all .out files in current directory
  python analyze_logs.py .
  
  # Analyze specific folder and save to custom CSV
  python analyze_logs.py /path/to/logs --output my_results.csv
  
  # Analyze recursively
  python analyze_logs.py /path/to/logs --recursive
  
  # Analyze only files matching a pattern
  python analyze_logs.py /path/to/logs --pattern "output_*.out"
"""
    )
    
    parser.add_argument('folder', type=str, nargs='?', default='.',
                       help='Folder containing .out log files (default: current directory)')
    parser.add_argument('--output', '-o', type=str, default='log_analysis_results.csv',
                       help='Output CSV file path (default: log_analysis_results.csv)')
    parser.add_argument('--pattern', '-p', type=str, default='*.out',
                       help='File pattern to match (default: *.out)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search recursively in subdirectories')
    parser.add_argument('--sort-by', '-s', type=str, default='avg_best_trainout_testin',
                       help='Column to sort results by (default: avg_best_trainout_testin)')
    parser.add_argument('--descending', '-d', action='store_true', default=True,
                       help='Sort in descending order (default: True)')
    parser.add_argument('--show-top', '-t', type=int, default=10,
                       help='Show top N results in console (default: 10)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("LOG ANALYSIS TOOL")
    print(f"{'='*80}")
    print(f"Folder: {args.folder}")
    print(f"Pattern: {args.pattern}")
    print(f"Recursive: {args.recursive}")
    print(f"Output: {args.output}")
    print(f"{'='*80}\n")
    
    # Analyze folder
    df = analyze_folder(args.folder, args.pattern, args.recursive)
    
    if df.empty:
        print("No results to save.")
        return
    
    # Sort results
    if args.sort_by in df.columns:
        df = df.sort_values(by=args.sort_by, ascending=not args.descending, na_position='last')
    
    # Save to CSV
    df.to_csv(args.output, index=False)
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
    
    display_cols = ['filename']
    for col in ['dataset', 'algorithm', 'best_acc', 'avg_best_trainout_testin', 'avg_final_test_in', 'lr', 'seed']:
        if col in df.columns:
            display_cols.append(col)
    
    top_df = df.head(args.show_top)[display_cols]
    
    # Format percentages
    for col in top_df.columns:
        if 'test' in col.lower():
            top_df[col] = top_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    print(top_df.to_string(index=False))
    print(f"\n{'='*80}")
    
    # --- Custom summary: Highest best_acc per dataset and their average ---
    datasets = ['PACS', 'OfficeHome', 'VLCS', 'TerraIncognita', 'DomainNet']
    best_accs = []
    print(f"\n{'='*80}")
    print("HIGHEST best_acc PER DATASET")
    print(f"{'='*80}")
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        if not ds_df.empty and 'best_acc' in ds_df.columns:
            # For DomainNet, handle possible missing/NaN best_acc
            ds_df = ds_df[pd.notnull(ds_df['best_acc'])]
            if not ds_df.empty:
                max_acc = ds_df['best_acc'].max()
                best_accs.append(max_acc)
                print(f"{ds:15s}: {max_acc:.2f}%")
            else:
                print(f"{ds:15s}: N/A")
        else:
            print(f"{ds:15s}: N/A")
    if best_accs:
        avg_acc = np.mean(best_accs)
        print(f"{'-'*80}")
        print(f"Average of bests: {avg_acc:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
