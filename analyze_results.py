#!/usr/bin/env python3
# filepath: /home/mosama97/scratch/MOKanA/analyze_results.py
"""
Analyze and compare results from different KAN-MOA experiments
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_output_file(output_path):
    """Parse output file to extract metrics"""
    metrics = {
        'best_val_acc': None,
        'final_test_acc': None,
        'training_time': None,
        'losses': {},
        'errors': []
    }
    
    try:
        with open(output_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Extract validation accuracy
            if 'Val acc' in line or 'validation accuracy' in line.lower():
                try:
                    acc = float(line.split(':')[-1].strip().rstrip('%'))
                    if metrics['best_val_acc'] is None or acc > metrics['best_val_acc']:
                        metrics['best_val_acc'] = acc
                except:
                    pass
            
            # Extract test accuracy
            if 'Test acc' in line or 'test accuracy' in line.lower():
                try:
                    acc = float(line.split(':')[-1].strip().rstrip('%'))
                    metrics['final_test_acc'] = acc
                except:
                    pass
            
            # Extract loss components
            for loss_name in ['Loss', 'Cls', 'Lip', 'Ortho', 'Comp', 'LoadBal', 'GridEnt', 'DomainRoute']:
                if f'{loss_name}:' in line:
                    try:
                        val = float(line.split(f'{loss_name}:')[1].split(',')[0].strip())
                        if loss_name not in metrics['losses']:
                            metrics['losses'][loss_name] = []
                        metrics['losses'][loss_name].append(val)
                    except:
                        pass
        
        # Average losses
        for key in metrics['losses']:
            if metrics['losses'][key]:
                metrics['losses'][f'{key}_mean'] = np.mean(metrics['losses'][key])
                metrics['losses'][f'{key}_std'] = np.std(metrics['losses'][key])
                metrics['losses'][f'{key}_final'] = metrics['losses'][key][-1]
    
    except Exception as e:
        logger.error(f"Error parsing {output_path}: {e}")
        metrics['errors'].append(str(e))
    
    return metrics


def parse_error_file(error_path):
    """Parse error file to detect issues"""
    issues = []
    
    try:
        with open(error_path, 'r') as f:
            content = f.read()
        
        # Check for common errors
        if 'CUDA out of memory' in content:
            issues.append('OOM_ERROR')
        if 'RuntimeError' in content:
            issues.append('RUNTIME_ERROR')
        if 'nan' in content.lower() or 'inf' in content.lower():
            issues.append('NAN_INF_ERROR')
        if 'UserWarning' in content and len(content) < 1000:
            issues.append('WARNING_ONLY')
    
    except Exception as e:
        logger.error(f"Error reading {error_path}: {e}")
    
    return issues


def collect_results(results_dir, pattern='output_*.out'):
    """Collect results from multiple experiments"""
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    for output_file in results_path.glob(pattern):
        exp_name = output_file.stem
        job_id = exp_name.replace('output_', '')
        
        logger.info(f"Processing {exp_name}...")
        
        # Parse output
        metrics = parse_output_file(output_file)
        
        # Parse error file
        error_file = output_file.parent / f"error_{job_id}.err"
        issues = parse_error_file(error_file) if error_file.exists() else []
        
        result = {
            'job_id': job_id,
            'exp_name': exp_name,
            'best_val_acc': metrics['best_val_acc'],
            'final_test_acc': metrics['final_test_acc'],
            'issues': ','.join(issues) if issues else 'NONE',
            'has_errors': len(metrics['errors']) > 0,
        }
        
        # Add loss metrics
        for key, value in metrics['losses'].items():
            if '_mean' in key or '_std' in key or '_final' in key:
                result[key] = value
        
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_experiments(df):
    """Analyze and compare experiments"""
    if df.empty:
        logger.warning("No results to analyze")
        return
    
    print("\n" + "="*80)
    print("EXPERIMENT ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal Experiments: {len(df)}")
    print(f"Successful: {len(df[df['issues'] == 'NONE'])}")
    print(f"With Issues: {len(df[df['issues'] != 'NONE'])}")
    
    # Best results
    if 'final_test_acc' in df.columns:
        best_idx = df['final_test_acc'].idxmax()
        if pd.notna(best_idx):
            best = df.loc[best_idx]
            print(f"\n{'='*80}")
            print("BEST PERFORMING EXPERIMENT")
            print(f"{'='*80}")
            print(f"Job ID: {best['job_id']}")
            print(f"Test Accuracy: {best['final_test_acc']:.2f}%")
            if 'best_val_acc' in best:
                print(f"Best Val Accuracy: {best['best_val_acc']:.2f}%")
    
    # Top 5 experiments
    print(f"\n{'='*80}")
    print("TOP 5 EXPERIMENTS")
    print(f"{'='*80}")
    
    top5 = df.nlargest(5, 'final_test_acc')[['job_id', 'final_test_acc', 'best_val_acc', 'issues']]
    print(top5.to_string(index=False))
    
    # Loss analysis
    loss_cols = [col for col in df.columns if '_mean' in col]
    if loss_cols:
        print(f"\n{'='*80}")
        print("AVERAGE LOSS COMPONENTS")
        print(f"{'='*80}")
        for col in loss_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"{col:20s}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Issues summary
    if 'issues' in df.columns:
        print(f"\n{'='*80}")
        print("ISSUES SUMMARY")
        print(f"{'='*80}")
        issue_counts = df['issues'].value_counts()
        print(issue_counts)
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze KAN-MOA experiment results')
    parser.add_argument('--results_dir', type=str, 
                       default='/home/mosama97/scratch/MOKanA/quick_tests',
                       help='Directory containing result files')
    parser.add_argument('--sweep_dir', type=str,
                       default='/home/mosama97/scratch/MOKanA/sweep_jobs',
                       help='Directory containing sweep results')
    parser.add_argument('--output', type=str, default='results_analysis.csv',
                       help='Output CSV file for results')
    parser.add_argument('--mode', choices=['quick', 'sweep', 'both'], default='both',
                       help='Which results to analyze')
    
    args = parser.parse_args()
    
    all_results = []
    
    # Analyze quick tests
    if args.mode in ['quick', 'both']:
        logger.info("Analyzing quick test results...")
        quick_df = collect_results(args.results_dir)
        if not quick_df.empty:
            quick_df['experiment_type'] = 'quick_test'
            all_results.append(quick_df)
            print(f"\n{'#'*80}")
            print("QUICK TEST RESULTS")
            print(f"{'#'*80}")
            analyze_experiments(quick_df)
    
    # Analyze sweep results
    if args.mode in ['sweep', 'both']:
        logger.info("Analyzing sweep results...")
        sweep_df = collect_results(args.sweep_dir)
        if not sweep_df.empty:
            sweep_df['experiment_type'] = 'sweep'
            all_results.append(sweep_df)
            print(f"\n{'#'*80}")
            print("SWEEP RESULTS")
            print(f"{'#'*80}")
            analyze_experiments(sweep_df)
    
    # Combined analysis
    if len(all_results) > 0:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
        print(f"\n{'#'*80}")
        print("COMBINED ANALYSIS")
        print(f"{'#'*80}")
        analyze_experiments(combined_df)
    else:
        logger.warning("No results found to analyze")


if __name__ == '__main__':
    main()
