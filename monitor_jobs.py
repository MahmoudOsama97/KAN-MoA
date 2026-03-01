#!/usr/bin/env python3
# filepath: /home/mosama97/scratch/MOKanA/monitor_jobs.py
"""
Monitor running jobs and provide real-time status updates
"""

import subprocess
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_running_jobs():
    """Get list of running jobs for the user"""
    try:
        result = subprocess.run(['squeue', '-u', os.environ.get('USER', 'mosama97'), 
                                '-o', '%.18i %.9P %.30j %.8T %.10M %.6D'],
                               capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return ""


def parse_latest_output(output_file, num_lines=50):
    """Parse latest lines from output file"""
    try:
        if not Path(output_file).exists():
            return None
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Get last N lines
        latest = lines[-num_lines:] if len(lines) > num_lines else lines
        
        # Extract key metrics
        metrics = {
            'last_step': None,
            'last_loss': None,
            'last_acc': None,
            'warnings': 0,
            'errors': 0
        }
        
        for line in latest:
            if 'Step' in line:
                try:
                    metrics['last_step'] = int(line.split('Step')[1].split()[0])
                except:
                    pass
            
            if 'Loss:' in line:
                try:
                    metrics['last_loss'] = float(line.split('Loss:')[1].split(',')[0].strip())
                except:
                    pass
            
            if 'acc' in line.lower():
                try:
                    # Try to extract accuracy
                    parts = line.lower().split('acc')
                    for part in parts[1:]:
                        num = ''.join(c for c in part.split(':')[-1].split()[0] if c.isdigit() or c == '.')
                        if num:
                            metrics['last_acc'] = float(num)
                            break
                except:
                    pass
            
            if 'warning' in line.lower():
                metrics['warnings'] += 1
            if 'error' in line.lower():
                metrics['errors'] += 1
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error parsing {output_file}: {e}")
        return None


def monitor_jobs(results_dir, interval=60, max_iterations=None):
    """Monitor jobs continuously"""
    iteration = 0
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 100)
        print(f"KAN-MOA JOB MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        
        # Get running jobs
        jobs_output = get_running_jobs()
        print("\nRUNNING JOBS:")
        print("-" * 100)
        print(jobs_output)
        
        # Check output files
        results_path = Path(results_dir)
        if results_path.exists():
            output_files = sorted(results_path.glob('output_*.out'))
            
            if output_files:
                print("\n" + "=" * 100)
                print("LATEST PROGRESS FROM OUTPUT FILES:")
                print("=" * 100)
                print(f"{'Job ID':<15} {'Step':<10} {'Loss':<12} {'Acc':<10} {'Warn':<8} {'Err':<8}")
                print("-" * 100)
                
                for output_file in output_files[-10:]:  # Show last 10
                    job_id = output_file.stem.replace('output_', '')
                    metrics = parse_latest_output(output_file)
                    
                    if metrics:
                        step = metrics['last_step'] or 'N/A'
                        loss = f"{metrics['last_loss']:.4f}" if metrics['last_loss'] else 'N/A'
                        acc = f"{metrics['last_acc']:.2f}%" if metrics['last_acc'] else 'N/A'
                        warn = metrics['warnings']
                        err = metrics['errors']
                        
                        print(f"{job_id:<15} {str(step):<10} {loss:<12} {acc:<10} {warn:<8} {err:<8}")
        
        print("\n" + "=" * 100)
        print(f"Press Ctrl+C to exit. Refreshing every {interval} seconds...")
        print("=" * 100)
        
        iteration += 1
        if max_iterations and iteration >= max_iterations:
            break
        
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break


def main():
    parser = argparse.ArgumentParser(description='Monitor KAN-MOA jobs')
    parser.add_argument('--results_dir', type=str,
                       default='/home/mosama97/scratch/MOKanA/quick_tests',
                       help='Directory to monitor for results')
    parser.add_argument('--interval', type=int, default=60,
                       help='Refresh interval in seconds')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Maximum number of iterations (None for infinite)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting job monitor for {args.results_dir}")
    monitor_jobs(args.results_dir, args.interval, args.iterations)


if __name__ == '__main__':
    main()
