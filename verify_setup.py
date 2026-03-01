#!/usr/bin/env python3
# filepath: /home/mosama97/scratch/MOKanA/verify_setup.py
"""
Verify that all enhancements are properly integrated
"""

import sys
import importlib.util
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_module_import(module_path, class_name):
    """Check if a class can be imported from a module"""
    try:
        spec = importlib.util.spec_from_file_location("temp_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, class_name):
            print(f"✅ {class_name} found in {Path(module_path).name}")
            return True
        else:
            print(f"❌ {class_name} NOT FOUND in {Path(module_path).name}")
            return False
    except Exception as e:
        print(f"❌ Error importing {class_name}: {e}")
        return False

def check_method_exists(module_path, class_name, method_name):
    """Check if a method exists in a class"""
    try:
        spec = importlib.util.spec_from_file_location("temp_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            if hasattr(cls, method_name):
                print(f"  ✓ Method {method_name} exists")
                return True
            else:
                print(f"  ✗ Method {method_name} NOT FOUND")
                return False
        return False
    except Exception as e:
        print(f"  ✗ Error checking method: {e}")
        return False

def main():
    print("=" * 80)
    print("MOKanA Enhanced Setup Verification")
    print("=" * 80)
    print()
    
    all_good = True
    
    # Check script files
    print("📝 Checking Script Files:")
    print("-" * 80)
    all_good &= check_file_exists("generate_quick_tests.sh", "Quick tests generator")
    all_good &= check_file_exists("generate_sweep.sh", "Sweep generator")
    all_good &= check_file_exists("monitor_jobs.py", "Job monitor")
    all_good &= check_file_exists("analyze_results.py", "Results analyzer")
    all_good &= check_file_exists("setup_scripts.sh", "Setup script")
    all_good &= check_file_exists("RUN_ME_FIRST.sh", "Master setup script")
    print()
    
    # Check documentation
    print("📚 Checking Documentation:")
    print("-" * 80)
    all_good &= check_file_exists("README_IMPROVEMENTS.md", "Improvements README")
    all_good &= check_file_exists("ENHANCEMENT_SUMMARY.md", "Enhancement summary")
    print()
    
    # Check core files
    print("🔧 Checking Core Implementation Files:")
    print("-" * 80)
    kan_adapters = "domainbed/models/kan_adapters.py"
    algorithms = "domainbed/algorithms/algorithms.py"
    hparams = "domainbed/hparams_registry.py"
    
    all_good &= check_file_exists(kan_adapters, "KAN adapters")
    all_good &= check_file_exists(algorithms, "Algorithms")
    all_good &= check_file_exists(hparams, "Hyperparameters")
    print()
    
    # Check if MOEBase has new methods
    print("🎯 Checking Enhanced MOEBase Class:")
    print("-" * 80)
    if Path(kan_adapters).exists():
        all_good &= check_method_exists(kan_adapters, "MOEBase", "compute_load_balance_loss")
        all_good &= check_method_exists(kan_adapters, "MOEBase", "compute_grid_entropy_loss")
        all_good &= check_method_exists(kan_adapters, "MOEBase", "compute_pairwise_orthogonality_loss")
        all_good &= check_method_exists(kan_adapters, "MOEBase", "scale_kan_gradients")
    print()
    
    # Check if KAN_MOA has new methods
    print("🎯 Checking Enhanced KAN_MOA Class:")
    print("-" * 80)
    if Path(algorithms).exists():
        all_good &= check_method_exists(algorithms, "KAN_MOA", "compute_domain_routing_loss")
        all_good &= check_method_exists(algorithms, "KAN_MOA", "compute_complexity_invariance_jsd")
    print()
    
    # Check directories
    print("📂 Checking Directories:")
    print("-" * 80)
    Path("quick_tests").mkdir(exist_ok=True)
    Path("sweep_jobs").mkdir(exist_ok=True)
    all_good &= check_file_exists("quick_tests", "Quick tests directory")
    all_good &= check_file_exists("sweep_jobs", "Sweep jobs directory")
    print()
    
    # Final summary
    print("=" * 80)
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Your MOKanA enhanced setup is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run: bash RUN_ME_FIRST.sh")
        print("  2. Or directly: bash submit_quick_tests.sh")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Please review the errors above and ensure all files are properly set up.")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
