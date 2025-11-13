#!/usr/bin/env python3
"""
Run all notebooks in sequence for the Network Security Capstone project.

This script executes notebooks 01-05 sequentially, capturing outputs and
handling errors gracefully.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Notebook execution order
NOTEBOOKS = [
    "01_data_overview.ipynb",
    "02_beth_unsupervised.ipynb",
    "03_unsw_supervised.ipynb",
    "04_results_comparison.ipynb",
    "05_presentation_visuals.ipynb"
]

def run_notebook(notebook_path):
    """Execute a notebook using jupyter nbconvert."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {notebook_path.name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        # Execute notebook and save output
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=3600",  # 1 hour timeout (3600 seconds)
                # Alternatives:
                # "--ExecutePreprocessor.timeout=1800",  # 30 minutes
                # "--ExecutePreprocessor.timeout=7200",  # 2 hours (very conservative)
                # "--ExecutePreprocessor.timeout=-1",     # No timeout (DANGEROUS - not recommended)
                str(notebook_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"\nSUCCESS: {notebook_path.name} completed!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[X] ERROR: {notebook_path.name} failed!")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"\n[X] UNEXPECTED ERROR: {notebook_path.name}")
        print(f"Error: {str(e)}")
        return False

def main():
    """Main execution function."""
    notebook_dir = Path(__file__).parent
    
    print("="*80)
    print("NETWORK SECURITY CAPSTONE - AUTOMATED NOTEBOOK EXECUTION")
    print("="*80)
    print(f"\nNotebook directory: {notebook_dir}")
    print(f"Total notebooks to run: {len(NOTEBOOKS)}")
    print(f"\nExecution order:")
    for i, nb in enumerate(NOTEBOOKS, 1):
        print(f"  {i}. {nb}")
    
    # Check all notebooks exist
    print(f"\n{'='*80}")
    print("CHECKING NOTEBOOK FILES")
    print(f"{'='*80}\n")
    
    missing = []
    for nb in NOTEBOOKS:
        nb_path = notebook_dir / nb
        if nb_path.exists():
            print(f"Found: {nb}")
        else:
            print(f"[X] Missing: {nb}")
            missing.append(nb)
    
    if missing:
        print(f"\n[X] ERROR: Missing notebooks: {', '.join(missing)}")
        print("Please ensure all notebooks exist before running.")
        sys.exit(1)
    
    # Execute notebooks sequentially
    print(f"\n{'='*80}")
    print("STARTING SEQUENTIAL EXECUTION")
    print(f"{'='*80}")
    
    start_time = datetime.now()
    results = []
    
    for i, nb in enumerate(NOTEBOOKS, 1):
        nb_path = notebook_dir / nb
        print(f"\n[{i}/{len(NOTEBOOKS)}] Processing: {nb}")
        
        success = run_notebook(nb_path)
        results.append((nb, success))
        
        if not success:
            print(f"\n[WARN]  WARNING: {nb} failed!")
            response = input("Continue with remaining notebooks? (y/n): ").strip().lower()
            if response != 'y':
                print("\nExecution halted by user.")
                break
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}\n")
    
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"Total notebooks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nTotal duration: {duration}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nDetailed results:")
    for nb, success in results:
        status = "SUCCESS" if success else "[X] FAILED"
        print(f"  {status}: {nb}")
    
    if failed == 0:
        print(f"\n? All notebooks executed successfully!")
        sys.exit(0)
    else:
        print(f"\n[WARN]  {failed} notebook(s) failed. Check logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
