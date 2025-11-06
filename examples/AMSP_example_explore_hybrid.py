"""
Parameter Space Exploration for Hybrid Quantum-Classical Portfolio Optimization

This script performs a comprehensive grid search over:
- Risk aversion (q): [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
- Budget levels: [1k, 5k, 10k, 25k, 50k, 100k, 250k, 500k]

Total: 7 × 8 = 56 backtests

Purpose: Find optimal (q, Budget) configurations independently for the hybrid method
before comparing with regular method.

Runtime: ~18-20 hours (20-25 min per backtest)

Usage:
    python AMSP_explore_hybrid.py

Output:
    results/results_hybrid_q*.json (56 files)
    logs/exploration_hybrid_log.csv

"""

import subprocess
import time
import pandas as pd
import os
from datetime import datetime

# ===============================================================================
# GRID SEARCH CONFIGURATION
# ===============================================================================

Q_VALUES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]  # 7 risk aversion levels
BUDGETS = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]  # 8 capital levels

# Fixed parameters (consistent across all runs)
LAMBDA1 = 1e12
K = 2
H_SCALE = 1e5
SOLVER_TYPE = 'classic' # classic or QAOA, dont foret cahnge here too: AMSP_test_bed_CVAR_hybrid.py Otherwise the script will still run the solver type in AMSP_test_bed_CVAR_hybrid.py.
SHARPE_N = 3
REBALANCE_FREQ = '63B'
TRAINING_LOOKBACK = 120
RISK_LOOKBACK = 120
NEW_INVEST = 0
SIM_START = '2020-07-01'
SIM_END = '2025-08-01'

# Script configuration
BASE_SCRIPT = 'AMSP_test_bed_CVAR_hybrid.py'
TIMEOUT_MINUTES = 30
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def modify_script_for_run(q, budget):
    """
    Create a temporary version of the script with specific q and budget values.

    Parameters:
    -----------
    q : float
        Risk aversion parameter
    budget : float
        Initial budget

    Returns:
    --------
    str : Path to temporary script
    """
    # Read original script
    with open(BASE_SCRIPT, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Modify parameters
    modified_lines = []
    for line in lines:
        # Replace q parameter
        if line.strip().startswith('q =') and not line.strip().startswith('q_risk'):
            modified_lines.append(f'q = {q}\n')
        # Replace initial_budget parameter
        elif line.strip().startswith('initial_budget ='):
            modified_lines.append(f'initial_budget = {int(budget)}\n')
        # Keep all other lines unchanged
        else:
            modified_lines.append(line)

    # Write temporary script
    temp_script = BASE_SCRIPT.replace('.py', f'_temp_q{q:.0e}_B{int(budget)}.py')
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)

    return temp_script


def run_single_backtest(q, budget, run_number, total_runs):
    """
    Run a single backtest with specified parameters.

    Parameters:
    -----------
    q : float
        Risk aversion parameter
    budget : float
        Initial budget
    run_number : int
        Current run number (for progress tracking)
    total_runs : int
        Total number of runs

    Returns:
    --------
    dict : Result summary with success status and duration
    """
    print(f"\n{'='*90}")
    print(f"RUN {run_number}/{total_runs}: HYBRID METHOD")
    print(f"Parameters: q={q:.0e}, Budget=${budget:,}")
    print(f"{'='*90}")

    # Create temporary script
    temp_script = modify_script_for_run(q, budget)

    # Track execution time
    start_time = time.time()

    try:
        # Run backtest using same Python interpreter as current process
        # This ensures we use the virtual environment if active
        import sys
        python_executable = sys.executable

        result = subprocess.run(
            [python_executable, temp_script],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_MINUTES * 60
        )

        duration_sec = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[OK] SUCCESS: Completed in {duration_sec/60:.1f} minutes")
            success = True
            error_msg = None
        else:
            print(f"\n[ERROR] FAILED: Exit code {result.returncode}")
            print(f"Error output:\n{result.stderr[-500:]}")  # Last 500 chars
            success = False
            error_msg = result.stderr[-200:]

    except subprocess.TimeoutExpired:
        duration_sec = TIMEOUT_MINUTES * 60
        print(f"\n[ERROR] TIMEOUT: Exceeded {TIMEOUT_MINUTES} minutes")
        success = False
        error_msg = f"Timeout after {TIMEOUT_MINUTES} minutes"

    except Exception as e:
        duration_sec = time.time() - start_time
        print(f"\n[ERROR] EXCEPTION: {str(e)}")
        success = False
        error_msg = str(e)

    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            try:
                os.remove(temp_script)
            except:
                pass  # Ignore cleanup errors

    # Return result summary
    return {
        'run_number': run_number,
        'method': 'hybrid',
        'q': q,
        'budget': budget,
        'success': success,
        'duration_sec': duration_sec,
        'duration_min': duration_sec / 60,
        'error_msg': error_msg,
        'timestamp': datetime.now().isoformat()
    }


def save_log(log_entries, log_path):
    """
    Save exploration log to CSV.

    Parameters:
    -----------
    log_entries : list
        List of dictionaries with run results
    log_path : str
        Path to save log CSV
    """
    df = pd.DataFrame(log_entries)
    df.to_csv(log_path, index=False)
    print(f"  Log updated: {log_path}")


def print_summary(log_entries, total_duration_sec):
    """
    Print summary statistics at the end of exploration.

    Parameters:
    -----------
    log_entries : list
        List of run result dictionaries
    total_duration_sec : float
        Total elapsed time
    """
    df = pd.DataFrame(log_entries)

    total_runs = len(df)
    successful_runs = df['success'].sum()
    failed_runs = total_runs - successful_runs
    success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0

    avg_duration = df['duration_min'].mean()
    total_duration_hours = total_duration_sec / 3600

    print(f"\n{'='*90}")
    print(f"EXPLORATION SUMMARY: HYBRID METHOD")
    print(f"{'='*90}")
    print(f"Total Runs:        {total_runs}")
    print(f"Successful:        {successful_runs} ({success_rate:.1f}%)")
    print(f"Failed:            {failed_runs}")
    print(f"Avg Duration:      {avg_duration:.1f} minutes/run")
    print(f"Total Time:        {total_duration_hours:.2f} hours")
    print(f"Results Directory: {os.path.abspath(RESULTS_DIR)}")
    print(f"Log File:          {os.path.abspath(os.path.join(LOGS_DIR, f'exploration_hybrid_{SOLVER_TYPE}_log.csv'))}")
    print(f"{'='*90}\n")

    # Show failed runs if any
    if failed_runs > 0:
        print("Failed Runs:")
        failed_df = df[~df['success']][['q', 'budget', 'error_msg']]
        print(failed_df.to_string(index=False))
        print()


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def check_dependencies():
    """Check that all required modules are installed."""
    print("Checking dependencies...")

    required_modules = [
        'pandas',
        'numpy',
        'matplotlib',
        'pandas_market_calendars',
        'quantstats',
        'scipy',
        'seaborn'
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print(f"\n❌ ERROR: Missing required modules: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("[OK] All dependencies installed\n")
    return True


def main():
    """
    Main grid search execution.
    """
    # Check dependencies first
    if not check_dependencies():
        return

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Calculate total runs
    total_runs = len(Q_VALUES) * len(BUDGETS)

    # Print header
    print(f"\n{'='*90}")
    print(f"PARAMETER SPACE EXPLORATION: HYBRID METHOD")
    print(f"{'='*90}")
    print(f"q values:      {len(Q_VALUES)} levels: {Q_VALUES}")
    print(f"Budgets:       {len(BUDGETS)} levels: {BUDGETS}")
    print(f"Total Runs:    {total_runs}")
    print(f"Est. Time:     {total_runs * 25 / 60:.1f} hours (@ 25 min/run)")
    print(f"Start Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}\n")

    """
    # Skip confirmation for automated runs
    # Confirmation
    response = input("Start grid search? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled by user.")
        return
    """
    
    # Run grid search
    log_entries = []
    run_number = 0
    start_time = time.time()
    log_path = os.path.join(LOGS_DIR, f'exploration_hybrid_{SOLVER_TYPE}_log.csv')

    for q in Q_VALUES:
        for budget in BUDGETS:
            run_number += 1

            # Run backtest
            result = run_single_backtest(q, budget, run_number, total_runs)
            log_entries.append(result)

            # Save log after each run (for safety)
            save_log(log_entries, log_path)

            # Estimate remaining time
            elapsed_sec = time.time() - start_time
            avg_sec_per_run = elapsed_sec / run_number
            remaining_runs = total_runs - run_number
            est_remaining_hours = (remaining_runs * avg_sec_per_run) / 3600

            print(f"\nProgress: {run_number}/{total_runs} ({run_number/total_runs*100:.1f}%)")
            print(f"Estimated remaining: {est_remaining_hours:.1f} hours")

    # Print final summary
    total_duration_sec = time.time() - start_time
    print_summary(log_entries, total_duration_sec)

    # Reminder to run analysis
    print("Next step: Run analyze_exploration.py to generate plots and tables\n")


if __name__ == '__main__':
    main()
