"""
Quick verification that multi-solver comparison framework is ready to run.

Checks:
1. All required CSV files exist
2. Data can be loaded successfully
3. Comparison cells can execute
"""

import os
import sys
import pandas as pd
import numpy as np

# Make sure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("="*80)
print("MULTI-SOLVER COMPARISON FRAMEWORK - VERIFICATION")
print("="*80)

# Expected solvers
SOLVERS_TO_RUN = ['classic', 'QAOA', 'QAOA_shots', 'SamplerVQE']

# Check 1: File existence
print("\n1. Checking for required CSV files...")
print("-"*80)

all_files_exist = True
for solver in SOLVERS_TO_RUN:
    roi_file = f"tutorial_results/{solver}_roi_optimized_portfolio.csv"
    values_file = f"tutorial_results/{solver}_values_optimized_portfolio.csv"

    roi_exists = os.path.exists(roi_file)
    values_exists = os.path.exists(values_file)

    status = "[OK]" if (roi_exists and values_exists) else "[MISSING]"
    print(f"{status} {solver:15s}: ROI={roi_exists}, Values={values_exists}")

    if not (roi_exists and values_exists):
        all_files_exist = False

if not all_files_exist:
    print("\n[WARNING] Some solver results are missing!")
    print("   Run the notebook for each missing solver before running comparison cells.")
else:
    print("\n[SUCCESS] All solver result files found!")

# Check 2: Data loading
print("\n2. Testing data loading...")
print("-"*80)

solver_data = {}
solvers_available = []

for solver in SOLVERS_TO_RUN:
    roi_file = f"tutorial_results/{solver}_roi_optimized_portfolio.csv"
    values_file = f"tutorial_results/{solver}_values_optimized_portfolio.csv"

    if os.path.exists(roi_file) and os.path.exists(values_file):
        try:
            roi = pd.read_csv(roi_file, index_col=0, parse_dates=True).squeeze()
            values = pd.read_csv(values_file, index_col=0, parse_dates=True).squeeze()

            solver_data[solver] = {
                'roi': roi,
                'values': values
            }
            solvers_available.append(solver)

            print(f"[OK] {solver:15s}: {len(roi)} dates, ROI range [{roi.min():.2f}%, {roi.max():.2f}%]")

        except Exception as e:
            print(f"[ERROR] {solver:15s}: ERROR loading - {e}")
    else:
        print(f"[SKIP] {solver:15s}: Files not found (skipped)")

# Check 3: Basic metrics calculation test
print("\n3. Testing metrics calculation...")
print("-"*80)

if solvers_available:
    try:
        test_solver = solvers_available[0]
        test_values = solver_data[test_solver]['values']
        test_roi = solver_data[test_solver]['roi']

        # Calculate test metrics
        daily_returns = test_values.pct_change().dropna()
        total_return = test_roi.iloc[-1]
        volatility = daily_returns.std() * np.sqrt(252) * 100
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        print(f"[OK] Test calculation successful for {test_solver}:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Volatility:   {volatility:.2f}%")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")

    except Exception as e:
        print(f"[ERROR] Metrics calculation test FAILED: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[SKIP] No solvers available for testing")

# Check 4: Directory structure
print("\n4. Checking output directory...")
print("-"*80)

if os.path.exists("tutorial_results"):
    files = os.listdir("tutorial_results")
    print(f"[OK] tutorial_results/ exists with {len(files)} files")

    # Check for expected output files
    expected_outputs = [
        'solver_comparison_table.csv',
        'master_roi_comparison.png',
        'master_risk_return_analysis.png'
    ]

    existing_outputs = [f for f in expected_outputs if f in files]
    if existing_outputs:
        print(f"   Found {len(existing_outputs)} comparison output(s):")
        for f in existing_outputs:
            print(f"   - {f}")
else:
    print("[ERROR] tutorial_results/ directory not found")

# Final summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if all_files_exist and len(solvers_available) == len(SOLVERS_TO_RUN):
    print("[SUCCESS] ALL CHECKS PASSED")
    print(f"   - {len(solvers_available)}/{len(SOLVERS_TO_RUN)} solvers ready")
    print("   - Data loading successful")
    print("   - Metrics calculation working")
    print("\n[READY] Ready to run comparison cells (20, 22, 24, 26, 30)!")
elif len(solvers_available) > 0:
    print("[PARTIAL] PARTIAL SUCCESS")
    print(f"   - {len(solvers_available)}/{len(SOLVERS_TO_RUN)} solvers ready")
    print(f"   - Missing: {', '.join(set(SOLVERS_TO_RUN) - set(solvers_available))}")
    print("\n[NOTE] You can run comparison cells with available solvers,")
    print("   or run notebook for missing solvers first.")
else:
    print("[NOT READY] NOT READY")
    print("   - No solver results found")
    print("\n[TODO] Run the notebook with solver_type set to each solver first:")
    for solver in SOLVERS_TO_RUN:
        print(f"   - solver_type = '{solver}'")

print("="*80)
