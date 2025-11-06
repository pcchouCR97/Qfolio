"""
Script to configure the 4 backtest files with correct settings.
"""

import re

def update_file(filepath, replacements):
    """
    Update a file with multiple string replacements.

    Parameters:
    -----------
    filepath : str
        Path to file to update
    replacements : list of tuples
        List of (old_pattern, new_pattern) tuples
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for old, new in replacements:
        content = content.replace(old, new)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Updated: {filepath}")

# File 1: AMSP_NumPy.py (already correct - regular + classic)
print("\n1. AMSP_NumPy.py - Pure NumPy, No Refinement")
update_file('AMSP_NumPy.py', [
    ("Quantum Portfolio Optimization - Academic Research Version",
     "Pure NumPy Portfolio Optimization - No Refinement"),
    ("solver_type = 'classic' # classic, QAOA, QAOA_MPS, QAOA_SPSA",
     "solver_type = 'classic'  # Pure NumPy solver (no refinement)")
])

# File 2: AMSP_QAOA.py (needs QAOA solver, regular method)
print("\n2. AMSP_QAOA.py - Pure QAOA, No Refinement")
update_file('AMSP_QAOA.py', [
    ("Quantum Portfolio Optimization - Academic Research Version",
     "Pure QAOA Portfolio Optimization - No Refinement"),
    ("solver_type = 'classic' # classic, QAOA, QAOA_MPS, QAOA_SPSA",
     "solver_type = 'QAOA'  # Pure QAOA solver (no refinement)")
])

# File 3: AMSP_NumPy_GradientDescent.py (needs classic solver, hybrid method)
print("\n3. AMSP_NumPy_GradientDescent.py - NumPy + SLSQP Refinement")
update_file('AMSP_NumPy_GradientDescent.py', [
    ("Hybrid Quantum-Classical Portfolio Optimization - Academic Research Version",
     "Hybrid NumPy + Gradient Descent Portfolio Optimization"),
    ("solver_type = 'QAOA'  # Use QAOA for Stage 1 quantum optimization (same λ/H as classical for fair comparison)",
     "solver_type = 'classic'  # Use NumPy for Stage 1, then SLSQP refinement")
])

# File 4: AMSP_QAOA_GradientDescent.py (already correct - hybrid + QAOA)
print("\n4. AMSP_QAOA_GradientDescent.py - QAOA + SLSQP Refinement")
print("[OK] Already configured correctly")

print("\n" + "="*70)
print("CONFIGURATION COMPLETE!")
print("="*70)
print("\nFile Summary:")
print("  1. AMSP_NumPy.py:                  regular + classic")
print("  2. AMSP_QAOA.py:                   regular + QAOA")
print("  3. AMSP_NumPy_GradientDescent.py:  hybrid + classic")
print("  4. AMSP_QAOA_GradientDescent.py:   hybrid + QAOA")
print("\nExpected result filenames:")
print("  results_regular_classic_*.json")
print("  results_regular_QAOA_*.json")
print("  results_hybrid_classic_*.json")
print("  results_hybrid_QAOA_*.json")
print("\n")
