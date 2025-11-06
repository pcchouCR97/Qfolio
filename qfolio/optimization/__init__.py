"""
Optimization module for Qfolio package.

This module provides portfolio optimization tools including:
- Hamiltonian construction (AMSP encoding)
- AMSP encoding utilities
- JPMorgan decomposition pipeline
"""

from qfolio.optimization.hamiltonian import PortfolioOptimizationHamiltonian
from qfolio.optimization.amsp_encoder import (
    estimate_num_bits_basek,
    AMSP_estimator,
    Estimate_num_bits
)
from qfolio.optimization.jpmorgan_pipeline import (
    run_jpmorgan_pipeline,
    rmt_preprocessing,
    spectral_clustering
)

__all__ = [
    'PortfolioOptimizationHamiltonian',
    'estimate_num_bits_basek',
    'AMSP_estimator',
    'Estimate_num_bits',
    'run_jpmorgan_pipeline',
    'rmt_preprocessing',
    'spectral_clustering',
]
