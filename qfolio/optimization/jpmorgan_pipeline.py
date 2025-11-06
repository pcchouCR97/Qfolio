"""
JPMorgan Quantum Portfolio Optimization Pipeline

This module implements the complete JPMorgan pipeline:
1. RMT preprocessing (Marchenko-Pastur denoising)
2. Spectral clustering for community detection
3. Per-community quantum optimization with QAOA
4. Solution aggregation

Based on: "JPMorgan decomposition" research paper approach
"""

import numpy as np
import pandas as pd
from qfolio.optimization.hamiltonian import PortfolioOptimizationHamiltonian
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def rmt_preprocessing(correlation: np.ndarray) -> np.ndarray:
    """
    Apply RMT (Random Matrix Theory) preprocessing using Marchenko-Pastur law.

    Removes noise from correlation matrix by filtering eigenvalues below
    the Marchenko-Pastur threshold.

    Args:
        correlation: Correlation matrix (N x N)

    Returns:
        Denoised correlation matrix C_star

    Reference:
        Marchenko-Pastur distribution for random matrix eigenvalues
    """
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marchenko-Pastur threshold
    beta = 1.0  # Ratio of observations to variables
    sigma = 1.0  # Noise level
    lambda_plus = sigma**2 * (1 + np.sqrt(beta))**2

    signal_mask = eigenvalues > lambda_plus
    num_signal = np.sum(signal_mask)

    # Keep signal eigenvalues (excluding first - represents market mode)
    if num_signal > 1:
        signal_eigenvalues = eigenvalues[signal_mask][1:]
        signal_eigenvectors = eigenvectors[:, signal_mask][:, 1:]
    else:
        signal_eigenvalues = eigenvalues[signal_mask]
        signal_eigenvectors = eigenvectors[:, signal_mask]

    # Reconstruct cleaned correlation matrix
    C_star = np.zeros_like(correlation)
    for eval, evec in zip(signal_eigenvalues, signal_eigenvectors.T):
        C_star += eval * np.outer(evec, evec)

    return C_star


def spectral_clustering(C_star: np.ndarray, num_assets: int, max_size: int) -> list:
    """
    Perform spectral clustering using Newman's bisection algorithm.

    Recursively splits assets into communities until all communities
    are at most max_size.

    Args:
        C_star: Cleaned correlation matrix from RMT preprocessing
        num_assets: Number of assets
        max_size: Maximum community size

    Returns:
        List of communities, where each community is a list of asset indices

    Reference:
        Newman's spectral clustering algorithm for community detection
    """
    def bisect_community(corr_matrix, community):
        """Bisect a community using leading eigenvector."""
        if len(community) <= 1:
            return [], []

        indices = np.array(community)
        C_sub = corr_matrix[np.ix_(indices, indices)]

        try:
            eigenvalues_sub, eigenvectors_sub = np.linalg.eigh(C_sub)
            leading_eigvec = eigenvectors_sub[:, -1]
        except:
            return [], []

        pos_mask = leading_eigvec >= 0
        neg_mask = leading_eigvec < 0

        if not np.any(pos_mask) or not np.any(neg_mask):
            return [], []

        subcomm1 = [community[i] for i in range(len(community)) if pos_mask[i]]
        subcomm2 = [community[i] for i in range(len(community)) if neg_mask[i]]

        return subcomm1, subcomm2

    communities = []
    pending = [list(range(num_assets))]
    iteration = 0
    max_iterations = 1000

    while pending and iteration < max_iterations:
        iteration += 1
        current_comm = pending.pop(0)

        if len(current_comm) <= max_size:
            communities.append(current_comm)
            continue

        sub1, sub2 = bisect_community(C_star, current_comm)

        if sub1 and sub2:
            pending.append(sub1)
            pending.append(sub2)
        else:
            # Sequential fallback if bisection fails
            for i in range(0, len(current_comm), max_size):
                sub_comm = current_comm[i:min(i+max_size, len(current_comm))]
                if len(sub_comm) <= max_size:
                    communities.append(sub_comm)
                else:
                    pending.append(sub_comm)

    return communities


def run_jpmorgan_pipeline(
    prices_data: pd.DataFrame,
    tickers: list,
    budget: float,
    current_prices: np.ndarray,
    quantum_solver: str = 'classical',
    max_community_size: int = 3,
    max_binary_bits: int = 7,
    k: int = 2,
    q_original: float = 0.001,
    lambda1: float = 1E12
) -> np.ndarray:
    """
    Run complete JPMorgan quantum portfolio optimization pipeline.

    Pipeline steps:
    1. Calculate returns and covariance
    2. RMT preprocessing (denoise correlation matrix)
    3. Spectral clustering (split into communities)
    4. Risk rebalancing (adjust q parameter)
    5. Per-community QAOA optimization
    6. Aggregate solutions

    Args:
        prices_data: DataFrame of historical prices
        tickers: List of asset tickers
        budget: Total budget for optimization
        current_prices: Current prices for each asset
        quantum_solver: Solver type ('classic', 'QAOA_shots', or 'QAOA_MPS_optimized')
                        'classic' uses NumPyMinimumEigensolver (exact classical solver)
        max_community_size: Maximum assets per community (default: 3)
        max_binary_bits: Number of bits for AMSP encoding (default: 7)
        k: Base for encoding (default: 2)
        q_original: Original risk aversion parameter (default: 0.001)
        lambda1: Budget constraint penalty (default: 1E12)

    Returns:
        Array of portfolio weights (normalized)

    Example:
        >>> prices_df = pd.DataFrame({...})
        >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
        >>> weights = run_jpmorgan_pipeline(
        ...     prices_df, tickers, budget=10000.0,
        ...     current_prices=np.array([150.0, 300.0, 2500.0])
        ... )
        >>> print(f"Weights: {weights}")
    """
    num_assets = len(tickers)

    # Step 1: Calculate returns and statistics
    returns = prices_data.pct_change(fill_method=None).dropna()
    expected_returns = returns.mean().values
    covariances = returns.cov().values

    # Step 2: RMT preprocessing
    std_devs = np.sqrt(np.diag(covariances))
    correlation = covariances / np.outer(std_devs, std_devs)
    C_star = rmt_preprocessing(correlation)

    # Step 3: Spectral clustering
    communities = spectral_clustering(C_star, num_assets, max_community_size)

    # Step 4: Risk rebalancing (Equation 15 from JPMorgan paper)
    mu_norm_sq = np.sum(expected_returns**2)
    sigma_frobenius = np.linalg.norm(covariances, 'fro')

    numerator = 0
    denominator = 0
    for comm in communities:
        mu_k = expected_returns[comm]
        sigma_k = covariances[np.ix_(comm, comm)]
        numerator += np.sum(mu_k**2)
        denominator += np.linalg.norm(sigma_k, 'fro')

    risk_rebalance_factor = (numerator / mu_norm_sq) / (denominator / sigma_frobenius)
    q_rebalanced = q_original * risk_rebalance_factor

    # Step 5: Solve each community with selected solver
    budget_per_community = budget / len(communities)
    max_shares = 2**max_binary_bits - 1
    amsp_per_community = budget_per_community / max_shares

    # Setup solver based on quantum_solver parameter
    if quantum_solver == 'classic':
        # Classical exact solver using NumPyMinimumEigensolver
        classical_algorithm = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(classical_algorithm)
    else:
        # QAOA-based quantum solver
        qaoa_optimizer = COBYLA(maxiter=100)
        sampler = Sampler()
        sampler.set_options(shots=512)
        qaoa_sampler = QAOA(sampler=sampler, optimizer=qaoa_optimizer, reps=1)
        optimizer = MinimumEigenOptimizer(qaoa_sampler)

    all_shares = np.zeros(num_assets)

    print(f"\n  JPMorgan Pipeline: {len(communities)} communities detected")
    for comm_idx, comm in enumerate(communities):
        print(f"    Community {comm_idx+1}/{len(communities)}: {len(comm)} assets")

        comm_returns = expected_returns[comm]
        comm_covariances = covariances[np.ix_(comm, comm)]
        comm_prices = current_prices[comm]

        # Create Hamiltonian for this community
        portfolio_opt = PortfolioOptimizationHamiltonian(
            expected_returns=comm_returns,
            covariances=comm_covariances,
            q=q_rebalanced,
            budget=budget_per_community,
            lambda1=lambda1,
            prices=comm_prices,
            max_binary_bits=max_binary_bits
        )

        qp = portfolio_opt.to_quadratic_program(k=k)

        try:
            result = optimizer.solve(qp)

            # Decode AMSP lots
            comm_amsp_lots = []
            for j in range(len(comm)):
                lot_count = 0
                for b in range(max_binary_bits):
                    var_idx = j * max_binary_bits + b
                    if result.x[var_idx] > 0.5:
                        lot_count += k**b
                comm_amsp_lots.append(lot_count)

            # Convert to real shares
            for j in range(len(comm)):
                if comm_amsp_lots[j] > 0:
                    real_shares = comm_amsp_lots[j] * (amsp_per_community / comm_prices[j])
                    all_shares[comm[j]] = real_shares

            print(f"      Optimized {np.sum(np.array(comm_amsp_lots) > 0)} assets")
        except Exception as e:
            print(f"      Warning: Community optimization failed: {e}")
            continue

    # Step 6: Return normalized weights
    portfolio_values = all_shares * current_prices
    total_value = np.sum(portfolio_values)

    if total_value > 0:
        weights = portfolio_values / total_value
    else:
        weights = np.zeros(num_assets)

    return weights


if __name__ == "__main__":
    # Test the pipeline
    print("JPMorgan Pipeline Test")
    print("=" * 80)
    print("Module loaded successfully!")
    print("\nAvailable functions:")
    print("  - rmt_preprocessing()")
    print("  - spectral_clustering()")
    print("  - run_jpmorgan_pipeline()")
