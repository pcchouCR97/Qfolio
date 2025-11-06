"""
Portfolio Optimization Hamiltonian Construction

This module provides the core Hamiltonian formulation for portfolio optimization
using AMSP encoding and QUBO representation.
"""

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model as AdvModel
from qfolio.optimization.amsp_encoder import estimate_num_bits_basek, AMSP_estimator
import numpy as np


class PortfolioOptimizationHamiltonian(OptimizationApplication):
    """
    Portfolio optimization using AMSP-encoded QUBO formulation.

    This class constructs a Hamiltonian for portfolio optimization that balances:
    1. Risk minimization (covariance)
    2. Return maximization (expected returns)
    3. Budget constraint enforcement

    The portfolio positions are encoded using AMSP (Adaptive Multi-Scale Pricing)
    which allows efficient binary representation of share counts.

    Attributes:
        _expected_returns: Expected return per asset
        _covariances: Covariance matrix of asset returns
        _budget: Total budget in monetary value
        q: Risk aversion parameter (weight on covariance term)
        _lambda1: Penalty parameter for budget constraint
        _prices: Price per asset (will be replaced with AMSP)
        _max_binary_bits: Number of binary bits per asset for encoding
    """

    def __init__(
            self,
            expected_returns: np.ndarray,
            covariances: np.ndarray,
            q: float,
            budget: float,  # Total budget in monetary value
            lambda1: float,  # penalty parameter
            prices: np.ndarray,  # Price per asset
            max_binary_bits: int,  # Number of binary bits per asset (adjustable)
    ) -> None:
        """
        Initialize portfolio optimization Hamiltonian.

        Args:
            expected_returns: Array of expected returns per asset
            covariances: Covariance matrix of asset returns (N x N)
            q: Risk aversion parameter (higher = more risk-averse)
            budget: Total budget available for investment
            lambda1: Penalty parameter for budget constraint violation
            prices: Current price per asset
            max_binary_bits: Number of binary bits for AMSP encoding

        Example:
            >>> expected_returns = np.array([0.1, 0.2, 0.15])
            >>> covariances = np.array([[0.1, 0.02, 0.01],
            ...                          [0.02, 0.08, 0.03],
            ...                          [0.01, 0.03, 0.09]])
            >>> prices = np.array([50.0, 300.0, 500.0])
            >>> opt = PortfolioOptimizationHamiltonian(
            ...     expected_returns=expected_returns,
            ...     covariances=covariances,
            ...     q=0.001,
            ...     budget=10000.0,
            ...     lambda1=1e12,
            ...     prices=prices,
            ...     max_binary_bits=7
            ... )
        """
        self._expected_returns = expected_returns
        self._covariances = covariances
        self._budget = budget
        self.q = q
        self._lambda1 = lambda1
        self._prices = prices
        self._max_binary_bits = max_binary_bits  # Define number of bits for binary encoding

    def to_quadratic_program(self, k: int = 2, H_scale: float = 1.0) -> QuadraticProgram:
        """
        Convert portfolio optimization to Qiskit QuadraticProgram (QUBO).

        Constructs the Hamiltonian:
        H = q * (risk) + lambda1 * (budget_penalty) - (return)

        Where:
        - risk = sum_ij cov[i,j] * price[i] * shares[i] * price[j] * shares[j]
        - budget_penalty = (sum_i shares[i] * price[i] - budget)^2
        - return = sum_i expected_return[i] * shares[i]

        Args:
            k: Base for encoding (default: 2 for binary)
            H_scale: Scaling factor for Hamiltonian (default: 1.0)

        Returns:
            QuadraticProgram ready for quantum or classical optimization

        Note:
            This method modifies self._prices to use AMSP encoding internally.
        """
        num_assets = len(self._expected_returns)
        mdl = AdvModel(name=f"Portfolio Optimization ")

        # Binary decision variables for encoding the number of shares
        x = {
            (i, j): mdl.binary_var(name=f"x_{i}_{j}")
            for i in range(num_assets)
            for j in range(self._max_binary_bits)
        }

        # Convert binary representation to integer share count
        # shares[i] = sum_j (k^j * x[i,j])
        s = {
            i: mdl.sum(k**j * x[i, j] for j in range(self._max_binary_bits))
            for i in range(num_assets)
        }

        # Calculate AMSP encoding
        lots, AMSP = AMSP_estimator(np.array(self._prices),
                                    max_binary_bits=self._max_binary_bits,
                                    budget=self._budget,
                                    k=k)

        # Replace prices with AMSP (uniform lot size)
        self._prices = np.full(len(self._expected_returns), AMSP)

        # H1: Quadratic risk term using covariance matrix
        h_1 = mdl.sum(
            self._covariances[i, j] * self._prices[i] * s[i] * self._prices[j] * s[j]
            for i in range(num_assets)
            for j in range(num_assets)
        )

        # H2: Budget constraint penalty (soft constraint)
        h_2 = (mdl.sum(s[i] * self._prices[i] for i in range(num_assets)) - self._budget) ** 2

        # H3: Return term (to maximize, so negative)
        h_3 = - mdl.sum(self._expected_returns[i] * s[i] for i in range(num_assets))

        # Final Hamiltonian function
        H = self.q * h_1 + self._lambda1 * h_2 + h_3

        # Scale H
        H = H * H_scale

        # Minimize the function
        mdl.minimize(H)

        # Convert to Qiskit's Quadratic Program
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result, k: int) -> dict:
        """
        Decode binary solution to number of AMSP lots per asset.

        Args:
            result: Optimization result with binary solution
            k: Base used for encoding (typically 2)

        Returns:
            Dictionary mapping asset index to number of AMSP lots
            Example: {0: 15, 2: 8} means asset 0 gets 15 lots, asset 2 gets 8 lots

        Example:
            >>> result = optimizer.solve(qp)
            >>> shares = hamiltonian.interpret(result, k=2)
            >>> print(shares)
            {0: 12, 1: 5, 2: 3}
        """
        num_assets = len(self._expected_returns)
        num_bits = self._max_binary_bits
        shares = {}

        for i in range(num_assets):
            # Decode binary solution to integer share count (AMSP lots)
            share_value = sum(k**j * int(result.x[i * num_bits + j]) for j in range(num_bits))
            if share_value > 0:
                shares[i] = share_value

        return shares


# Test
if __name__ == "__main__":
    # Import dependencies
    import numpy as np

    # Example data for testing
    expected_returns = np.array([0.1, 0.2, 0.15])
    covariances = np.array([[0.1, 0.02, 0.01],
                             [0.02, 0.08, 0.03],
                             [0.01, 0.03, 0.09]])
    prices = np.array([50, 300, 500])
    budget = 10000
    lambda1 = 100

    prices = prices/100
    budget = budget/100

    base_k = 2
    num_bits = estimate_num_bits_basek(budget, prices, k=base_k)
    max_binary_bits = num_bits

    # Instantiate class with test values
    optimizer = PortfolioOptimizationHamiltonian(
        expected_returns=expected_returns,
        covariances=covariances,
        q=0.5,
        budget=budget,
        lambda1=lambda1,
        prices=prices,
        max_binary_bits=max_binary_bits,
    )

    # Test if `to_quadratic_program()` runs correctly
    qp = optimizer.to_quadratic_program(k=base_k)

    # Print result to confirm it's working
    print("Test successful! Quadratic Program generated:")
    print(f"Number of bits: {num_bits}")
