"""
AMSP (Adaptive Multi-Scale Pricing) Encoding Utilities

This module provides utilities for encoding portfolio optimization problems
using AMSP encoding, which allows efficient binary representation of share counts.
"""

import numpy as np


def Estimate_num_bits(budget: float, prices: list) -> int:
    """
    Estimate the number of binary bits required for base-2 encoding.

    Args:
        budget: Total budget in monetary value
        prices: List of asset prices

    Returns:
        Number of bits required for binary encoding
    """
    min_price = min(prices)  # Find the lowest stock price
    num_bits = int(np.ceil(np.log2(budget / min_price)))  # Compute required bits
    return num_bits


def estimate_num_bits_basek(b: float, p: list, k: int) -> int:
    """
    Estimate the number of bits required to encode a value using base-k encoding.

    Args:
        b: The budget or maximum reference value
        p: A list of stock prices
        k: The base for encoding (typically 2)

    Returns:
        The number of bits required for base-k encoding

    Example:
        >>> prices = [100.0, 200.0, 300.0]
        >>> budget = 10000.0
        >>> estimate_num_bits_basek(budget, prices, k=2)
        7
    """
    min_price = min(p)  # Find the lowest stock price
    ratio = b / min_price  # Compute the ratio of budget to minimum price
    num_bits = int(np.ceil(np.emath.logn(k, ratio)))  # Calculate bits needed using base-k log
    return num_bits


def AMSP_estimator(prices: np.ndarray, max_binary_bits: int, budget: float, k: int = 2) -> tuple:
    """
    Calculate AMSP (Adaptive Multi-Scale Pricing) value and lot sizes.

    AMSP encoding allows efficient representation of portfolio positions by
    using a fixed lot size (AMSP) that can be purchased in integer quantities
    encoded in binary.

    Args:
        prices: Array of asset prices
        max_binary_bits: Maximum number of binary bits for encoding
        budget: Total budget available
        k: Base for encoding (default: 2 for binary)

    Returns:
        Tuple of (lots, AMSP) where:
        - lots: Array of lot sizes per asset (budget/price)
        - AMSP: Adaptive Multi-Scale Pricing value (common lot size)

    Example:
        >>> prices = np.array([100.0, 200.0, 300.0])
        >>> lots, amsp = AMSP_estimator(prices, max_binary_bits=7, budget=10000.0, k=2)
        >>> print(f"AMSP value: {amsp:.2f}")
        AMSP value: 78.74
    """
    AMSP = budget / sum(k ** i for i in range(max_binary_bits))
    print(f"AMSP value: {AMSP}")
    lots = AMSP / prices
    return lots, AMSP


if __name__ == "__main__":
    # Test the functions
    prices = np.array([100.0, 200.0, 300.0])
    budget = 10000.0
    k = 2

    # Test estimate_num_bits_basek
    num_bits = estimate_num_bits_basek(budget, prices, k)
    print(f"Estimated bits required: {num_bits}")

    # Test AMSP_estimator
    lots, amsp = AMSP_estimator(prices, max_binary_bits=num_bits, budget=budget, k=k)
    print(f"Lots per asset: {lots}")
    print(f"AMSP value: {amsp}")
