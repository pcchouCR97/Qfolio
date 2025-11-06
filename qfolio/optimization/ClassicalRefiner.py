
import numpy as np
from scipy.optimize import minimize
import pandas as pd

def _classical_objective(weights, expected_returns, cov_matrix, q_risk_factor):
    """
    The classical Markowitz objective function to be minimized.
    """
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # We want to MINIMIZE the NEGATIVE of return, plus risk
    return -portfolio_return + q_risk_factor * portfolio_risk

def refine_portfolio(qaoa_result: dict,
                     latest_prices: pd.Series,
                     expected_returns: pd.Series,
                     cov_matrix: np.ndarray,
                     q_risk_factor: float,
                     budget: float,
                     assets: list) -> dict:
    """
    Takes a 'chunky' integer-based portfolio solution from a quantum algorithm
    and refines it using a classical continuous-variable optimizer.

    Args:
        qaoa_result: The result dictionary from the quantum optimization step.
                     Expected format: {'allocation': {'AAPL': 10, ...}, 'value': 20450.0}
        latest_prices: A pandas Series of the latest asset prices, indexed by asset name.
        expected_returns: A pandas Series of expected returns for the assets.
        cov_matrix: The numpy covariance matrix of the assets.
        q_risk_factor: The 'q' risk aversion parameter.
        budget: The total budget for this period.
        assets: The list of asset tickers in the correct order.

    Returns:
        A dictionary containing the refined, high-precision portfolio allocation and its value.
    """
    qaoa_shares = qaoa_result.get('allocation', {})
    num_assets = len(assets)

    if not qaoa_shares or num_assets == 0:
        print("--- Classical Refiner: Initial solution has no assets. Skipping. ---")
        return {"allocation": {}, "value": budget}

    # 1. Convert the chunky QAOA solution into continuous weights for the starting point (x0)
    qaoa_value = sum(qaoa_shares.get(asset, 0) * latest_prices[asset] for asset in assets)
    
    if qaoa_value == 0:
        # Avoid division by zero if QAOA returns an empty portfolio; start with equal weights.
        initial_weights = np.full(num_assets, 1 / num_assets)
    else:
        initial_weights = np.array([(qaoa_shares.get(asset, 0) * latest_prices[asset]) / qaoa_value for asset in assets])

    print("--- Running Classical Refinement ---")
    print(f"Initial weights from QAOA: {np.round(initial_weights, 3)}")

    # 2. Define constraints (budget must sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # 3. Define bounds for each weight (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # 4. Run the classical optimizer
    refined_result = minimize(_classical_objective,
                              initial_weights,
                              args=(expected_returns.reindex(assets).values, cov_matrix, q_risk_factor),
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)

    if not refined_result.success:
        print(f"--- Classical Refiner Warning: Optimization failed. Reason: {refined_result.message} ---")
        print("--- Returning original QAOA solution. ---")
        return qaoa_result

    # 5. Convert the final continuous weights back into a share allocation
    final_weights = refined_result.x
    print(f"Refined continuous weights: {np.round(final_weights, 3)}")

    final_dollar_allocation = final_weights * budget
    
    # Floor the shares to get whole numbers, which is a common approach
    final_share_allocation = {assets[i]: int(final_dollar_allocation[i] / latest_prices[assets[i]]) for i in range(num_assets)}

    final_portfolio_value = sum(final_share_allocation[asset] * latest_prices[asset] for asset in assets)
    cash_leftover = budget - final_portfolio_value

    print(f"Refined Portfolio Value: ${final_portfolio_value:,.2f} (Cash: ${cash_leftover:,.2f})")

    return {
        "allocation": final_share_allocation,
        "value": final_portfolio_value,
        "date": qaoa_result.get("date")
    }
