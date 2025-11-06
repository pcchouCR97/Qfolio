"""
Standardized Results Saving Module for Quantum Portfolio Optimization

This module provides a unified interface for saving backtest results from both
regular and hybrid quantum portfolio optimization methods. Ensures consistent
format for fair comparison and reproducible research.

Usage:
    from save_results_module import save_backtest_results

    save_backtest_results(
        method='regular',
        q=1e-3,
        budget=10000,
        # ... all other parameters
    )

Author: Quantum Portfolio Optimization Research Team
Date: 2025-10-19
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

def save_backtest_results(
    # === Metadata ===
    method: str,  # 'regular' or 'hybrid'
    q: float,
    budget: float,
    lambda1: float,
    k: int,
    solver_type: str,
    sharpe_n: int,
    H_scale: float,

    # === Simulation Configuration ===
    sim_start_date: str,
    sim_end_date: str,
    rebalance_freq: str,
    training_lookback_days: int,
    risk_lookback_period: int,
    new_invest_per_period: float,
    exchange: str,
    confidence_level: float,

    # === Performance Metrics ===
    total_return_pct: float,
    cagr_pct: float,
    sharpe_ratio: float,
    sortino_ratio: float,
    calmar_ratio: float,
    treynor_ratio: float,
    information_ratio: float,
    max_drawdown_pct: float,
    volatility_pct: float,
    alpha_pct: float,
    beta: float,
    r_squared: float,
    omega_ratio: float,
    profit_factor: float,
    gain_to_pain: float,
    recovery_factor: float,
    payoff_ratio: float,
    tail_ratio: float,
    win_rate_pct: float,
    skewness: float,
    kurtosis: float,

    # === Market Capture ===
    upside_capture_pct: float,
    downside_capture_pct: float,
    capture_ratio: float,

    # === Portfolio Characteristics ===
    avg_num_assets: float,
    avg_turnover_pct: float,
    num_rebalances: int,
    avg_sector_hhi: float,
    avg_sectors_held: float,
    max_single_asset_weight_pct: float,
    avg_max_asset_weight_pct: float,

    # === Benchmark Comparison ===
    spy_total_return_pct: float,
    excess_return_pct: float,
    outperformance_pct: float,
    correlation: float,
    tracking_error_pct: float,

    # === Statistical Tests ===
    t_statistic: float,
    p_value: float,
    significance_level: str,

    # === Drawdown Analysis ===
    max_drawdown_duration_days: int,
    avg_drawdown_duration_days: float,

    # === Time Series Data (Optional) ===
    dates: Optional[List[str]] = None,
    portfolio_values: Optional[List[float]] = None,
    daily_returns: Optional[List[float]] = None,
    drawdown_pct: Optional[List[float]] = None,

    # === Hybrid-Specific (Optional) ===
    avg_refinement_improvement_pct: Optional[float] = None,
    refinement_successes: Optional[int] = None,
    refinement_failures: Optional[int] = None,

    # === Output Configuration ===
    output_dir: str = "results",
    include_timeseries: bool = False
) -> str:
    """
    Save backtest results in standardized JSON format.

    Parameters:
    -----------
    method : str
        'regular' or 'hybrid'
    q : float
        Risk aversion parameter
    budget : float
        Initial portfolio budget
    ... (see function signature for all parameters)

    Returns:
    --------
    str : Path to saved JSON file

    Notes:
    ------
    - All percentage values should be in [0, 100] range (not [0, 1])
    - Ratios should be unitless (Sharpe, Sortino, etc.)
    - Dates should be in 'YYYY-MM-DD' format
    - Set include_timeseries=False for smaller file sizes
    """

    # Build result dictionary
    results = {
        "metadata": {
            "method": str(method),
            "q": float(q),
            "budget": float(budget),
            "lambda1": float(lambda1),
            "k": int(k),
            "solver_type": str(solver_type),
            "sharpe_n": int(sharpe_n),
            "H_scale": float(H_scale),
            "sim_start_date": str(sim_start_date),
            "sim_end_date": str(sim_end_date),
            "rebalance_freq": str(rebalance_freq),
            "training_lookback": int(training_lookback_days),
            "risk_lookback": int(risk_lookback_period),
            "new_invest_per_period": float(new_invest_per_period),
            "confidence_level": float(confidence_level),
            "exchange": str(exchange),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        },

        "performance": {
            "total_return_pct": float(total_return_pct),
            "cagr_pct": float(cagr_pct),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "treynor_ratio": float(treynor_ratio),
            "information_ratio": float(information_ratio),
            "max_drawdown_pct": float(max_drawdown_pct),
            "volatility_pct": float(volatility_pct),
            "alpha_pct": float(alpha_pct),
            "beta": float(beta),
            "r_squared": float(r_squared),
            "omega_ratio": float(omega_ratio),
            "profit_factor": float(profit_factor),
            "gain_to_pain": float(gain_to_pain),
            "recovery_factor": float(recovery_factor),
            "payoff_ratio": float(payoff_ratio),
            "tail_ratio": float(tail_ratio),
            "win_rate_pct": float(win_rate_pct),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis)
        },

        "market_capture": {
            "upside_capture_pct": float(upside_capture_pct),
            "downside_capture_pct": float(downside_capture_pct),
            "capture_ratio": float(capture_ratio)
        },

        "portfolio_characteristics": {
            "avg_num_assets": float(avg_num_assets),
            "avg_turnover_pct": float(avg_turnover_pct),
            "num_rebalances": int(num_rebalances),
            "avg_sector_hhi": float(avg_sector_hhi),
            "avg_sectors_held": float(avg_sectors_held),
            "max_single_asset_weight_pct": float(max_single_asset_weight_pct),
            "avg_max_asset_weight_pct": float(avg_max_asset_weight_pct)
        },

        "benchmark_comparison": {
            "spy_total_return_pct": float(spy_total_return_pct),
            "excess_return_pct": float(excess_return_pct),
            "outperformance_pct": float(outperformance_pct),
            "correlation": float(correlation),
            "tracking_error_pct": float(tracking_error_pct)
        },

        "statistical_tests": {
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "significance_level": str(significance_level)
        },

        "drawdown_analysis": {
            "max_drawdown_duration_days": int(max_drawdown_duration_days),
            "avg_drawdown_duration_days": float(avg_drawdown_duration_days)
        }
    }

    # Add hybrid-specific metrics if applicable
    if method == 'hybrid':
        if avg_refinement_improvement_pct is not None:
            results["hybrid_specific"] = {
                "avg_refinement_improvement_pct": float(avg_refinement_improvement_pct),
                "refinement_successes": int(refinement_successes) if refinement_successes is not None else 0,
                "refinement_failures": int(refinement_failures) if refinement_failures is not None else 0
            }

    # Optionally include time series data
    if include_timeseries and dates is not None:
        results["time_series"] = {
            "dates": dates,
            "portfolio_values": [float(v) for v in portfolio_values] if portfolio_values else [],
            "daily_returns": [float(r) for r in daily_returns] if daily_returns else [],
            "drawdown_pct": [float(dd) for dd in drawdown_pct] if drawdown_pct else []
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    filename = f"results_{method}_{solver_type}_q{q:.0e}_B{int(budget)}_lambda{lambda1:.0e}.json"
    filepath = os.path.join(output_dir, filename)

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Results saved to: {filepath}")
    return filepath


def load_results(filepath: str) -> Dict:
    """
    Load backtest results from JSON file.

    Parameters:
    -----------
    filepath : str
        Path to JSON results file

    Returns:
    --------
    dict : Loaded results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_consistency_score(
    sharpe: float,
    max_drawdown_pct: float,
    win_rate_pct: float
) -> float:
    """
    Calculate custom consistency score for parameter optimization.

    Consistency Score = Sharpe × (1 - MaxDD/100) × sqrt(WinRate/50)

    This metric balances:
    - Risk-adjusted returns (Sharpe)
    - Capital preservation (low MaxDD)
    - Reliability (high win rate)

    Parameters:
    -----------
    sharpe : float
        Sharpe ratio
    max_drawdown_pct : float
        Maximum drawdown percentage [0, 100]
    win_rate_pct : float
        Win rate percentage [0, 100]

    Returns:
    --------
    float : Consistency score (higher is better)

    Examples:
    ---------
    >>> calculate_consistency_score(2.0, 25.0, 55.0)
    1.574  # Good consistent portfolio

    >>> calculate_consistency_score(3.0, 50.0, 48.0)
    1.469  # High Sharpe but large drawdown
    """
    mdd_factor = 1 - (max_drawdown_pct / 100)
    win_factor = np.sqrt(win_rate_pct / 50)
    consistency = sharpe * mdd_factor * win_factor
    return float(consistency)


def extract_metrics_dataframe(results_dir: str = "results") -> pd.DataFrame:
    """
    Load all results from directory and create summary DataFrame.

    Parameters:
    -----------
    results_dir : str
        Directory containing JSON result files

    Returns:
    --------
    pd.DataFrame : Summary dataframe with all metrics
    """
    import glob

    files = glob.glob(os.path.join(results_dir, 'results_*.json'))

    if not files:
        print(f"Warning: No result files found in {results_dir}")
        return pd.DataFrame()

    data = []
    for filepath in files:
        result = load_results(filepath)

        # Flatten structure
        row = {
            # Metadata
            'method': result['metadata']['method'],
            'q': result['metadata']['q'],
            'budget': result['metadata']['budget'],
            'lambda1': result['metadata']['lambda1'],
            'solver_type': result['metadata']['solver_type'],
            'timestamp': result['metadata']['timestamp'],

            # Performance
            **result['performance'],

            # Portfolio characteristics
            **result['portfolio_characteristics'],

            # Benchmark comparison
            **result['benchmark_comparison'],

            # Statistical tests
            **result['statistical_tests'],

            # Market capture
            **result['market_capture'],

            # Drawdown
            **result['drawdown_analysis']
        }

        # Add hybrid-specific if available
        if 'hybrid_specific' in result:
            row.update(result['hybrid_specific'])

        # Calculate consistency score
        row['consistency_score'] = calculate_consistency_score(
            row['sharpe_ratio'],
            row['max_drawdown_pct'],
            row['win_rate_pct']
        )

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by method, q, budget
    df = df.sort_values(['method', 'q', 'budget'])

    print(f"[OK] Loaded {len(df)} backtest results")
    return df


if __name__ == '__main__':
    """
    Example usage and testing.
    """
    print("save_results_module.py - Testing mode")

    # Example: Save dummy results
    save_backtest_results(
        method='regular',
        q=1e-3,
        budget=10000,
        lambda1=1e12,
        k=2,
        solver_type='classic',
        sharpe_n=3,
        H_scale=1e5,
        sim_start_date='2020-07-01',
        sim_end_date='2025-08-01',
        rebalance_freq='63B',
        training_lookback_days=120,
        risk_lookback_period=120,
        new_invest_per_period=0,
        exchange='NYSE',
        confidence_level=0.95,
        total_return_pct=213.5,
        cagr_pct=24.3,
        sharpe_ratio=1.25,
        sortino_ratio=1.58,
        calmar_ratio=0.95,
        treynor_ratio=0.18,
        information_ratio=0.45,
        max_drawdown_pct=-25.6,
        volatility_pct=28.5,
        alpha_pct=8.5,
        beta=1.15,
        r_squared=0.82,
        omega_ratio=1.35,
        profit_factor=2.1,
        gain_to_pain=1.8,
        recovery_factor=8.3,
        payoff_ratio=1.9,
        tail_ratio=1.12,
        win_rate_pct=54.2,
        skewness=-0.15,
        kurtosis=2.8,
        upside_capture_pct=125.3,
        downside_capture_pct=85.7,
        capture_ratio=1.46,
        avg_num_assets=2.95,
        avg_turnover_pct=45.2,
        num_rebalances=20,
        avg_sector_hhi=42.3,
        avg_sectors_held=2.8,
        max_single_asset_weight_pct=65.2,
        avg_max_asset_weight_pct=58.3,
        spy_total_return_pct=85.2,
        excess_return_pct=128.25,
        outperformance_pct=60.5,
        correlation=0.78,
        tracking_error_pct=12.5,
        t_statistic=3.25,
        p_value=0.0012,
        significance_level='***',
        max_drawdown_duration_days=180,
        avg_drawdown_duration_days=45.5,
        output_dir='results_test',
        include_timeseries=False
    )

    # Example: Load and display
    df = extract_metrics_dataframe('results_test')
    print("\nLoaded DataFrame:")
    print(df[['method', 'q', 'budget', 'sharpe_ratio', 'consistency_score']].head())

    print("\nModule test complete!")
